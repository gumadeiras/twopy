"""Save workflow run metadata beside custom output files.

twopy writes this metadata after a workflow runs. Workflow authors can write the
result file and let twopy record the workflow id, version, source hash,
parameters, and recording path.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import yaml

from twopy._version import __version__
from twopy.analysis.response_plotting import EpochResponsePlotData, ResponsePlotData
from twopy.converted import RecordingData
from twopy.custom.parameters import CustomParameterValue
from twopy.custom.results import CustomLineBand, CustomLinePlot, CustomResult
from twopy.custom.workflows import CustomWorkflow

__all__ = [
    "CustomWorkflowProvenance",
    "custom_result_artifact_paths",
    "provenance_sidecar_path",
    "validate_custom_result",
    "write_result_provenance",
    "write_workflow_provenance_for_path",
    "workflow_provenance",
]

_HDF5_SUFFIXES = frozenset((".h5", ".hdf5"))


@dataclass(frozen=True)
class CustomWorkflowProvenance:
    """Workflow run metadata saved beside custom outputs.

    Args:
        workflow_id: Stable workflow id.
        workflow_name: Workflow name.
        workflow_version: Workflow version supplied by the author.
        workflow_source_path: Python source file used for this run.
        workflow_source_hash: SHA-256 hash of the workflow source file.
        twopy_version: twopy package version used for the run.
        run_started_at: UTC timestamp for the start of the run.
        parameters: Parameter values used for the run.
        recording_path: Converted recording path analyzed by the workflow.

    Returns:
        Provenance record saved as YAML or HDF5 metadata.
    """

    workflow_id: str
    workflow_name: str
    workflow_version: str
    workflow_source_path: Path
    workflow_source_hash: str
    twopy_version: str
    run_started_at: str
    parameters: Mapping[str, CustomParameterValue]
    recording_path: Path

    def as_mapping(self) -> dict[str, object]:
        """Return workflow, package, parameter, and recording metadata."""
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "workflow_version": self.workflow_version,
            "workflow_source_path": str(self.workflow_source_path),
            "workflow_source_hash": self.workflow_source_hash,
            "twopy_version": self.twopy_version,
            "run_started_at": self.run_started_at,
            "parameters": dict(self.parameters),
            "recording_path": str(self.recording_path),
        }


def workflow_provenance(
    workflow: CustomWorkflow,
    *,
    parameters: Mapping[str, CustomParameterValue],
    recording: RecordingData,
) -> CustomWorkflowProvenance:
    """Create metadata for one custom workflow run.

    Args:
        workflow: Validated workflow definition.
        parameters: Parameter values used for the run.
        recording: Active converted recording.

    Returns:
        Run metadata with current twopy version and UTC timestamp.
    """
    return CustomWorkflowProvenance(
        workflow_id=workflow.id,
        workflow_name=workflow.name,
        workflow_version=workflow.version,
        workflow_source_path=workflow.source_path,
        workflow_source_hash=workflow.source_hash,
        twopy_version=__version__,
        run_started_at=datetime.now(UTC).isoformat(),
        parameters=dict(parameters),
        recording_path=recording.path,
    )


def provenance_sidecar_path(path: Path) -> Path:
    """Return the YAML metadata path for one output file.

    Args:
        path: Output file path.

    Returns:
        Sidecar path beside the output file.
    """
    return path.with_suffix(".twopy-workflow.yml")


def validate_custom_result(
    result: object,
    *,
    output_dir: Path,
    expected_roi_shape: tuple[int, int],
) -> CustomResult:
    """Check one workflow result before the GUI uses it.

    Args:
        result: Object returned by a custom workflow.
        output_dir: Workflow output folder that must contain returned files.
        expected_roi_shape: Full-frame movie shape expected for returned ROIs.

    Returns:
        The checked ``CustomResult``.
    """
    if not isinstance(result, CustomResult):
        msg = "Custom workflow must return CustomResult."
        raise ValueError(msg)
    if result.message == "":
        msg = "CustomResult.message must not be blank."
        raise ValueError(msg)
    for path in result.files:
        _validate_output_file_path(path, output_dir=output_dir)
        _validate_existing_file(path)
    for table in result.tables:
        if table.title == "":
            msg = "CustomTable.title must not be blank."
            raise ValueError(msg)
        _validate_table_highlighted_rows(table.highlighted_rows)
        _validate_output_file_path(table.path, output_dir=output_dir)
        _validate_existing_file(table.path)
    for plot in result.plots:
        _validate_line_plot(plot)
    if (
        result.roi_set is not None
        and result.roi_set.masks.shape[1:] != expected_roi_shape
    ):
        msg = (
            "CustomResult.roi_set masks must match the full movie frame shape; "
            f"got {result.roi_set.masks.shape[1:]}, expected {expected_roi_shape}"
        )
        raise ValueError(msg)
    if result.response_plot_data is not None:
        _validate_response_plot_data(result.response_plot_data)
    _validate_custom_result_visible_rois(result.visible_roi_indices)
    return result


def _validate_table_highlighted_rows(highlighted_rows: object) -> None:
    """Check that highlighted table rows are non-negative integers."""
    if not isinstance(highlighted_rows, tuple):
        msg = "CustomTable.highlighted_rows must be a tuple of integers."
        raise ValueError(msg)
    for row_index in highlighted_rows:
        if type(row_index) is not int or row_index < 0:
            msg = "CustomTable.highlighted_rows must contain non-negative integers."
            raise ValueError(msg)


def _validate_custom_result_visible_rois(visible_roi_indices: object) -> None:
    """Check optional ROI rows selected by a workflow result."""
    if visible_roi_indices is None:
        return
    if not isinstance(visible_roi_indices, tuple):
        msg = "CustomResult.visible_roi_indices must be a tuple of integers."
        raise ValueError(msg)
    for index in visible_roi_indices:
        if type(index) is not int or index < 0:
            msg = "CustomResult.visible_roi_indices must contain non-negative integers."
            raise ValueError(msg)


def custom_result_artifact_paths(result: CustomResult) -> tuple[Path, ...]:
    """Return every output file path declared by a custom result.

    Args:
        result: Validated custom workflow result.

    Returns:
        Unique file and table paths in first-seen order.
    """
    return _unique_paths((*result.files, *(table.path for table in result.tables)))


def write_result_provenance(
    result: CustomResult,
    provenance: CustomWorkflowProvenance,
) -> tuple[Path, ...]:
    """Write run metadata for every file in a result.

    Args:
        result: Validated custom workflow result.
        provenance: Workflow metadata for the run.

    Returns:
        Paths written or updated with workflow metadata.
    """
    provenance_paths: list[Path] = []
    for path in custom_result_artifact_paths(result):
        provenance_paths.append(write_workflow_provenance_for_path(path, provenance))
    return _unique_paths(tuple(provenance_paths))


def write_workflow_provenance_for_path(
    path: Path,
    provenance: CustomWorkflowProvenance,
) -> Path:
    """Attach workflow metadata to one existing output file.

    Args:
        path: Existing output file path.
        provenance: Workflow metadata for the run.

    Returns:
        HDF5 files are updated in place. Other files get a YAML sidecar.
    """
    resolved = path.expanduser()
    if resolved.suffix.lower() in _HDF5_SUFFIXES:
        _write_hdf5_provenance(resolved, provenance)
        return resolved
    return _write_sidecar_provenance(resolved, provenance)


def _validate_output_file_path(path: Path, *, output_dir: Path) -> None:
    """Check that one output file stays below the workflow output folder."""
    output_root = output_dir.expanduser().resolve(strict=False)
    resolved = path.expanduser().resolve(strict=False)
    if _is_relative_to(resolved, output_root):
        return
    msg = (
        f"Custom workflow result files must be written below ctx.output_dir; got {path}"
    )
    raise ValueError(msg)


def _validate_existing_file(path: Path) -> None:
    """Check that one result path exists and is a file."""
    if not path.expanduser().is_file():
        msg = f"Custom workflow result file does not exist: {path}"
        raise ValueError(msg)


def _validate_line_plot(plot: CustomLinePlot) -> None:
    """Check one custom line plot for usable array shapes."""
    if plot.title == "":
        msg = "CustomLinePlot.title must not be blank."
        raise ValueError(msg)
    if plot.y_label == "":
        msg = "CustomLinePlot.y_label must not be blank."
        raise ValueError(msg)
    if plot.x.ndim != 1:
        msg = f"CustomLinePlot.x must be one-dimensional; got {plot.x.shape}"
        raise ValueError(msg)
    series_count = _validate_line_plot_y(plot)
    _validate_line_plot_colors(plot.colors, series_count=series_count)
    _validate_line_plot_bands(
        plot.bands, x_length=plot.x.shape[0], series_count=series_count
    )


def _validate_line_plot_y(plot: CustomLinePlot) -> int:
    """Check line values and return the number of plotted series."""
    if plot.y.ndim == 1:
        if plot.y.shape[0] != plot.x.shape[0]:
            msg = "CustomLinePlot 1D y length must match x length."
            raise ValueError(msg)
        if len(plot.labels) not in {0, 1}:
            msg = "CustomLinePlot 1D y can have zero or one label."
            raise ValueError(msg)
        return 1
    if plot.y.ndim == 2:
        if plot.y.shape[1] != plot.x.shape[0]:
            msg = "CustomLinePlot 2D y sample count must match x length."
            raise ValueError(msg)
        if len(plot.labels) not in {0, plot.y.shape[0]}:
            msg = "CustomLinePlot labels must match the number of y rows."
            raise ValueError(msg)
        return int(plot.y.shape[0])
    msg = f"CustomLinePlot.y must be one- or two-dimensional; got {plot.y.shape}"
    raise ValueError(msg)


def _validate_line_plot_colors(
    colors: tuple[str, ...],
    *,
    series_count: int,
) -> None:
    """Check optional custom line colors before GUI rendering."""
    if len(colors) not in {0, series_count}:
        msg = "CustomLinePlot colors must match the number of plotted series."
        raise ValueError(msg)
    for color in colors:
        if not _is_hex_color(color):
            msg = f"CustomLinePlot colors must be #RRGGBB strings; got {color!r}."
            raise ValueError(msg)


def _is_hex_color(value: str) -> bool:
    """Return whether a string is a six-digit RGB hex color."""
    if not isinstance(value, str) or len(value) != 7 or value[0] != "#":
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value[1:])


def _validate_line_plot_bands(
    bands: tuple[CustomLineBand, ...],
    *,
    x_length: int,
    series_count: int,
) -> None:
    """Check custom line-plot bands before GUI rendering."""
    for band in bands:
        if band.series_index < 0 or band.series_index >= series_count:
            msg = (
                "CustomLineBand.series_index must point at a plot series; "
                f"got {band.series_index} for {series_count} series"
            )
            raise ValueError(msg)
        if band.lower.ndim != 1 or band.upper.ndim != 1:
            msg = "CustomLineBand lower and upper values must be one-dimensional."
            raise ValueError(msg)
        if band.lower.shape[0] != x_length or band.upper.shape[0] != x_length:
            msg = "CustomLineBand lower and upper values must match x length."
            raise ValueError(msg)


def _validate_response_plot_data(plot_data: object) -> None:
    """Check returned response plot data before it reaches the plot dock."""
    if not isinstance(plot_data, ResponsePlotData):
        msg = "CustomResult.response_plot_data must be a ResponsePlotData object."
        raise ValueError(msg)
    epochs = plot_data.epochs
    if len(epochs) == 0:
        msg = "CustomResult.response_plot_data must contain at least one epoch."
        raise ValueError(msg)
    _validate_response_plot_visible_rois(
        getattr(plot_data, "visible_roi_indices", None),
        roi_count=len(getattr(epochs[0], "roi_labels", ())),
    )
    for epoch_index, epoch in enumerate(epochs):
        _validate_response_epoch(epoch, epoch_index=epoch_index)


def _validate_response_plot_visible_rois(
    visible_roi_indices: object,
    *,
    roi_count: int,
) -> None:
    """Check optional initial visible ROI indices."""
    if visible_roi_indices is None:
        return
    if not isinstance(visible_roi_indices, tuple):
        msg = "Response plot visible_roi_indices must be a tuple of integers."
        raise ValueError(msg)
    for index in visible_roi_indices:
        if type(index) is not int or index < 0 or index >= roi_count:
            msg = "Response plot visible_roi_indices contains an invalid ROI index."
            raise ValueError(msg)


def _validate_response_epoch(epoch: object, *, epoch_index: int) -> None:
    """Check one returned response-plot epoch."""
    if not isinstance(epoch, EpochResponsePlotData):
        msg = (
            f"Response plot epoch {epoch_index} must be an "
            "EpochResponsePlotData object."
        )
        raise ValueError(msg)
    roi_labels = getattr(epoch, "roi_labels", None)
    time_seconds = getattr(epoch, "time_seconds", None)
    mean_values = getattr(epoch, "mean_values", None)
    sem_values = getattr(epoch, "sem_values", None)
    if not isinstance(roi_labels, tuple) or len(roi_labels) == 0:
        msg = f"Response plot epoch {epoch_index} must contain ROI labels."
        raise ValueError(msg)
    if any(not isinstance(label, str) or label == "" for label in roi_labels):
        msg = f"Response plot epoch {epoch_index} ROI labels must be non-empty strings."
        raise ValueError(msg)
    if not isinstance(time_seconds, np.ndarray) or time_seconds.ndim != 1:
        msg = f"Response plot epoch {epoch_index} time_seconds must be 1D."
        raise ValueError(msg)
    if time_seconds.shape[0] == 0:
        msg = f"Response plot epoch {epoch_index} time_seconds must not be empty."
        raise ValueError(msg)
    if not np.all(np.isfinite(time_seconds)):
        msg = f"Response plot epoch {epoch_index} time_seconds must be finite."
        raise ValueError(msg)
    if np.any(np.diff(time_seconds) < 0):
        msg = f"Response plot epoch {epoch_index} time_seconds must be ordered."
        raise ValueError(msg)
    expected_shape = (len(roi_labels), time_seconds.shape[0])
    if not isinstance(mean_values, np.ndarray) or mean_values.shape != expected_shape:
        msg = (
            f"Response plot epoch {epoch_index} mean_values must have shape "
            f"{expected_shape}."
        )
        raise ValueError(msg)
    if not isinstance(sem_values, np.ndarray) or sem_values.shape != expected_shape:
        msg = (
            f"Response plot epoch {epoch_index} sem_values must have shape "
            f"{expected_shape}."
        )
        raise ValueError(msg)


def _write_sidecar_provenance(
    path: Path,
    provenance: CustomWorkflowProvenance,
) -> Path:
    """Write a YAML sidecar beside one non-HDF5 file."""
    sidecar = provenance_sidecar_path(path)
    with sidecar.open("w", encoding="utf-8") as sidecar_file:
        yaml.safe_dump(provenance.as_mapping(), sidecar_file, sort_keys=True)
    return sidecar


def _write_hdf5_provenance(
    path: Path,
    provenance: CustomWorkflowProvenance,
) -> None:
    """Write workflow metadata into a HDF5 file."""
    metadata = provenance.as_mapping()
    with h5py.File(path, "a") as h5_file:
        if "twopy_workflow" in h5_file:
            del h5_file["twopy_workflow"]
        group = h5_file.create_group("twopy_workflow")
        for key, value in metadata.items():
            if key != "parameters":
                group.attrs[key] = value
        parameter_group = group.create_group("parameters")
        for parameter_name, parameter_value in provenance.parameters.items():
            parameter_group.attrs[parameter_name] = parameter_value


def _unique_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Return paths in first-seen order without duplicates."""
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return tuple(unique)


def _is_relative_to(path: Path, root: Path) -> bool:
    """Return whether ``path`` is inside ``root``."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
