"""Public objects used by twopy custom workflows.

Workflow files import these plain Python objects from ``twopy.custom``. The
objects stay independent of napari so the same workflow can run from the GUI or
from tests.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from twopy._version import __version__
from twopy.analysis.dff_options import DeltaFOverFOptions
from twopy.analysis.response_processing import ResponseProcessingOptions
from twopy.analysis.responses import GroupedRoiResponses
from twopy.analysis.timing import resolve_recording_timing
from twopy.analysis.workflow import (
    AnalysisResponseComputation,
    compute_recording_responses,
)
from twopy.converted import RecordingData
from twopy.napari.plotting.data import ResponsePlotData
from twopy.roi import RoiSet, make_roi_set
from twopy.stimulus import stimulus_epoch_names_by_number

__all__ = [
    "CustomEpoch",
    "CustomLinePlot",
    "CustomParameterValue",
    "CustomResult",
    "CustomRunContext",
    "CustomTable",
    "CustomWorkflow",
    "CustomWorkflowFunction",
    "CustomWorkflowProvenance",
    "WorkflowDeclaration",
]

CustomParameterValue = str | int | float | bool | None


class CustomWorkflowFunction(Protocol):
    """Function shape accepted by custom workflow discovery."""

    def __call__(self, *args: object) -> CustomResult:
        """Run the workflow.

        Args:
            *args: Validated workflow call arguments.

        Returns:
            Workflow result.
        """
        ...


@dataclass(frozen=True)
class CustomTable:
    """CSV or TSV table shown in the Custom tab.

    Args:
        title: Short title shown above the table.
        path: CSV or TSV file written by the workflow.
        highlighted_rows: Zero-based data rows to highlight.
        display_path: Optional path shown to the user when preview reads from a
            local cache path.

    Returns:
        Immutable table descriptor.
    """

    title: str
    path: Path
    highlighted_rows: tuple[int, ...] = ()
    display_path: Path | None = None


@dataclass(frozen=True)
class CustomLinePlot:
    """Line plot shown in the Custom tab.

    Args:
        title: Short display title for the GUI.
        x: One-dimensional x-axis values.
        y: One-dimensional values or two-dimensional ``(series, samples)``
            values.
        labels: Optional series labels.
        y_label: Y-axis label shown beside the plot.

    Returns:
        Immutable plot descriptor.
    """

    title: str
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    labels: tuple[str, ...] = ()
    y_label: str = "Value"


@dataclass(frozen=True)
class CustomEpoch:
    """One stimulus epoch available to workflow code.

    Args:
        number: One-based epoch number.
        name: Epoch name from stimulus metadata.
        duration_seconds: Shortest observed duration, or ``None`` when timing
            is unavailable.

    Returns:
        Epoch descriptor used by parameter controls and response helpers.
    """

    number: int
    name: str
    duration_seconds: float | None

    @property
    def label(self) -> str:
        """Return the label shown in epoch dropdowns, such as ``2: Odor A``."""
        if self.name == "":
            return f"Epoch {self.number}"
        return f"{self.number}: {self.name}"

    @property
    def selector(self) -> int:
        """Return the epoch number accepted by ``epoch_metric``."""
        return self.number


@dataclass(frozen=True)
class CustomResult:
    """Outputs returned by one custom workflow run.

    Args:
        message: Status text shown after the workflow finishes.
        files: Files written by the workflow.
        tables: Tables to preview in the Custom tab.
        plots: Line plots to show in the Custom tab.
        roi_set: Optional ROIs to replace the active Labels layer.
        response_plot_data: Optional responses to show in the response dock.

    Returns:
        Immutable workflow result.
    """

    message: str
    files: tuple[Path, ...] = ()
    tables: tuple[CustomTable, ...] = ()
    plots: tuple[CustomLinePlot, ...] = ()
    roi_set: RoiSet | None = None
    response_plot_data: ResponsePlotData | None = None


@dataclass(frozen=True)
class WorkflowDeclaration:
    """Raw metadata attached by the ``@workflow`` decorator.

    Args:
        id: Stable workflow id.
        name: GUI label.
        version: Workflow version.
        description: One-sentence description.
        params_type: Optional frozen dataclass for GUI controls.
        author: Optional workflow author string.
        requires_rois: Whether ROIs must exist before running.
        output_prefix: Optional output subfolder name.
        function: Workflow function.

    Returns:
        Declaration attached to a workflow function.
    """

    id: str
    name: str
    version: str
    description: str
    params_type: type[object] | None
    author: str | None
    requires_rois: bool
    output_prefix: str | None
    function: CustomWorkflowFunction


@dataclass(frozen=True)
class CustomWorkflow:
    """One workflow that passed discovery checks.

    Args:
        id: Stable workflow id.
        name: GUI label.
        version: Workflow version.
        description: One-sentence description.
        params_type: Optional frozen dataclass for GUI controls.
        author: Optional workflow author string.
        requires_rois: Whether ROIs must exist before running.
        output_prefix: Safe output subfolder name.
        function: Workflow function.
        source_path: Python file that defines the workflow.
        source_hash: SHA-256 hash of the source file at discovery time.

    Returns:
        Workflow definition shown in the Custom tab.
    """

    id: str
    name: str
    version: str
    description: str
    params_type: type[object] | None
    author: str | None
    requires_rois: bool
    output_prefix: str
    function: CustomWorkflowFunction
    source_path: Path
    source_hash: str


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


@dataclass(frozen=True)
class CustomRunContext:
    """Helpers passed to workflow code at run time.

    Args:
        recording: Active converted recording.
        roi_set: Current ROI masks, if any.
        output_dir: Folder for this workflow run.
        delta_f_over_f_options: Current Plot tab dF/F settings.
        response_processing_options: Current Plot tab processing settings.
        response_pre_window_seconds: Seconds before each epoch in response analysis.
        response_post_window_seconds: Seconds after each epoch in response analysis.
        provenance: Workflow metadata saved with files written through this context.
        visible_roi_indices: Zero-based ROI indices currently visible in the
            response dock. Empty means no visibility state is available.

    Returns:
        Run context with file, ROI, and analysis helpers.
    """

    recording: RecordingData
    roi_set: RoiSet | None
    output_dir: Path
    delta_f_over_f_options: DeltaFOverFOptions
    response_processing_options: ResponseProcessingOptions
    response_pre_window_seconds: float
    response_post_window_seconds: float
    provenance: CustomWorkflowProvenance
    visible_roi_indices: tuple[int, ...] = ()

    def current_rois(self) -> RoiSet:
        """Return current ROI masks or raise a clear error."""
        if self.roi_set is None:
            msg = "This custom workflow needs ROIs, but no ROI Labels are available."
            raise ValueError(msg)
        return self.roi_set

    def rois_for_selector(self, selector: str) -> RoiSet:
        """Return the ROI subset selected in a ``roi_selector`` control.

        Args:
            selector: ``all_rois`` or ``visible_rois``.

        Returns:
            All ROIs or the visible ROI subset.
        """
        normalized = selector.strip().casefold()
        if normalized in {"", "all", "all_rois"}:
            return self.current_rois()
        if normalized in {"visible", "visible_rois"}:
            return self._visible_rois()
        msg = f"Unknown ROI selector {selector!r}."
        raise ValueError(msg)

    def compute_standard_responses(
        self,
        roi_set: RoiSet | None = None,
    ) -> AnalysisResponseComputation:
        """Run the same response analysis used by the Plot tab.

        Args:
            roi_set: Optional ROI masks. When omitted, current ROIs are used.

        Returns:
            In-memory response computation.
        """
        selected_rois = roi_set if roi_set is not None else self.current_rois()
        dff_options = self.delta_f_over_f_options
        return compute_recording_responses(
            self.recording,
            selected_rois,
            baseline_mode=dff_options.baseline_mode,
            baseline_epoch_number=dff_options.baseline_epoch_number,
            baseline_epoch_name=dff_options.baseline_epoch_name,
            background_method=dff_options.background_method,
            baseline_sample_seconds=dff_options.baseline_sample_seconds,
            fit_mode=dff_options.fit_mode,
            apply_motion_mask=dff_options.apply_motion_mask,
            response_pre_window_seconds=self.response_pre_window_seconds,
            response_post_window_seconds=self.response_post_window_seconds,
            response_processing_options=self.response_processing_options,
        )

    def epoch_names(self) -> dict[int, str]:
        """Return stimulus epoch names keyed by one-based epoch number."""
        return stimulus_epoch_names_by_number(self.recording)

    def epoch_choices(self) -> tuple[CustomEpoch, ...]:
        """Return sorted epochs with names and observed durations."""
        durations = self.epoch_durations_seconds()
        return tuple(
            CustomEpoch(
                number=epoch_number,
                name=epoch_name,
                duration_seconds=durations.get(epoch_number),
            )
            for epoch_number, epoch_name in sorted(self.epoch_names().items())
        )

    def epoch_durations_seconds(self) -> dict[int, float]:
        """Return shortest observed duration per stimulus epoch.

        Args:
            None.

        Returns:
            Mapping from epoch number to shortest positive duration in seconds.
            Empty when timing cannot be resolved.
        """
        try:
            timing = resolve_recording_timing(self.recording)
        except ValueError:
            return {}
        durations: dict[int, float] = {}
        for epoch_window in timing.epoch_windows:
            frame_count = (
                epoch_window.window.stop_frame - epoch_window.window.start_frame
            )
            if frame_count <= 0:
                continue
            duration = frame_count / timing.frame_rate_hz
            existing = durations.get(epoch_window.epoch_number)
            if existing is None or duration < existing:
                durations[epoch_window.epoch_number] = float(duration)
        return durations

    def min_epoch_duration_seconds(self) -> float | None:
        """Return the shortest positive epoch duration, when timing is available."""
        durations = tuple(self.epoch_durations_seconds().values())
        if len(durations) == 0:
            return None
        return min(durations)

    def response_plot_data(
        self,
        grouped: GroupedRoiResponses,
        *,
        source_path: Path | None = None,
        max_rois: int | None = None,
        roi_indices: Sequence[int] | None = None,
    ) -> ResponsePlotData:
        """Build response plot data from grouped responses.

        Args:
            grouped: Responses returned by ``compute_standard_responses``.
            source_path: Optional path shown as the plot source.
            max_rois: Optional cap on visible ROI traces.
            roi_indices: Optional ROI indices to mark visible. All ROI rows are
                kept so hidden ROIs stay available in the ROIs tab.

        Returns:
            Plot data for ``CustomResult.response_plot_data``.
        """
        from twopy.napari.plotting.data import (
            response_plot_data_from_grouped,
        )

        plot_data = response_plot_data_from_grouped(
            grouped,
            source_path=source_path,
        )
        keep_indices = tuple(range(len(grouped.roi_labels)))
        if roi_indices is not None:
            keep_indices = tuple(
                index for index in roi_indices if 0 <= index < len(grouped.roi_labels)
            )
        if max_rois is not None:
            keep_indices = keep_indices[: max(0, max_rois)]
        if roi_indices is None and max_rois is None:
            return plot_data
        return replace(
            plot_data,
            visible_roi_indices=keep_indices,
        )

    def epoch_window(
        self,
        start_seconds: float,
        stop_seconds: float,
    ) -> tuple[float, float]:
        """Return a valid epoch-relative metric window.

        Args:
            start_seconds: Epoch-relative window start in seconds.
            stop_seconds: Epoch-relative window stop in seconds.

        Returns:
            ``(start_seconds, stop_seconds)`` as floats.
        """
        start = float(start_seconds)
        stop = float(stop_seconds)
        if stop <= start:
            msg = "epoch window stop must be greater than start"
            raise ValueError(msg)
        return start, stop

    def response_window(
        self,
        start_seconds: float,
        stop_seconds: float,
    ) -> tuple[float, float]:
        """Return a valid response-plot time window.

        Args:
            start_seconds: Window start in seconds relative to epoch onset.
            stop_seconds: Window stop in seconds relative to epoch onset.

        Returns:
            ``(start_seconds, stop_seconds)`` as floats. Starts can be negative
            because response plots may include pre-epoch baseline.
        """
        start = float(start_seconds)
        stop = float(stop_seconds)
        if stop <= start:
            msg = "response window stop must be greater than start"
            raise ValueError(msg)
        return start, stop

    def epoch_selector(self, epoch: CustomEpoch | str | int) -> str | int:
        """Return an epoch selector accepted by response helpers.

        Args:
            epoch: Epoch object, epoch number, epoch name, or dropdown label
                such as ``"2: Odor A"``.

        Returns:
            Epoch number when the label starts with one, otherwise the input.
        """
        if isinstance(epoch, CustomEpoch):
            return epoch.selector
        if isinstance(epoch, int):
            return epoch
        label_prefix = epoch.split(":", maxsplit=1)[0].strip()
        try:
            return int(label_prefix)
        except ValueError:
            return epoch

    def output_path(self, filename: str | Path) -> Path:
        """Return a safe path inside the workflow output folder.

        Args:
            filename: Relative output filename or subpath.

        Returns:
            Path below this run's output folder.

        Raises:
            ValueError: If the requested path is absolute or escapes the output
                directory.
        """
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            msg = f"Custom workflow output path must stay relative: {filename}"
            raise ValueError(msg)
        output_path = self.output_dir / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _visible_rois(self) -> RoiSet:
        """Return the ROI subset currently visible in the response plot."""
        rois = self.current_rois()
        indices = tuple(
            index for index in self.visible_roi_indices if 0 <= index < len(rois.labels)
        )
        if len(indices) == 0:
            msg = "ROI selector 'visible_rois' has no visible ROIs."
            raise ValueError(msg)
        return make_roi_set(
            rois.masks[np.array(indices, dtype=np.int64), :, :],
            labels=tuple(rois.labels[index] for index in indices),
        )

    def write_roi_table(
        self,
        path: Path,
        columns: Mapping[str, Sequence[object]],
        *,
        roi_labels: tuple[str, ...] | None = None,
    ) -> None:
        """Write an ROI-indexed CSV table and workflow metadata.

        Args:
            path: CSV path, usually from ``output_path``.
            columns: Mapping from column name to one value per ROI.
            roi_labels: Optional ROI labels. When omitted, current ROI labels
                are used.

        Returns:
            None.
        """
        labels = roi_labels if roi_labels is not None else self.current_rois().labels
        _validate_column_lengths(columns, row_count=len(labels))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            column_names = tuple(columns)
            writer.writerow(("roi_label", *column_names))
            for row_index, label in enumerate(labels):
                writer.writerow(
                    (label, *(columns[name][row_index] for name in column_names)),
                )
        from twopy.custom.provenance import write_workflow_provenance_for_path

        write_workflow_provenance_for_path(path, self.provenance)

    def write_matrix_csv(
        self,
        path: Path,
        values: npt.ArrayLike,
        *,
        row_labels: tuple[str, ...],
        column_prefix: str = "sample",
        column_labels: Sequence[str] | None = None,
    ) -> None:
        """Write a matrix CSV and workflow metadata.

        Args:
            path: CSV path, usually from ``output_path``.
            values: Matrix shaped ``(rows, columns)``.
            row_labels: Label for each matrix row.
            column_prefix: Prefix used for generated sample column names.
            column_labels: Optional explicit label for each matrix column.

        Returns:
            None.
        """
        matrix = np.asarray(values)
        if matrix.ndim != 2:
            msg = f"Matrix CSV values must be two-dimensional; got {matrix.shape}"
            raise ValueError(msg)
        if matrix.shape[0] != len(row_labels):
            msg = (
                "Matrix row count must match row labels; "
                f"got {matrix.shape[0]} rows and {len(row_labels)} labels"
            )
            raise ValueError(msg)
        resolved_column_labels = _matrix_column_labels(
            column_count=matrix.shape[1],
            column_prefix=column_prefix,
            column_labels=column_labels,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(("label", *resolved_column_labels))
            for row_index, label in enumerate(row_labels):
                writer.writerow((label, *matrix[row_index, :].tolist()))
        from twopy.custom.provenance import write_workflow_provenance_for_path

        write_workflow_provenance_for_path(path, self.provenance)

    def epoch_metric(
        self,
        grouped: GroupedRoiResponses,
        epoch: CustomEpoch | str | int,
        metric: str,
        *,
        window_seconds: tuple[float, float] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute one response value per ROI for an epoch.

        Args:
            grouped: Grouped response trials.
            epoch: Epoch object, dropdown label, name, or number to summarize.
            metric: ``mean``, ``peak``, or ``minimum``.
            window_seconds: Optional epoch-relative time window.

        Returns:
            One value per ROI, averaged across matching trials.
        """
        selector = self.epoch_selector(epoch)
        matching_trials = tuple(
            trial
            for trial in grouped.trials
            if trial.epoch_name == selector or trial.epoch_number == selector
        )
        if len(matching_trials) == 0:
            msg = f"No grouped response trials match epoch {epoch!r}."
            raise ValueError(msg)
        trial_metrics = []
        for trial in matching_trials:
            values = trial.values
            if window_seconds is not None:
                start, stop = window_seconds
                keep = (trial.time_seconds >= start) & (trial.time_seconds <= stop)
                values = values[keep, :]
            if values.shape[0] == 0:
                msg = f"Metric window has no samples for epoch {epoch!r}."
                raise ValueError(msg)
            trial_metrics.append(_metric_by_roi(values, metric))
        return np.nanmean(np.stack(trial_metrics, axis=0), axis=0)


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


def _validate_column_lengths(
    columns: Mapping[str, Sequence[object]],
    *,
    row_count: int,
) -> None:
    """Confirm all table columns match the expected row count."""
    for name, values in columns.items():
        if len(values) != row_count:
            msg = f"Column {name!r} has {len(values)} rows; expected {row_count} rows"
            raise ValueError(msg)


def _matrix_column_labels(
    *,
    column_count: int,
    column_prefix: str,
    column_labels: Sequence[str] | None,
) -> tuple[str, ...]:
    """Return explicit or generated matrix CSV column labels."""
    if column_labels is None:
        return tuple(f"{column_prefix}_{index:04d}" for index in range(column_count))
    labels = tuple(column_labels)
    if len(labels) != column_count:
        msg = (
            "Matrix column label count must match matrix columns; "
            f"got {len(labels)} labels and {column_count} columns"
        )
        raise ValueError(msg)
    if any(label == "" for label in labels):
        msg = "Matrix column labels must not be blank."
        raise ValueError(msg)
    return labels


def _metric_by_roi(
    values: npt.NDArray[np.float64],
    metric: str,
) -> npt.NDArray[np.float64]:
    """Return one metric value per ROI for a trial value matrix."""
    if metric == "mean":
        return np.nanmean(values, axis=0)
    if metric == "peak":
        return np.nanmax(values, axis=0)
    if metric == "minimum":
        return np.nanmin(values, axis=0)
    msg = f"Unknown custom metric {metric!r}."
    raise ValueError(msg)
