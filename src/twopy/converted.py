"""Load twopy-converted HDF5 recordings for analysis.

Inputs: ``recording_data.h5`` plus its referenced ``aligned_movie.h5`` file.
Outputs: typed Python objects with metadata, stimulus data, photodiode signals,
and a lazy movie reader.

Analysis code should start here after source conversion. This module does not
read MATLAB or raw TIFF files.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt

from twopy.conversion.types import FrameCountAudit

__all__ = [
    "ConvertedMovie",
    "RecordingData",
    "load_converted_recording",
]


@dataclass(frozen=True)
class ConvertedMovie:
    """Lazy reader for the converted aligned movie.

    Inputs: ``aligned_movie.h5`` path, dataset name, shape, and dtype.
    Outputs: frame-range reads or frame batches from the converted movie.

    The aligned movie is usually the largest file in a recording. This object
    keeps analysis code from loading the full movie unless the caller asks for
    it explicitly.
    """

    path: Path
    dataset_name: str
    shape: tuple[int, int, int]
    dtype: str

    def read_frames(
        self,
        start: int | None = None,
        stop: int | None = None,
    ) -> npt.NDArray[np.float64]:
        """Read a frame range from the converted aligned movie.

        Args:
            start: First frame index. ``None`` starts at frame zero.
            stop: Exclusive stop frame. ``None`` reads through the final frame.

        Returns:
            Movie block with shape ``(frames, x, y)``.

        This is the direct escape hatch for previews, tests, and small frame
        ranges. Trace extraction should prefer ``iter_frame_batches``.
        """
        frame_slice = slice(start, stop)
        with h5py.File(self.path, "r") as h5_file:
            dataset = h5_file[self.dataset_name]
            return cast(npt.NDArray[np.float64], dataset[frame_slice, :, :])

    def iter_frame_batches(
        self,
        *,
        chunk_frames: int = 128,
        start: int | None = None,
        stop: int | None = None,
    ) -> Iterator[tuple[int, int, npt.NDArray[np.float64]]]:
        """Read movie frames in bounded chunks.

        Args:
            chunk_frames: Maximum number of frames per returned block.
            start: First frame index. ``None`` starts at frame zero.
            stop: Exclusive stop frame. ``None`` reads through the final frame.

        Yields:
            ``(start, stop, frames)`` blocks.

        The iterator streams from disk and keeps memory bounded by the chunk
        size. This is the intended access pattern for trace extraction.
        """
        start_frame, stop_frame = _normalize_movie_frame_range(
            frame_count=self.shape[0],
            start=start,
            stop=stop,
        )
        if chunk_frames < 1:
            msg = f"chunk_frames must be at least 1; got {chunk_frames}"
            raise ValueError(msg)

        with h5py.File(self.path, "r") as h5_file:
            dataset = h5_file[self.dataset_name]
            for chunk_start in range(start_frame, stop_frame, chunk_frames):
                chunk_stop = min(chunk_start + chunk_frames, stop_frame)
                frames = cast(
                    npt.NDArray[np.float64],
                    dataset[chunk_start:chunk_stop, :, :],
                )
                yield chunk_start, chunk_stop, frames


@dataclass(frozen=True)
class RecordingData:
    """Loaded non-source recording data ready for analysis.

    Inputs: twopy-converted HDF5 files.
    Outputs: metadata dictionaries, stimulus arrays with column names,
    photodiode vectors, mean image, frame-count audit values, and a lazy movie
    reader.

    This object is the analysis-facing representation of one converted
    recording. It deliberately excludes source MATLAB/TIFF ownership.
    """

    path: Path
    movie: ConvertedMovie
    source_session_dir: Path
    acquisition_metadata: dict[str, object]
    run_metadata: dict[str, object]
    synchronization_metadata: dict[str, str]
    stimulus_timeline: npt.NDArray[np.float64]
    stimulus_timeline_column_names: tuple[str, ...]
    stimulus_parameters: tuple[dict[str, object], ...]
    imaging_res_pd: npt.NDArray[np.float64]
    high_res_pd: npt.NDArray[np.float64]
    mean_image: npt.NDArray[np.float64]
    frame_counts: FrameCountAudit


def load_converted_recording(
    recording_data_path: Path,
    *,
    movie_path: Path | None = None,
) -> RecordingData:
    """Load one twopy-converted recording from HDF5.

    Args:
        recording_data_path: Path to ``recording_data.h5``.
        movie_path: Optional explicit path to ``aligned_movie.h5``. When omitted,
            the path is read from the HDF5 movie reference.

    Returns:
        A ``RecordingData`` object with small arrays loaded and movie frames kept
        lazy.

    Raises:
        ValueError: If required HDF5 groups, attributes, or frame contracts are
            inconsistent.

    The main HDF5 file is treated as the manifest. It names the separate movie
    file so scripts can usually pass only ``recording_data.h5``.
    """
    data_path = recording_data_path.expanduser()
    with h5py.File(data_path, "r") as h5_file:
        _require_format(h5_file, expected="converted-recording", path=data_path)
        resolved_movie_path = _resolve_movie_path(
            h5_file=h5_file,
            recording_data_path=data_path,
            explicit_movie_path=movie_path,
        )
        movie_dataset = _read_str_attr(h5_file["movie"], "aligned_movie_dataset")
        movie_shape = _read_int_tuple_attr(h5_file["movie"], "aligned_movie_shape")
        movie_dtype = _read_str_attr(h5_file["movie"], "aligned_movie_dtype")
        if len(movie_shape) != 3:
            msg = f"aligned movie shape must have 3 dimensions: {movie_shape}"
            raise ValueError(msg)

        recording = RecordingData(
            path=data_path,
            movie=ConvertedMovie(
                path=resolved_movie_path,
                dataset_name=movie_dataset,
                shape=movie_shape,
                dtype=movie_dtype,
            ),
            source_session_dir=Path(
                _read_str_attr(h5_file, "source_session_dir"),
            ),
            acquisition_metadata=_read_attrs(h5_file["metadata"]),
            run_metadata=_read_attrs(h5_file["run"]),
            synchronization_metadata=_read_str_attrs(h5_file["photodiode"]),
            stimulus_timeline=cast(
                npt.NDArray[np.float64],
                h5_file["stimulus/timeline"][()],
            ),
            stimulus_timeline_column_names=_read_string_dataset(
                h5_file,
                "stimulus/timeline_column_names",
            ),
            stimulus_parameters=_read_stimulus_parameters(h5_file),
            imaging_res_pd=cast(
                npt.NDArray[np.float64],
                h5_file["photodiode/imaging_res_pd"][()],
            ),
            high_res_pd=cast(
                npt.NDArray[np.float64],
                h5_file["photodiode/high_res_pd"][()],
            ),
            mean_image=cast(
                npt.NDArray[np.float64],
                h5_file["movie/mean_image"][()],
            ),
            frame_counts=_read_frame_counts(h5_file),
        )

    _validate_loaded_recording(recording)
    return recording


def _resolve_movie_path(
    *,
    h5_file: h5py.File,
    recording_data_path: Path,
    explicit_movie_path: Path | None,
) -> Path:
    """Resolve the aligned movie file path for one recording.

    Args:
        h5_file: Open ``recording_data.h5`` file.
        recording_data_path: Path to the manifest file.
        explicit_movie_path: Caller-provided movie path, if any.

    Returns:
        Expanded movie file path.
    """
    if explicit_movie_path is not None:
        return explicit_movie_path.expanduser()

    movie_name = _read_str_attr(h5_file["movie"], "aligned_movie_file")
    return recording_data_path.parent / movie_name


def _validate_loaded_recording(recording: RecordingData) -> None:
    """Check the converted files agree on the core frame contract.

    Args:
        recording: Loaded recording to validate.

    Returns:
        None.

    Raises:
        ValueError: If movie file metadata or photodiode samples disagree.

    This keeps frame alignment failures close to loading instead of letting them
    surface later as incorrect ROI responses.
    """
    if not recording.movie.path.is_file():
        msg = f"Missing converted aligned movie file: {recording.movie.path}"
        raise ValueError(msg)

    with h5py.File(recording.movie.path, "r") as h5_file:
        _require_format(h5_file, expected="aligned-movie", path=recording.movie.path)
        if recording.movie.dataset_name not in h5_file:
            msg = (
                f"Missing aligned movie dataset {recording.movie.dataset_name!r} "
                f"in {recording.movie.path}"
            )
            raise ValueError(msg)
        dataset = h5_file[recording.movie.dataset_name]
        if tuple(dataset.shape) != recording.movie.shape:
            msg = (
                f"Movie shape mismatch: manifest has {recording.movie.shape}, "
                f"movie file has {tuple(dataset.shape)}"
            )
            raise ValueError(msg)

    frame_count = recording.movie.shape[0]
    if recording.imaging_res_pd.shape != (frame_count,):
        msg = (
            "imaging_res_pd must have one sample per aligned movie frame; "
            f"got {recording.imaging_res_pd.shape} for {frame_count} frames"
        )
        raise ValueError(msg)
    if (
        len(recording.stimulus_timeline_column_names)
        != recording.stimulus_timeline.shape[1]
    ):
        msg = (
            "stimulus timeline column names must match timeline width; "
            f"got {len(recording.stimulus_timeline_column_names)} names for "
            f"{recording.stimulus_timeline.shape[1]} columns"
        )
        raise ValueError(msg)


def _require_format(h5_file: h5py.File, *, expected: str, path: Path) -> None:
    """Validate a twopy HDF5 file format marker.

    Args:
        h5_file: Open HDF5 file.
        expected: Expected ``twopy_format`` attribute.
        path: File path used in clear error messages.

    Returns:
        None.
    """
    actual = _read_str_attr(h5_file, "twopy_format")
    if actual != expected:
        msg = f"Expected {expected!r} HDF5 file at {path}; got {actual!r}"
        raise ValueError(msg)


def _read_attrs(group: h5py.Group | h5py.File) -> dict[str, object]:
    """Read HDF5 attributes into plain Python values.

    Args:
        group: HDF5 object with attributes.

    Returns:
        Dictionary keyed by attribute name.
    """
    return {key: _plain_attr_value(value) for key, value in group.attrs.items()}


def _read_str_attrs(group: h5py.Group | h5py.File) -> dict[str, str]:
    """Read HDF5 attributes that should all be text.

    Args:
        group: HDF5 object with attributes.

    Returns:
        Dictionary of string attributes.
    """
    return {key: str(_plain_attr_value(value)) for key, value in group.attrs.items()}


def _read_str_attr(group: h5py.Group | h5py.File, key: str) -> str:
    """Read one required string HDF5 attribute.

    Args:
        group: HDF5 object with attributes.
        key: Attribute name.

    Returns:
        Attribute value as text.
    """
    if key not in group.attrs:
        msg = f"Missing required HDF5 attribute {key!r}"
        raise ValueError(msg)
    value = _plain_attr_value(group.attrs[key])
    return str(value)


def _read_int_attr(group: h5py.Group | h5py.File, key: str) -> int:
    """Read one required integer HDF5 attribute.

    Args:
        group: HDF5 object with attributes.
        key: Attribute name.

    Returns:
        Attribute value as an integer.
    """
    if key not in group.attrs:
        msg = f"Missing required HDF5 attribute {key!r}"
        raise ValueError(msg)
    return int(group.attrs[key])


def _read_int_tuple_attr(group: h5py.Group | h5py.File, key: str) -> tuple[int, ...]:
    """Read one integer tuple HDF5 attribute.

    Args:
        group: HDF5 object with attributes.
        key: Attribute name.

    Returns:
        Tuple of integer values.
    """
    if key not in group.attrs:
        msg = f"Missing required HDF5 attribute {key!r}"
        raise ValueError(msg)
    value = group.attrs[key]
    if isinstance(value, np.ndarray):
        return tuple(int(item) for item in value.tolist())
    return (int(value),)


def _plain_attr_value(value: object) -> object:
    """Convert a small HDF5 attribute value into plain Python data.

    Args:
        value: Raw h5py attribute value.

    Returns:
        Python scalar, string, or list.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _read_stimulus_parameters(
    h5_file: h5py.File,
) -> tuple[dict[str, object], ...]:
    """Read converted stimulus epoch parameter dictionaries.

    Args:
        h5_file: Open ``recording_data.h5`` file.

    Returns:
        Tuple of per-epoch dictionaries.
    """
    raw_json = h5_file["stimulus/parameters_json"][()]
    if isinstance(raw_json, bytes):
        json_text = raw_json.decode("utf-8")
    else:
        json_text = str(raw_json)
    decoded = json.loads(json_text)
    if not isinstance(decoded, list):
        msg = "stimulus/parameters_json must decode to a list"
        raise ValueError(msg)

    parameters: list[dict[str, object]] = []
    for item in decoded:
        if not isinstance(item, dict):
            msg = "each stimulus parameter entry must be a dictionary"
            raise ValueError(msg)
        parameters.append(cast(dict[str, object], item))

    return tuple(parameters)


def _read_string_dataset(h5_file: h5py.File, dataset_name: str) -> tuple[str, ...]:
    """Read a one-dimensional HDF5 string dataset.

    Args:
        h5_file: Open ``recording_data.h5`` file.
        dataset_name: Dataset path.

    Returns:
        Tuple of decoded strings.
    """
    values = h5_file[dataset_name][()]
    return tuple(_decode_string_value(value) for value in values)


def _decode_string_value(value: object) -> str:
    """Decode one HDF5 string scalar.

    Args:
        value: Raw scalar value from h5py.

    Returns:
        Text value.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_frame_counts(h5_file: h5py.File) -> FrameCountAudit:
    """Read frame-count audit metadata from converted HDF5.

    Args:
        h5_file: Open ``recording_data.h5`` file.

    Returns:
        ``FrameCountAudit`` object with all saved counts and deltas.
    """
    group = h5_file["frame_counts"]
    return FrameCountAudit(
        aligned_movie_frames=_read_int_attr(group, "aligned_movie_frames"),
        imaging_res_pd_samples=_read_int_attr(group, "imaging_res_pd_samples"),
        acquisition_number_of_frames=_read_int_attr(
            group,
            "acquisition_number_of_frames",
        ),
        imaging_res_pd_minus_movie=_read_int_attr(
            group,
            "imaging_res_pd_minus_movie",
        ),
        acquisition_minus_movie=_read_int_attr(group, "acquisition_minus_movie"),
    )


def _normalize_movie_frame_range(
    *,
    frame_count: int,
    start: int | None,
    stop: int | None,
) -> tuple[int, int]:
    """Normalize and validate a converted movie frame range.

    Args:
        frame_count: Total number of frames in the movie.
        start: Optional first frame index.
        stop: Optional exclusive stop frame.

    Returns:
        ``(start, stop)`` frame bounds.
    """
    start_frame = 0 if start is None else start
    stop_frame = frame_count if stop is None else stop
    if start_frame < 0 or stop_frame > frame_count or start_frame >= stop_frame:
        msg = (
            f"Invalid frame range [{start_frame}, {stop_frame}) for "
            f"{frame_count} frames"
        )
        raise ValueError(msg)
    return start_frame, stop_frame
