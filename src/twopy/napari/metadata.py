"""Recording metadata formatting for the twopy napari adapter.

Inputs: loaded converted recordings.
Outputs: short human-readable text for the napari metadata panel.

This module only formats data already present in ``recording_data.h5``. It does
not read source files or decide analysis behavior.
"""

from twopy.converted import RecordingData

__all__ = ["format_recording_metadata"]


def format_recording_metadata(recording: RecordingData | None) -> str:
    """Format general recording metadata for display in napari.

    Args:
        recording: Loaded converted recording, or ``None`` before a recording
            has been loaded.

    Returns:
        Multi-line text with the most useful general recording fields.
    """
    if recording is None:
        return "No recording loaded."

    lines = [
        f"Recording data: {recording.path}",
        f"Source folder: {recording.source_session_dir}",
        f"Movie file: {recording.movie.path}",
        (
            "Movie shape: "
            f"{recording.movie.shape[0]} frames x "
            f"{recording.movie.shape[1]} x {recording.movie.shape[2]} pixels"
        ),
        f"Movie dtype: {recording.movie.dtype}",
        f"Mean image shape: {_shape_text(recording.mean_image.shape)}",
        (
            "Alignment-valid crop: "
            f"axis0 [{recording.alignment_valid_crop.axis0_start}, "
            f"{recording.alignment_valid_crop.axis0_stop}), "
            f"axis1 [{recording.alignment_valid_crop.axis1_start}, "
            f"{recording.alignment_valid_crop.axis1_stop})"
        ),
        (
            "Frame counts: "
            f"movie={recording.frame_counts.aligned_movie_frames}, "
            f"imaging photodiode={recording.frame_counts.imaging_res_pd_samples}, "
            f"acquisition metadata="
            f"{recording.frame_counts.acquisition_number_of_frames}"
        ),
    ]
    lines.extend(_metadata_lines("Run", recording.run_metadata))
    lines.extend(_metadata_lines("Acquisition", recording.acquisition_metadata))
    return "\n".join(lines)


def _metadata_lines(title: str, values: dict[str, object]) -> list[str]:
    """Format a metadata dictionary with stable ordering.

    Args:
        title: Section name.
        values: Metadata fields from the converted recording.

    Returns:
        Lines to append to the metadata panel.
    """
    if not values:
        return []
    lines = [f"{title}:"]
    for key in sorted(values):
        lines.append(f"  {key}: {values[key]}")
    return lines


def _shape_text(shape: tuple[int, ...]) -> str:
    """Format an array shape in plain text.

    Args:
        shape: Array shape tuple.

    Returns:
        Shape with dimensions separated by ``x``.
    """
    return " x ".join(str(value) for value in shape)
