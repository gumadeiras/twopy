"""Command-line launcher for the twopy napari app.

Inputs: optional ``recording_data.h5`` path and preview/ROI command-line
arguments.
Outputs: a napari viewer, either empty or populated with a converted recording.

The launcher is the application entry point. It should stay thin: query and
analysis workflows belong in dedicated helpers that the controls call.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from twopy.napari.constants import DEFAULT_MOVIE_PREVIEW_FRAMES
from twopy.napari.controls import add_twopy_magicgui_controls
from twopy.napari.types import NapariRecordingView
from twopy.napari.viewer import create_viewer, open_recording_in_napari

__all__ = ["launch_napari", "main"]


def launch_napari(
    recording_data_path: Path | None = None,
    *,
    roi_set: Path | None = None,
    roi_output_path: Path | None = None,
    movie_start_frame: int = 0,
    movie_stop_frame: int | None = DEFAULT_MOVIE_PREVIEW_FRAMES,
) -> NapariRecordingView | None:
    """Launch napari and start the event loop.

    Args:
        recording_data_path: Optional path to ``recording_data.h5``. When
            omitted, twopy opens the current directory or ``./twopy`` recording
            if one exists; otherwise it opens an empty viewer with a Load
            Recording button.
        roi_set: Optional saved ROI HDF5 path to reopen.
        roi_output_path: Optional ROI output path for the Save ROIs button.
        movie_start_frame: First movie preview frame.
        movie_stop_frame: Exclusive movie preview stop frame. ``None`` skips
            the movie preview and shows only the mean image.

    Returns:
        ``NapariRecordingView`` when a recording is loaded at startup,
        otherwise ``None``.

    This is the script-facing launcher. It keeps command-line convenience
    separate from the core GUI adapter so a future plugin can still call
    ``open_recording_in_napari`` directly.
    """
    resolved_recording_path = resolve_launch_recording_path(recording_data_path)
    movie_range = (
        None
        if movie_stop_frame is None
        else (int(movie_start_frame), int(movie_stop_frame))
    )
    viewer = create_viewer()
    if resolved_recording_path is None:
        add_twopy_magicgui_controls(
            viewer,
            roi_labels_layer=None,
            roi_output_path=(
                roi_output_path.expanduser()
                if roi_output_path is not None
                else Path.cwd() / "rois.h5"
            ),
        )
        view = None
    else:
        view = open_recording_in_napari(
            resolved_recording_path,
            viewer=viewer,
            roi_set=roi_set,
            roi_output_path=roi_output_path,
            movie_frame_range=movie_range,
        )

    import napari

    napari.run()
    return view


def resolve_launch_recording_path(recording_data_path: Path | None) -> Path | None:
    """Resolve a command-line recording path.

    Args:
        recording_data_path: Optional caller path.

    Returns:
        Existing ``recording_data.h5`` path, or ``None`` when no path was
        supplied and no default recording exists.

    Raises:
        ValueError: If an explicit path was supplied and does not exist.

    Running from a converted output directory or from a source recording
    directory should be enough for day-to-day use.
    """
    candidates: list[Path] = []
    if recording_data_path is not None:
        candidates.append(recording_data_path.expanduser())
    else:
        cwd = Path.cwd()
        candidates.extend(
            (
                cwd / "recording_data.h5",
                cwd / "twopy" / "recording_data.h5",
            ),
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    if recording_data_path is None:
        return None

    candidate_list = "\n".join(f"- {candidate}" for candidate in candidates)
    msg = (
        "Could not find recording_data.h5. Pass a path, run from a twopy output "
        "directory, or run from a recording directory that contains "
        "twopy/recording_data.h5.\n"
        f"Checked:\n{candidate_list}"
    )
    raise ValueError(msg)


def parse_launch_args() -> Namespace:
    """Parse command-line arguments for ``python -m twopy.napari``.

    Args:
        None.

    Returns:
        Parsed argument namespace.
    """
    parser = ArgumentParser(
        description="Launch napari for a twopy-converted recording.",
    )
    parser.add_argument(
        "recording_data_path",
        nargs="?",
        type=Path,
        help=(
            "Path to recording_data.h5. If omitted, twopy checks the current "
            "directory and ./twopy/recording_data.h5, then opens an empty "
            "viewer if neither exists."
        ),
    )
    parser.add_argument(
        "--roi-set",
        type=Path,
        default=None,
        help="Optional existing rois.h5 file to load into the Labels layer.",
    )
    parser.add_argument(
        "--roi-output",
        type=Path,
        default=None,
        help="Optional output path for the Save ROIs button.",
    )
    parser.add_argument(
        "--movie-start",
        type=int,
        default=0,
        help="First frame for the movie preview.",
    )
    parser.add_argument(
        "--movie-stop",
        type=int,
        default=DEFAULT_MOVIE_PREVIEW_FRAMES,
        help="Exclusive stop frame for the movie preview.",
    )
    parser.add_argument(
        "--no-movie",
        action="store_true",
        help="Skip loading movie preview frames and show only the mean image.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the twopy napari launcher from the command line.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_launch_args()
    launch_napari(
        args.recording_data_path,
        roi_set=args.roi_set,
        roi_output_path=args.roi_output,
        movie_start_frame=args.movie_start,
        movie_stop_frame=None if args.no_movie else args.movie_stop,
    )
