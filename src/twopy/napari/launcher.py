"""Command-line launcher for the twopy napari app.

Inputs: optional ``recording_data.h5`` path and preview/ROI command-line
arguments.
Outputs: a napari viewer, either empty or populated with a converted recording.

The launcher is the application entry point. It should stay thin: query and
analysis workflows belong in dedicated helpers that the controls call.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from twopy.napari.controls import add_twopy_magicgui_controls
from twopy.napari.paths import resolve_launch_recording_path, resolve_recording_paths
from twopy.napari.types import NapariRecordingView
from twopy.napari.viewer import create_viewer, open_recording_in_napari

__all__ = ["launch_napari", "main"]


def launch_napari(
    recording_data_path: Path | None = None,
    *,
    roi_file_to_load: Path | None = None,
    roi_save_file: Path | None = None,
    movie_start_frame: int = 0,
    movie_end_frame: int | None = None,
    load_movie: bool = True,
) -> NapariRecordingView | None:
    """Launch napari and start the event loop.

    Args:
        recording_data_path: Optional path to ``recording_data.h5``. When
            omitted, twopy opens the current directory or ``./twopy`` recording
            if one exists; otherwise it opens an empty viewer with a Load
            Recording button.
        roi_file_to_load: Optional saved ROI HDF5 path to reopen.
        roi_save_file: Optional ROI output path for the Save ROIs button.
        movie_start_frame: First movie preview frame.
        movie_end_frame: Last movie frame to load. ``None`` means the final
            movie frame.
        load_movie: Whether to add the aligned movie layer.

    Returns:
        ``NapariRecordingView`` when a recording is loaded at startup,
        otherwise ``None``.

    This is the script-facing launcher. It keeps command-line convenience
    separate from the core GUI adapter so a future plugin can still call
    ``open_recording_in_napari`` directly.
    """
    resolved_recording_path = resolve_launch_recording_path(recording_data_path)
    movie_range = (int(movie_start_frame), movie_end_frame) if load_movie else None
    viewer = create_viewer()
    if resolved_recording_path is None:
        add_twopy_magicgui_controls(
            viewer,
            roi_labels_layer=None,
            roi_save_file=(
                roi_save_file.expanduser()
                if roi_save_file is not None
                else Path.cwd() / "rois.h5"
            ),
        )
        view = None
    else:
        paths = resolve_recording_paths(resolved_recording_path)
        view = open_recording_in_napari(
            paths.recording_data_path,
            viewer=viewer,
            roi_set=roi_file_to_load,
            roi_save_file=roi_save_file,
            movie_path=paths.movie_path,
            movie_frame_range=movie_range,
        )

    import napari

    napari.run()
    return view


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
        "--roi-file-to-load",
        dest="roi_file_to_load",
        type=Path,
        default=None,
        help="Optional existing ROI HDF5 file to load into the Labels layer.",
    )
    parser.add_argument(
        "--roi-save-file",
        dest="roi_save_file",
        type=Path,
        default=None,
        help="Optional ROI HDF5 output path for the Save ROIs button.",
    )
    parser.add_argument(
        "--movie-start",
        type=int,
        default=0,
        help="First frame for the movie preview.",
    )
    parser.add_argument(
        "--movie-end",
        type=int,
        default=None,
        help="Last frame for the movie preview. Defaults to the final frame.",
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
        roi_file_to_load=args.roi_file_to_load,
        roi_save_file=args.roi_save_file,
        movie_start_frame=args.movie_start,
        movie_end_frame=args.movie_end,
        load_movie=not args.no_movie,
    )
