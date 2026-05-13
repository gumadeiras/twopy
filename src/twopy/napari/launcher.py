"""Command-line launcher for the twopy napari app.

Inputs: optional ``recording_data.h5`` path and preview/ROI command-line
arguments.
Outputs: a napari viewer, either empty or populated with a converted recording.

The launcher is the application entry point. It should stay thin: query and
analysis workflows belong in dedicated helpers that the controls call.
"""

from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path

from twopy._version import __version__
from twopy.filenames import ROI_FILENAME
from twopy.napari.controls import add_twopy_magicgui_controls
from twopy.napari.loading import resolve_or_convert_launch_recording
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
        recording_data_path: Optional path to a source recording, converted
            folder, or ``recording_data.h5``. When omitted, twopy opens the
            current directory or ``./twopy`` recording if one exists; otherwise
            it opens an empty viewer with folder-based recording selection.
        roi_file_to_load: Optional saved ROI HDF5 path to reopen.
        roi_save_file: Optional ROI output path used when saving ROIs and
            analysis from the response options.
        movie_start_frame: First movie preview frame.
        movie_end_frame: Last movie frame to load. ``None`` means the final
            movie frame.
        load_movie: Whether to add the aligned movie layer.

    Returns:
        ``NapariRecordingView`` when a recording is loaded at startup,
        otherwise ``None``.

    This is the script-facing launcher. It may convert source data before
    opening napari, but all analysis still starts from converted HDF5 files.
    """
    resolved_recording = resolve_or_convert_launch_recording(recording_data_path)
    movie_range = (int(movie_start_frame), movie_end_frame) if load_movie else None
    viewer = create_viewer()
    if resolved_recording is None:
        from twopy.napari.plotting import (
            add_twopy_response_plot_widget,
            create_twopy_response_options_widget,
        )
        from twopy.napari.trial_timeline import TrialTimelineController

        response_plot_widget, _response_plot_dock = add_twopy_response_plot_widget(
            viewer,
            recording=None,
        )
        response_options_widget = create_twopy_response_options_widget(
            response_plot_widget,
        )
        trial_timeline_controller = TrialTimelineController(viewer)
        add_twopy_magicgui_controls(
            viewer,
            roi_labels_layer=None,
            roi_save_file=(
                roi_save_file.expanduser()
                if roi_save_file is not None
                else Path.cwd() / ROI_FILENAME
            ),
            response_plot_widget=response_plot_widget,
            response_options_widget=response_options_widget,
            trial_timeline_controller=trial_timeline_controller,
        )
        view = None
    else:
        paths = resolved_recording.paths
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


def parse_launch_args(args: Sequence[str] | None = None) -> Namespace:
    """Parse command-line arguments for ``python -m twopy.napari``.

    Args:
        args: Optional command-line arguments. ``None`` reads from
            ``sys.argv``.

    Returns:
        Parsed argument namespace.
    """
    parser = ArgumentParser(
        description="Launch napari for a twopy-converted recording.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"twopy {__version__}",
        help="Print the current twopy package version and exit.",
    )
    parser.add_argument(
        "recording_data_path",
        nargs="?",
        type=Path,
        help=(
            "Path to a source recording, converted folder, or recording_data.h5. "
            "If omitted, twopy checks the current directory and ./twopy output, "
            "then opens an empty viewer if neither exists."
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
        help="Optional ROI HDF5 output path for saved ROIs and analysis.",
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
    return parser.parse_args(args)


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
