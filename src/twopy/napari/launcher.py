"""Command-line launcher for the twopy napari app.

Inputs: optional ``recording_data.h5`` path and preview/ROI command-line
arguments.
Outputs: a napari viewer, either empty or populated with a converted recording.

The launcher is the application entry point. It should stay thin: query and
analysis workflows belong in dedicated helpers that the controls call.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from twopy._version import __version__
from twopy.config import (
    config_search_paths,
    discover_config_path,
    load_config,
    preferred_config_path,
    write_config_template,
)

if TYPE_CHECKING:
    from twopy.napari.protocols import NapariViewer
    from twopy.napari.types import NapariRecordingView

__all__ = ["launch_napari", "main"]

_CONFIG_SETUP_MESSAGE = (
    "Created twopy config template:\n"
    "  {path}\n\n"
    "Edit this file with your lab paths, then run twopy again."
)
_INVALID_CONFIG_MESSAGE = (
    "Invalid twopy config:\n"
    "  {path}\n\n"
    "{error}\n\n"
    "Edit this file, then run twopy again."
)


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

    This launcher may convert source data before opening napari, but all
    analysis still starts from converted HDF5 files. The command-line entry
    point creates a first-launch config template before calling this function;
    direct Python callers should run ``twopy config setup`` first.
    """
    from twopy.napari.loading import resolve_or_convert_launch_recording
    from twopy.napari.viewer import create_viewer, open_recording_in_napari

    resolved_recording = resolve_or_convert_launch_recording(recording_data_path)
    movie_range = (int(movie_start_frame), movie_end_frame) if load_movie else None
    viewer = create_viewer()
    if resolved_recording is None:
        _install_empty_launch_controls(viewer, roi_save_file=roi_save_file)
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
            output_route=resolved_recording.output_route,
        )

    import napari

    napari.run()
    return view


def _install_empty_launch_controls(
    viewer: NapariViewer, *, roi_save_file: Path | None
) -> None:
    """Add empty-viewer controls before napari starts drawing.

    Args:
        viewer: Empty napari viewer receiving twopy docks.
        roi_save_file: Optional ROI output path for later saved analysis.

    Returns:
        None.

    The empty app should appear with its full layout in one pass. Installing
    docks before ``napari.run()`` avoids a visible layout shift after the window
    appears.
    """
    from twopy.filenames import ROI_FILENAME
    from twopy.napari.controls import add_twopy_magicgui_controls
    from twopy.napari.plotting import (
        add_twopy_response_plot_widget,
        create_twopy_response_options_widget,
    )
    from twopy.napari.trial_timeline import TrialTimelineController

    response_plot_widget, response_plot_dock = add_twopy_response_plot_widget(
        viewer,
        recording=None,
    )
    response_options_widget = create_twopy_response_options_widget(
        response_plot_widget,
    )
    trial_timeline_controller = TrialTimelineController(viewer)
    control_docks = add_twopy_magicgui_controls(
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
    vars(viewer)["_twopy_empty_launch_controls"] = (
        response_plot_widget,
        response_plot_dock,
        response_options_widget,
        trial_timeline_controller,
        control_docks,
    )


def parse_launch_args(args: Sequence[str] | None = None) -> Namespace:
    """Parse command-line arguments for ``python -m twopy.napari``.

    Args:
        args: Optional command-line arguments. ``None`` reads from
            ``sys.argv``.

    Returns:
        Parsed argument namespace.
    """
    raw_args = list(sys.argv[1:] if args is None else args)
    if raw_args and raw_args[0] == "config":
        parsed_config_args = _build_config_parser().parse_args(raw_args[1:])
        parsed_config_args.command = "config"
        return parsed_config_args

    parser = _build_parser()
    parsed_args = parser.parse_args(raw_args)
    parsed_args.command = None
    return parsed_args


def _build_config_parser() -> ArgumentParser:
    """Build the parser for ``twopy config`` commands.

    Args:
        None.

    Returns:
        Configured config-command parser.
    """
    config_parser = ArgumentParser(
        prog="twopy config",
        description="Show or create the twopy config file.",
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    setup_parser = config_subparsers.add_parser(
        "setup",
        help="Create an editable twopy config template.",
    )
    setup_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing config file.",
    )
    return config_parser


def _build_parser() -> ArgumentParser:
    """Build the command-line parser for the twopy app.

    Args:
        None.

    Returns:
        Configured argument parser.
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
    return parser


def main() -> None:
    """Run the twopy napari launcher from the command line.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_launch_args()
    if args.command == "config":
        _run_config_command(args)
        return
    if not _ensure_config_for_launch():
        return

    launch_napari(
        args.recording_data_path,
        roi_file_to_load=args.roi_file_to_load,
        roi_save_file=args.roi_save_file,
        movie_start_frame=args.movie_start,
        movie_end_frame=args.movie_end,
        load_movie=not args.no_movie,
    )


def _run_config_command(args: Namespace) -> None:
    """Run a ``twopy config`` subcommand.

    Args:
        args: Parsed CLI arguments for the config command.

    Returns:
        None.
    """
    if args.config_command is None:
        _show_config()
        return

    if args.config_command == "setup":
        _setup_config(force=args.force)
        return

    msg = f"Unsupported twopy config command: {args.config_command}"
    raise ValueError(msg)


def _show_config() -> None:
    """Print the active config file path and contents.

    Args:
        None.

    Returns:
        None.
    """
    config_path = discover_config_path()
    if config_path is None:
        print("No twopy config file found.")
        print(f"Setup path: {preferred_config_path()}")
        print("Searched:")
        for candidate in config_search_paths():
            print(f"  {candidate}")
        print("Run `twopy config setup` to create a template.")
        return

    _validate_config(config_path)
    print(f"twopy config: {config_path}")
    print("status: valid")
    print()
    print(config_path.read_text(encoding="utf-8"), end="")


def _ensure_config_for_launch() -> bool:
    """Create a first-launch config template when none exists yet.

    Args:
        None.

    Returns:
        ``True`` when twopy can keep launching, or ``False`` after creating a
        template that the user must edit before running twopy again.
    """
    config_path = discover_config_path()
    if config_path is not None:
        _validate_config(config_path)
        return True

    config_path = write_config_template()
    print(_CONFIG_SETUP_MESSAGE.format(path=config_path))
    return False


def _validate_config(config_path: Path) -> None:
    """Validate one config file and stop the command when it is invalid.

    Args:
        config_path: Config file selected for this command.

    Returns:
        None.
    """
    try:
        load_config(config_path)
    except (FileNotFoundError, ValueError) as error:
        print(
            _INVALID_CONFIG_MESSAGE.format(path=config_path, error=error),
            file=sys.stderr,
        )
        raise SystemExit(2) from error


def _setup_config(*, force: bool = False) -> None:
    """Create the editable config template for ``twopy config setup``.

    Args:
        force: Whether to replace an existing config file.

    Returns:
        None.
    """
    try:
        config_path = write_config_template(force=force)
    except FileExistsError:
        config_path = preferred_config_path()
        print(f"twopy config already exists:\n  {config_path}")
        print("Use `twopy config setup --force` to replace it.")
        return

    print(_CONFIG_SETUP_MESSAGE.format(path=config_path))
