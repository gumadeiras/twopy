"""Recording loading helpers for the twopy napari adapter.

Inputs: user-selected recording folders, converted folders, or recording files.
Outputs: converted recording paths ready for napari.

This module may trigger conversion when a selected source recording does not
yet have twopy HDF5 output. It keeps that write behavior out of the read-only
path resolver.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py

from twopy.conversion import convert_recording_to_twopy
from twopy.napari.constants import RECORDING_DATA_FILENAME
from twopy.napari.paths import NapariRecordingPaths, PathInput, is_default_path
from twopy.napari.paths import resolve_recording_paths as resolve_converted_paths
from twopy.session import discover_session_files

__all__ = [
    "ResolvedNapariRecording",
    "resolve_or_convert_launch_recording",
    "resolve_or_convert_recording",
]


@dataclass(frozen=True)
class ResolvedNapariRecording:
    """Resolved recording paths plus whether napari had to convert first.

    Inputs: one selected path.
    Outputs: converted paths and a conversion flag for user-facing status.
    """

    paths: NapariRecordingPaths
    was_converted: bool


def resolve_or_convert_recording(path: PathInput) -> ResolvedNapariRecording:
    """Resolve a selected recording path, converting source data if needed.

    Args:
        path: User-selected source folder, converted folder, or
            ``recording_data.h5`` path.

    Returns:
        Converted recording paths and whether conversion was run.

    Raises:
        ValueError: If the path is neither converted twopy output nor a source
            recording folder.

    Existing converted files win. If ``recording_data.h5`` exists but
    ``aligned_movie.h5`` is missing and the source recording can be identified,
    conversion rewrites that same output folder to repair the pair.
    """
    selected = Path.cwd() if is_default_path(path) else Path(path).expanduser()
    try:
        paths = resolve_converted_paths(selected)
    except ValueError as error:
        source_dir = _source_recording_dir(selected)
        if source_dir is None:
            raise error
        converted = convert_recording_to_twopy(source_dir)
        return ResolvedNapariRecording(
            paths=resolve_converted_paths(converted.path),
            was_converted=True,
        )

    if paths.movie_path is not None:
        return ResolvedNapariRecording(paths=paths, was_converted=False)

    source_dir = _source_recording_dir_for_converted_selection(
        selected=selected,
        recording_data_path=paths.recording_data_path,
    )
    if source_dir is None:
        return ResolvedNapariRecording(paths=paths, was_converted=False)

    converted = convert_recording_to_twopy(
        source_dir,
        output_dir=paths.recording_data_path.parent,
    )
    return ResolvedNapariRecording(
        paths=resolve_converted_paths(converted.path),
        was_converted=True,
    )


def resolve_or_convert_launch_recording(
    recording_data_path: PathInput | None,
) -> ResolvedNapariRecording | None:
    """Resolve startup recording paths for the ``twopy`` command.

    Args:
        recording_data_path: Optional command-line path.

    Returns:
        Converted recording paths, or ``None`` when no startup recording is
        available.

    An explicit command-line path may be source or converted data. With no path,
    twopy checks the current folder for converted output first, then treats the
    current folder as a source recording only if it has the expected source file
    shape.
    """
    if recording_data_path is not None:
        return resolve_or_convert_recording(recording_data_path)

    cwd = Path.cwd()
    for candidate in (cwd / RECORDING_DATA_FILENAME, cwd / "twopy", cwd):
        try:
            return resolve_or_convert_recording(candidate)
        except ValueError:
            continue
    return None


def _source_recording_dir_for_converted_selection(
    *,
    selected: Path,
    recording_data_path: Path,
) -> Path | None:
    """Infer the source recording folder for incomplete converted output.

    Args:
        selected: User-selected path.
        recording_data_path: Existing converted ``recording_data.h5`` path.

    Returns:
        Source recording folder, or ``None`` when it cannot be inferred.
    """
    candidates = [
        _source_dir_from_recording_data(recording_data_path),
        selected,
        selected.parent,
        recording_data_path.parent.parent,
    ]
    if recording_data_path.parent.name == "twopy":
        candidates.insert(0, recording_data_path.parent.parent)
    return _first_source_recording_dir(tuple(candidates))


def _source_recording_dir(path: Path) -> Path | None:
    """Return the source recording folder implied by one selected path.

    Args:
        path: User-selected path.

    Returns:
        Source recording folder, or ``None``.
    """
    candidates = [path]
    if path.is_file():
        candidates.append(path.parent)
    if path.name == "twopy":
        candidates.append(path.parent)
    if path.parent.name == "twopy":
        candidates.append(path.parent.parent)
    return _first_source_recording_dir(tuple(candidates))


def _source_dir_from_recording_data(recording_data_path: Path) -> Path | None:
    """Read the source recording folder recorded during conversion.

    Args:
        recording_data_path: Existing converted recording file.

    Returns:
        Source recording path, or ``None`` if the file cannot be read.
    """
    try:
        with h5py.File(recording_data_path, "r") as h5_file:
            value = h5_file.attrs.get("source_session_dir")
    except OSError:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or value == "":
        return None
    return Path(value).expanduser()


def _first_source_recording_dir(candidates: tuple[Path | None, ...]) -> Path | None:
    """Return the first candidate with the expected source recording files.

    Args:
        candidates: Possible source recording folders.

    Returns:
        Resolved source folder, or ``None``.
    """
    seen: set[Path] = set()
    validation_error: FileNotFoundError | ValueError | None = None
    for candidate in candidates:
        if candidate is None:
            continue
        expanded = candidate.expanduser()
        if expanded in seen:
            continue
        seen.add(expanded)
        try:
            return discover_session_files(expanded).session_dir.resolve()
        except (FileNotFoundError, ValueError) as error:
            if validation_error is None and _looks_like_source_recording_dir(expanded):
                validation_error = error
            continue
    if validation_error is not None:
        raise validation_error
    return None


def _looks_like_source_recording_dir(path: Path) -> bool:
    """Return whether a folder appears to be microscope source output.

    Args:
        path: Candidate folder.

    Returns:
        ``True`` when the folder contains stable source-recording markers.

    Discovery errors from source-like folders should reach the user. Errors
    from arbitrary folders should stay quiet so twopy can continue checking
    other candidates such as a parent source folder.
    """
    if not path.is_dir():
        return False
    fixed_markers = (
        "stimulusData",
        "alignedMovie.mat",
        "defaultAlignChannel.txt",
        "highResPd.mat",
        "imageDescription.mat",
        "imagingResPd.mat",
    )
    if any((path / marker).exists() for marker in fixed_markers):
        return True
    return any(_source_candidate_files(path, "*.tif")) or any(
        _source_candidate_files(path, "*_alignment.txt")
    )


def _source_candidate_files(path: Path, pattern: str) -> tuple[Path, ...]:
    """Return source files matching a flexible microscope filename pattern.

    Args:
        path: Candidate source folder.
        pattern: Glob pattern.

    Returns:
        Matching files excluding macOS AppleDouble sidecars.
    """
    return tuple(
        candidate
        for candidate in path.glob(pattern)
        if candidate.is_file() and not candidate.name.startswith("._")
    )
