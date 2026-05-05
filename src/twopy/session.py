"""Find the required files inside one two-photon imaging session folder.

Inputs: a session directory such as a timestamped microscope output folder.
Outputs: typed paths to the required files that twopy needs before analysis.

This module does not read large imaging data. It only checks the folder shape so
later analysis code can fail early with clear, auditable messages.
"""

from dataclasses import dataclass
from pathlib import Path

__all__ = ["TwoPhotonSessionFiles", "discover_session_files"]


@dataclass(frozen=True)
class TwoPhotonSessionFiles:
    """Paths that define one two-photon imaging session.

    Inputs: filesystem paths discovered under one session directory.
    Outputs: a frozen record that analysis and GUI code can pass around safely.

    This object exists so the rest of the code does not repeat path-glob logic
    or silently depend on filenames that change with the stimulus name.
    """

    session_dir: Path
    stimulus_data_dir: Path
    aligned_movie_mat: Path
    alignment_text: Path
    raw_tif: Path
    default_align_channel_text: Path
    high_res_pd_mat: Path
    image_description_mat: Path
    imaging_res_pd_mat: Path
    saved_analysis_dir: Path | None


def discover_session_files(session_dir: Path) -> TwoPhotonSessionFiles:
    """Discover the required files for one two-photon imaging session.

    Args:
        session_dir: Timestamped microscope output folder to inspect.

    Returns:
        A typed record containing the required paths and optional
        ``savedAnalysis`` directory.

    Raises:
        FileNotFoundError: If the session directory or a required path is
            missing.
        ValueError: If a flexible filename pattern, such as the raw TIFF movie,
            matches zero or multiple files.

    The goal is to keep downstream analysis honest: before loading any large
    MATLAB or TIFF data, twopy first proves the session has the expected shape.
    """
    root = session_dir.expanduser()
    _require_directory(root, "session directory")

    # The stimulus name changes between experiments, so glob only the stable
    # parts of filenames: one raw TIFF movie and one alignment text file.
    raw_tif = _find_one_file(root, "*.tif", "raw TIFF movie")
    alignment_text = _find_one_file(root, "*_alignment.txt", "alignment text")

    saved_analysis = root / "savedAnalysis"

    return TwoPhotonSessionFiles(
        session_dir=root,
        stimulus_data_dir=_require_directory(root / "stimulusData", "stimulus data"),
        aligned_movie_mat=_require_file(root / "alignedMovie.mat", "aligned movie"),
        alignment_text=alignment_text,
        raw_tif=raw_tif,
        default_align_channel_text=_require_file(
            root / "defaultAlignChannel.txt",
            "default alignment channel",
        ),
        high_res_pd_mat=_require_file(root / "highResPd.mat", "high-resolution PD"),
        image_description_mat=_require_file(
            root / "imageDescription.mat",
            "image description",
        ),
        imaging_res_pd_mat=_require_file(
            root / "imagingResPd.mat",
            "imaging-resolution PD",
        ),
        saved_analysis_dir=saved_analysis if saved_analysis.is_dir() else None,
    )


def _require_directory(path: Path, label: str) -> Path:
    """Return a directory path or raise a clear missing-data error.

    Args:
        path: Directory path that should exist.
        label: Human-readable name used in the error message.

    Returns:
        The same path when it exists and is a directory.

    Raises:
        FileNotFoundError: If the directory is absent or is not a directory.

    This helper keeps validation messages consistent and easy to interpret from
    the GUI.
    """
    if not path.is_dir():
        msg = f"Missing required {label} directory: {path}"
        raise FileNotFoundError(msg)
    return path


def _require_file(path: Path, label: str) -> Path:
    """Return a file path or raise a clear missing-data error.

    Args:
        path: File path that should exist.
        label: Human-readable name used in the error message.

    Returns:
        The same path when it exists and is a file.

    Raises:
        FileNotFoundError: If the file is absent or is not a regular file.

    This helper separates fixed filenames from globbed filenames while keeping
    the same error style for both.
    """
    if not path.is_file():
        msg = f"Missing required {label} file: {path}"
        raise FileNotFoundError(msg)
    return path


def _find_one_file(session_dir: Path, pattern: str, label: str) -> Path:
    """Find exactly one required file that has a stimulus-dependent name.

    Args:
        session_dir: Session directory to search.
        pattern: Glob pattern that captures the stable part of the filename.
        label: Human-readable name used in the error message.

    Returns:
        The single matching file path.

    Raises:
        ValueError: If the pattern finds no files or more than one file.

    The raw movie and alignment text include stimulus/run details in their
    names. A strict one-match check avoids guessing when the folder is messy.
    """
    matches = sorted(path for path in session_dir.glob(pattern) if path.is_file())

    if len(matches) != 1:
        joined = ", ".join(str(path) for path in matches) if matches else "none"
        msg = (
            f"Expected exactly one {label} matching {pattern!r} in "
            f"{session_dir}; found {len(matches)}: {joined}"
        )
        raise ValueError(msg)

    return matches[0]
