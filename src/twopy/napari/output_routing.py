"""Resolve local and published output folders for napari recordings.

Inputs: loaded recording paths, source session metadata, and twopy config.
Outputs: one explicit route saying where napari saves locally and where those
files should be visible after sync.

Napari often works from the local analysis cache even when the user selected a
network source folder or an existing converted output folder. This module keeps
that local-vs-published distinction explicit so Save ROIs + analysis does not
silently infer the wrong destination later.
"""

from dataclasses import dataclass
from pathlib import Path

from twopy.config import (
    DEFAULT_CONFIG_PATH,
    load_config,
    resolve_analysis_output_dir,
)
from twopy.converted import RecordingData

__all__ = [
    "NapariOutputRoute",
    "default_output_route",
    "recording_output_route",
    "same_output_path",
]


@dataclass(frozen=True)
class NapariOutputRoute:
    """Local and published folders for one loaded napari recording.

    Args:
        local_root: Folder where napari writes files during interactive work.
        publish_root: Folder where saved outputs should be visible after sync.

    Returns:
        Immutable output routing information for save and status code.

    ``local_root`` is usually a cache folder. ``publish_root`` is either the
    configured ``analysis_output`` folder or a conservative fallback beside the
    manually selected recording output.
    """

    local_root: Path
    publish_root: Path


def default_output_route(
    *,
    local_root: Path,
    source_session_dir: Path | None,
    fallback_publish_root: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> NapariOutputRoute:
    """Return the output route for one loaded recording.

    Args:
        local_root: Folder containing the loaded local ``recording_data.h5``.
        source_session_dir: Source microscope folder recorded in HDF5 metadata,
            or ``None`` when a converted file has no source metadata.
        fallback_publish_root: Folder to use when config routing cannot map
            the source. Manual converted loads pass the selected converted
            folder; manual source loads pass ``source/twopy``.
        config_path: YAML config path used for normal ``analysis_output``.

    Returns:
        Output route for the loaded recording.

    Configured output wins when the source maps cleanly. The fallback keeps
    manual loads outside configured data roots auditable instead of losing the
    copy-back step.
    """
    local = local_root.expanduser()
    fallback = fallback_publish_root.expanduser()
    if source_session_dir is None:
        return NapariOutputRoute(
            local_root=local,
            publish_root=fallback,
        )

    try:
        config = load_config(config_path)
        publish = resolve_analysis_output_dir(config, source_session_dir)
    except ValueError:
        return NapariOutputRoute(
            local_root=local,
            publish_root=fallback,
        )

    return NapariOutputRoute(
        local_root=local,
        publish_root=publish,
    )


def recording_output_route(recording: RecordingData) -> NapariOutputRoute:
    """Return the fallback output route for an already-loaded recording.

    Args:
        recording: Loaded converted recording whose folder is the local output
            root.

    Returns:
        Output route using configured ``analysis_output`` when the source maps,
        otherwise the converted recording folder.
    """
    return default_output_route(
        local_root=recording.path.parent,
        source_session_dir=recording.source_session_dir,
        fallback_publish_root=recording.path.parent,
    )


def same_output_path(left: Path, right: Path) -> bool:
    """Return whether two output paths point at the same filesystem location."""
    return left.expanduser().resolve(strict=False) == right.expanduser().resolve(
        strict=False,
    )
