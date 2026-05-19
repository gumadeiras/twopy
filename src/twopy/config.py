"""Load twopy configuration from a small YAML file.

The config layer keeps machine-specific paths out of analysis code. GUI and
script callers get one typed object with database paths, cache paths, custom
workflow paths, and output settings.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from twopy.pixel_calibration import DEFAULT_PIXEL_CALIBRATION_PATH

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_ANALYSIS_CACHE_DIR",
    "AnalysisOutputMode",
    "TwopyConfig",
    "data_path_match",
    "load_config",
    "resolve_data_recording_path",
    "resolve_analysis_cache_dir",
    "resolve_analysis_output_dir",
    "resolve_analysis_work_dir",
]

DEFAULT_CONFIG_PATH = Path("config.yml")
DEFAULT_ANALYSIS_CACHE_DIR = Path.home() / ".cache" / "twopy" / "recordings"
DatabaseAccess = Literal["direct", "copy"]
AnalysisOutputMode = Path | Literal["source"]


@dataclass(frozen=True)
class TwopyConfig:
    """Machine-local paths used by twopy.

    The paths are not required to exist during parsing so tests and machines
    without mounted lab drives can still load and inspect configuration.
    """

    database_path: Path
    data_paths: tuple[Path, ...]
    database_access: DatabaseAccess
    analysis_output: AnalysisOutputMode
    analysis_caching: bool = True
    analysis_cache_dir: Path = DEFAULT_ANALYSIS_CACHE_DIR
    pixel_calibration_path: Path = DEFAULT_PIXEL_CALIBRATION_PATH
    custom_workflow_paths: tuple[Path, ...] = ()


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> TwopyConfig:
    """Load twopy configuration from YAML.

    Args:
        path: YAML config file path. Defaults to ``config.yml`` in the current
            working directory.

    Returns:
        A typed configuration object with paths, DB access, and output mode.

    Raises:
        FileNotFoundError: If the config file is absent.
        ValueError: If the YAML is not a mapping or required keys are missing.

    YAML is used because the file is edited by people and benefits from comments.
    """
    config_path = path.expanduser()
    if not config_path.is_file():
        msg = f"Missing twopy config file: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open("r", encoding="utf-8") as config_file:
        try:
            loaded: object = yaml.safe_load(config_file)
        except yaml.YAMLError as error:
            msg = f"Could not parse twopy config file {config_path}: {error}"
            raise ValueError(msg) from error

    if not isinstance(loaded, dict):
        msg = f"twopy config must be a YAML mapping: {config_path}"
        raise ValueError(msg)

    raw_config: dict[object, object] = {key: value for key, value in loaded.items()}

    return TwopyConfig(
        database_path=_required_path(raw_config, "database_path", config_path),
        data_paths=_required_path_tuple(raw_config, "data_paths", config_path),
        database_access=_database_access(raw_config, config_path),
        analysis_output=_analysis_output(raw_config, config_path),
        analysis_caching=_optional_bool(
            raw_config,
            "analysis_caching",
            config_path,
            default=True,
        ),
        analysis_cache_dir=_optional_path(
            raw_config,
            "analysis_cache_dir",
            config_path,
            default=DEFAULT_ANALYSIS_CACHE_DIR,
        ),
        pixel_calibration_path=_optional_path(
            raw_config,
            "pixel_calibration_path",
            config_path,
            default=DEFAULT_PIXEL_CALIBRATION_PATH,
        ),
        custom_workflow_paths=_optional_path_tuple(
            raw_config,
            "custom_workflow_paths",
            config_path,
        ),
    )


def resolve_data_recording_path(
    config: TwopyConfig,
    relative_parts: tuple[str, ...],
) -> Path:
    """Resolve a database-relative recording path against ordered data roots.

    Args:
        config: Loaded twopy configuration with ordered data roots.
        relative_parts: Safe relative path pieces from a database row.

    Returns:
        The first existing recording candidate under ``data_paths``. If none
        exists, the candidate under the first configured root is returned so the
        loader can fail with the concrete path it tried.

    The lab can keep the same relative recording tree on multiple volumes. This
    function is the ordered availability check that picks the preferred mounted
    copy without hiding missing-data errors behind a second abstraction.
    """
    candidates = tuple(
        data_root.expanduser().joinpath(*relative_parts)
        for data_root in config.data_paths
    )
    if not candidates:
        msg = "twopy config must include at least one data path"
        raise ValueError(msg)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def data_path_match(
    config: TwopyConfig, recording_dir: Path
) -> tuple[Path, Path] | None:
    """Return the configured data root and relative path for one recording.

    Args:
        config: Loaded twopy configuration with ordered data roots.
        recording_dir: Source recording folder.

    Returns:
        ``(data_root, relative_recording)`` for the first configured root that
        contains ``recording_dir``, or ``None`` when the recording is external.

    Output and cache routing need the same root-relative path that database
    search used. Keeping that rule here avoids each caller inventing its own
    root membership test.
    """
    source_dir = recording_dir.expanduser().resolve(strict=False)
    for data_root in config.data_paths:
        expanded_root = data_root.expanduser().resolve(strict=False)
        try:
            return expanded_root, source_dir.relative_to(expanded_root)
        except ValueError:
            continue
    return None


def resolve_analysis_output_dir(config: TwopyConfig, recording_dir: Path) -> Path:
    """Resolve the output directory for one recording.

    Args:
        config: Loaded twopy configuration.
        recording_dir: Recording folder being analyzed.

    Returns:
        Directory where twopy should write analysis output for the recording.

    Raises:
        ValueError: If ``analysis_output`` is a root path and the recording is
            not inside any configured data root.

    ``analysis_output: source`` writes into ``recording_dir/twopy``. A path
    value mirrors the recording's path relative to its matching data root under
    that output root.
    """
    source_dir = recording_dir.expanduser()
    if config.analysis_output == "source":
        return source_dir / "twopy"

    match = data_path_match(config, source_dir)
    if match is None:
        msg = (
            f"Recording {source_dir} is not inside configured data_paths; "
            "cannot mirror analysis output directory"
        )
        raise ValueError(msg)

    _data_root, relative_recording = match
    return config.analysis_output / relative_recording


def resolve_analysis_cache_dir(config: TwopyConfig, recording_dir: Path) -> Path:
    """Resolve the local cache directory for one recording.

    Args:
        config: Loaded twopy configuration.
        recording_dir: Source recording folder being analyzed.

    Returns:
        Directory where twopy should keep local working files.

    The cache mirrors the recording-relative directory shape under whichever
    configured data root contains the recording. External recordings still get
    a stable cache path under ``_external``.
    """
    source_dir = recording_dir.expanduser()
    match = data_path_match(config, source_dir)
    if match is None:
        return _external_analysis_cache_dir(config, source_dir)
    _data_root, relative_recording = match
    return config.analysis_cache_dir / relative_recording


def resolve_analysis_work_dir(config: TwopyConfig, recording_dir: Path) -> Path:
    """Resolve the directory used for interactive analysis work.

    Args:
        config: Loaded twopy configuration.
        recording_dir: Source recording folder being analyzed.

    Returns:
        Local cache directory when ``analysis_caching`` is true; otherwise the
        configured publish/output directory.

    The work directory is where converted HDF5, ROIs, and analysis outputs are
    read during normal analysis. ``analysis_output`` remains the publish
    destination used for sync.
    """
    if config.analysis_caching:
        return resolve_analysis_cache_dir(config, recording_dir)
    return resolve_analysis_output_dir(config, recording_dir)


def _required_path(config: dict[object, object], key: str, config_path: Path) -> Path:
    """Read one required path value from parsed YAML.

    Args:
        config: Parsed YAML mapping.
        key: Required key to read.
        config_path: Source config file path, used for clear errors.

    Returns:
        Expanded ``Path`` value for the requested key.

    Raises:
        ValueError: If the key is missing or is not a string.

    This helper keeps config validation explicit and avoids hidden defaults.
    """
    value = config.get(key)
    if not isinstance(value, str) or value == "":
        msg = f"twopy config key {key!r} must be a non-empty string: {config_path}"
        raise ValueError(msg)

    return _path_from_config_value(value)


def _required_path_tuple(
    config: dict[object, object],
    key: str,
    config_path: Path,
) -> tuple[Path, ...]:
    """Read one required non-empty path list from parsed YAML.

    Args:
        config: Parsed YAML mapping.
        key: Required key to read.
        config_path: Source config file path, used for clear errors.

    Returns:
        Expanded paths in config order.

    Raises:
        ValueError: If the key is missing, empty, or contains non-string items.
    """
    value = config.get(key)
    if not isinstance(value, list) or not value:
        msg = (
            f"twopy config key {key!r} must be a non-empty list of paths: {config_path}"
        )
        raise ValueError(msg)

    paths: list[Path] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or item == "":
            msg = (
                f"twopy config key {key!r} item {index} must be a "
                f"non-empty path string: {config_path}"
            )
            raise ValueError(msg)
        paths.append(_path_from_config_value(item))
    return tuple(paths)


def _external_analysis_cache_dir(
    config: TwopyConfig,
    source_dir: Path,
) -> Path:
    """Return a stable cache directory for recordings outside ``data_paths``.

    Args:
        config: Loaded twopy configuration.
        source_dir: Source recording folder.

    Returns:
        External cache directory path.
    """
    source_key = str(source_dir.resolve(strict=False))
    digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]
    folder_name = source_dir.name or "recording"
    return config.analysis_cache_dir / "_external" / f"{folder_name}-{digest}"


def _database_access(
    config: dict[object, object],
    config_path: Path,
) -> DatabaseAccess:
    """Read the optional database access mode from parsed YAML.

    Args:
        config: Parsed YAML mapping.
        config_path: Source config file path, used for clear errors.

    Returns:
        ``copy`` or ``direct``. Missing values default to ``copy``.

    The default favors local DB copies because database queries over the network
    can be slow, but transferring the DB file locally is usually fast. twopy
    verifies whether a copy is stale before copying again.
    """
    value = config.get("database_access", "copy")
    if value == "copy":
        return "copy"
    if value == "direct":
        return "direct"

    msg = (
        f"twopy config key 'database_access' must be 'copy' or 'direct': {config_path}"
    )
    raise ValueError(msg)


def _analysis_output(
    config: dict[object, object],
    config_path: Path,
) -> AnalysisOutputMode:
    """Read the required analysis output setting from parsed YAML.

    Args:
        config: Parsed YAML mapping.
        config_path: Source config file path, used for clear errors.

    Returns:
        ``source`` or an expanded output root path.

    Raises:
        ValueError: If the key is absent or not a non-empty string.

    Keeping this explicit avoids accidental writes to unclear locations.
    """
    value = config.get("analysis_output")
    if not isinstance(value, str) or value == "":
        msg = (
            "twopy config key 'analysis_output' must be 'source' or a "
            f"non-empty path string: {config_path}"
        )
        raise ValueError(msg)

    if value == "source":
        return "source"

    return _path_from_config_value(value)


def _optional_bool(
    config: dict[object, object],
    key: str,
    config_path: Path,
    *,
    default: bool,
) -> bool:
    """Read an optional boolean config value.

    Args:
        config: Parsed YAML mapping.
        key: Optional key to read.
        config_path: Source config file path, used for clear errors.
        default: Value returned when the key is absent.

    Returns:
        Parsed boolean value.
    """
    value = config.get(key, default)
    if isinstance(value, bool):
        return value

    msg = f"twopy config key {key!r} must be true or false: {config_path}"
    raise ValueError(msg)


def _optional_path(
    config: dict[object, object],
    key: str,
    config_path: Path,
    *,
    default: Path,
) -> Path:
    """Read an optional path config value.

    Args:
        config: Parsed YAML mapping.
        key: Optional key to read.
        config_path: Source config file path, used for clear errors.
        default: Value returned when the key is absent.

    Returns:
        Expanded path value.
    """
    value = config.get(key)
    if value is None:
        return default.expanduser()
    return _optional_path_value(value, key, config_path)


def _optional_path_value(value: object, key: str, config_path: Path) -> Path:
    """Validate and expand one optional path value.

    Args:
        value: Raw YAML value.
        key: Config key, used for clear errors.
        config_path: Source config file path, used for clear errors.

    Returns:
        Expanded path.
    """
    if not isinstance(value, str) or value == "":
        msg = f"twopy config key {key!r} must be a non-empty string: {config_path}"
        raise ValueError(msg)
    return _path_from_config_value(value)


def _optional_path_tuple(
    config: dict[object, object],
    key: str,
    config_path: Path,
) -> tuple[Path, ...]:
    """Read an optional list of paths from config.

    Args:
        config: Parsed YAML mapping.
        key: Optional path-list key to read.
        config_path: Source config file path, used for clear errors.

    Returns:
        Expanded paths in config order.
    """
    value = config.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        msg = f"twopy config key {key!r} must be a list of paths: {config_path}"
        raise ValueError(msg)
    paths: list[Path] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or item == "":
            msg = (
                f"twopy config key {key!r} item {index} must be a "
                f"non-empty path string: {config_path}"
            )
            raise ValueError(msg)
        paths.append(_path_from_config_value(item))
    return tuple(paths)


def _path_from_config_value(value: str) -> Path:
    """Expand one validated config path string.

    Args:
        value: Non-empty path string from YAML.

    Returns:
        Expanded path.
    """
    return Path(value).expanduser()
