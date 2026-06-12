"""Load twopy configuration from a small YAML file.

The config layer keeps machine-specific paths out of analysis code. GUI and
script callers get one typed object with database paths, cache paths, custom
workflow paths, and output settings.
"""

import hashlib
import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal

import yaml

from twopy.pixel_calibration import DEFAULT_PIXEL_CALIBRATION_PATH

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_ANALYSIS_CACHE_DIR",
    "CONFIG_ENV_VAR",
    "AnalysisOutputMode",
    "DatabaseAccess",
    "TwopyConfig",
    "config_search_paths",
    "config_template_text",
    "data_path_match",
    "data_path_recording_candidates",
    "discover_config_path",
    "load_config",
    "preferred_config_path",
    "resolve_data_recording_path",
    "resolve_analysis_cache_dir",
    "resolve_analysis_output_dir",
    "resolve_analysis_work_dir",
    "resolve_config_path",
    "user_config_path",
    "write_config_template",
]

DEFAULT_CONFIG_PATH = Path("config.yml")
DEFAULT_ANALYSIS_CACHE_DIR = Path.home() / ".cache" / "twopy" / "recordings"
CONFIG_ENV_VAR = "TWOPY_CONFIG"
_CONFIG_DIRNAME = "twopy"
_CONFIG_FILENAME = "config.yml"
_CONFIG_TEMPLATE_RESOURCE = "config.example.yml"
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


def user_config_path() -> Path:
    """Return the usual twopy config path for this user.

    Args:
        None.

    Returns:
        ``~/.config/twopy/config.yml`` on macOS and Linux, or the user's
        roaming app-data path on Windows.

    This path stays the same no matter where twopy starts, so installed GUI
    users can set up their paths once.
    """
    if os.name == "nt":
        roaming_root = os.environ.get("APPDATA")
        if roaming_root is not None and roaming_root != "":
            return Path(roaming_root) / _CONFIG_DIRNAME / _CONFIG_FILENAME
        return Path.home() / "AppData" / "Roaming" / _CONFIG_DIRNAME / _CONFIG_FILENAME

    return Path.home() / ".config" / _CONFIG_DIRNAME / _CONFIG_FILENAME


def preferred_config_path() -> Path:
    """Return where setup should write a new config file.

    Args:
        None.

    Returns:
        The path from ``TWOPY_CONFIG`` when set, otherwise the user config path.

    Setup uses ``TWOPY_CONFIG`` when it is set. Otherwise, first launch writes
    the usual config file for this user.
    """
    env_path = _env_config_path()
    if env_path is not None:
        return env_path
    return user_config_path()


def config_search_paths() -> tuple[Path, ...]:
    """Return the default config search order.

    Args:
        None.

    Returns:
        Config paths in the order twopy checks them.

    ``TWOPY_CONFIG`` comes first when it is set. A source checkout can still
    use a local ``config.yml``. Installed users get one stable config file in
    their user folder.
    """
    candidates: list[Path] = []
    env_path = _env_config_path()
    if env_path is not None:
        candidates.append(env_path)
    candidates.append(DEFAULT_CONFIG_PATH)
    candidates.append(user_config_path())
    return tuple(candidates)


def discover_config_path() -> Path | None:
    """Find the first existing default config file.

    Args:
        None.

    Returns:
        The first existing config path, or ``None`` when setup is still needed.

    Callers that need a clear "not configured yet" branch use this before
    loading and validating YAML.
    """
    env_path = _env_config_path()
    if env_path is not None:
        return env_path if env_path.is_file() else None

    for candidate in (DEFAULT_CONFIG_PATH, user_config_path()):
        expanded = candidate.expanduser()
        if expanded.is_file():
            return expanded
    return None


def resolve_config_path(path: Path | None = None) -> Path:
    """Return the config path twopy should load.

    Args:
        path: Optional config path. ``None`` or the old ``config.yml`` default
            uses twopy's usual config search.

    Returns:
        The given path, an existing config found by search, or the path that
        setup would write when no config exists yet.

    Returning the setup path for a missing default keeps error messages and
    first-launch setup pointed at the same file.
    """
    if path is not None and path != DEFAULT_CONFIG_PATH:
        return path.expanduser()

    discovered = discover_config_path()
    if discovered is not None:
        return discovered
    return preferred_config_path()


def config_template_text() -> str:
    """Read the packaged example config template.

    Args:
        None.

    Returns:
        YAML text for a new editable ``config.yml``.

    The template is included in the installed package, so pip-installed users
    do not need a source checkout to create their first config file.
    """
    template = files("twopy").joinpath(_CONFIG_TEMPLATE_RESOURCE)
    return template.read_text(encoding="utf-8")


def write_config_template(path: Path | None = None, *, force: bool = False) -> Path:
    """Create an editable config template.

    Args:
        path: Optional destination. Defaults to the preferred setup path.
        force: Whether to replace an existing file.

    Returns:
        The path that was written.

    Raises:
        FileExistsError: If the target exists and ``force`` is false.

    Setup writes a plain YAML file that users can inspect before twopy reads
    any paths from it.
    """
    config_path = (path if path is not None else preferred_config_path()).expanduser()
    if config_path.exists() and not force:
        msg = f"twopy config already exists: {config_path}"
        raise FileExistsError(msg)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_template_text(), encoding="utf-8")
    return config_path


def load_config(path: Path | None = None) -> TwopyConfig:
    """Load twopy configuration from YAML.

    Args:
        path: Optional YAML config file path. Defaults to twopy's usual config
            search: ``TWOPY_CONFIG``, local ``config.yml``, then the user
            config path.

    Returns:
        A typed configuration object with paths, DB access, and output mode.

    Raises:
        FileNotFoundError: If the config file is absent.
        ValueError: If the YAML is not a mapping or required keys are missing.

    YAML is used because the file is edited by people and benefits from comments.
    """
    config_path = resolve_config_path(path)
    if not config_path.is_file():
        msg = _missing_config_message(path, config_path)
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

    raw_config: dict[object, object] = dict(loaded.items())

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


def _missing_config_message(path: Path | None, resolved_path: Path) -> str:
    """Return the missing-config error shown to scripts and tests."""
    if path is not None and path != DEFAULT_CONFIG_PATH:
        return f"Missing twopy config file: {resolved_path}"
    if _env_config_path() is not None:
        return f"Missing twopy config file from {CONFIG_ENV_VAR}: {resolved_path}"

    searched = "\n".join(f"- {candidate}" for candidate in config_search_paths())
    return (
        "Missing twopy config file. Run `twopy config setup`, edit the file, "
        f"then try again.\nSearched:\n{searched}"
    )


def _env_config_path() -> Path | None:
    """Return the explicit config path from the environment, when set."""
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path is None or env_path == "":
        return None
    return Path(env_path).expanduser()


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
    candidates = _recording_candidates_for_relative_parts(config, relative_parts)

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

    Output and cache paths need the same path below the data root that database
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


def data_path_recording_candidates(
    config: TwopyConfig,
    recording_dir: Path,
) -> tuple[Path, ...]:
    """Return every configured mirror candidate for one recording path.

    Args:
        config: Loaded twopy configuration with ordered data roots.
        recording_dir: Recording folder under one configured data root.

    Returns:
        Candidate recording folders under every configured root with the same
        root-relative path, or an empty tuple when ``recording_dir`` is external.

    Database loads may fall back to the first configured root when no source
    copy is mounted so cache-only loads can still work. This helper exposes the
    full mirrored candidate set for diagnostics and audits without changing
    that load behavior.
    """
    match = data_path_match(config, recording_dir)
    if match is None:
        return ()
    _matched_root, relative_recording = match
    return _recording_candidates_for_relative_parts(
        config,
        tuple(str(part) for part in relative_recording.parts),
    )


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
        configured output directory.

    The work directory is where converted HDF5, ROIs, and analysis outputs are
    read during normal analysis. ``analysis_output`` remains the folder where
    finished files are copied.
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


def _recording_candidates_for_relative_parts(
    config: TwopyConfig,
    relative_parts: tuple[str, ...],
) -> tuple[Path, ...]:
    """Return ordered data-root candidates for one relative recording path."""
    candidates = tuple(
        data_root.expanduser().joinpath(*relative_parts)
        for data_root in config.data_paths
    )
    if not candidates:
        msg = "twopy config must include at least one data path"
        raise ValueError(msg)
    return candidates


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
