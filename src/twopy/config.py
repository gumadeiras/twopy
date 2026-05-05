"""Load twopy configuration from a small YAML file.

Inputs: a YAML config file with lab paths and database access mode.
Outputs: a typed ``TwopyConfig`` object with normalized values.

The config layer keeps machine-specific paths out of analysis code and makes the
GUI easy to point at the right database and recording roots.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from twopy.database.types import DatabaseAccess

__all__ = ["DEFAULT_CONFIG_PATH", "TwopyConfig", "load_config"]

DEFAULT_CONFIG_PATH = Path("config.yml")


@dataclass(frozen=True)
class TwopyConfig:
    """Machine-local paths used by twopy.

    Inputs: database path, data path, and DB access mode from ``config.yml``.
    Outputs: immutable config values that GUI and analysis code can pass around.

    The paths are not required to exist during parsing so tests and machines
    without mounted lab drives can still load and inspect configuration.
    """

    database_path: Path
    data_path: Path
    database_access: DatabaseAccess


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> TwopyConfig:
    """Load twopy configuration from YAML.

    Args:
        path: YAML config file path. Defaults to ``config.yml`` in the current
            working directory.

    Returns:
        A typed configuration object with paths and database access mode.

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
        loaded: object = yaml.safe_load(config_file)

    if not isinstance(loaded, dict):
        msg = f"twopy config must be a YAML mapping: {config_path}"
        raise ValueError(msg)

    raw_config = cast(dict[object, object], loaded)

    return TwopyConfig(
        database_path=_required_path(raw_config, "database_path", config_path),
        data_path=_required_path(raw_config, "data_path", config_path),
        database_access=_database_access(raw_config, config_path),
    )


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

    return Path(value).expanduser()


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

    The default favors local DB copies because mounted lab volumes can be slow
    and we already verify whether a copy is stale before copying again.
    """
    value = config.get("database_access", "copy")
    if value in {"copy", "direct"}:
        return cast(DatabaseAccess, value)

    msg = (
        f"twopy config key 'database_access' must be 'copy' or 'direct': {config_path}"
    )
    raise ValueError(msg)
