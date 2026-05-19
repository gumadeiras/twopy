"""Public script-friendly API for twopy.

Inputs: simple Python arguments such as year, genotype, stimulus, and paths.
Outputs: typed twopy objects returned by lower-level modules.

This module is the stable place for scripts to import from. It keeps scripts away
from implementation details like SQLite table names.
"""

from pathlib import Path

from twopy.config import DEFAULT_CONFIG_PATH, load_config
from twopy.database import DatabaseAccess, DatabaseExperiment
from twopy.database import find_recordings as _find_recordings

__all__ = ["find_recordings"]


def find_recordings(
    *,
    year: int | None = None,
    month: int | None = None,
    day: int | None = None,
    genotype: str | None = None,
    stimulus: str | None = None,
    sensor: str | None = None,
    cell_type: str | None = None,
    hemisphere: str | None = None,
    person: str | None = None,
    database_access: DatabaseAccess | None = None,
    database_cache_dir: Path | None = None,
    limit: int = 100,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> tuple[DatabaseExperiment, ...]:
    """Find recording experiments using the configured lab database path.

    Args:
        year: Optional experiment year.
        month: Optional experiment month.
        day: Optional experiment day-of-month.
        genotype: Optional genotype substring.
        stimulus: Optional stimulus-function substring.
        sensor: Optional sensor substring from fly ``fluorescentProtein``.
        cell_type: Optional cell-type substring from fly ``cellType``.
        hemisphere: Optional exact recording hemisphere, such as ``left`` or
            ``right``.
        person: Optional experimenter substring from fly ``surgeon``.
        database_access: Optional override for configured DB access. ``direct``
            queries mounted DB files directly. ``copy`` copies DB files locally
            because network DB queries can be slow while file transfer is fast.
        database_cache_dir: Optional cache directory for local DB copies.
        limit: Maximum number of recordings to return.
        config_path: YAML config file containing ``database_path``.

    Returns:
        Typed recording experiment rows.

    Example:
        ``find_recordings(year=2023, month=10, genotype="gh146")``

    The function loads ``config.yml`` by default so scripts can start with a
    single import and call.
    """
    config = load_config(config_path)
    return _find_recordings(
        config.database_path,
        year=year,
        month=month,
        day=day,
        genotype=genotype,
        stimulus=stimulus,
        sensor=sensor,
        cell_type=cell_type,
        hemisphere=hemisphere,
        person=person,
        database_access=database_access or config.database_access,
        cache_dir=database_cache_dir,
        limit=limit,
    )
