"""Database search helpers for browsing recording experiments.

Inputs: configured lab database paths, user-facing text filters, and typed
``DatabaseExperiment`` rows.
Outputs: filtered experiment rows, browsable hierarchy nodes, and source
recording paths.

The napari search dialog uses this module so database querying, grouping, and
path resolution stay testable without Qt imports.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

from twopy.database.modeled import find_stimulus_presentations
from twopy.database.types import DatabaseExperiment

if TYPE_CHECKING:
    from twopy.config import TwopyConfig

__all__ = [
    "ExperimentHierarchyField",
    "ExperimentSearchFilters",
    "ExperimentSearchNode",
    "build_experiment_search_tree",
    "date_label_for_experiment",
    "database_hemisphere_for_recording_path",
    "find_database_experiment_for_recording_path",
    "find_recording_search_results",
    "normalize_experiment_date_filter",
    "recording_path_for_database_experiment",
    "recording_paths_for_database_experiments",
    "time_label_for_experiment",
]

ExperimentHierarchyField = Literal[
    "user",
    "cell_type",
    "sensor",
    "stimulus",
    "date",
    "time",
]

UNKNOWN_LABEL = "(unknown)"
DEFAULT_EXPERIMENT_SEARCH_LIMIT = 50_000
_DATE_FILTER_PATTERN = re.compile(
    r"^\s*(\d{4})[^0-9A-Za-z]+(\d{2})[^0-9A-Za-z]+(\d{2})\s*$"
)


@dataclass(frozen=True)
class ExperimentSearchFilters:
    """User-facing filters for recording database search.

    Inputs: optional text from the database-search dialog.
    Outputs: immutable normalized filter values passed to database search.

    Blank strings mean no filtering, matching the GUI contract.
    """

    user: str | None = None
    cell_type: str | None = None
    sensor: str | None = None
    stimulus: str | None = None
    date: str | None = None
    limit: int = DEFAULT_EXPERIMENT_SEARCH_LIMIT


@dataclass(frozen=True)
class ExperimentSearchNode:
    """One node in the recording search hierarchy.

    Inputs: a hierarchy label, the field it represents, child nodes, and the
    experiments below it.
    Outputs: an immutable tree the Qt dialog can render directly.

    Every non-root node carries all descendant experiments so selecting a user,
    cell type, stimulus, date, or time can load the matching recordings.
    """

    label: str
    field: ExperimentHierarchyField | None
    experiments: tuple[DatabaseExperiment, ...]
    children: tuple[ExperimentSearchNode, ...] = ()


def find_recording_search_results(
    config: TwopyConfig,
    filters: ExperimentSearchFilters,
    *,
    cache_dir: Path | None = None,
) -> tuple[DatabaseExperiment, ...]:
    """Find recording experiments for the database-search dialog.

    Args:
        config: Loaded twopy configuration.
        filters: User-facing text filters. Blank values are ignored.
        cache_dir: Optional local database-copy cache.

    Returns:
        Typed database experiments matching all active filters.

    The SQL query stays in the database package. This helper maps GUI labels
    such as ``user`` and ``sensor`` onto the stable modeled query fields.
    """
    experiments = find_stimulus_presentations(
        config.database_path,
        date_contains=normalize_experiment_date_filter(filters.date),
        stimulus_contains=_clean_filter(filters.stimulus),
        sensor_contains=_clean_filter(filters.sensor),
        cell_type_contains=_clean_filter(filters.cell_type),
        person_contains=_clean_filter(filters.user),
        limit=filters.limit,
        database_access=config.database_access,
        cache_dir=cache_dir,
    )
    return _unique_experiments_by_relative_path(experiments)


def normalize_experiment_date_filter(value: str | None) -> str | None:
    r"""Normalize one user-facing experiment date filter.

    Args:
        value: Optional date text from the search dialog or scripting API.

    Returns:
        ``YYYY-MM-DD`` for date-like input with non-letter separators, a
        stripped non-empty fallback string for other input, or ``None`` for
        blank input.

    This keeps the GUI forgiving when a user types ``YYYY/MM/DD``,
    ``YYYY\\MM\\DD``, ``YYYY.MM.DD``, or similar separator variants while
    preserving the existing substring-search behavior for non-date text.
    """
    cleaned = _clean_filter(value)
    if cleaned is None:
        return None

    match = _DATE_FILTER_PATTERN.fullmatch(cleaned)
    if match is None:
        return cleaned

    year, month, day = (int(part) for part in match.groups())
    return date(year, month, day).isoformat()


def build_experiment_search_tree(
    experiments: tuple[DatabaseExperiment, ...],
) -> ExperimentSearchNode:
    """Build the folder-like hierarchy for database search results.

    Args:
        experiments: Database rows to group.

    Returns:
        Root node with descendants grouped by user, cell type, sensor, stimulus,
        date, then experiment time.

    Nodes are sorted by display label so repeated searches render in a stable
    order. Unknown values are grouped explicitly instead of disappearing.
    """
    return ExperimentSearchNode(
        label="Results",
        field=None,
        experiments=experiments,
        children=_group_nodes(
            experiments,
            field="user",
            label_for_experiment=lambda experiment: _optional_label(experiment.person),
            next_builder=_cell_type_nodes,
        ),
    )


def find_database_experiment_for_recording_path(
    config: TwopyConfig,
    recording_dir: Path,
    *,
    cache_dir: Path | None = None,
) -> DatabaseExperiment | None:
    """Return the database row that owns one source recording folder.

    Args:
        config: Loaded twopy configuration with database and data roots.
        recording_dir: Source microscope recording folder.
        cache_dir: Optional local database-copy cache.

    Returns:
        Matching database experiment, or ``None`` when no row matches.

    The microscope folder itself does not contain fly-eye metadata. The lab
    database stores that mandatory recording metadata in ``fly.eye``; this
    helper resolves the converted recording's source path back to that row.
    """
    candidates = _candidate_database_relative_paths(config, recording_dir)
    if len(candidates) == 0:
        return None

    experiments: list[DatabaseExperiment] = []
    seen_ids: set[tuple[Path, int]] = set()
    for query_term in _path_query_terms(recording_dir):
        for experiment in find_stimulus_presentations(
            config.database_path,
            path_contains=query_term,
            limit=DEFAULT_EXPERIMENT_SEARCH_LIMIT,
            database_access=config.database_access,
            cache_dir=cache_dir,
        ):
            key = (
                experiment.database_path.expanduser().resolve(strict=False),
                experiment.stimulus_presentation_id,
            )
            if key in seen_ids:
                continue
            seen_ids.add(key)
            experiments.append(experiment)

    matches_by_path: dict[str, DatabaseExperiment] = {}
    for experiment in experiments:
        relative_path = _normalized_relative_path(experiment.relative_data_path)
        if any(
            _relative_path_matches(relative_path, candidate) for candidate in candidates
        ):
            matches_by_path.setdefault(relative_path, experiment)

    if len(matches_by_path) > 1:
        matched = ", ".join(sorted(matches_by_path))
        msg = f"Multiple database rows match recording path {recording_dir}: {matched}"
        raise ValueError(msg)
    if len(matches_by_path) == 0:
        return None
    return next(iter(matches_by_path.values()))


def database_hemisphere_for_recording_path(
    config: TwopyConfig,
    recording_dir: Path,
    *,
    cache_dir: Path | None = None,
) -> str | None:
    """Return ``fly.eye`` for one source recording folder.

    Args:
        config: Loaded twopy configuration.
        recording_dir: Source microscope recording folder.
        cache_dir: Optional local database-copy cache.

    Returns:
        ``"left"``, ``"right"``, or ``None`` when the database row is absent or
        missing the field.
    """
    experiment = find_database_experiment_for_recording_path(
        config,
        recording_dir,
        cache_dir=cache_dir,
    )
    if experiment is None or experiment.hemisphere is None:
        return None
    normalized = experiment.hemisphere.strip().casefold()
    if normalized not in {"left", "right"}:
        msg = (
            f"Database fly.eye for recording {recording_dir} must be "
            f"'left' or 'right'; got {experiment.hemisphere!r}."
        )
        raise ValueError(msg)
    return normalized


def recording_path_for_database_experiment(
    config: TwopyConfig,
    experiment: DatabaseExperiment,
) -> Path:
    """Resolve one database experiment into a source recording folder.

    Args:
        config: Loaded twopy configuration with ``data_path``.
        experiment: Database row with a relative data path.

    Returns:
        Source recording folder under the configured data root.

    Raises:
        ValueError: If the database row does not contain a safe relative path.

    The database stores Windows-style separators in observed rows. Twopy treats
    the value as relative path text and resolves it beneath ``data_path`` before
    passing it to the existing recording loader.
    """
    relative_path = _normalized_relative_path(experiment.relative_data_path)
    if relative_path == "":
        msg = (
            "Database experiment has an empty relativeDataPath: "
            f"{experiment.stimulus_presentation_id}"
        )
        raise ValueError(msg)

    path_parts = PurePosixPath(relative_path).parts
    if any(part in {"", ".", ".."} for part in path_parts):
        msg = (
            "Database experiment relativeDataPath must stay under data_path: "
            f"{experiment.relative_data_path!r}"
        )
        raise ValueError(msg)

    return config.data_path.expanduser().joinpath(*path_parts)


def recording_paths_for_database_experiments(
    config: TwopyConfig,
    experiments: tuple[DatabaseExperiment, ...],
) -> tuple[Path, ...]:
    """Resolve database experiments into unique source recording folders.

    Args:
        config: Loaded twopy configuration with ``data_path``.
        experiments: Database rows to load.

    Returns:
        Unique source recording folders in search-result order.

    Parent hierarchy nodes can contain multiple database rows. Deduplicating
    here prevents the GUI from loading the same folder twice when duplicate
    rows point at one recording.
    """
    paths: list[Path] = []
    seen: set[Path] = set()
    for experiment in experiments:
        path = recording_path_for_database_experiment(config, experiment)
        if path not in seen:
            paths.append(path)
            seen.add(path)
    return tuple(paths)


def date_label_for_experiment(experiment: DatabaseExperiment) -> str:
    """Return the date folder label for one experiment.

    Args:
        experiment: Database experiment row.

    Returns:
        ``YYYY-MM-DD`` when available, otherwise ``(unknown)``.
    """
    date_parts = _date_time_parts_from_relative_path(experiment.relative_data_path)
    if date_parts is not None:
        year, month_day, _time = date_parts
        return f"{year}-{month_day.replace('_', '-')}"

    if experiment.date is not None and len(experiment.date) >= 10:
        return experiment.date[:10]
    return UNKNOWN_LABEL


def time_label_for_experiment(experiment: DatabaseExperiment) -> str:
    """Return the experiment-time label beneath one date node.

    Args:
        experiment: Database experiment row.

    Returns:
        ``HH:MM:SS`` when available, otherwise a stable experiment id label.
    """
    date_parts = _date_time_parts_from_relative_path(experiment.relative_data_path)
    if date_parts is not None:
        _year, _month_day, time = date_parts
        return time.replace("_", ":")

    if experiment.date is not None and len(experiment.date) >= 19:
        return experiment.date[11:19]
    return f"id {experiment.stimulus_presentation_id}"


def _unique_experiments_by_relative_path(
    experiments: tuple[DatabaseExperiment, ...],
) -> tuple[DatabaseExperiment, ...]:
    """Return the first experiment row for each source recording path.

    Args:
        experiments: Database rows from all configured SQLite files.

    Returns:
        Deduplicated rows in original query order.

    The lab keeps the same stimulus rows in ``experimentLog.db`` and
    ``experimentInitLog.db``. The GUI browses source recordings, so duplicate DB
    rows for one ``relativeDataPath`` should appear as one experiment.
    """
    unique: list[DatabaseExperiment] = []
    seen: set[str] = set()
    for experiment in experiments:
        key = _normalized_relative_path(experiment.relative_data_path)
        if key in seen:
            continue
        unique.append(experiment)
        seen.add(key)
    return tuple(unique)


def _date_time_parts_from_relative_path(
    relative_data_path: str,
) -> tuple[str, str, str] | None:
    """Return ``YYYY``, ``MM_DD``, and ``HH_MM_SS`` from a source path."""
    parts = PurePosixPath(_normalized_relative_path(relative_data_path)).parts
    for index in range(len(parts) - 2):
        year = parts[index]
        month_day = parts[index + 1]
        time = parts[index + 2]
        if (
            len(year) == 4
            and year.isdigit()
            and _is_underscore_date(month_day)
            and _is_underscore_time(time)
        ):
            return year, month_day, time
    return None


def _candidate_database_relative_paths(
    config: TwopyConfig,
    recording_dir: Path,
) -> tuple[str, ...]:
    """Return normalized DB path candidates for one source folder."""
    candidates: list[str] = []
    path = recording_dir.expanduser()
    try:
        relative = path.relative_to(config.data_path.expanduser())
    except ValueError:
        relative = None
    if relative is not None:
        candidates.append(_normalized_relative_path(relative.as_posix()))

    parts = path.parts
    for index in range(2, len(parts) - 2):
        if (
            len(parts[index]) == 4
            and parts[index].isdigit()
            and _is_underscore_date(parts[index + 1])
            and _is_underscore_time(parts[index + 2])
        ):
            candidates.append(
                _normalized_relative_path(
                    PurePosixPath(*parts[index - 2 : index + 3]).as_posix()
                )
            )
            break
    return tuple(dict.fromkeys(candidates))


def _path_query_terms(recording_dir: Path) -> tuple[str, ...]:
    """Return narrow SQL ``LIKE`` terms for path lookup before exact matching."""
    parts = recording_dir.expanduser().parts
    terms = [recording_dir.name]
    if len(parts) >= 3:
        terms.append("\\".join(parts[-3:]))
        terms.append("/".join(parts[-3:]))
    return tuple(dict.fromkeys(term for term in terms if term))


def _relative_path_matches(relative_path: str, candidate: str) -> bool:
    """Return whether a normalized DB path matches one candidate path."""
    return relative_path == candidate or relative_path.endswith(f"/{candidate}")


def _cell_type_nodes(
    experiments: tuple[DatabaseExperiment, ...],
) -> tuple[ExperimentSearchNode, ...]:
    """Group user children by cell type."""
    return _group_nodes(
        experiments,
        field="cell_type",
        label_for_experiment=lambda experiment: _optional_label(experiment.cell_type),
        next_builder=_sensor_nodes,
    )


def _sensor_nodes(
    experiments: tuple[DatabaseExperiment, ...],
) -> tuple[ExperimentSearchNode, ...]:
    """Group cell-type children by sensor."""
    return _group_nodes(
        experiments,
        field="sensor",
        label_for_experiment=lambda experiment: _optional_label(
            experiment.fluorescent_protein
        ),
        next_builder=_stimulus_nodes,
    )


def _stimulus_nodes(
    experiments: tuple[DatabaseExperiment, ...],
) -> tuple[ExperimentSearchNode, ...]:
    """Group sensor children by stimulus function."""
    return _group_nodes(
        experiments,
        field="stimulus",
        label_for_experiment=lambda experiment: _optional_label(
            experiment.stimulus_function
        ),
        next_builder=_date_nodes,
    )


def _date_nodes(
    experiments: tuple[DatabaseExperiment, ...],
) -> tuple[ExperimentSearchNode, ...]:
    """Group stimulus children by date."""
    return _group_nodes(
        experiments,
        field="date",
        label_for_experiment=date_label_for_experiment,
        next_builder=_time_nodes,
    )


def _time_nodes(
    experiments: tuple[DatabaseExperiment, ...],
) -> tuple[ExperimentSearchNode, ...]:
    """Group date children by experiment time."""
    return _group_nodes(
        experiments,
        field="time",
        label_for_experiment=time_label_for_experiment,
        next_builder=lambda grouped: (),
    )


def _group_nodes(
    experiments: tuple[DatabaseExperiment, ...],
    *,
    field: ExperimentHierarchyField,
    label_for_experiment: Callable[[DatabaseExperiment], str],
    next_builder: Callable[
        [tuple[DatabaseExperiment, ...]],
        tuple[ExperimentSearchNode, ...],
    ],
) -> tuple[ExperimentSearchNode, ...]:
    """Group experiments for one hierarchy level."""
    grouped: dict[str, list[DatabaseExperiment]] = {}
    for experiment in experiments:
        grouped.setdefault(label_for_experiment(experiment), []).append(experiment)

    nodes: list[ExperimentSearchNode] = []
    for label in sorted(grouped, key=_sort_label):
        node_experiments = tuple(grouped[label])
        nodes.append(
            ExperimentSearchNode(
                label=label,
                field=field,
                experiments=node_experiments,
                children=next_builder(node_experiments),
            )
        )
    return tuple(nodes)


def _optional_label(value: str | None) -> str:
    """Return a display label for a nullable database value."""
    if value is None or value == "":
        return UNKNOWN_LABEL
    return value


def _normalized_relative_path(relative_data_path: str) -> str:
    """Normalize a database source path for grouping and deduplication."""
    return relative_data_path.replace("\\", "/").strip("/")


def _is_underscore_date(value: str) -> bool:
    """Return whether text looks like a source-folder ``MM_DD`` date."""
    return (
        len(value) == 5
        and value[2] == "_"
        and value[:2].isdigit()
        and value[3:].isdigit()
    )


def _is_underscore_time(value: str) -> bool:
    """Return whether text looks like a source-folder ``HH_MM_SS`` time."""
    return (
        len(value) == 8
        and value[2] == "_"
        and value[5] == "_"
        and value[:2].isdigit()
        and value[3:5].isdigit()
        and value[6:].isdigit()
    )


def _clean_filter(value: str | None) -> str | None:
    """Return a non-empty stripped filter string or ``None``."""
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _sort_label(label: str) -> tuple[bool, str]:
    """Sort unknown values last and all other labels case-insensitively."""
    return (label == UNKNOWN_LABEL, label.casefold())
