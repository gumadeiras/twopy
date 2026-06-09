"""Persist reusable napari database-search filter sets.

Inputs: user-entered database search filters and a machine-local YAML file.
Outputs: typed favorite records that the Qt search dialog can list and apply.

Favorites are GUI workflow state, not database query logic. Keeping this module
Qt-free lets tests validate persistence and normalization without opening a
dialog while the napari dialog stays responsible only for widgets.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from twopy.database.search import (
    ExperimentSearchFilters,
    normalize_experiment_date_filter,
)

__all__ = [
    "DEFAULT_DATABASE_FAVORITES_PATH",
    "ExperimentSearchFavorite",
    "database_search_favorite_filters_are_empty",
    "database_search_favorite_tooltip",
    "default_database_search_favorite_name",
    "load_database_search_favorites",
    "normalized_database_search_favorite",
    "normalized_database_search_filters",
    "replace_database_search_favorite",
    "save_database_search_favorites",
    "update_database_search_favorite",
]

DEFAULT_DATABASE_FAVORITES_PATH = (
    Path.home() / ".config" / "twopy" / "database_search_favorites.yml"
)
_FAVORITE_FIELDS = ("user", "cell_type", "sensor", "stimulus", "date")
_FAVORITE_LABELS = {
    "user": "User",
    "cell_type": "Cell type",
    "sensor": "Sensor",
    "stimulus": "Stimulus",
    "date": "Date",
}
_GENERATED_NAME_FIELD_LIMIT = 32
_GENERATED_NAME_LIMIT = 96


@dataclass(frozen=True)
class ExperimentSearchFavorite:
    """One named set of reusable database search filters.

    Inputs: a user-facing name and the filter values to restore.
    Outputs: immutable favorite records for persistence and dialog display.

    The stored filters mirror the database-search form. Result rows and source
    paths are intentionally excluded because the database can change while a
    reusable query remains valid.
    """

    name: str
    filters: ExperimentSearchFilters


def normalized_database_search_filters(
    filters: ExperimentSearchFilters,
) -> ExperimentSearchFilters:
    """Return cleaned search filters suitable for saving or comparing.

    Args:
        filters: Search filters from the dialog or from storage.

    Returns:
        Filters with blank text collapsed to ``None`` and date-like text
        normalized with the same rule used by database search.

    Normalizing before persistence keeps duplicate detection honest and makes
    saved favorites stable even when users type equivalent dates with different
    separators.
    """
    return ExperimentSearchFilters(
        user=_clean_filter(filters.user),
        cell_type=_clean_filter(filters.cell_type),
        sensor=_clean_filter(filters.sensor),
        stimulus=_clean_filter(filters.stimulus),
        date=normalize_experiment_date_filter(filters.date),
        limit=filters.limit,
    )


def normalized_database_search_favorite(
    favorite: ExperimentSearchFavorite,
) -> ExperimentSearchFavorite:
    """Return a favorite with a clean name and normalized filters.

    Args:
        favorite: Favorite from the dialog or storage.

    Returns:
        A normalized favorite safe to display and save.

    Raises:
        ValueError: If the name is blank or all filters are blank.

    Favorites must carry one real search constraint so the dialog does not save
    a broad "search everything" shortcut by accident.
    """
    name = favorite.name.strip()
    if name == "":
        msg = "Database search favorite name must not be blank."
        raise ValueError(msg)

    filters = normalized_database_search_filters(favorite.filters)
    if database_search_favorite_filters_are_empty(filters):
        msg = "Database search favorite must include at least one filter."
        raise ValueError(msg)

    return ExperimentSearchFavorite(name=name, filters=filters)


def database_search_favorite_filters_are_empty(
    filters: ExperimentSearchFilters,
) -> bool:
    """Return whether a filter set contains no reusable search constraints.

    Args:
        filters: Search filters to inspect.

    Returns:
        ``True`` when every saved filter field is blank.

    The search limit is not part of the GUI favorite contract, so it does not
    make an otherwise blank favorite valid.
    """
    normalized = normalized_database_search_filters(filters)
    return all(_filter_value(normalized, field) is None for field in _FAVORITE_FIELDS)


def default_database_search_favorite_name(filters: ExperimentSearchFilters) -> str:
    """Return a compact readable default name for a filter set.

    Args:
        filters: Search filters currently shown in the dialog.

    Returns:
        A human-readable label built from the active fields.

    The dialog uses this as the editable default when prompting for a favorite
    name, so users usually only need to confirm.
    """
    normalized = normalized_database_search_filters(filters)
    parts: list[str] = []
    for field in _FAVORITE_FIELDS:
        value = _filter_value(normalized, field)
        if value is None:
            continue
        label = _FAVORITE_LABELS[field]
        parts.append(f"{label}: {_shorten(value, _GENERATED_NAME_FIELD_LIMIT)}")
    return _shorten(", ".join(parts), _GENERATED_NAME_LIMIT)


def database_search_favorite_tooltip(favorite: ExperimentSearchFavorite) -> str:
    """Return full filter details for a favorite list tooltip.

    Args:
        favorite: Favorite to describe.

    Returns:
        Multi-line text containing the favorite name and every active filter.

    Long stimulus names stay visible in the tooltip while the list row remains
    compact enough for the lower-left panel.
    """
    normalized = normalized_database_search_favorite(favorite)
    lines = [normalized.name]
    for field in _FAVORITE_FIELDS:
        value = _filter_value(normalized.filters, field)
        if value is None:
            continue
        lines.append(f"{_FAVORITE_LABELS[field]}: {value}")
    return "\n".join(lines)


def load_database_search_favorites(
    path: Path = DEFAULT_DATABASE_FAVORITES_PATH,
) -> tuple[ExperimentSearchFavorite, ...]:
    """Load database search favorites from YAML.

    Args:
        path: Machine-local YAML file path.

    Returns:
        Normalized favorites in persisted order, deduplicated by filters.

    Raises:
        OSError: If the file exists but cannot be read.
        ValueError: If the YAML structure is not the expected favorite list.

    Missing files mean the user has no saved favorites yet.
    """
    favorites_path = path.expanduser()
    if not favorites_path.exists():
        return ()

    with favorites_path.open("r", encoding="utf-8") as favorite_file:
        loaded: object = yaml.safe_load(favorite_file)

    if loaded is None:
        return ()
    if not isinstance(loaded, dict):
        msg = f"Database search favorites must be a YAML mapping: {favorites_path}"
        raise ValueError(msg)

    raw_mapping = cast(dict[object, object], loaded)
    raw_favorites = raw_mapping.get("favorites", ())
    if raw_favorites is None:
        return ()
    if not isinstance(raw_favorites, list):
        msg = f"Database search favorites key must be a list: {favorites_path}"
        raise ValueError(msg)

    favorites: tuple[ExperimentSearchFavorite, ...] = ()
    for index, raw_favorite in enumerate(raw_favorites):
        favorite = _favorite_from_storage(raw_favorite, favorites_path, index)
        favorites = replace_database_search_favorite(favorites, favorite)
    return favorites


def save_database_search_favorites(
    favorites: tuple[ExperimentSearchFavorite, ...],
    path: Path = DEFAULT_DATABASE_FAVORITES_PATH,
) -> None:
    """Save database search favorites to YAML.

    Args:
        favorites: Favorite records to persist.
        path: Machine-local YAML file path.

    Returns:
        None.

    Raises:
        OSError: If the parent directory or file cannot be written.
        ValueError: If any favorite is invalid.

    The file is plain YAML so scientists can inspect or repair common searches
    without depending on twopy internals.
    """
    normalized: tuple[ExperimentSearchFavorite, ...] = ()
    for favorite in favorites:
        normalized = replace_database_search_favorite(normalized, favorite)

    favorites_path = path.expanduser()
    favorites_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "favorites": [
            {
                "name": favorite.name,
                "filters": _filters_to_storage(favorite.filters),
            }
            for favorite in normalized
        ],
    }
    with favorites_path.open("w", encoding="utf-8") as favorite_file:
        yaml.safe_dump(data, favorite_file, sort_keys=False)


def replace_database_search_favorite(
    favorites: tuple[ExperimentSearchFavorite, ...],
    favorite: ExperimentSearchFavorite,
) -> tuple[ExperimentSearchFavorite, ...]:
    """Return favorites with one normalized favorite inserted or replaced.

    Args:
        favorites: Existing favorite records.
        favorite: New or updated favorite.

    Returns:
        Favorites with duplicate filter sets and duplicate names removed.

    An exact filter duplicate updates that row instead of appending. A reused
    name updates that name. This keeps the list tidy without making the dialog
    own deduplication rules.
    """
    normalized = normalized_database_search_favorite(favorite)

    replaced = False
    output: list[ExperimentSearchFavorite] = []
    for existing in favorites:
        existing_normalized = normalized_database_search_favorite(existing)
        if _favorites_have_same_identity(existing_normalized, normalized):
            if not replaced:
                output.append(normalized)
                replaced = True
            continue
        output.append(existing_normalized)

    if not replaced:
        output.append(normalized)
    return tuple(output)


def update_database_search_favorite(
    favorites: tuple[ExperimentSearchFavorite, ...],
    index: int,
    favorite: ExperimentSearchFavorite,
) -> tuple[ExperimentSearchFavorite, ...]:
    """Return favorites with one selected row edited.

    Args:
        favorites: Existing favorite records.
        index: Row the user selected for editing.
        favorite: Edited favorite values from the dialog.

    Returns:
        Favorites with the selected row replaced by the normalized edited
        favorite and any duplicate name or filter row removed.

    Raises:
        IndexError: If ``index`` does not point at an existing favorite.
        ValueError: If the edited favorite is invalid.

    Editing targets the selected row. If the edited favorite matches another
    row by name or filters, the edited row wins and the duplicate is removed so
    the persisted list remains unique.
    """
    if index < 0 or index >= len(favorites):
        msg = f"Database search favorite index is out of range: {index}"
        raise IndexError(msg)

    normalized = normalized_database_search_favorite(favorite)
    output: list[ExperimentSearchFavorite] = []
    for existing_index, existing in enumerate(favorites):
        if existing_index == index:
            continue
        existing_normalized = normalized_database_search_favorite(existing)
        if _favorites_have_same_identity(existing_normalized, normalized):
            continue
        output.append(existing_normalized)

    replacement_index = min(index, len(output))
    output.insert(replacement_index, normalized)
    return tuple(output)


def _favorite_from_storage(
    raw_favorite: object,
    path: Path,
    index: int,
) -> ExperimentSearchFavorite:
    """Parse one favorite item from decoded YAML."""
    if not isinstance(raw_favorite, dict):
        msg = f"Favorite #{index + 1} must be a mapping: {path}"
        raise ValueError(msg)
    favorite_mapping = cast(dict[object, object], raw_favorite)

    raw_name = favorite_mapping.get("name")
    if not isinstance(raw_name, str):
        msg = f"Favorite #{index + 1} name must be text: {path}"
        raise ValueError(msg)

    raw_filters = favorite_mapping.get("filters")
    if not isinstance(raw_filters, dict):
        msg = f"Favorite #{index + 1} filters must be a mapping: {path}"
        raise ValueError(msg)
    filter_mapping = cast(dict[object, object], raw_filters)

    return normalized_database_search_favorite(
        ExperimentSearchFavorite(
            name=raw_name,
            filters=ExperimentSearchFilters(
                user=_optional_storage_text(filter_mapping, "user", path, index),
                cell_type=_optional_storage_text(
                    filter_mapping,
                    "cell_type",
                    path,
                    index,
                ),
                sensor=_optional_storage_text(filter_mapping, "sensor", path, index),
                stimulus=_optional_storage_text(
                    filter_mapping,
                    "stimulus",
                    path,
                    index,
                ),
                date=_optional_storage_text(filter_mapping, "date", path, index),
            ),
        )
    )


def _optional_storage_text(
    mapping: dict[object, object],
    key: str,
    path: Path,
    index: int,
) -> str | None:
    """Read one optional text filter from decoded YAML."""
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"Favorite #{index + 1} filter {key!r} must be text: {path}"
        raise ValueError(msg)
    return value


def _filters_to_storage(filters: ExperimentSearchFilters) -> dict[str, str]:
    """Return storage fields for one normalized filter set."""
    normalized = normalized_database_search_filters(filters)
    stored: dict[str, str] = {}
    for field in _FAVORITE_FIELDS:
        value = _filter_value(normalized, field)
        if value is None:
            continue
        stored[field] = value
    return stored


def _filter_key(filters: ExperimentSearchFilters) -> tuple[str | None, ...]:
    """Return a stable duplicate-detection key for filters."""
    normalized = normalized_database_search_filters(filters)
    return tuple(_filter_value(normalized, field) for field in _FAVORITE_FIELDS)


def _favorites_have_same_identity(
    left: ExperimentSearchFavorite,
    right: ExperimentSearchFavorite,
) -> bool:
    """Return whether two normalized favorites should occupy one row."""
    return (
        _filter_key(left.filters) == _filter_key(right.filters)
        or left.name.casefold() == right.name.casefold()
    )


def _filter_value(filters: ExperimentSearchFilters, field: str) -> str | None:
    """Return one named filter value from the known favorite field set."""
    if field == "user":
        return filters.user
    if field == "cell_type":
        return filters.cell_type
    if field == "sensor":
        return filters.sensor
    if field == "stimulus":
        return filters.stimulus
    if field == "date":
        return filters.date
    msg = f"Unknown database search favorite field: {field}"
    raise ValueError(msg)


def _clean_filter(value: str | None) -> str | None:
    """Return stripped text or ``None`` for blank filters."""
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _shorten(value: str, limit: int) -> str:
    """Return compact display text without hiding full values in storage."""
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."
