"""Query Clark lab experiment SQLite databases.

Inputs: a DB folder. Outputs: typed table metadata and experiment records.

The package keeps public database helpers in one import location while splitting
catalog discovery, generic table search, and modeled recording search into
focused modules.
"""

from twopy.database.catalog import read_database_catalog
from twopy.database.generic import find_experiments
from twopy.database.modeled import find_recordings, find_stimulus_presentations
from twopy.database.search import (
    ExperimentHierarchyField,
    ExperimentSearchFilters,
    ExperimentSearchNode,
    build_experiment_search_tree,
    database_hemisphere_for_recording_path,
    find_database_experiment_for_recording_path,
    find_recording_search_results,
    normalize_experiment_date_filter,
    recording_path_for_database_experiment,
    recording_paths_for_database_experiments,
)
from twopy.database.types import (
    DatabaseAccess,
    DatabaseCatalog,
    DatabaseColumn,
    DatabaseExperiment,
    DatabaseRecord,
    DatabaseTable,
)

__all__ = [
    "DatabaseAccess",
    "DatabaseCatalog",
    "DatabaseColumn",
    "DatabaseExperiment",
    "DatabaseRecord",
    "DatabaseTable",
    "ExperimentHierarchyField",
    "ExperimentSearchFilters",
    "ExperimentSearchNode",
    "build_experiment_search_tree",
    "database_hemisphere_for_recording_path",
    "find_experiments",
    "find_database_experiment_for_recording_path",
    "find_recordings",
    "find_recording_search_results",
    "find_stimulus_presentations",
    "normalize_experiment_date_filter",
    "read_database_catalog",
    "recording_path_for_database_experiment",
    "recording_paths_for_database_experiments",
]
