"""Qt database-search dialog for the twopy napari Load tab.

Inputs: text filters entered by the user and a callback that can load recording
paths.
Outputs: a separate search window with a collapsible experiment hierarchy.

The dialog owns Qt widgets only. Database querying, grouping, and path
resolution live in ``twopy.database.search``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from twopy.config import DEFAULT_CONFIG_PATH, TwopyConfig, load_config
from twopy.database.search import (
    ExperimentSearchFilters,
    ExperimentSearchNode,
    build_experiment_search_tree,
    find_recording_search_results,
    normalize_experiment_date_filter,
    recording_path_for_database_experiment,
)
from twopy.database.types import DatabaseExperiment
from twopy.napari.database_favorites import (
    DEFAULT_DATABASE_FAVORITES_PATH,
    ExperimentSearchFavorite,
    database_search_favorite_filters_are_empty,
    database_search_favorite_tooltip,
    default_database_search_favorite_name,
    load_database_search_favorites,
    normalized_database_search_favorite,
    normalized_database_search_filters,
    replace_database_search_favorite,
    save_database_search_favorites,
)
from twopy.napari.errors import exception_message_for_user
from twopy.napari.load_workflow import RecordingLoadFailure, RecordingLoadResult

__all__ = [
    "ExperimentFavoriteErrorDialog",
    "ExperimentSearchErrorDialog",
    "ExperimentSearchDialog",
    "RecordingLoadErrorDialog",
]

COUNT_COLUMN_WIDTH = 72
FILTER_PANEL_WIDTH = 340
FAVORITE_NAME_DIALOG_WIDTH = FILTER_PANEL_WIDTH * 3
TEXT_FIELD_MARGIN_PX = 6
PRIMARY_ACTION_BUTTON_STYLE = (
    "QPushButton {"
    "background-color: #2d7dd2;"
    "color: white;"
    "border: 1px solid #1f5f9f;"
    "border-radius: 4px;"
    "padding: 4px 10px;"
    "}"
    "QPushButton:disabled {"
    "background-color: #8fb5d8;"
    "color: #edf4fb;"
    "border-color: #7aa6cc;"
    "}"
)
SECONDARY_ACTION_BUTTON_STYLE = (
    "QPushButton {"
    "background-color: #f2f2f2;"
    "color: #202020;"
    "border: 1px solid #a8a8a8;"
    "border-radius: 4px;"
    "padding: 4px 10px;"
    "}"
    "QPushButton:disabled {"
    "background-color: #eeeeee;"
    "color: #8a8a8a;"
    "border-color: #c8c8c8;"
    "}"
)

FILTER_HINTS = {
    "user": "gustavo",
    "cell_type": "ALPN",
    "sensor": "g6f",
    "stimulus": "bars_singles_stim=5s_bars=3x400ms_dt=200ms_int=20",
    "date": "2025-10-22",
}


@dataclass(frozen=True)
class _ResolvedExperimentPaths:
    """Resolved source paths plus database-row failures from one selection."""

    paths: tuple[Path, ...]
    failures: tuple[RecordingLoadFailure, ...] = ()


class ExperimentSearchDialog(QDialog):
    """Search and load database experiments from a separate Qt window.

    Args:
        on_load_recording_paths: Callback that receives source recording paths
            selected from the result hierarchy and returns structured load
            results.
        config_path: Optional twopy config path. Defaults to ``config.yml``.
        parent: Optional Qt parent widget.

    Outputs:
        Dialog containing text filters and a collapsible result tree.
    """

    def __init__(
        self,
        *,
        on_load_recording_paths: Callable[[tuple[Path, ...]], RecordingLoadResult],
        config_path: Path = DEFAULT_CONFIG_PATH,
        favorites_path: Path = DEFAULT_DATABASE_FAVORITES_PATH,
        parent: QWidget | None = None,
    ) -> None:
        """Create the database experiment search dialog."""
        super().__init__(parent)
        self._on_load_recording_paths = on_load_recording_paths
        self._config_path = config_path
        self._favorites_path = favorites_path
        self._config: TwopyConfig | None = None
        self._favorites_load_error: Exception | None = None
        self._favorites = self._load_favorites()

        self._user_filter = self._filter_line_edit(FILTER_HINTS["user"])
        self._cell_type_filter = self._filter_line_edit(FILTER_HINTS["cell_type"])
        self._sensor_filter = self._filter_line_edit(FILTER_HINTS["sensor"])
        self._stimulus_filter = self._filter_line_edit(FILTER_HINTS["stimulus"])
        self._date_filter = self._filter_line_edit(FILTER_HINTS["date"])
        self._favorites_list = QListWidget()
        self._favorites_list.itemDoubleClicked.connect(
            lambda _item: self.use_selected_favorite(),
        )
        self._favorites_list.itemSelectionChanged.connect(
            self._update_favorite_action_states,
        )
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(("Experiment", "Count"))
        self._tree.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection,
        )
        self._configure_result_columns()
        self._tree.itemActivated.connect(self._load_tree_item)

        search_button = QPushButton("Search")
        search_button.setStyleSheet(SECONDARY_ACTION_BUTTON_STYLE)
        search_button.clicked.connect(self.search)
        self._save_favorite_button = QPushButton("Save favorite...")
        self._save_favorite_button.setStyleSheet(PRIMARY_ACTION_BUTTON_STYLE)
        self._save_favorite_button.clicked.connect(self.save_current_favorite)
        self._use_favorite_button = QPushButton("Use")
        self._use_favorite_button.clicked.connect(self.use_selected_favorite)
        self._remove_favorite_button = QPushButton("Remove")
        self._remove_favorite_button.clicked.connect(self.remove_selected_favorite)
        load_button = QPushButton("Load selected")
        load_button.clicked.connect(self.load_selected)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)

        for widget in self._filter_widgets():
            widget.textChanged.connect(self._update_favorite_action_states)
        self._render_favorites()
        self._update_favorite_action_states()

        self.setWindowTitle("Search database")
        self.resize(1040, 560)

        layout = QVBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._filter_panel(search_button))
        splitter.addWidget(self._tree)
        splitter.setSizes((FILTER_PANEL_WIDTH, 700))
        layout.addWidget(splitter)
        layout.addLayout(_bottom_button_row(load_button, close_button))
        self.setLayout(layout)

    def search(self) -> None:
        """Run the database search and render grouped results.

        Args:
            None.

        Returns:
            None.
        """
        try:
            raw_date_filter = self._date_filter.text()
            date_filter = normalize_experiment_date_filter(raw_date_filter)
            if date_filter is not None and date_filter != raw_date_filter:
                self._date_filter.setText(date_filter)
            config = self._load_config()
            experiments = find_recording_search_results(
                config,
                ExperimentSearchFilters(
                    user=self._user_filter.text(),
                    cell_type=self._cell_type_filter.text(),
                    sensor=self._sensor_filter.text(),
                    stimulus=self._stimulus_filter.text(),
                    date=date_filter,
                ),
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            self._tree.clear()
            self._show_search_error(error)
            return

        self._render_tree(build_experiment_search_tree(experiments))

    def save_current_favorite(self) -> None:
        """Save the current search filters as a reusable favorite.

        Args:
            None.

        Returns:
            None.

        Blank filter sets are ignored because a favorite should represent a
        real reusable query, not an accidental all-database search.
        """
        if self._favorites_load_error is not None:
            self._show_favorite_error(self._favorites_load_error)
            return

        try:
            filters = normalized_database_search_filters(self._current_filters())
            if database_search_favorite_filters_are_empty(filters):
                return

            name = self._favorite_name_from_user(
                default_database_search_favorite_name(filters),
            )
            if name is None:
                return

            favorite = normalized_database_search_favorite(
                ExperimentSearchFavorite(name=name, filters=filters),
            )
            updated_favorites = replace_database_search_favorite(
                self._favorites,
                favorite,
            )
            save_database_search_favorites(updated_favorites, self._favorites_path)
        except (OSError, ValueError) as error:
            self._show_favorite_error(error)
            return

        self._favorites = updated_favorites
        self._render_favorites(selected=favorite)
        self._update_favorite_action_states()

    def use_selected_favorite(self) -> None:
        """Apply the selected favorite to the filters and search.

        Args:
            None.

        Returns:
            None.

        Running the search only after an explicit Use action or double-click
        keeps ordinary selection safe for review and removal.
        """
        favorite = self._selected_favorite()
        if favorite is None:
            return
        self._apply_filters(favorite.filters)
        self.search()

    def remove_selected_favorite(self) -> None:
        """Remove the selected favorite from the local favorite list.

        Args:
            None.

        Returns:
            None.

        Removal targets the selected favorite row, not the current filter text,
        so users can safely edit filters while a favorite remains selected.
        """
        if self._favorites_load_error is not None:
            self._show_favorite_error(self._favorites_load_error)
            return

        selected = self._selected_favorite_index()
        if selected is None:
            return
        updated_favorites = self._favorites[:selected] + self._favorites[selected + 1 :]
        try:
            save_database_search_favorites(updated_favorites, self._favorites_path)
        except (OSError, ValueError) as error:
            self._show_favorite_error(error)
            return
        self._favorites = updated_favorites
        self._render_favorites(
            selected_index=min(selected, len(self._favorites) - 1),
        )
        self._update_favorite_action_states()

    def load_selected(self) -> None:
        """Load recordings for the selected tree node.

        Args:
            None.

        Returns:
            None.
        """
        selected = self._tree.selectedItems()
        if not selected:
            return
        self._load_tree_items(tuple(selected))

    def _filter_panel(
        self,
        search_button: QPushButton,
    ) -> QWidget:
        """Return the left search filter panel."""
        panel = QWidget()
        panel.setMinimumWidth(FILTER_PANEL_WIDTH)

        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.addStretch(1)
        action_layout.addWidget(self._save_favorite_button)
        action_layout.addWidget(search_button)

        layout = QVBoxLayout()
        layout.addLayout(self._filter_form())
        layout.addLayout(action_layout)
        layout.addWidget(self._favorites_panel())
        layout.addStretch(1)
        panel.setLayout(layout)
        return panel

    def _filter_form(self) -> QFormLayout:
        """Return the left filter form layout."""
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("User", self._user_filter)
        form.addRow("Cell type", self._cell_type_filter)
        form.addRow("Sensor", self._sensor_filter)
        form.addRow("Stimulus", self._stimulus_filter)
        form.addRow("Date", self._date_filter)
        return form

    def _favorites_panel(self) -> QWidget:
        """Return the lower-left favorites panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Favorites"))
        layout.addWidget(self._favorites_list)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.addStretch(1)
        actions.addWidget(self._use_favorite_button)
        actions.addWidget(self._remove_favorite_button)
        layout.addLayout(actions)

        panel.setLayout(layout)
        return panel

    def _filter_line_edit(self, placeholder: str) -> QLineEdit:
        """Create one filter field with an italic placeholder hint."""
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        line_edit.setTextMargins(
            TEXT_FIELD_MARGIN_PX,
            0,
            TEXT_FIELD_MARGIN_PX,
            0,
        )
        line_edit.setClearButtonEnabled(True)
        line_edit.returnPressed.connect(self.search)
        line_edit.textChanged.connect(
            lambda text, widget=line_edit: _set_filter_hint_font(
                widget,
                is_hint_visible=text == "",
            )
        )
        _set_filter_hint_font(line_edit, is_hint_visible=True)
        return line_edit

    def _filter_widgets(self) -> tuple[QLineEdit, ...]:
        """Return filter widgets in saved-favorite field order."""
        return (
            self._user_filter,
            self._cell_type_filter,
            self._sensor_filter,
            self._stimulus_filter,
            self._date_filter,
        )

    def _current_filters(self) -> ExperimentSearchFilters:
        """Return search filters from the current dialog fields."""
        return ExperimentSearchFilters(
            user=self._user_filter.text(),
            cell_type=self._cell_type_filter.text(),
            sensor=self._sensor_filter.text(),
            stimulus=self._stimulus_filter.text(),
            date=self._date_filter.text(),
        )

    def _apply_filters(self, filters: ExperimentSearchFilters) -> None:
        """Apply one saved favorite filter set to the dialog fields."""
        normalized = normalized_database_search_filters(filters)
        self._user_filter.setText(normalized.user or "")
        self._cell_type_filter.setText(normalized.cell_type or "")
        self._sensor_filter.setText(normalized.sensor or "")
        self._stimulus_filter.setText(normalized.stimulus or "")
        self._date_filter.setText(normalized.date or "")

    def _load_favorites(self) -> tuple[ExperimentSearchFavorite, ...]:
        """Load persisted favorites for this dialog instance."""
        try:
            favorites = load_database_search_favorites(self._favorites_path)
        except (OSError, ValueError) as error:
            self._favorites_load_error = error
            self._show_favorite_error(error)
            return ()
        self._favorites_load_error = None
        return favorites

    def _render_favorites(
        self,
        *,
        selected: ExperimentSearchFavorite | None = None,
        selected_index: int | None = None,
    ) -> None:
        """Render persisted favorites into the lower-left list."""
        self._favorites_list.clear()
        selected_row = selected_index
        for index, favorite in enumerate(self._favorites):
            item = QListWidgetItem(favorite.name)
            item.setToolTip(database_search_favorite_tooltip(favorite))
            self._favorites_list.addItem(item)
            if selected is not None and favorite == selected:
                selected_row = index

        if (
            selected_row is not None
            and 0 <= selected_row < self._favorites_list.count()
        ):
            self._favorites_list.setCurrentRow(selected_row)

    def _selected_favorite_index(self) -> int | None:
        """Return the selected favorite row, if any."""
        selected = self._favorites_list.currentRow()
        if selected < 0 or selected >= len(self._favorites):
            return None
        return selected

    def _selected_favorite(self) -> ExperimentSearchFavorite | None:
        """Return the selected favorite, if any."""
        selected = self._selected_favorite_index()
        if selected is None:
            return None
        return self._favorites[selected]

    def _update_favorite_action_states(self) -> None:
        """Enable favorite actions only when their targets are valid."""
        can_write_favorites = self._favorites_load_error is None
        self._save_favorite_button.setEnabled(
            can_write_favorites and self._current_filters_can_be_saved()
        )
        has_selection = self._selected_favorite_index() is not None
        self._use_favorite_button.setEnabled(has_selection)
        self._remove_favorite_button.setEnabled(can_write_favorites and has_selection)

    def _current_filters_can_be_saved(self) -> bool:
        """Return whether current filters are valid favorite contents."""
        try:
            return not database_search_favorite_filters_are_empty(
                self._current_filters(),
            )
        except ValueError:
            return False

    def _favorite_name_from_user(self, default_name: str) -> str | None:
        """Prompt the user for a favorite name."""
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Save Favorite")
        dialog.setLabelText("Favorite name")
        dialog.setInputMode(QInputDialog.InputMode.TextInput)
        dialog.setTextValue(default_name)
        dialog.resize(FAVORITE_NAME_DIALOG_WIDTH, dialog.sizeHint().height())
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.textValue()

    def _configure_result_columns(self) -> None:
        """Keep Experiment wide and Count compact."""
        header = self._tree.header()
        if header is None:
            return
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self._tree.setColumnWidth(1, COUNT_COLUMN_WIDTH)

    def _load_config(self) -> TwopyConfig:
        """Return cached twopy configuration for DB search."""
        if self._config is None:
            self._config = load_config(self._config_path)
        return self._config

    def _render_tree(self, root: ExperimentSearchNode) -> None:
        """Render one immutable search tree into the Qt tree widget."""
        self._tree.clear()
        for child in root.children:
            self._tree.addTopLevelItem(self._tree_item(child))
        if self._tree.topLevelItemCount() == 1:
            top_item = self._tree.topLevelItem(0)
            if top_item is not None:
                top_item.setExpanded(True)
        self._tree.setColumnWidth(1, COUNT_COLUMN_WIDTH)

    def _tree_item(self, node: ExperimentSearchNode) -> QTreeWidgetItem:
        """Create one Qt tree item for a search node."""
        item = QTreeWidgetItem((node.label, str(len(node.experiments))))
        item.setData(
            0,
            Qt.ItemDataRole.UserRole,
            node.experiments,
        )
        for child in node.children:
            item.addChild(self._tree_item(child))
        return item

    def _load_tree_item(self, item: QTreeWidgetItem, _column: int) -> None:
        """Load source recording paths represented by one tree item."""
        self._load_tree_items((item,))

    def _load_tree_items(self, items: tuple[QTreeWidgetItem, ...]) -> None:
        """Load source recording paths represented by selected tree items."""
        experiments = _experiments_for_tree_items(items)
        if len(experiments) == 0:
            return
        resolved = _resolve_experiment_paths(
            self._load_config(),
            experiments,
        )

        result = _merge_load_failures(
            self._on_load_recording_paths(resolved.paths),
            resolved.failures,
        )
        self.accept()
        if result.failures:
            self._show_load_errors(result)

    def _show_search_error(self, error: Exception) -> None:
        """Show one database-search failure with clear details."""
        ExperimentSearchErrorDialog(error, parent=self.parentWidget()).exec()

    def _show_load_errors(self, result: RecordingLoadResult) -> None:
        """Show database-result load failures with path details."""
        RecordingLoadErrorDialog(result, parent=self.parentWidget()).exec()

    def _show_favorite_error(self, error: Exception) -> None:
        """Show a favorites persistence failure with clear details."""
        ExperimentFavoriteErrorDialog(error, parent=self.parentWidget()).exec()


def _set_filter_hint_font(line_edit: QLineEdit, *, is_hint_visible: bool) -> None:
    """Italicize empty filter fields so placeholder hints read as hints.

    Args:
        line_edit: Filter text field.
        is_hint_visible: Whether the field is empty and showing its placeholder.

    Returns:
        None.
    """
    font = line_edit.font()
    font.setItalic(is_hint_visible)
    line_edit.setFont(font)


def _bottom_button_row(
    load_button: QPushButton, close_button: QPushButton
) -> QHBoxLayout:
    """Return the explicit bottom-right button row for the search dialog.

    Args:
        load_button: Button that loads selected experiments.
        close_button: Button that closes the dialog.

    Returns:
        Horizontal layout with action buttons right-aligned.

    Qt's platform button box places action-role buttons on the left on macOS,
    so the search dialog owns this row directly.
    """
    layout = QHBoxLayout()
    layout.addStretch(1)
    layout.addWidget(load_button)
    layout.addWidget(close_button)
    return layout


class ExperimentSearchErrorDialog(QDialog):
    """Dialog showing database-search failures.

    Args:
        error: Exception raised while querying configured database files.
        parent: Optional Qt parent widget.

    Outputs:
        Modal dialog with a clear summary and scrollable diagnostic details.
    """

    def __init__(
        self,
        error: Exception,
        *,
        parent: QWidget | None = None,
    ) -> None:
        """Create an error dialog for failed database searches."""
        super().__init__(parent)
        self.setWindowTitle("Database Search Error")
        self.resize(720, 360)

        summary = QLabel("Could not search the experiment database.")
        layout = QVBoxLayout()
        layout.addWidget(summary)
        layout.addWidget(_read_only_details(exception_message_for_user(error)))
        layout.addWidget(_close_button_box(self))
        self.setLayout(layout)


class ExperimentFavoriteErrorDialog(QDialog):
    """Dialog showing database-search favorite persistence failures.

    Args:
        error: Exception raised while loading or saving favorites.
        parent: Optional Qt parent widget.

    Outputs:
        Modal dialog with a clear summary and scrollable diagnostic details.
    """

    def __init__(
        self,
        error: Exception,
        *,
        parent: QWidget | None = None,
    ) -> None:
        """Create an error dialog for failed favorite operations."""
        super().__init__(parent)
        self.setWindowTitle("Database Favorites Error")
        self.resize(720, 360)

        summary = QLabel("Could not update database search favorites.")
        layout = QVBoxLayout()
        layout.addWidget(summary)
        layout.addWidget(_read_only_details(exception_message_for_user(error)))
        layout.addWidget(_close_button_box(self))
        self.setLayout(layout)


class RecordingLoadErrorDialog(QDialog):
    """Dialog showing recording-load failures.

    Args:
        result: Structured load result containing failed paths and messages.
        parent: Optional Qt parent widget.

    Outputs:
        Modal dialog with a clear summary and scrollable failure list.
    """

    def __init__(
        self,
        result: RecordingLoadResult,
        *,
        parent: QWidget | None = None,
    ) -> None:
        """Create an error dialog for failed recording loads."""
        super().__init__(parent)
        self.setWindowTitle("Recording Load Errors")
        self.resize(720, 420)

        summary = QLabel(_load_failure_summary(result))
        layout = QVBoxLayout()
        layout.addWidget(summary)
        layout.addWidget(_read_only_details(_load_failure_text(result.failures)))
        layout.addWidget(_close_button_box(self))
        self.setLayout(layout)


def _load_failure_summary(result: RecordingLoadResult) -> str:
    """Return a short load-failure summary."""
    failed_count = len(result.failures)
    if result.loaded_count == 0:
        return f"No recordings loaded. Failed to load {failed_count} recordings."
    return (
        f"Loaded {result.loaded_count} recordings. "
        f"Failed to load {failed_count} recordings."
    )


def _load_failure_text(failures: tuple[RecordingLoadFailure, ...]) -> str:
    """Return scrollable plain text for all failed recording paths."""
    entries: list[str] = []
    for failure in failures:
        path_text = "Selection" if failure.path is None else str(failure.path)
        entries.append(f"{path_text}\n{failure.message}")
    return "\n\n".join(entries)


def _read_only_details(text: str) -> QPlainTextEdit:
    """Return a scrollable read-only details box for error dialogs."""
    details = QPlainTextEdit()
    details.setReadOnly(True)
    details.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
    details.setPlainText(text)
    return details


def _close_button_box(dialog: QDialog) -> QDialogButtonBox:
    """Return a Close button box wired to one dialog."""
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    return buttons


def _resolve_experiment_paths(
    config: TwopyConfig,
    experiments: tuple[DatabaseExperiment, ...],
) -> _ResolvedExperimentPaths:
    """Resolve selected experiments while preserving per-row failures."""
    paths: list[Path] = []
    failures: list[RecordingLoadFailure] = []
    seen: set[Path] = set()
    for experiment in experiments:
        try:
            path = recording_path_for_database_experiment(config, experiment)
        except (FileNotFoundError, OSError, ValueError) as error:
            failures.append(
                RecordingLoadFailure(path=None, message=_experiment_error(error)),
            )
            continue
        if path in seen:
            continue
        paths.append(path)
        seen.add(path)
    return _ResolvedExperimentPaths(paths=tuple(paths), failures=tuple(failures))


def _experiments_for_tree_items(
    items: tuple[QTreeWidgetItem, ...],
) -> tuple[DatabaseExperiment, ...]:
    """Return unique experiments represented by selected tree items.

    Args:
        items: Selected hierarchy rows from the search result tree.

    Returns:
        Unique experiments in tree-selection order.

    A parent and child can both be selected. Deduplicating here prevents the
    same database row from being resolved or loaded twice while still allowing
    users to combine unrelated rows in one load action.
    """
    experiments: list[DatabaseExperiment] = []
    seen: set[tuple[Path, int, str]] = set()
    for item in items:
        item_experiments = cast(
            tuple[DatabaseExperiment, ...],
            item.data(0, Qt.ItemDataRole.UserRole),
        )
        for experiment in item_experiments:
            key = (
                experiment.database_path.expanduser().resolve(strict=False),
                experiment.stimulus_presentation_id,
                experiment.relative_data_path,
            )
            if key in seen:
                continue
            experiments.append(experiment)
            seen.add(key)
    return tuple(experiments)


def _merge_load_failures(
    result: RecordingLoadResult,
    failures: tuple[RecordingLoadFailure, ...],
) -> RecordingLoadResult:
    """Return one load result with path-resolution and loader failures."""
    if len(failures) == 0:
        return result
    return RecordingLoadResult(
        loaded_count=result.loaded_count,
        failures=failures + result.failures,
        source_warnings=result.source_warnings,
        status_text=result.status_text,
    )


def _experiment_error(error: Exception) -> str:
    """Return a clear path-resolution failure message."""
    return (
        "Could not resolve database experiment path.\n"
        f"{exception_message_for_user(error)}"
    )
