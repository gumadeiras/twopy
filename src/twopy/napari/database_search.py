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

from napari.settings import get_settings
from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtGui import QColor, QKeyEvent, QPainter, QPainterPath, QPen
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
    QProxyStyle,
    QPushButton,
    QSplitter,
    QStyle,
    QStyleOption,
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
from twopy.napari.database_favorite_editor import ExperimentFavoriteEditDialog
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
    update_database_search_favorite,
)
from twopy.napari.errors import exception_message_for_user
from twopy.napari.load_workflow import RecordingLoadFailure, RecordingLoadResult
from twopy.napari.text import configure_placeholder
from twopy.napari.theme import (
    TwopyThemeColors,
    active_twopy_theme_colors,
    apply_twopy_theme,
    style_action_button,
    style_caption,
    style_section_title,
)

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


class _TreeBranchStyle(QProxyStyle):
    """Draw database-tree chevrons with the active napari text color."""

    def __init__(self, tree: QTreeWidget) -> None:
        """Create a branch style that follows live napari theme changes."""
        super().__init__()
        self.setParent(tree)
        self._tree = tree
        self._color = QColor()
        get_settings().appearance.events.theme.connect(self._theme_changed)
        self._theme_changed()

    def _theme_changed(self, _event: object | None = None) -> None:
        """Refresh the chevron color and repaint visible branches."""
        theme = active_twopy_theme_colors(self._tree.palette())
        self._color = QColor(theme.text)
        viewport = self._tree.viewport()
        if viewport is not None:
            viewport.update()

    def drawPrimitive(
        self,
        element: QStyle.PrimitiveElement,
        option: QStyleOption | None,
        painter: QPainter | None,
        widget: QWidget | None = None,
    ) -> None:
        """Draw one branch chevron or use the base style for other elements."""
        if (
            element != QStyle.PrimitiveElement.PE_IndicatorBranch
            or option is None
            or painter is None
        ):
            super().drawPrimitive(element, option, painter, widget)
            return
        if not option.state & QStyle.StateFlag.State_Children:
            return

        center = option.rect.center()
        path = QPainterPath()
        if option.state & QStyle.StateFlag.State_Open:
            path.moveTo(float(center.x() - 4), float(center.y() - 2))
            path.lineTo(float(center.x()), float(center.y() + 2))
            path.lineTo(float(center.x() + 4), float(center.y() - 2))
        else:
            path.moveTo(float(center.x() - 2), float(center.y() - 4))
            path.lineTo(float(center.x() + 2), float(center.y()))
            path.lineTo(float(center.x() - 2), float(center.y() + 4))

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self._color, 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawPath(path)
        painter.restore()


class ExperimentSearchDialog(QDialog):
    """Search and load database experiments from a separate Qt window.

    Args:
        on_load_recording_paths: Callback that receives source recording paths
            selected from the result hierarchy and returns structured load
            results.
        config_path: Optional twopy config path. The default uses twopy's usual
            config search.
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
        self._configure_favorites_list()
        self._favorites_list.itemDoubleClicked.connect(
            lambda _item: self.use_selected_favorite(),
        )
        self._favorites_list.itemSelectionChanged.connect(
            self._update_favorite_action_states,
        )
        self._tree = QTreeWidget()
        self._tree.setObjectName("database_experiment_tree")
        self._tree_branch_style = _TreeBranchStyle(self._tree)
        self._tree.setStyle(self._tree_branch_style)
        self._tree.setHeaderLabels(("Experiment", "Count"))
        self._tree.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection,
        )
        self._configure_result_columns()
        self._tree.itemActivated.connect(self._load_tree_item)
        self._tree.installEventFilter(self)
        self._result_status = QLabel("Set filters, then select Search.")
        style_caption(self._result_status)

        search_button = QPushButton("Search")
        style_action_button(search_button, role="primary")
        search_button.clicked.connect(self.search)
        self._save_favorite_button = QPushButton("Save as favorite...")
        style_action_button(self._save_favorite_button)
        self._save_favorite_button.clicked.connect(self.save_current_favorite)
        self._use_favorite_button = QPushButton("Use")
        style_action_button(self._use_favorite_button)
        self._use_favorite_button.clicked.connect(self.use_selected_favorite)
        self._edit_favorite_button = QPushButton("Edit...")
        style_action_button(self._edit_favorite_button, role="quiet")
        self._edit_favorite_button.clicked.connect(self.edit_selected_favorite)
        self._remove_favorite_button = QPushButton("Remove")
        style_action_button(self._remove_favorite_button, role="danger")
        self._remove_favorite_button.clicked.connect(self.remove_selected_favorite)
        load_button = QPushButton("Load selected")
        style_action_button(load_button, role="primary")
        load_button.clicked.connect(self.load_selected)
        close_button = QPushButton("Close")
        style_action_button(close_button, role="quiet")
        close_button.clicked.connect(self.reject)
        for button in (
            search_button,
            self._save_favorite_button,
            self._use_favorite_button,
            self._edit_favorite_button,
            self._remove_favorite_button,
            load_button,
            close_button,
        ):
            button.setAutoDefault(False)
            button.setDefault(False)

        for widget in self._filter_widgets():
            widget.textChanged.connect(self._update_favorite_action_states)
        self._render_favorites()
        self._update_favorite_action_states()

        self.setWindowTitle("Search database")
        self.resize(1040, 640)

        layout = QVBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("database_search_splitter")
        splitter.setHandleWidth(12)
        splitter.addWidget(self._filter_panel(search_button))
        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(8)
        results_layout.addWidget(self._result_status)
        results_layout.addWidget(self._tree)
        splitter.addWidget(results_panel)
        splitter.setSizes((FILTER_PANEL_WIDTH, 700))
        layout.addWidget(splitter)
        layout.addLayout(_bottom_button_row(load_button, close_button))
        self.setLayout(layout)
        apply_twopy_theme(
            self,
            name="twopy_database_search",
            additional_style=_database_search_style,
        )

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        """Load selected search results when Return is pressed in the result tree.

        Args:
            a0: Widget that received the event.
            a1: Qt event sent to the watched widget.

        Returns:
            Whether this dialog handled the event.

        The result tree accepts Return without emitting ``itemActivated`` on all
        Qt backends, so the dialog handles that key explicitly while filter
        fields keep their own Return-to-search behavior.
        """
        watched = a0
        event = a1
        if (
            watched is self._tree
            and isinstance(event, QKeyEvent)
            and event.type() == QEvent.Type.KeyPress
            and event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}
            and self._tree.selectedItems()
        ):
            self.load_selected()
            event.accept()
            return True
        return super().eventFilter(watched, event)

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
            self._result_status.setText("Search failed.")
            self._show_search_error(error)
            return

        self._render_tree(build_experiment_search_tree(experiments))
        count = len(experiments)
        self._result_status.setText(
            f"{count} recording{'s' if count != 1 else ''} found."
        )

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

    def edit_selected_favorite(self) -> None:
        """Edit the selected favorite name and saved filter values.

        Args:
            None.

        Returns:
            None.

        Editing targets the selected favorite row. The dialog writes the edited
        list before updating memory so failed YAML writes cannot desynchronize
        the visible list from the persisted favorites file.
        """
        if self._favorites_load_error is not None:
            self._show_favorite_error(self._favorites_load_error)
            return

        selected = self._selected_favorite_index()
        if selected is None:
            return
        dialog = ExperimentFavoriteEditDialog(
            self._favorites[selected],
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            favorite = normalized_database_search_favorite(dialog.favorite())
            updated_favorites = update_database_search_favorite(
                self._favorites,
                selected,
                favorite,
            )
            save_database_search_favorites(updated_favorites, self._favorites_path)
        except (IndexError, OSError, ValueError) as error:
            self._show_favorite_error(error)
            return

        self._favorites = updated_favorites
        self._render_favorites(selected=favorite)
        self._update_favorite_action_states()

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

    def _reorder_favorites_from_visible_rows(
        self,
        *,
        moved_source_start: int | None = None,
        moved_source_end: int | None = None,
        moved_destination_row: int | None = None,
    ) -> None:
        """Persist the favorite order currently shown in the list.

        Args:
            moved_source_start: First source row from the completed Qt move.
            moved_source_end: Last source row from the completed Qt move.
            moved_destination_row: Destination row from the completed Qt move.

        Returns:
            None.

        The dragged list order is the user's requested order. The dialog writes
        that order before changing its stored tuple, so a disk failure cannot
        leave the visible rows disagreeing with the YAML file.
        """
        if self._favorites_load_error is not None:
            self._render_favorites()
            self._show_favorite_error(self._favorites_load_error)
            return

        previous_favorites = self._favorites
        selected = self._favorite_for_completed_move(
            previous_favorites,
            source_start=moved_source_start,
            source_end=moved_source_end,
            destination_row=moved_destination_row,
        )
        if selected is None:
            selected = self._favorite_for_rendered_row(
                self._favorites_list.currentRow(),
                previous_favorites,
            )
        try:
            updated_favorites = self._favorites_from_visible_rows(previous_favorites)
            if updated_favorites == previous_favorites:
                return
            save_database_search_favorites(updated_favorites, self._favorites_path)
        except (OSError, ValueError) as error:
            self._render_favorites(selected=selected)
            self._show_favorite_error(error)
            return

        self._favorites = updated_favorites
        self._render_favorites(selected=selected)
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
        layout.addWidget(self._favorites_panel(), 1)
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
        title = QLabel("Favorites")
        style_section_title(title)
        layout.addWidget(title)
        layout.addWidget(self._favorites_list, 1)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.addStretch(1)
        actions.addWidget(self._use_favorite_button)
        actions.addWidget(self._edit_favorite_button)
        actions.addWidget(self._remove_favorite_button)
        layout.addLayout(actions)

        panel.setLayout(layout)
        return panel

    def _configure_favorites_list(self) -> None:
        """Configure the favorites list for explicit internal row moves."""
        self._favorites_list.setObjectName("database_favorites_list")
        self._favorites_list.setAccessibleName("Saved database searches")
        self._favorites_list.setToolTip(
            "Double-click a saved search to use it. Drag rows to reorder them."
        )
        self._favorites_list.setMinimumHeight(140)
        self._favorites_list.setUniformItemSizes(True)
        self._favorites_list.setSpacing(0)
        self._favorites_list.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel,
        )
        self._favorites_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self._favorites_list.setDragEnabled(True)
        self._favorites_list.setAcceptDrops(True)
        self._favorites_list.setDropIndicatorShown(True)
        self._favorites_list.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove,
        )
        self._favorites_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._favorites_list.setDragDropOverwriteMode(False)
        model = self._favorites_list.model()
        if model is None:
            msg = "Favorites list must have a Qt model for row reordering."
            raise RuntimeError(msg)
        model.rowsMoved.connect(self._handle_favorites_rows_moved)

    def _handle_favorites_rows_moved(
        self,
        _source_parent: object,
        source_start: int,
        source_end: int,
        _destination_parent: object,
        destination_row: int,
    ) -> None:
        """Persist a completed favorite row move from the Qt list model."""
        self._reorder_favorites_from_visible_rows(
            moved_source_start=source_start,
            moved_source_end=source_end,
            moved_destination_row=destination_row,
        )

    def _filter_line_edit(self, placeholder: str) -> QLineEdit:
        """Create one filter field with an italic placeholder hint."""
        line_edit = QLineEdit()
        configure_placeholder(line_edit, placeholder)
        line_edit.setTextMargins(
            TEXT_FIELD_MARGIN_PX,
            0,
            TEXT_FIELD_MARGIN_PX,
            0,
        )
        line_edit.setClearButtonEnabled(True)
        line_edit.returnPressed.connect(self.search)
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
            item.setData(Qt.ItemDataRole.UserRole, index)
            self._favorites_list.addItem(item)
            if selected is not None and favorite == selected:
                selected_row = index

        if (
            selected_row is not None
            and 0 <= selected_row < self._favorites_list.count()
        ):
            self._favorites_list.setCurrentRow(selected_row)

    def _favorites_from_visible_rows(
        self,
        favorites: tuple[ExperimentSearchFavorite, ...],
    ) -> tuple[ExperimentSearchFavorite, ...]:
        """Return favorites in the order currently shown by the list widget."""
        if self._favorites_list.count() != len(favorites):
            msg = "Favorite rows do not match the stored favorite list."
            raise ValueError(msg)

        ordered: list[ExperimentSearchFavorite] = []
        seen: set[int] = set()
        for row in range(self._favorites_list.count()):
            item = self._favorites_list.item(row)
            if item is None:
                msg = f"Favorite row {row + 1} is missing."
                raise ValueError(msg)
            index = _favorite_index_from_item(item, len(favorites))
            if index in seen:
                msg = "Favorite rows contain a duplicate item."
                raise ValueError(msg)
            seen.add(index)
            ordered.append(favorites[index])
        return tuple(ordered)

    def _favorite_for_rendered_row(
        self,
        row: int,
        favorites: tuple[ExperimentSearchFavorite, ...],
    ) -> ExperimentSearchFavorite | None:
        """Return the favorite represented by a rendered row, if any."""
        if row < 0 or row >= self._favorites_list.count():
            return None
        item = self._favorites_list.item(row)
        if item is None:
            return None
        try:
            return favorites[_favorite_index_from_item(item, len(favorites))]
        except ValueError:
            return None

    def _favorite_for_completed_move(
        self,
        favorites: tuple[ExperimentSearchFavorite, ...],
        *,
        source_start: int | None,
        source_end: int | None,
        destination_row: int | None,
    ) -> ExperimentSearchFavorite | None:
        """Return the first favorite moved by a completed Qt row move."""
        if source_start is None or source_end is None or destination_row is None:
            return None
        moved_count = source_end - source_start + 1
        if moved_count < 1:
            return None
        moved_row = destination_row
        if destination_row > source_start:
            moved_row -= moved_count
        return self._favorite_for_rendered_row(moved_row, favorites)

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
        self._edit_favorite_button.setEnabled(can_write_favorites and has_selection)
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
        dialog.setWindowTitle("Save as Favorite")
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


def _database_search_style(color: TwopyThemeColors) -> str:
    """Return compact database-search styles from the active theme.

    Args:
        color: Active twopy colors from napari.

    Returns:
        Styles for the search splitter and favorites list.

    The styles keep the resize gutter quiet and make favorite rows easy to scan.
    """
    return f"""
QSplitter#database_search_splitter::handle {{
    background: transparent;
}}
QListWidget#database_favorites_list {{
    background: {color.field};
    border: 1px solid {color.control_outline};
    border-radius: 7px;
}}
QListWidget#database_favorites_list::item {{
    min-height: 24px;
    padding: 2px 7px;
    border-bottom: 1px solid {color.divider};
}}
QTreeWidget#database_experiment_tree {{
    border-radius: 0;
}}
QTreeWidget#database_experiment_tree QHeaderView {{
    background: {color.surface};
    border: none;
    border-bottom: 1px solid {color.control_outline};
    border-radius: 0;
}}
QTreeWidget#database_experiment_tree QHeaderView::section {{
    background: transparent;
    border: none;
    border-radius: 0;
}}
"""


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
    layout.addWidget(close_button)
    layout.addWidget(load_button)
    return layout


def _favorite_index_from_item(item: QListWidgetItem, favorite_count: int) -> int:
    """Return the stored favorite index for one list item.

    Args:
        item: Rendered favorite row.
        favorite_count: Number of favorites in the tuple used for rendering.

    Returns:
        Index into the tuple used when the item was rendered.

    Raises:
        ValueError: If the row metadata is missing or stale.

    The row stores an index instead of duplicating favorite data, so drag/drop
    reorders the existing records without making widget text part of the data
    contract.
    """
    raw_index = item.data(Qt.ItemDataRole.UserRole)
    if not isinstance(raw_index, int):
        msg = "Favorite row is missing its stored favorite index."
        raise ValueError(msg)
    if raw_index < 0 or raw_index >= favorite_count:
        msg = "Favorite row refers to a favorite that no longer exists."
        raise ValueError(msg)
    return raw_index


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
        apply_twopy_theme(self, name="twopy_database_search_error")


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
        apply_twopy_theme(self, name="twopy_database_favorite_error")


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
        apply_twopy_theme(self, name="twopy_recording_load_error")


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
