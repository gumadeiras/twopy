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
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
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
    recording_path_for_database_experiment,
)
from twopy.database.types import DatabaseExperiment
from twopy.napari.errors import exception_message_for_user

__all__ = [
    "ExperimentLoadErrorDialog",
    "ExperimentLoadFailure",
    "ExperimentLoadResult",
    "ExperimentSearchErrorDialog",
    "ExperimentSearchDialog",
]

COUNT_COLUMN_WIDTH = 72
FILTER_PANEL_WIDTH = 340
TEXT_FIELD_MARGIN_PX = 6

FILTER_HINTS = {
    "user": "gustavo",
    "cell_type": "ALPN",
    "sensor": "g6f",
    "stimulus": "bars_singles_stim=5s_bars=3x400ms_dt=200ms_int=20",
    "date": "2025-10-22",
}


@dataclass(frozen=True)
class ExperimentLoadFailure:
    """One recording path that failed to load from database search.

    Inputs: source recording path and the error message from the loader.
    Outputs: immutable failure item for the error dialog.
    """

    path: Path | None
    message: str


@dataclass(frozen=True)
class ExperimentLoadResult:
    """Result from loading recordings selected in database search.

    Inputs: count of successfully loaded recordings and per-path failures.
    Outputs: structured load result the dialog can use for user feedback.
    """

    loaded_count: int
    failures: tuple[ExperimentLoadFailure, ...] = ()


@dataclass(frozen=True)
class _ResolvedExperimentPaths:
    """Resolved source paths plus database-row failures from one selection."""

    paths: tuple[Path, ...]
    failures: tuple[ExperimentLoadFailure, ...] = ()


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
        on_load_recording_paths: Callable[[tuple[Path, ...]], ExperimentLoadResult],
        config_path: Path = DEFAULT_CONFIG_PATH,
        parent: QWidget | None = None,
    ) -> None:
        """Create the database experiment search dialog."""
        super().__init__(parent)
        self._on_load_recording_paths = on_load_recording_paths
        self._config_path = config_path
        self._config: TwopyConfig | None = None

        self._user_filter = self._filter_line_edit(FILTER_HINTS["user"])
        self._cell_type_filter = self._filter_line_edit(FILTER_HINTS["cell_type"])
        self._sensor_filter = self._filter_line_edit(FILTER_HINTS["sensor"])
        self._stimulus_filter = self._filter_line_edit(FILTER_HINTS["stimulus"])
        self._date_filter = self._filter_line_edit(FILTER_HINTS["date"])
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(("Experiment", "Count"))
        self._configure_result_columns()
        self._tree.itemActivated.connect(self._load_tree_item)

        search_button = QPushButton("Search")
        search_button.clicked.connect(self.search)
        load_button = QPushButton("Load Selected")
        load_button.clicked.connect(self.load_selected)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)

        self.setWindowTitle("Search Database")
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
            config = self._load_config()
            experiments = find_recording_search_results(
                config,
                ExperimentSearchFilters(
                    user=self._user_filter.text(),
                    cell_type=self._cell_type_filter.text(),
                    sensor=self._sensor_filter.text(),
                    stimulus=self._stimulus_filter.text(),
                    date=self._date_filter.text(),
                ),
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            self._tree.clear()
            self._show_search_error(error)
            return

        self._render_tree(build_experiment_search_tree(experiments))

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
        self._load_tree_item(selected[0], 0)

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
        action_layout.addWidget(search_button)

        layout = QVBoxLayout()
        layout.addLayout(self._filter_form())
        layout.addLayout(action_layout)
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
        experiments = cast(
            tuple[DatabaseExperiment, ...],
            item.data(0, Qt.ItemDataRole.UserRole),
        )
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

    def _show_load_errors(self, result: ExperimentLoadResult) -> None:
        """Show database-result load failures with path details."""
        ExperimentLoadErrorDialog(result, parent=self.parentWidget()).exec()


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


class ExperimentLoadErrorDialog(QDialog):
    """Dialog showing database-search load failures.

    Args:
        result: Structured load result containing failed paths and messages.
        parent: Optional Qt parent widget.

    Outputs:
        Modal dialog with a clear summary and scrollable failure list.
    """

    def __init__(
        self,
        result: ExperimentLoadResult,
        *,
        parent: QWidget | None = None,
    ) -> None:
        """Create an error dialog for failed experiment loads."""
        super().__init__(parent)
        self.setWindowTitle("Experiment Load Errors")
        self.resize(720, 420)

        summary = QLabel(_load_failure_summary(result))
        layout = QVBoxLayout()
        layout.addWidget(summary)
        layout.addWidget(_read_only_details(_load_failure_text(result.failures)))
        layout.addWidget(_close_button_box(self))
        self.setLayout(layout)


def _load_failure_summary(result: ExperimentLoadResult) -> str:
    """Return a short load-failure summary."""
    failed_count = len(result.failures)
    if result.loaded_count == 0:
        return f"No recordings loaded. Failed to load {failed_count} recordings."
    return (
        f"Loaded {result.loaded_count} recordings. "
        f"Failed to load {failed_count} recordings."
    )


def _load_failure_text(failures: tuple[ExperimentLoadFailure, ...]) -> str:
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
    failures: list[ExperimentLoadFailure] = []
    seen: set[Path] = set()
    for experiment in experiments:
        try:
            path = recording_path_for_database_experiment(config, experiment)
        except (FileNotFoundError, OSError, ValueError) as error:
            failures.append(
                ExperimentLoadFailure(path=None, message=_experiment_error(error)),
            )
            continue
        if path in seen:
            continue
        paths.append(path)
        seen.add(path)
    return _ResolvedExperimentPaths(paths=tuple(paths), failures=tuple(failures))


def _merge_load_failures(
    result: ExperimentLoadResult,
    failures: tuple[ExperimentLoadFailure, ...],
) -> ExperimentLoadResult:
    """Return one load result with path-resolution and loader failures."""
    if len(failures) == 0:
        return result
    return ExperimentLoadResult(
        loaded_count=result.loaded_count,
        failures=failures + result.failures,
    )


def _experiment_error(error: Exception) -> str:
    """Return a clear path-resolution failure message."""
    return (
        "Could not resolve database experiment path.\n"
        f"{exception_message_for_user(error)}"
    )
