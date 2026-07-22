"""File dialogs for recording folders and saved recording lists.

Inputs: separate starting folders for manual loads and CSV-list loads.
Outputs: paths selected through one consistent, napari-themed dialog.

The Qt dialog supports multiple recording folders and keeps both load actions
visually consistent. Native platform dialogs cannot provide both behaviors.
"""

from pathlib import Path

from qtpy.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QListView,
    QSplitter,
    QTreeView,
)

from twopy.napari.theme import TwopyThemeColors, apply_twopy_theme

__all__ = [
    "choose_loaded_recording_csvs",
    "choose_recording_folders",
    "selected_browse_folder",
]

_LOAD_DIALOG_WIDTH = 960
_LOAD_DIALOG_HEIGHT = 600
_LOAD_DIALOG_SIDEBAR_WIDTH = 180
_LOAD_DIALOG_COLUMN_WIDTHS = (320, 90, 140, 180)


def choose_recording_folders(start_folder: Path | None) -> tuple[Path, ...]:
    """Choose one or more recording folders.

    Args:
        start_folder: Folder to show when the dialog opens.

    Returns:
        Selected recording folders, or an empty tuple after cancellation.

    The non-native Qt dialog is required because native directory dialogs do
    not support multiple folder selection consistently.
    """
    return _choose_existing_paths(
        title="Load recording folders",
        start_folder=start_folder,
        file_mode=QFileDialog.FileMode.Directory,
        show_directories_only=True,
    )


def choose_loaded_recording_csvs(start_folder: Path | None) -> tuple[Path, ...]:
    """Choose one or more saved recording-list CSV files.

    Args:
        start_folder: Folder to show when the dialog opens.

    Returns:
        Selected CSV files, or an empty tuple after cancellation.

    This function uses the same dialog surface as manual recording selection.
    """
    return _choose_existing_paths(
        title="Load recording lists",
        start_folder=start_folder,
        file_mode=QFileDialog.FileMode.ExistingFiles,
        show_directories_only=False,
        name_filters=("CSV files (*.csv)", "All files (*)"),
    )


def selected_browse_folder(paths: tuple[Path, ...]) -> Path | None:
    """Return the folder that contains the first accepted selection.

    Args:
        paths: Paths accepted by one load dialog.

    Returns:
        The containing folder, or ``None`` when the user selected nothing.

    Opening the containing folder shows the last selection and nearby items.
    """
    if len(paths) == 0:
        return None
    return paths[0].expanduser().parent


def _choose_existing_paths(
    *,
    title: str,
    start_folder: Path | None,
    file_mode: QFileDialog.FileMode,
    show_directories_only: bool,
    name_filters: tuple[str, ...] = (),
) -> tuple[Path, ...]:
    """Choose paths through the shared twopy load dialog."""
    dialog = _build_load_dialog(
        title=title,
        start_folder=start_folder,
        file_mode=file_mode,
        show_directories_only=show_directories_only,
        name_filters=name_filters,
    )
    if dialog.exec() != QFileDialog.DialogCode.Accepted:
        return ()
    return tuple(Path(path).expanduser() for path in dialog.selectedFiles())


def _build_load_dialog(
    *,
    title: str,
    start_folder: Path | None,
    file_mode: QFileDialog.FileMode,
    show_directories_only: bool,
    name_filters: tuple[str, ...] = (),
) -> QFileDialog:
    """Return one fully configured twopy load dialog."""
    dialog = QFileDialog()
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    apply_twopy_theme(
        dialog,
        name="twopy_load_path_dialog",
        additional_style=_load_dialog_style,
    )
    dialog.setWindowTitle(title)
    dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
    dialog.setLabelText(QFileDialog.DialogLabel.Accept, "Load")
    dialog.setFileMode(file_mode)
    dialog.setOption(QFileDialog.Option.ShowDirsOnly, show_directories_only)
    if start_folder is not None:
        dialog.setDirectory(str(start_folder.expanduser().resolve()))
    if name_filters:
        dialog.setNameFilters(list(name_filters))
        dialog.selectNameFilter(name_filters[0])
    _allow_extended_selection(dialog)
    _configure_load_dialog_layout(dialog)
    return dialog


def _allow_extended_selection(dialog: QFileDialog) -> None:
    """Allow multiple rows in both Qt file-dialog view modes."""
    for view in (*dialog.findChildren(QListView), *dialog.findChildren(QTreeView)):
        view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)


def _configure_load_dialog_layout(dialog: QFileDialog) -> None:
    """Set readable initial pane and detail-column sizes."""
    dialog.resize(_LOAD_DIALOG_WIDTH, _LOAD_DIALOG_HEIGHT)

    sidebar = dialog.findChild(QListView, "sidebar")
    splitter = dialog.findChild(QSplitter, "splitter")
    if sidebar is not None:
        sidebar.setMinimumWidth(_LOAD_DIALOG_SIDEBAR_WIDTH)
    if splitter is not None:
        splitter.setCollapsible(0, False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes(
            [
                _LOAD_DIALOG_SIDEBAR_WIDTH,
                _LOAD_DIALOG_WIDTH - _LOAD_DIALOG_SIDEBAR_WIDTH,
            ]
        )

    tree_view = dialog.findChild(QTreeView, "treeView")
    if tree_view is None:
        return
    header = tree_view.header()
    if header is None:
        return
    for index, width in enumerate(_LOAD_DIALOG_COLUMN_WIDTHS[: header.count()]):
        header.resizeSection(index, width)


def _load_dialog_style(_colors: TwopyThemeColors) -> str:
    """Return consistent rounded corners for both load-dialog panes."""
    return """
QFileDialog#twopy_load_path_dialog QListView#sidebar,
QFileDialog#twopy_load_path_dialog QTreeView#treeView {
    border-radius: 7px;
}
"""
