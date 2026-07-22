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
    QTreeView,
)

from twopy.napari.theme import apply_twopy_theme

__all__ = [
    "choose_loaded_recording_csvs",
    "choose_recording_folders",
    "selected_browse_folder",
]


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
    dialog = QFileDialog()
    apply_twopy_theme(dialog, name="twopy_load_path_dialog")
    dialog.setWindowTitle(title)
    dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
    dialog.setLabelText(QFileDialog.DialogLabel.Accept, "Load")
    dialog.setFileMode(file_mode)
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setOption(QFileDialog.Option.ShowDirsOnly, show_directories_only)
    if start_folder is not None:
        dialog.setDirectory(str(start_folder.expanduser().resolve()))
    if name_filters:
        dialog.setNameFilters(list(name_filters))
        dialog.selectNameFilter(name_filters[0])
    _allow_extended_selection(dialog)
    if dialog.exec() != QFileDialog.DialogCode.Accepted:
        return ()
    return tuple(Path(path).expanduser() for path in dialog.selectedFiles())


def _allow_extended_selection(dialog: QFileDialog) -> None:
    """Allow multiple rows in both Qt file-dialog view modes."""
    for view in (*dialog.findChildren(QListView), *dialog.findChildren(QTreeView)):
        view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
