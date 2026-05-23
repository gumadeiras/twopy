"""Progress dialog for asynchronous napari recording loads.

Inputs: batch counts, current path text, and a cancel callback.
Outputs: a modeless Qt dialog that reports load progress without owning the
recording-load workflow.
"""

from collections.abc import Callable
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

__all__ = ["RecordingLoadProgressDialog"]

_MINIMUM_DIALOG_WIDTH = 760
_MAXIMUM_DIALOG_SCREEN_FRACTION = 0.92
_PATH_HEIGHT_PADDING = 8


class RecordingLoadProgressDialog(QDialog):
    """Modeless progress dialog for a recording-load batch.

    Args:
        total_count: Number of recordings requested.
        on_cancel_pending: Callback invoked when the user cancels queued work.
        parent: Optional Qt parent widget.

    Outputs:
        A small non-modal dialog with path, phase, progress, and cancel state.
    """

    def __init__(
        self,
        *,
        total_count: int,
        on_cancel_pending: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        """Create the progress dialog."""
        super().__init__(parent)
        self._on_cancel_pending = on_cancel_pending
        self.setWindowTitle("Loading recordings")
        self.setModal(False)
        self.setMinimumWidth(_MINIMUM_DIALOG_WIDTH)

        self._summary = QLabel(
            _summary_text(completed_count=0, total_count=total_count)
        )
        self._summary.setWordWrap(True)
        self._path = QLabel("")
        self._path.setWordWrap(True)
        self._path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._phase = QLabel("Preparing load...")
        self._phase.setWordWrap(True)
        self._progress = QProgressBar()
        self._progress.setRange(0, max(total_count, 1))
        self._progress.setValue(0)
        self._cancel_button = QPushButton("Cancel pending")
        self._cancel_button.clicked.connect(self._cancel_pending)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._cancel_button)

        layout = QVBoxLayout()
        layout.addWidget(self._summary)
        layout.addWidget(self._path)
        layout.addWidget(self._phase)
        layout.addWidget(self._progress)
        layout.addLayout(buttons)
        self.setLayout(layout)
        self.resize(_MINIMUM_DIALOG_WIDTH, self.sizeHint().height())

    def update_progress(
        self,
        *,
        completed_count: int,
        total_count: int,
        current_path: Path | None,
        phase: str,
    ) -> None:
        """Update the visible load progress.

        Args:
            completed_count: Number of recordings finished or failed.
            total_count: Total requested recording count.
            current_path: Path currently being prepared or applied.
            phase: Short user-facing phase text.

        Returns:
            None.
        """
        self._summary.setText(
            _summary_text(completed_count=completed_count, total_count=total_count)
        )
        path_text = "" if current_path is None else str(current_path)
        self._path.setText(path_text)
        self._phase.setText(phase)
        self._progress.setRange(0, max(total_count, 1))
        self._progress.setValue(min(completed_count, total_count))
        self._fit_to_path(path_text)

    def mark_finishing(self, *, phase: str) -> None:
        """Show final status and disable cancellation.

        Args:
            phase: Final user-facing phase text.

        Returns:
            None.
        """
        self._phase.setText(phase)
        self._cancel_button.setEnabled(False)

    def _cancel_pending(self) -> None:
        """Ask the controller to skip queued recordings after the active load."""
        self._cancel_button.setEnabled(False)
        self._phase.setText("Finishing current recording; pending loads cancelled.")
        self._on_cancel_pending()

    def _fit_to_path(self, path_text: str) -> None:
        """Resize the dialog so wrapped path text is fully visible."""
        screen = self.screen() or QApplication.primaryScreen()
        maximum_width = _MINIMUM_DIALOG_WIDTH
        if screen is not None:
            maximum_width = max(
                _MINIMUM_DIALOG_WIDTH,
                int(
                    screen.availableGeometry().width() * _MAXIMUM_DIALOG_SCREEN_FRACTION
                ),
            )
        text_width = self.fontMetrics().horizontalAdvance(path_text)
        width = max(_MINIMUM_DIALOG_WIDTH, min(maximum_width, text_width))
        self.resize(width, self.height())
        content_width = width
        layout = self.layout()
        if layout is not None:
            margins = layout.contentsMargins()
            content_width = max(width - margins.left() - margins.right(), 1)
        path_height = 0
        if path_text != "":
            rect = self._path.fontMetrics().boundingRect(
                0,
                0,
                content_width,
                10_000,
                Qt.TextFlag.TextWordWrap | Qt.TextFlag.TextWrapAnywhere,
                path_text,
            )
            path_height = rect.height() + _PATH_HEIGHT_PADDING
        self._path.setMinimumHeight(path_height)
        self._path.updateGeometry()
        self.adjustSize()


def _summary_text(*, completed_count: int, total_count: int) -> str:
    """Return compact batch progress text."""
    return f"Processed {completed_count} of {total_count} selected recordings."
