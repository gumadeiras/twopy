"""Shared table widgets for Group Matching views.

Inputs: Qt table mouse events from FOV and ROI saved-group tables.
Outputs: row-selection behavior that lets users click the selected row again
to clear it.
"""

from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QTableWidget

__all__ = ["ToggleSelectionTable"]


class ToggleSelectionTable(QTableWidget):
    """Table that clears selection when the selected row is clicked again."""

    def mousePressEvent(self, e: QMouseEvent | None) -> None:  # noqa: N802
        """Toggle the current row off when users click it a second time."""
        if e is None:
            super().mousePressEvent(e)
            return
        row_index = self.rowAt(e.pos().y())
        if row_index >= 0 and row_index == self.currentRow() and self.selectedItems():
            self.clearSelection()
            self.setCurrentCell(-1, -1)
            return
        super().mousePressEvent(e)
