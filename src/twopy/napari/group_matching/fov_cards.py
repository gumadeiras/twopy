"""FOV mean-image cards for manual group matching.

Inputs: one loaded napari recording, existing FOV assignment text, and row notes.
Outputs: a fixed-size selectable card used by the FOV assignment grid.

The card owns only display and per-recording edit state. The assignment view owns
batch actions, CSV persistence, and table selection.
"""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QVBoxLayout,
)

from twopy.napari.group_matching.cards import fov_group_display_id, image_overlay_stack
from twopy.napari.group_matching.images import (
    THUMBNAIL_SIZE,
    mean_image_thumbnail_pixmap,
)
from twopy.napari.group_matching.style import group_matching_button
from twopy.napari.session import LoadedNapariRecording
from twopy.napari.text import configure_placeholder

__all__ = [
    "FOV_CARD_WIDTH",
    "FovRecordingCard",
]

FOV_CARD_WIDTH = THUMBNAIL_SIZE + 72
_FOV_CARD_HEIGHT = THUMBNAIL_SIZE + 90


class FovRecordingCard(QFrame):
    """One selectable mean-image card for FOV assignment."""

    def __init__(
        self,
        *,
        recording: LoadedNapariRecording,
        fov_group_id: str,
        note: str,
    ) -> None:
        """Create a FOV-assignment card.

        Args:
            recording: Loaded recording represented by this card.
            fov_group_id: Existing FOV assignment, if any.
            note: Existing row-specific FOV note, if any.
        """
        super().__init__()
        self.recording_path = recording.recording.source_session_dir.expanduser()
        self._mean_image_layer = recording.mean_image_layer
        self._image_label = QLabel()
        self._overlay_label = QLabel()
        self._note_edit = QLineEdit(note)
        self._note_edit.setObjectName("fov_recording_note")
        configure_placeholder(self._note_edit, "Optional note")
        self._select_button = group_matching_button("Select", role="quiet")
        self._select_button.setCheckable(True)
        self._select_button.toggled.connect(self._sync_selection_style)
        self._contrast_slider = QSlider(Qt.Orientation.Vertical)
        self._contrast_slider.setRange(1, 100)
        self._contrast_slider.setValue(100)
        self._contrast_slider.setFixedHeight(THUMBNAIL_SIZE)
        self._contrast_slider.setToolTip("Contrast")
        self._contrast_slider.valueChanged.connect(self._refresh_image)

        image_stack = image_overlay_stack(
            self._image_label,
            self._overlay_label,
            size=THUMBNAIL_SIZE,
        )

        preview_row = QHBoxLayout()
        preview_row.setContentsMargins(0, 0, 0, 0)
        preview_row.setSpacing(8)
        preview_row.addWidget(image_stack)
        preview_row.addWidget(self._contrast_slider)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        layout.addLayout(preview_row)
        layout.addWidget(self._note_edit)
        layout.addWidget(self._select_button)
        self.setLayout(layout)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("fov_recording_card")
        self.setFixedSize(FOV_CARD_WIDTH, _FOV_CARD_HEIGHT)
        self.set_fov_group_id(fov_group_id)
        self._refresh_image()
        self._sync_selection_style()

    def is_selected(self) -> bool:
        """Return whether this card is selected for batch assignment."""
        return self._select_button.isChecked()

    def set_selected(self, selected: bool) -> None:
        """Set whether this card is selected for batch assignment."""
        self._select_button.setChecked(selected)

    def set_fov_group_id(self, fov_group_id: str) -> None:
        """Set the visual FOV assignment label."""
        fov_text = fov_group_id if fov_group_id else "none"
        self._overlay_label.setText(
            f"{self.recording_path.name} - FOV ID: {fov_group_display_id(fov_text)}",
        )

    def note(self) -> str:
        """Return the row-specific note for this recording."""
        return self._note_edit.text()

    def set_note(self, note: str) -> None:
        """Set the row-specific note without rebuilding the card."""
        self._note_edit.setText(note)

    def _refresh_image(self) -> None:
        """Refresh the card thumbnail using the current contrast slider."""
        pixmap = mean_image_thumbnail_pixmap(
            self._mean_image_layer,
            contrast_percentile=float(self._contrast_slider.value()),
        )
        if pixmap is not None:
            self._image_label.setPixmap(pixmap)

    def _sync_selection_style(self) -> None:
        """Show selected cards with a visible border."""
        if self.is_selected():
            self._select_button.setText("Selected")
        else:
            self._select_button.setText("Select")
        self.setProperty("selected", self.is_selected())
        style = self.style()
        if style is not None:
            style.unpolish(self)
            style.polish(self)
