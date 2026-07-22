"""Small Qt panel builders for response plotting widgets.

Inputs: already-created buttons, labels, layouts, and plot widgets.
Outputs: composed Qt widgets used by the napari response docks.

This module owns generic panel layout only. It does not load data, compute
responses, or export figures.
"""

from html import escape

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGroupBox,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from twopy.napari.plotting.widgets import EpochPlotWidget

__all__ = [
    "SidebarTextLabel",
    "epoch_plot_panel",
    "response_metadata_tab",
    "scrolling_tab",
]


class SidebarTextLabel(QLabel):
    """Selectable sidebar text that wraps long paths inside the dock width.

    Args:
        text: Initial plain text shown in the sidebar label.

    Returns:
        A label that keeps a copy-clean plain text value while giving Qt
        display-only break opportunities for long path-like tokens.

    Metadata and export status often contain unspaced filesystem paths. A plain
    QLabel wraps words but not every path-like token, so this label renders
    escaped rich text with zero-width break hints and keeps the original text in
    ``text()`` for programmatic reads.
    """

    def __init__(self, text: str = "") -> None:
        """Create a copyable, wrapping sidebar text label."""
        super().__init__()
        self._plain_text = ""
        self._configure_text_label()
        self.setText(text)

    def setText(self, a0: str | None) -> None:
        """Replace the visible text with plain copyable text.

        Args:
            a0: Text to display and expose for copying.

        Returns:
            None.
        """
        self._plain_text = a0 or ""
        super().setText(_sidebar_label_markup(self._plain_text))
        self.setVisible(bool(self._plain_text))

    def text(self) -> str:
        """Return the plain text currently visible in the label.

        Args:
            None.

        Returns:
            Copy-clean text without inserted line-break characters.
        """
        return self._plain_text

    def _configure_text_label(self) -> None:
        """Apply label behavior needed for narrow sidebar tabs."""
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(1)


def _sidebar_label_markup(text: str) -> str:
    """Return rich text with break opportunities for path-like sidebar values.

    Args:
        text: Plain user-visible text.

    Returns:
        Escaped rich text that preserves explicit newlines and lets Qt wrap long
        path-like tokens at natural separators.
    """
    break_after = frozenset(("/", "\\", "_", "-", ".", ",", ";", ":", "+"))
    parts: list[str] = []
    for character in text:
        if character == "\n":
            parts.append("<br>")
            continue
        parts.append(escape(character))
        if character in break_after:
            parts.append("&#8203;")
    return "".join(parts)


def epoch_plot_panel(*, title: str, plot: EpochPlotWidget) -> QWidget:
    """Create one titled plot panel for the horizontal response strip.

    Args:
        title: Plot title.
        plot: Epoch response plot widget.

    Returns:
        Qt widget containing the title and plot.
    """
    panel = QWidget()
    layout = QVBoxLayout()
    title_label = QLabel(title)
    title_label.setObjectName("twopy_plot_title")
    title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    layout.addWidget(title_label)
    layout.addWidget(plot)
    panel.setLayout(layout)
    return panel


def response_metadata_tab(
    *,
    recording_summary_label: SidebarTextLabel,
    microscope_summary_label: SidebarTextLabel,
    analysis_output_label: SidebarTextLabel,
    roi_output_label: SidebarTextLabel,
    status_label: SidebarTextLabel,
    update_notice_label: SidebarTextLabel,
    motion_summary_widget: QWidget | None = None,
) -> QWidget:
    """Create the tab that shows selected-recording metadata.

    Args:
        recording_summary_label: Label showing root, genotype, stimulus, and
            recording time for the selected recording.
        microscope_summary_label: Label showing microscope acquisition fields.
        analysis_output_label: Label showing the analysis output path.
        roi_output_label: Label showing the ROI output path.
        status_label: Label showing the latest app action or update command.
        update_notice_label: Label showing the copyable update command.
        motion_summary_widget: Optional section showing alignment movement.

    Returns:
        Scrollable Qt widget for the Metadata tab.
    """
    layout = QVBoxLayout()
    layout.setSpacing(6)
    layout.addWidget(_metadata_group("Recording", recording_summary_label))
    layout.addWidget(_metadata_group("Microscope", microscope_summary_label))
    layout.addWidget(
        _metadata_group("Outputs", analysis_output_label, roi_output_label),
    )
    layout.addWidget(_metadata_group("Status", status_label, update_notice_label))
    if motion_summary_widget is not None:
        layout.addWidget(motion_summary_widget)
    layout.addStretch(1)
    tab = scrolling_tab(layout)
    for label in (
        recording_summary_label,
        microscope_summary_label,
        analysis_output_label,
        roi_output_label,
        status_label,
        update_notice_label,
    ):
        label.setVisible(bool(label.text()))
    return tab


def _metadata_group(title: str, *labels: SidebarTextLabel) -> QGroupBox:
    """Create one Metadata-tab subsection using Plot-tab group styling."""
    group = QGroupBox(title)
    layout = QVBoxLayout()
    layout.setSpacing(4)
    group.setLayout(layout)
    for label in labels:
        layout.addWidget(label)
    return group


def scrolling_tab(layout: QVBoxLayout) -> QScrollArea:
    """Create a scrollable options tab around one layout.

    Args:
        layout: Layout that callers will populate with controls.

    Returns:
        Scroll area suitable for a tab body.
    """
    content = QWidget()
    content.setLayout(layout)
    content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    content.setMinimumWidth(1)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(content)
    return scroll
