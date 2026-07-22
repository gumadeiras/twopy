"""Shared visual theme for twopy Qt views.

Inputs: napari themes, Qt palette fallbacks, and twopy widgets.
Outputs: live theme colors, widget styles, and action roles.

The theme keeps all twopy views consistent without changing napari itself.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

from napari.settings import get_settings
from napari.utils.theme import Theme, get_theme
from qtpy.QtCore import QEvent, QObject
from qtpy.QtGui import QColor, QFontDatabase, QPalette
from qtpy.QtWidgets import QLabel, QPushButton, QWidget

__all__ = [
    "TWOPY_CONTROL_HEIGHT",
    "TwopyThemeColors",
    "active_twopy_theme_colors",
    "apply_twopy_theme",
    "readable_text_color",
    "style_action_button",
    "style_caption",
    "style_section_title",
    "twopy_theme_colors",
    "twopy_theme_style_sheet",
]

TWOPY_CONTROL_HEIGHT = 40
_LIGHT_THEME_ACCENT = QColor("#c45116")
_LIGHT_THEME_ACCENT_TEXT = QColor("#ffffff")
type AdditionalStyle = Callable[["TwopyThemeColors"], str]


class _ThemeChangeWatcher(QObject):
    """Refresh one local stylesheet after a napari theme change."""

    def __init__(
        self,
        target: QWidget,
        additional_style: AdditionalStyle | None,
    ) -> None:
        """Store the themed root and its optional view-specific style."""
        super().__init__(target)
        self._target = target
        self._additional_style = additional_style
        self._is_refreshing = False
        get_settings().appearance.events.theme.connect(self._napari_theme_changed)

    def _napari_theme_changed(self, _event: object | None = None) -> None:
        """Refresh colors after napari selects a different theme."""
        self._refresh()

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:  # noqa: N802
        """Refresh theme colors when the watched palette changes."""
        if (
            a0 is self._target
            and a1 is not None
            and a1.type()
            in (QEvent.Type.ApplicationPaletteChange, QEvent.Type.PaletteChange)
            and not self._is_refreshing
        ):
            self._refresh()
        return super().eventFilter(a0, a1)

    def _refresh(self) -> None:
        """Refresh the watched widget once without recursive updates."""
        if self._is_refreshing:
            return
        self._is_refreshing = True
        try:
            _refresh_twopy_theme(self._target, self._additional_style)
        finally:
            self._is_refreshing = False


@dataclass(frozen=True)
class TwopyThemeColors:
    """Store CSS colors for one active theme.

    Inputs: colors calculated from one napari theme or Qt palette fallback.
    Outputs: immutable color values for twopy stylesheets.

    One color set keeps all views consistent after a napari theme change.
    """

    window: str
    surface: str
    raised_surface: str
    field: str
    text: str
    muted_text: str
    faint_text: str
    border: str
    control_outline: str
    divider: str
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_text: str
    primary_outline: str
    selected_surface: str
    button: str
    button_hover: str
    button_pressed: str
    disabled_surface: str
    disabled_text: str
    is_dark: bool


def apply_twopy_theme(
    widget: QWidget,
    *,
    name: str,
    additional_style: AdditionalStyle | None = None,
) -> None:
    """Apply the shared theme to one twopy widget tree.

    Args:
        widget: Root widget that owns the twopy view.
        name: Stable object name for tests and scoped additions.
        additional_style: Optional theme-aware style for one special view.

    Returns:
        None.

    The local stylesheet preserves napari styling outside this widget tree.
    """
    widget.setObjectName(name)
    _refresh_twopy_theme(widget, additional_style)
    watcher = _ThemeChangeWatcher(widget, additional_style)
    widget.installEventFilter(watcher)


def style_action_button(button: QPushButton, *, role: str = "secondary") -> None:
    """Set the visual role and hit area for one action button.

    Args:
        button: Action button to configure.
        role: One of ``primary``, ``secondary``, ``quiet``, or ``danger``.

    Returns:
        None.

    The role shows action priority and keeps repeated workflows easy to scan.
    """
    allowed_roles = {"primary", "secondary", "quiet", "danger"}
    if role not in allowed_roles:
        msg = f"Unknown twopy button role: {role}."
        raise ValueError(msg)
    button.setProperty("twopyRole", role)
    button.setMinimumHeight(TWOPY_CONTROL_HEIGHT)


def style_section_title(label: QLabel) -> None:
    """Style one label as a view section title.

    Args:
        label: Visible section title.

    Returns:
        None.

    Section titles make dense scientific controls easier to scan.
    """
    label.setObjectName("twopy_section_title")


def style_caption(label: QLabel) -> None:
    """Style one label as quiet supporting text.

    Args:
        label: Visible supporting text.

    Returns:
        None.

    Captions keep instructions available without competing with actions.
    """
    label.setObjectName("twopy_caption")


def readable_text_color(background: QColor) -> str:
    """Return black or white text for one data color.

    Args:
        background: Data color behind the text.

    Returns:
        A readable black or white CSS color.

    Data chips keep their scientific colors in every napari theme.
    """
    return "#202124" if background.lightness() >= 150 else "#ffffff"


def twopy_theme_colors(palette: QPalette) -> TwopyThemeColors:
    """Return twopy colors from one active Qt palette.

    Args:
        palette: Active palette from napari or the operating system.

    Returns:
        CSS color values for twopy widget styles.

    Palette colors keep the interface readable in light and dark themes.
    """
    window = palette.window().color()
    base = palette.base().color()
    text = palette.windowText().color()
    button = palette.button().color()
    button_text = palette.buttonText().color()
    accent = palette.highlight().color()
    accent_text = palette.highlightedText().color()
    is_dark = window.lightness() < 128

    surface = _mix(base, window, 0.18 if is_dark else 0.10)
    raised_surface = _mix(base, window, 0.08 if is_dark else 0.04)
    field = _mix(base, window, 0.02 if is_dark else 0.01)
    border = _mix(text, window, 0.20 if is_dark else 0.16)
    divider = _mix(text, window, 0.13 if is_dark else 0.10)
    selected_surface = _mix(accent, surface, 0.24 if is_dark else 0.13)
    button_hover = button.lighter(118) if is_dark else button.darker(105)
    button_pressed = button.lighter(128) if is_dark else button.darker(112)
    accent_hover = accent.lighter(112) if is_dark else accent.darker(106)
    accent_pressed = accent.lighter(122) if is_dark else accent.darker(114)
    return TwopyThemeColors(
        window=_css(window),
        surface=_css(surface),
        raised_surface=_css(raised_surface),
        field=_css(field),
        text=_css(text),
        muted_text=_css(_mix(text, window, 0.62 if is_dark else 0.58)),
        faint_text=_css(_mix(text, window, 0.42 if is_dark else 0.38)),
        border=_css(border),
        control_outline=_css(_mix(text, window, 0.62 if is_dark else 0.68)),
        divider=_css(divider),
        accent=_css(accent),
        accent_hover=_css(accent_hover),
        accent_pressed=_css(accent_pressed),
        accent_text=_css(accent_text),
        primary_outline=_css(_mix(accent_text, accent, 0.58)),
        selected_surface=_css(selected_surface),
        button=_css(button),
        button_hover=_css(button_hover),
        button_pressed=_css(button_pressed),
        disabled_surface=_css(_mix(button, window, 0.46)),
        disabled_text=_css(_mix(button_text, window, 0.42)),
        is_dark=is_dark,
    )


def active_twopy_theme_colors(palette: QPalette) -> TwopyThemeColors:
    """Return colors from the active napari theme.

    Args:
        palette: Qt palette used only if napari has no valid active theme.

    Returns:
        CSS color values for the current napari theme.

    Napari changes its stylesheet without changing the Qt application palette.
    This function reads the same theme model that napari uses.
    """
    theme_id = str(get_settings().appearance.theme)
    try:
        theme = get_theme(theme_id)
    except (RuntimeError, ValueError):
        return twopy_theme_colors(palette)
    if not isinstance(theme, Theme):
        return twopy_theme_colors(palette)
    return _colors_from_napari_theme(theme)


def twopy_theme_style_sheet(source: QPalette | TwopyThemeColors) -> str:
    """Return the shared twopy stylesheet for one color source.

    Args:
        source: Active theme colors or a Qt palette fallback.

    Returns:
        Qt stylesheet text for twopy views.

    The stylesheet gives all views one hierarchy and one interaction language.
    """
    color = (
        source if isinstance(source, TwopyThemeColors) else twopy_theme_colors(source)
    )
    interface_font = _installed_font(
        "Avenir Next",
        "Avenir",
        "Helvetica Neue",
        "Arial",
        "Noto Sans",
        "DejaVu Sans",
    )
    number_font = _installed_font("Menlo", "Monaco", "DejaVu Sans Mono")
    return f"""
QWidget {{
    color: {color.text};
    font-family: "{interface_font}";
    font-size: 13px;
}}
QDialog, QTabWidget {{
    background: {color.window};
}}
QLabel#twopy_section_title {{
    color: {color.text};
    font-size: 17px;
    font-weight: 700;
    padding: 0 0 2px 0;
}}
QLabel#twopy_plot_title {{
    color: {color.text};
    font-size: 14px;
    font-weight: 700;
    padding: 3px 0 1px 0;
}}
QLabel#twopy_caption,
QLabel[twopyRole="caption"] {{
    color: {color.muted_text};
}}
QLabel[twopyRole="status"] {{
    color: {color.muted_text};
    background: {color.surface};
    border-radius: 6px;
    padding: 8px 10px;
}}
QLabel[twopyRole="number"] {{
    font-family: "{number_font}";
}}
QWidget[twopyRole="colorSwatch"] {{
    border: 1px solid {color.control_outline};
}}
QTabWidget::pane {{
    background: {color.window};
    border: 1px solid {color.divider};
    border-radius: 8px;
    top: -1px;
}}
QTabBar::tab {{
    background: {color.window};
    color: {color.muted_text};
    border: none;
    border-bottom: 2px solid transparent;
    min-height: 26px;
    min-width: 42px;
    padding: 7px 10px;
}}
QTabBar::tab:!selected {{
    background: {color.window};
}}
QTabBar::tab:hover {{
    color: {color.text};
    background: {color.surface};
}}
QTabBar::tab:selected {{
    color: {color.text};
    background: {color.selected_surface};
    border-bottom: 2px solid {color.accent};
    font-weight: 700;
}}
QGroupBox {{
    background: {color.window};
    border: 1px solid {color.control_outline};
    border-radius: 10px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: 700;
}}
QGroupBox::title {{
    color: {color.text};
    background: {color.window};
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
}}
QGroupBox QLabel,
QGroupBox QCheckBox,
QGroupBox QRadioButton {{
    font-weight: 400;
}}
QPushButton {{
    background: {color.button};
    color: {color.text};
    border: 1px solid {color.control_outline};
    border-radius: 7px;
    min-height: 38px;
    padding: 0 13px;
}}
QPushButton:hover {{
    background: {color.button_hover};
    border-color: {color.accent};
}}
QPushButton:pressed {{
    background: {color.button_pressed};
    padding-top: 1px;
}}
QPushButton:focus {{
    border: 2px solid {color.accent};
}}
QPushButton[twopyRole="primary"] {{
    background: {color.accent};
    color: {color.accent_text};
    border-color: {color.primary_outline};
    font-weight: 700;
}}
QPushButton[twopyRole="primary"]:hover {{
    background: {color.accent_hover};
}}
QPushButton[twopyRole="primary"]:pressed {{
    background: {color.accent_pressed};
}}
QPushButton[twopyRole="quiet"] {{
    background: transparent;
    color: {color.muted_text};
    border-color: {color.control_outline};
}}
QPushButton[twopyRole="quiet"]:hover {{
    background: {color.surface};
    color: {color.text};
    border-color: {color.divider};
}}
QPushButton:disabled {{
    background: {color.disabled_surface};
    color: {color.disabled_text};
    border-color: {color.control_outline};
    font-weight: 400;
}}
QLineEdit,
QPlainTextEdit,
QTextEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox,
QDateEdit {{
    background: {color.field};
    color: {color.text};
    border: 1px solid {color.border};
    border-radius: 7px;
    min-height: 28px;
    padding: 4px 8px;
    selection-background-color: {color.accent};
    selection-color: {color.accent_text};
}}
QLineEdit:hover,
QPlainTextEdit:hover,
QTextEdit:hover,
QComboBox:hover,
QSpinBox:hover,
QDoubleSpinBox:hover,
QDateEdit:hover {{
    border-color: {color.accent};
}}
QLineEdit:focus,
QPlainTextEdit:focus,
QTextEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QDateEdit:focus {{
    border: 2px solid {color.accent};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QAbstractItemView {{
    background: {color.field};
    color: {color.text};
    border: 1px solid {color.control_outline};
    border-radius: 7px;
    outline: none;
    selection-background-color: {color.selected_surface};
    selection-color: {color.text};
}}
QAbstractItemView::item {{
    min-height: 30px;
    padding: 4px 7px;
}}
QAbstractItemView::item:hover {{
    background: {color.surface};
}}
QAbstractItemView::item:selected {{
    background: {color.selected_surface};
    color: {color.text};
}}
QTableView, QTreeView {{
    border-radius: 0;
}}
QHeaderView {{
    background: {color.surface};
    border: none;
    border-bottom: 1px solid {color.control_outline};
}}
QHeaderView::section {{
    background: transparent;
    color: {color.muted_text};
    border: none;
    padding: 7px 8px;
    font-weight: 700;
}}
QTableWidget {{
    gridline-color: {color.divider};
}}
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    border: none;
    margin: 2px;
    width: 10px;
}}
QScrollBar::handle:vertical {{
    background: {color.border};
    border-radius: 4px;
    min-height: 32px;
}}
QScrollBar::handle:vertical:hover {{
    background: {color.muted_text};
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    border: none;
    margin: 2px;
    height: 10px;
}}
QScrollBar::handle:horizontal {{
    background: {color.border};
    border-radius: 4px;
    min-width: 32px;
}}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0;
}}
QSplitter::handle {{
    background: {color.divider};
    width: 1px;
    height: 1px;
}}
QProgressBar {{
    background: {color.field};
    color: {color.text};
    border: 1px solid {color.divider};
    border-radius: 6px;
    min-height: 20px;
    text-align: center;
    font-family: "{number_font}";
}}
QProgressBar::chunk {{
    background: {color.accent};
    border-radius: 5px;
}}
QToolTip {{
    background: {color.raised_surface};
    color: {color.text};
    border: 1px solid {color.border};
    border-radius: 5px;
    padding: 5px 7px;
}}
"""


def _mix(foreground: QColor, background: QColor, foreground_weight: float) -> QColor:
    """Return one weighted mix of two colors."""
    background_weight = 1.0 - foreground_weight
    return QColor(
        round(
            foreground.red() * foreground_weight + background.red() * background_weight,
        ),
        round(
            foreground.green() * foreground_weight
            + background.green() * background_weight,
        ),
        round(
            foreground.blue() * foreground_weight
            + background.blue() * background_weight,
        ),
    )


def _css(color: QColor) -> str:
    """Return one Qt color as CSS text."""
    return color.name()


def _refresh_twopy_theme(
    widget: QWidget,
    additional_style: AdditionalStyle | None,
) -> None:
    """Refresh shared and view-specific styles for one widget."""
    colors = active_twopy_theme_colors(widget.palette())
    stylesheet = twopy_theme_style_sheet(colors)
    if additional_style is not None:
        stylesheet += additional_style(colors)
    widget.setStyleSheet(stylesheet)


def _colors_from_napari_theme(theme: Theme) -> TwopyThemeColors:
    """Return twopy colors from one validated napari theme."""
    background = QColor(theme.background.as_hex())
    foreground = QColor(theme.foreground.as_hex())
    primary = QColor(theme.primary.as_hex())
    secondary = QColor(theme.secondary.as_hex())
    highlight = QColor(theme.highlight.as_hex())
    text = QColor(theme.text.as_hex())
    icon = QColor(theme.icon.as_hex())
    current = (
        QColor(_LIGHT_THEME_ACCENT)
        if theme.id == "light"
        else QColor(theme.current.as_hex())
    )
    accent_text = (
        QColor(_LIGHT_THEME_ACCENT_TEXT) if theme.id == "light" else QColor(text)
    )
    is_dark = theme.type == "dark"
    return TwopyThemeColors(
        window=_css(background),
        surface=_css(foreground),
        raised_surface=_css(primary),
        field=_css(foreground),
        text=_css(text),
        muted_text=_css(icon),
        faint_text=_css(secondary),
        border=_css(primary),
        control_outline=_css(secondary),
        divider=_css(primary),
        accent=_css(current),
        accent_hover=_css(_mix(text, current, 0.10)),
        accent_pressed=_css(_mix(text, current, 0.18)),
        accent_text=_css(accent_text),
        primary_outline=_css(_mix(text, current, 0.42)),
        selected_surface=_css(highlight),
        button=_css(foreground),
        button_hover=_css(primary),
        button_pressed=_css(highlight),
        disabled_surface=_css(_mix(foreground, background, 0.55)),
        disabled_text=_css(_mix(text, background, 0.44)),
        is_dark=is_dark,
    )


@cache
def _installed_font(*families: str) -> str:
    """Return the first requested font that Qt can use."""
    installed = set(QFontDatabase.families())
    return next((family for family in families if family in installed), families[-1])
