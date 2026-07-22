"""Tests for the shared twopy napari visual theme.

Inputs: small Qt widgets and light or dark palettes.
Outputs: assertions for theme colors, action roles, and control hit areas.
"""

import unittest

from napari.settings import get_settings
from napari.settings._fields import Theme as NapariThemeId
from napari.utils.theme import get_theme
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QApplication, QPushButton, QWidget

from twopy.napari.group_matching.style import style_group_matching_panel
from twopy.napari.theme import (
    TWOPY_CONTROL_HEIGHT,
    active_twopy_theme_colors,
    apply_twopy_theme,
    style_action_button,
    twopy_theme_colors,
    twopy_theme_style_sheet,
)


class NapariThemeTest(unittest.TestCase):
    """Test the visual rules shared by all twopy Qt views."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create the Qt application used by theme widgets."""
        cls._application = QApplication.instance() or QApplication([])

    def test_action_roles_have_consistent_hit_areas(self) -> None:
        """Confirm action roles keep a 40-pixel minimum hit area.

        Inputs: one button for each supported visual role.
        Outputs: each button stores its role and meets the shared height.
        """
        for role in ("primary", "secondary", "quiet", "danger"):
            button = QPushButton(role)

            style_action_button(button, role=role)

            self.assertEqual(button.property("twopyRole"), role)
            self.assertGreaterEqual(button.minimumHeight(), TWOPY_CONTROL_HEIGHT)

    def test_unknown_action_role_fails(self) -> None:
        """Confirm an unknown button role fails before it reaches the UI.

        Inputs: one unsupported action role.
        Outputs: a clear value error.
        """
        with self.assertRaisesRegex(ValueError, "Unknown twopy button role"):
            style_action_button(QPushButton("Unknown"), role="unknown")

    def test_theme_uses_palette_without_bright_dark_surfaces(self) -> None:
        """Confirm dark surfaces do not use a bright alternate color.

        Inputs: a dark napari-like palette with a bright alternate base.
        Outputs: dark derived surfaces that ignore the alternate base.
        """
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#323232"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#181818"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#989898"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#2d7dd2"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))

        colors = twopy_theme_colors(palette)

        self.assertTrue(colors.is_dark)
        self.assertNotEqual(colors.surface, "#989898")
        self.assertLess(QColor(colors.surface).lightness(), 80)

    def test_theme_stays_scoped_to_its_root(self) -> None:
        """Confirm the theme styles one root without changing its neighbor.

        Inputs: two separate widgets under one Qt application.
        Outputs: only the selected root receives twopy style text.
        """
        root = QWidget()
        neighbor = QWidget()

        apply_twopy_theme(root, name="twopy_test_root")

        self.assertEqual(root.objectName(), "twopy_test_root")
        self.assertIn('font-family: "Avenir Next"', root.styleSheet())
        self.assertIn("QTabBar::tab:selected", root.styleSheet())
        self.assertEqual(neighbor.styleSheet(), "")

    def test_selected_tabs_own_the_highlight(self) -> None:
        """Confirm selected tabs use emphasis and other tabs match the window.

        Inputs: the shared stylesheet for the current application palette.
        Outputs: distinct selected and unselected tab backgrounds.
        """
        palette = QWidget().palette()
        colors = twopy_theme_colors(palette)
        stylesheet = twopy_theme_style_sheet(palette)

        unselected_rule = stylesheet.split("QTabBar::tab:!selected", maxsplit=1)[1]
        selected_rule = stylesheet.split("QTabBar::tab:selected", maxsplit=1)[1]
        self.assertIn(
            f"background: {colors.window}",
            unselected_rule.split("}", maxsplit=1)[0],
        )
        self.assertIn(
            f"background: {colors.selected_surface}",
            selected_rule.split("}", maxsplit=1)[0],
        )
        self.assertNotEqual(colors.window, colors.selected_surface)

    def test_sections_use_the_window_background(self) -> None:
        """Confirm section outlines provide structure without a raised fill.

        Inputs: shared and Group Matching styles for the current palette.
        Outputs: every group section uses the palette window background.
        """
        palette = QWidget().palette()
        colors = twopy_theme_colors(palette)
        stylesheet = twopy_theme_style_sheet(palette)

        group_rule = stylesheet.split("QGroupBox {", maxsplit=1)[1]
        group_title_rule = stylesheet.split("QGroupBox::title", maxsplit=1)[1]
        self.assertIn(
            f"background: {colors.window}",
            group_rule.split("}", maxsplit=1)[0],
        )
        self.assertIn(
            f"background: {colors.window}",
            group_title_rule.split("}", maxsplit=1)[0],
        )

        group_matching_root = QWidget()
        style_group_matching_panel(group_matching_root)
        group_matching_style = group_matching_root.styleSheet()
        group_matching_colors = active_twopy_theme_colors(group_matching_root.palette())
        matching_group_rule = group_matching_style.split(
            "QGroupBox#group_matching_section {",
            maxsplit=1,
        )[1]
        matching_title_rule = group_matching_style.split(
            "QGroupBox#group_matching_section::title {",
            maxsplit=1,
        )[1]
        self.assertIn(
            f"background: {group_matching_colors.window}",
            matching_group_rule.split("}", maxsplit=1)[0],
        )
        self.assertIn(
            f"background: {group_matching_colors.window}",
            matching_title_rule.split("}", maxsplit=1)[0],
        )

    def test_disabled_actions_use_regular_text(self) -> None:
        """Confirm disabled primary actions do not keep primary emphasis.

        Inputs: the shared stylesheet for the current application palette.
        Outputs: disabled buttons use regular font weight.
        """
        stylesheet = twopy_theme_style_sheet(QWidget().palette())

        disabled_rule = stylesheet.split("QPushButton:disabled", maxsplit=1)[1]
        self.assertIn("font-weight: 400", disabled_rule.split("}", maxsplit=1)[0])

    def test_every_action_role_keeps_a_visible_outline(self) -> None:
        """Confirm each action role and state has a palette-aware outline.

        Inputs: the shared stylesheet for the current application palette.
        Outputs: base, primary, quiet, and disabled buttons keep an outline.
        """
        palette = QWidget().palette()
        colors = twopy_theme_colors(palette)
        stylesheet = twopy_theme_style_sheet(palette)

        self.assertIn(f"border: 1px solid {colors.control_outline}", stylesheet)
        self.assertIn(f"border-color: {colors.primary_outline}", stylesheet)
        self.assertGreaterEqual(
            stylesheet.count(f"border-color: {colors.control_outline}"),
            2,
        )
        self.assertNotIn("border-color: transparent", stylesheet)

    def test_tables_keep_their_outer_outline(self) -> None:
        """Confirm scroll-area styling does not remove table outlines.

        Inputs: the shared stylesheet for the current application palette.
        Outputs: item views keep a visible outline after scroll-area rules.
        """
        palette = QWidget().palette()
        colors = twopy_theme_colors(palette)
        stylesheet = twopy_theme_style_sheet(palette)

        item_view_rule = stylesheet.split("QAbstractItemView", maxsplit=1)[1]
        self.assertIn(
            f"border: 1px solid {colors.control_outline}",
            item_view_rule.split("}", maxsplit=1)[0],
        )
        self.assertNotIn("QScrollArea, QAbstractScrollArea", stylesheet)
        self.assertIn("QTableView, QTreeView", stylesheet)
        header_rule = stylesheet.split("QHeaderView {", maxsplit=1)[1]
        self.assertIn(
            f"border-bottom: 1px solid {colors.control_outline}",
            header_rule.split("}", maxsplit=1)[0],
        )
        header_section_rule = stylesheet.split("QHeaderView::section", maxsplit=1)[1]
        self.assertNotIn(
            "border-bottom",
            header_section_rule.split("}", maxsplit=1)[0],
        )

    def test_theme_refreshes_after_napari_theme_change(self) -> None:
        """Confirm a live napari theme change refreshes local theme colors.

        Inputs: one themed root while napari changes through all theme modes.
        Outputs: stylesheet colors from each selected napari theme.
        """
        settings = get_settings()
        original_theme = settings.appearance.theme
        try:
            settings.appearance.theme = NapariThemeId("dark")
            self._application.processEvents()
            root = QWidget()
            apply_twopy_theme(root, name="twopy_theme_test")
            dark_colors = active_twopy_theme_colors(root.palette())
            dark_style = root.styleSheet()

            settings.appearance.theme = NapariThemeId("light")
            self._application.processEvents()
            light_colors = active_twopy_theme_colors(root.palette())
            light_style = root.styleSheet()

            settings.appearance.theme = NapariThemeId("system")
            self._application.processEvents()
            system_colors = active_twopy_theme_colors(root.palette())
            system_style = root.styleSheet()

            self.assertIn(dark_colors.window, dark_style)
            self.assertIn(light_colors.window, light_style)
            self.assertIn(light_colors.text, light_style)
            self.assertIn(system_colors.window, system_style)
            self.assertIn(system_colors.text, system_style)
            self.assertNotEqual(dark_style, light_style)
        finally:
            settings.appearance.theme = original_theme
            self._application.processEvents()

    def test_light_theme_uses_a_contrasting_orange_accent(self) -> None:
        """Confirm the built-in light theme replaces its yellow accent.

        Inputs: napari light and dark theme selections.
        Outputs: a white-on-orange light accent and unchanged dark accent.
        """
        settings = get_settings()
        original_theme = settings.appearance.theme
        try:
            settings.appearance.theme = NapariThemeId("light")
            light = active_twopy_theme_colors(QWidget().palette())
            settings.appearance.theme = NapariThemeId("dark")
            dark = active_twopy_theme_colors(QWidget().palette())

            self.assertEqual(light.accent, "#c45116")
            self.assertEqual(light.accent_text, "#ffffff")
            self.assertNotEqual(light.accent, get_theme("light").current.as_hex())
            self.assertEqual(dark.accent, get_theme("dark").current.as_hex())
        finally:
            settings.appearance.theme = original_theme
            self._application.processEvents()

    def test_special_view_style_refreshes_with_napari_theme(self) -> None:
        """Confirm view-specific styles use the same live theme source.

        Inputs: one Group Matching root while napari changes theme.
        Outputs: special view colors from the new napari theme.
        """
        settings = get_settings()
        original_theme = settings.appearance.theme
        try:
            settings.appearance.theme = NapariThemeId("dark")
            root = QWidget()
            style_group_matching_panel(root)
            dark_style = root.styleSheet()

            settings.appearance.theme = NapariThemeId("light")
            self._application.processEvents()
            light_colors = active_twopy_theme_colors(root.palette())

            self.assertNotEqual(dark_style, root.styleSheet())
            self.assertIn(
                f"background: {light_colors.window}",
                root.styleSheet(),
            )
            self.assertIn(f"color: {light_colors.text}", root.styleSheet())
        finally:
            settings.appearance.theme = original_theme
            self._application.processEvents()


if __name__ == "__main__":
    unittest.main()
