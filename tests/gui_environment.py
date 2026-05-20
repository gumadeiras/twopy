"""Configure GUI libraries so tests do not open desktop windows.

Inputs: process environment before Qt, napari, or matplotlib are imported.
Outputs: headless defaults for GUI-heavy tests while preserving explicit caller
overrides.
"""

from collections.abc import Mapping
from os import environ

HEADLESS_GUI_ENVIRONMENT = {
    "QT_QPA_PLATFORM": "offscreen",
    "QT_LOGGING_RULES": "qt.qpa.*=false",
    "MPLBACKEND": "Agg",
}


def configure_headless_gui_tests() -> None:
    """Set quiet GUI defaults before any GUI toolkit import.

    Args:
        None.

    Returns:
        None.

    Qt creates a real desktop application as soon as ``QApplication`` is
    constructed. The offscreen platform keeps widget tests measurable without
    drawing windows or stealing focus from the user's current app.
    """
    environ.update(headless_gui_environment(environ))


def headless_gui_environment(base: Mapping[str, str]) -> dict[str, str]:
    """Return GUI-quiet environment values without replacing explicit choices.

    Args:
        base: Existing environment values to preserve.

    Returns:
        Copy of ``base`` with missing headless GUI defaults filled in.

    Test runners use this to give subprocesses the same quiet GUI behavior as
    direct ``tests.*`` imports.
    """
    env = dict(base)
    for key, value in HEADLESS_GUI_ENVIRONMENT.items():
        env.setdefault(key, value)
    return env


configure_headless_gui_tests()
