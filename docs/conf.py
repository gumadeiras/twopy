"""Build the twopy documentation site."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
    PYPROJECT = tomllib.load(pyproject_file)

project = "twopy"
author = "Gustavo Madeira Santana"
release = PYPROJECT["project"]["version"]

extensions = [
    "myst_parser",
]
source_suffix = {
    ".md": "markdown",
}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
html_extra_path = ["CNAME", ".nojekyll"]
html_show_sourcelink = False
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#1f6f78",
        "color-brand-content": "#165b63",
        "color-api-name": "#165b63",
    },
    "dark_css_variables": {
        "color-brand-primary": "#67d5df",
        "color-brand-content": "#67d5df",
        "color-api-name": "#67d5df",
    },
    "navigation_with_keys": True,
}
