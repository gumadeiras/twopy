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
    "sphinx_rtd_theme",
]
source_suffix = {
    ".md": "markdown",
}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_extra_path = ["CNAME", ".nojekyll"]
html_show_sourcelink = False
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}
