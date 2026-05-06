"""Package-version metadata for twopy.

Inputs: installed Python package metadata.
Outputs: the current twopy package version string.

Keeping this in one small module prevents the public Python API and command-line
interface from drifting when ``pyproject.toml`` changes.
"""

from importlib.metadata import version

__all__ = ["__version__"]

__version__ = version("twopy")
