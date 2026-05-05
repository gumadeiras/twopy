"""Top-level package for the twopy two-photon imaging analysis tool.

The package exposes project metadata and script-friendly public APIs.
Inputs: none.
Outputs: package constants and helper functions imported by users and scripts.
"""

from twopy.api import find_recordings

__all__ = ["__version__", "find_recordings"]

__version__ = "0.1.0"
