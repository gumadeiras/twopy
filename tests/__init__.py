"""Shared test package setup.

Inputs: unittest importing modules under the ``tests`` package.
Outputs: process-wide test defaults that must exist before GUI imports.
"""

import tests.gui_environment  # noqa: F401
