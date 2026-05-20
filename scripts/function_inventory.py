"""Inventory Python functions, documentation, static uses, and direct tests.

Inputs: repository Python source files and test files.
Outputs: CSV or Markdown rows with per-function size, documentation, fan-in,
test-use, API-surface, git-churn, runtime, risk, and complexity metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from function_inventory_support import (
    build_inventory,
    render_csv_for_tests,
)
from function_inventory_support.cli import main

__all__ = ["build_inventory", "main", "render_csv_for_tests"]


if __name__ == "__main__":
    raise SystemExit(main())
