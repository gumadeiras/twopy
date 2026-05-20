"""Command-line entry point for the function inventory report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from function_inventory_support.builder import _sort_metric, build_inventory
from function_inventory_support.model import write_csv, write_markdown


def main(argv: list[str] | None = None) -> int:
    """Run the inventory CLI over repo source and tests.

    Inputs: optional argv, source paths, test path, output format, and limit.
    Outputs: stdout report plus process status for developer audit scripts.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=Path(),
        type=Path,
        help="Repository root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--src",
        action="append",
        default=None,
        help="Source directory to inventory. May be repeated. Defaults to src/twopy.",
    )
    parser.add_argument(
        "--tests",
        default="tests",
        help="Test directory used for direct test attribution. Defaults to tests.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "markdown"),
        default="csv",
        help="Output format. Defaults to csv.",
    )
    parser.add_argument(
        "--limit",
        default=0,
        type=int,
        help="Maximum number of rows to print after sorting. Zero means all rows.",
    )
    parsed = parser.parse_args(argv)

    root = Path(parsed.root).resolve()
    source_dirs = tuple(
        root / source
        for source in (tuple(parsed.src) if parsed.src is not None else ("src/twopy",))
    )
    test_dir = root / str(parsed.tests)

    metrics = build_inventory(
        root=root,
        source_dirs=source_dirs,
        test_dir=test_dir,
    )
    rows = [metric.row() for metric in sorted(metrics.values(), key=_sort_metric)]
    limit = int(parsed.limit)
    if limit > 0:
        rows = rows[:limit]

    if str(parsed.format) == "markdown":
        write_markdown(rows, sys.stdout)
    else:
        write_csv(rows, sys.stdout)
    return 0
