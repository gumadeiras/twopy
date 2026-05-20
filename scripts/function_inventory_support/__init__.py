"""Support package for the function inventory developer tool."""

from function_inventory_support.builder import build_inventory
from function_inventory_support.model import FunctionKey, FunctionMetric, write_csv

__all__ = ["build_inventory", "render_csv_for_tests"]


def render_csv_for_tests(metrics: dict[FunctionKey, FunctionMetric]) -> str:
    """Render metrics to CSV text for script-level tests."""
    import io

    output = io.StringIO()
    rows = [
        metric.row()
        for metric in sorted(
            metrics.values(),
            key=lambda metric: (metric.path.as_posix(), metric.key.qualname),
        )
    ]
    write_csv(rows, output)
    return output.getvalue()
