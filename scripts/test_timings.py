"""Time unittest modules and groups for local test-suite tuning.

Inputs: optional group names or unittest module names.
Outputs: elapsed wall-clock time per module plus a slowest-first summary.

The script runs each module in a fresh Python process. That makes cold import
cost visible, which is useful for deciding whether a test belongs in a pure
analysis group or a Qt/napari group.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

TEST_GROUPS = {
    "analysis": (
        "tests.test_background_subtraction",
        "tests.test_dff",
        "tests.test_motion",
        "tests.test_response_maps",
        "tests.test_response_processing",
        "tests.test_response_roi_extraction",
        "tests.test_responses",
        "tests.test_roi_extraction",
        "tests.test_roi_mask_cleanup",
        "tests.test_trials",
    ),
    "conversion": (
        "tests.test_conversion",
        "tests.test_converted",
        "tests.test_hdf5_utils",
        "tests.test_inspection",
        "tests.test_matlab",
        "tests.test_real_fixture",
        "tests.test_saved_analysis",
        "tests.test_session",
        "tests.test_stimulus",
        "tests.test_synchronization",
    ),
    "metadata": (
        "tests.test_api",
        "tests.test_config",
        "tests.test_database",
        "tests.test_epoch_mapping",
        "tests.test_filenames",
        "tests.test_frame_ranges",
        "tests.test_group_matching",
        "tests.test_import",
        "tests.test_packaging",
        "tests.test_photodiode_classification",
        "tests.test_pixel_calibration",
        "tests.test_pixel_calibration_profiles",
        "tests.test_response_map_area",
        "tests.test_response_window_options",
    ),
    "napari": (
        "tests.test_load_workflow",
        "tests.test_napari",
        "tests.test_napari_database_search",
        "tests.test_napari_export",
        "tests.test_napari_live_controller",
        "tests.test_napari_load_tab",
        "tests.test_napari_loaded_recordings",
        "tests.test_napari_path_resolution",
        "tests.test_napari_plot_controls",
        "tests.test_napari_plot_data",
        "tests.test_napari_plot_widgets",
        "tests.test_napari_processing_controls",
        "tests.test_napari_response_workflow",
        "tests.test_napari_roi_generation",
        "tests.test_napari_roi_visibility",
        "tests.test_napari_timeline",
    ),
    "parity": (
        "tests.test_parity",
        "tests.test_parity_psycho5",
        "tests.test_parity_roi_extraction",
    ),
}


def main() -> int:
    """Run selected unittest modules and print elapsed times.

    Args:
        None. Command-line arguments are parsed from ``sys.argv``.

    Returns:
        Process exit code. Non-zero means at least one module failed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="*",
        help="Group names or unittest modules. Defaults to every test_*.py module.",
    )
    parsed = parser.parse_args()
    modules = _resolve_targets(tuple(parsed.targets))
    results: list[tuple[float, str, int]] = []

    for module in modules:
        start = time.perf_counter()
        completed = subprocess.run(
            (sys.executable, "-m", "unittest", module),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        elapsed = time.perf_counter() - start
        results.append((elapsed, module, completed.returncode))
        status = "ok" if completed.returncode == 0 else "fail"
        print(f"{elapsed:6.2f}s {status:4} {module}")
        if completed.returncode != 0:
            print(completed.stdout)

    print("\nSlowest modules:")
    for elapsed, module, status in sorted(results, reverse=True)[:10]:
        label = "ok" if status == 0 else "fail"
        print(f"{elapsed:6.2f}s {label:4} {module}")

    return 1 if any(status != 0 for _, _, status in results) else 0


def _resolve_targets(targets: tuple[str, ...]) -> tuple[str, ...]:
    """Expand group names and module names into unittest module paths.

    Args:
        targets: Group names from ``TEST_GROUPS`` or explicit module names.

    Returns:
        De-duplicated module names in requested order.
    """
    if len(targets) == 0:
        targets = tuple(
            f"tests.{path.stem}" for path in sorted(Path("tests").glob("test_*.py"))
        )

    modules: list[str] = []
    seen: set[str] = set()
    for target in targets:
        expanded = TEST_GROUPS.get(target, (target,))
        for module in expanded:
            if module not in seen:
                modules.append(module)
                seen.add(module)
    return tuple(modules)


if __name__ == "__main__":
    raise SystemExit(main())
