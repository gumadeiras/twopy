"""Time unittest modules, groups, or the full discover run.

Inputs: optional group names, unittest module names, or the discover flag.
Outputs: elapsed wall-clock time and optional local timing-history trends.

The script runs each module in a fresh Python process. That makes cold import
cost visible, which is useful for deciding whether a test belongs in a pure
analysis group or a Qt/napari group.

The discover mode runs the whole unittest suite once. That mode is intended for
pre-commit, where running the suite twice would make timing data too expensive.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TimingRecord(TypedDict):
    """One persisted full-suite timing sample.

    Args:
        timestamp_utc: ISO-8601 UTC time when the run finished.
        elapsed_seconds: Full unittest discover wall-clock duration.
        status: ``ok`` when tests passed, otherwise ``fail``.
        command: Command that was timed.
        python: Python version used for the run.
        git_commit: Short commit hash when available.
        git_dirty: Whether the worktree had local changes during the run.

    Returns:
        A JSON-serializable timing-history record.

    This exists so pre-commit can catch suite-runtime drift without writing
    repository-tracked files or re-running tests.
    """

    timestamp_utc: str
    elapsed_seconds: float
    status: str
    command: list[str]
    python: str
    git_commit: str
    git_dirty: bool


TEST_GROUPS = {
    "analysis": (
        "tests.test_background_subtraction",
        "tests.test_dff",
        "tests.test_motion",
        "tests.test_recording_timing",
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
        "tests.test_napari_display",
        "tests.test_napari_export",
        "tests.test_napari_live_controller",
        "tests.test_napari_load_tab",
        "tests.test_napari_loaded_recordings",
        "tests.test_napari_path_resolution",
        "tests.test_napari_plot_controls",
        "tests.test_napari_plot_data",
        "tests.test_napari_plot_widgets",
        "tests.test_napari_processing_controls",
        "tests.test_napari_response_map_display",
        "tests.test_napari_response_workflow",
        "tests.test_napari_roi_generation",
        "tests.test_napari_roi_visibility",
        "tests.test_napari_text",
        "tests.test_napari_timeline",
    ),
    "parity": (
        "tests.test_parity",
        "tests.test_parity_psycho5",
        "tests.test_parity_roi_extraction",
    ),
}


def main() -> int:
    """Run selected unittest timing mode and print elapsed times.

    Args:
        None. Command-line arguments are parsed from ``sys.argv``.

    Returns:
        Process exit code. Non-zero means at least one test command failed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Run unittest discovery once instead of timing modules separately.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Persist discover timing history under .git for trend reporting.",
    )
    parser.add_argument(
        "--history-limit",
        default=50,
        type=int,
        help="Number of full-suite timing records to keep when --record is set.",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="Group names or unittest modules. Defaults to every test_*.py module.",
    )
    parsed = parser.parse_args()

    if bool(parsed.discover):
        targets = tuple(str(target) for target in parsed.targets)
        if targets:
            parser.error("--discover does not accept module or group targets")
        return _run_discover(
            record=bool(parsed.record),
            history_limit=int(parsed.history_limit),
        )

    if bool(parsed.record):
        parser.error("--record is only supported with --discover")

    modules = _resolve_targets(tuple(str(target) for target in parsed.targets))
    results: list[tuple[float, str, int]] = []

    for module in modules:
        start = time.perf_counter()
        completed = subprocess.run(
            (sys.executable, "-m", "unittest", module),
            check=False,
            env=_test_environment(),
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


def _run_discover(*, record: bool, history_limit: int) -> int:
    """Run the full unittest suite once and optionally save timing history.

    Args:
        record: Whether to persist a local timing sample under ``.git``.
        history_limit: Maximum number of persisted samples to retain.

    Returns:
        Process exit code from unittest discovery.

    Pre-commit needs timing visibility but should not duplicate test execution.
    This path times the same full-suite command that the hook already ran.
    """
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
    start = time.perf_counter()
    completed = subprocess.run(command, check=False, env=_test_environment())
    elapsed = time.perf_counter() - start
    status = "ok" if completed.returncode == 0 else "fail"

    print(f"\nFull unittest discover: {elapsed:6.2f}s {status}")
    if record:
        record_entry: TimingRecord = {
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "elapsed_seconds": elapsed,
            "status": status,
            "command": command,
            "python": sys.version.split()[0],
            "git_commit": _git_text("rev-parse", "--short", "HEAD") or "unknown",
            "git_dirty": bool(_git_text("status", "--short")),
        }
        history_path = _history_path()
        records = _load_history(history_path)
        records.append(record_entry)
        records = records[-max(history_limit, 1) :]
        _write_history(history_path, records)
        _print_trend(records, record_entry)

    return completed.returncode


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


def _history_path() -> Path:
    """Return the untracked timing-history path inside the git directory.

    Args:
        None.

    Returns:
        Path to the local JSON file that stores recent full-suite timings.

    The git directory keeps timing trends machine-local, so pre-commit does not
    create repository churn and each developer sees their own hardware baseline.
    """
    git_dir = _git_text("rev-parse", "--git-dir")
    if not git_dir:
        return Path(".twopy-test-timings.json")
    return Path(git_dir).resolve() / "twopy-test-timings.json"


def _test_environment() -> dict[str, str]:
    """Return process environment defaults for quiet GUI tests.

    Args:
        None.

    Returns:
        Environment variables for unittest subprocesses.

    Napari and Qt tests should measure widgets without opening desktop windows
    or taking focus from the developer's active app. Test subprocesses must
    import the working tree before any installed twopy package.
    """
    from tests.gui_environment import headless_gui_environment

    env = headless_gui_environment(os.environ)
    source_paths = (str(REPO_ROOT / "src"), str(REPO_ROOT))
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        source_paths += (existing_pythonpath,)
    env["PYTHONPATH"] = os.pathsep.join(source_paths)
    return env


def _git_text(*args: str) -> str:
    """Run a small git query and return stripped stdout.

    Args:
        *args: Arguments passed after the ``git`` executable.

    Returns:
        Standard output without surrounding whitespace, or an empty string when
        git is unavailable.

    Timing history is useful metadata, not part of test correctness, so git
    lookup failures should never fail the verification gate.
    """
    completed = subprocess.run(
        ("git", *args),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _load_history(path: Path) -> list[TimingRecord]:
    """Load valid persisted timing records.

    Args:
        path: JSON timing-history file.

    Returns:
        Valid timing records, with malformed local data ignored.

    The history file is an optimization signal. Ignoring malformed local history
    keeps one bad write from blocking commits.
    """
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(loaded, dict):
        return []
    raw_records = loaded.get("runs")
    if not isinstance(raw_records, list):
        return []

    records: list[TimingRecord] = []
    for raw_record in raw_records:
        record = _validate_record(raw_record)
        if record is not None:
            records.append(record)
    return records


def _validate_record(raw_record: object) -> TimingRecord | None:
    """Validate one decoded timing-history record.

    Args:
        raw_record: JSON-decoded object from the local history file.

    Returns:
        A typed timing record, or ``None`` when required fields are missing.

    Validation keeps JSON loading honest without spreading casts through the
    trend-reporting code.
    """
    if not isinstance(raw_record, dict):
        return None
    timestamp = raw_record.get("timestamp_utc")
    elapsed = raw_record.get("elapsed_seconds")
    status = raw_record.get("status")
    command = raw_record.get("command")
    python = raw_record.get("python")
    git_commit = raw_record.get("git_commit")
    git_dirty = raw_record.get("git_dirty")
    if not isinstance(timestamp, str):
        return None
    if not isinstance(elapsed, int | float):
        return None
    if not isinstance(status, str):
        return None
    if not isinstance(command, list) or not all(
        isinstance(part, str) for part in command
    ):
        return None
    if not isinstance(python, str):
        return None
    if not isinstance(git_commit, str):
        return None
    if not isinstance(git_dirty, bool):
        return None
    return {
        "timestamp_utc": timestamp,
        "elapsed_seconds": float(elapsed),
        "status": status,
        "command": command,
        "python": python,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def _write_history(path: Path, records: list[TimingRecord]) -> None:
    """Persist local full-suite timing records.

    Args:
        path: JSON file path under the git directory.
        records: Timing records to write.

    Returns:
        None.

    The file stores a compact rolling window so trend checks stay cheap and
    human-readable when debugging a local slowdown.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"runs": records}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _print_trend(records: list[TimingRecord], current: TimingRecord) -> None:
    """Print a rolling timing trend for the full unittest suite.

    Args:
        records: Persisted timing records including the current run.
        current: Timing record for the just-finished run.

    Returns:
        None.

    Failed runs are recorded but excluded from the baseline median so broken
    test runs do not make future successful runs look artificially slow.
    """
    baseline = [
        record["elapsed_seconds"] for record in records[:-1] if record["status"] == "ok"
    ]
    if not baseline:
        print("Timing trend: first recorded run")
        return
    median = statistics.median(baseline)
    delta = current["elapsed_seconds"] - median
    sign = "+" if delta >= 0 else "-"
    print(
        "Timing trend: "
        f"current {current['elapsed_seconds']:.2f}s, "
        f"median(last {len(baseline)}) {median:.2f}s, "
        f"delta {sign}{abs(delta):.2f}s"
    )


if __name__ == "__main__":
    raise SystemExit(main())
