# Development guide

This page is for people working on twopy itself. If you just want to use twopy, see [Getting started](getting_started.md).

## Install from source

```sh
micromamba env create -f environment.yml
micromamba activate twopy
micromamba run -n twopy pre-commit install
```

The environment installs twopy as an editable package, so the `twopy` terminal command is available after activating it. If the environment already existed before the package was added, refresh the editable install:

```sh
micromamba run -n twopy python -m pip install -e .
```

## Run the gate before handoff

```sh
micromamba run -n twopy pre-commit run --all-files
```

The installed pre-commit hook runs ruff, ty, and unit tests before each commit. The unittest hook uses the timing helper's discover mode, so the full suite runs once and records a rolling local timing history in `.git/twopy-test-timings.json`.

GUI tests set `QT_QPA_PLATFORM=offscreen`, `QT_LOGGING_RULES=qt.qpa.*=false`, and `MPLBACKEND=Agg` by default before importing Qt, napari, or matplotlib, so widget coverage should not draw windows, steal focus, or print Qt platform noise. Set those environment variables explicitly before a test command when you need a different backend for debugging.

## Targeted test runs while iterating

```sh
micromamba run -n twopy python -m unittest tests.test_conversion tests.test_converted
micromamba run -n twopy python -m unittest tests.test_napari_load_tab tests.test_napari_path_resolution
micromamba run -n twopy python -m unittest tests.test_response_maps tests.test_response_roi_extraction
```

## The test-timing helper

```sh
micromamba run -n twopy python scripts/test_timings.py --discover --record
micromamba run -n twopy python scripts/test_timings.py analysis conversion metadata napari parity
micromamba run -n twopy python scripts/test_timings.py napari
```

`--discover --record` is the pre-commit path: normal unittest discovery once, last 50 local runs kept, current run printed against the prior successful median. The group / module mode runs each module in a fresh process so slow cold imports are visible. Keep pure analysis, display-coordinate, and plot-data tests out of Qt / napari modules when the same behavior can be tested through non-Qt helpers.

## Function inventory

Use the function inventory when planning documentation, testing, or simplification passes:

```sh
micromamba run -n twopy python scripts/function_inventory.py --format csv > build/function_inventory.csv
micromamba run -n twopy python scripts/function_inventory.py --format markdown --limit 25
```

The inventory reports action-oriented columns only: code lines, docstring lines, direct static call sites, direct test functions, lightweight complexity, API-surface classification, domain bucket, git file churn, current-body blame span, and a risk score. Direct test attribution follows normal `twopy` imports plus local `tests.*` helper re-exports when the called function name resolves uniquely, so shared test fixtures do not hide existing coverage.

The risk score is a review-prioritization heuristic: `(code_lines + 5 * complexity) * api_weight * test_gap_weight * churn_weight`. Exported API is weighted above public internal code, functions without direct static tests are doubled, and file commit count raises the score up to a capped churn multiplier. Complexity is a lightweight cyclomatic estimate (one base path plus branches, loops, exception handlers, boolean decision operands, match cases, and ternary expressions).
