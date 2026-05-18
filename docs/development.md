# Development Guide

Install from the repository so local code edits are used:

```sh
micromamba env create -f environment.yml
micromamba activate twopy
micromamba run -n twopy pre-commit install
```

The development environment installs twopy as an editable package, so the `twopy` terminal command is available after activating the environment. If the environment already existed before the command was added, refresh the editable install:

```sh
micromamba run -n twopy python -m pip install -e .
```

Run the full gate before handoff:

```sh
micromamba run -n twopy pre-commit run --all-files
```

The installed pre-commit hook runs ruff, ty, and unit tests before each commit. The unittest hook uses the timing helper's discover mode, so the suite runs once and records a rolling local timing history in `.git/twopy-test-timings.json`.

Use targeted unittest groups while iterating:

```sh
micromamba run -n twopy python -m unittest tests.test_conversion tests.test_converted
micromamba run -n twopy python -m unittest tests.test_napari_load_tab tests.test_napari_path_resolution
micromamba run -n twopy python -m unittest tests.test_response_maps tests.test_response_roi_extraction
```

Use the timing helper when changing test layout or imports:

```sh
micromamba run -n twopy python scripts/test_timings.py --discover --record
micromamba run -n twopy python scripts/test_timings.py analysis conversion metadata napari parity
micromamba run -n twopy python scripts/test_timings.py napari
```

`--discover --record` is the pre-commit path: it runs normal unittest discovery once, keeps the last 50 local runs, and prints the current run against the prior successful median. The group/module mode runs each module in a fresh process so slow cold imports are visible. Keep pure analysis, display-coordinate, and plot-data tests out of Qt/napari modules when they can test the same behavior through non-Qt helpers.
