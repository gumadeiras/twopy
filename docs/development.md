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

The installed pre-commit hook runs ruff, ty, and unit tests before each commit.

Use targeted unittest groups while iterating:

```sh
micromamba run -n twopy python -m unittest tests.test_conversion tests.test_converted
micromamba run -n twopy python -m unittest tests.test_napari_load_tab tests.test_napari_path_resolution
micromamba run -n twopy python -m unittest tests.test_response_maps tests.test_response_roi_extraction
```

Use the timing helper when changing test layout or imports:

```sh
micromamba run -n twopy python scripts/test_timings.py analysis conversion metadata napari parity
micromamba run -n twopy python scripts/test_timings.py napari
```

The timing helper runs each module in a fresh process so slow cold imports are visible. Keep pure analysis tests out of Qt/napari modules when they can test the same behavior through analysis helpers.
