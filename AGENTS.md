# AGENTS.MD

twopy is a two-photon imaging analysis tool.

## Working Principles

- Prefer simple, auditable code.
- Use Occam's razor: add complexity only when a real observed need requires it.
- Keep data flow explicit. Avoid hidden global state, surprising side effects, and clever indirection.
- Favor small functions with clear names over broad abstractions.
- When tradeoffs exist, choose the path that a scientist can inspect and trust.

## GUI

- Use napari for the user interface.
- Keep core modules GUI-independent. Session discovery, MATLAB loading, data conversion, and analysis code must not import napari.
- Keep napari code thin: load recordings, display movies, collect or select ROIs, call documented analysis functions, and display responses.
- Do not hide analysis decisions inside napari callbacks. Put scientific logic in typed, tested modules.
- Use napari image/shape/label layers for movie and ROI interaction where possible.

## Data Layout

- A two-photon session is a timestamped microscope output directory.
- Required session contents:
  - `stimulusData/`
  - `alignedMovie.mat`
  - one raw `*.tif` movie
  - one `*_alignment.txt` file
  - `defaultAlignChannel.txt`
  - `highResPd.mat`
  - `imageDescription.mat`
  - `imagingResPd.mat`
- Stimulus names and raw movie stems can change. Detect those files by stable patterns, not hard-coded full names.
- `savedAnalysis/` is optional. It exists only when the session was already analyzed by the lab MATLAB package.

## Converted Data

- Read microscope source data through a MATLAB file layer.
- Convert MATLAB-derived data into Python objects for analysis.
- Persist converted data as HDF5 files with gzip compression.
- Keep HDF5 groups and datasets named in plain language so files remain inspectable outside twopy.

## Environment

- Keep the environment reproducible.
- When adding, removing, or changing dependencies, update `environment.yml` or a requirements file in the same change.
- Prefer `environment.yml` for the micromamba development environment.
- Keep GUI dependencies in a separate package extra when possible so core data code stays usable without launching a GUI.

## Configuration

- Keep machine-local variables in `config.yml`.
- Load config through typed Python helpers instead of reading YAML ad hoc.
- Initial required config keys: `database_path` and `data_path`.

## Database

- Query the lab database through typed helper modules, not ad hoc SQL in GUI code.
- Open lab SQLite files read-only.
- Keep database querying schema-aware at runtime: discover tables and columns before filtering.
- Support direct DB reads and local-copy DB reads through explicit API arguments.
- When copying DB files locally, avoid unnecessary copies by checking source metadata and content hash.
- Do not write to `_Database` files from twopy unless Gustavo explicitly asks for write support.

## Documentation

- Keep `roadmap.md` updated as work lands.
- Roadmap updates should be high signal and low noise: record completed progress, explicit next work, and decisions; avoid speculative feature lists.
- Every file must start with a plain-language docstring explaining its purpose.
- Every public function, class, and method must document:
  - inputs
  - outputs
  - what it does
  - why it exists in the analysis workflow
- Use human language. Explain the goal and intuition, not just mechanics.
- Key analysis code must include inline comments that explain the reasoning behind the logic.
- Comments should clarify scientific or data-processing intent. Do not restate obvious syntax.

## Typing

- All Python functions and methods must have type annotations for inputs and return values.
- Use Python 3.13 for development and verification.
- Avoid `Any` unless there is no honest narrower type.
- Use standard library types where possible.
- Type checking is part of the gate:

```sh
micromamba run -n twopy python -m mypy
```

## Linting And Formatting

- Use ruff for linting, import ordering, docstring checks, and formatting.
- Keep ruff fixes small and reviewable.
- Run before handoff:

```sh
micromamba run -n twopy python -m ruff check .
micromamba run -n twopy python -m ruff format --check .
```

## Tests

- Add regression tests for bugs when feasible.
- Tests should be readable evidence for expected behavior.
- Prefer small fixtures and explicit expected values over opaque golden files.

## Verification

Run the touched-surface gate before handoff:

```sh
micromamba run -n twopy python -m ruff check .
micromamba run -n twopy python -m ruff format --check .
micromamba run -n twopy python -m mypy
micromamba run -n twopy python -m unittest discover -s tests
```
