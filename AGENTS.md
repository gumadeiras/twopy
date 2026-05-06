# AGENTS.MD

twopy is a two-photon imaging analysis tool.

## Working Principles

- Prefer simple, auditable code.
- Use Occam's razor: add complexity only when a real observed need requires it.
- Start from first principles: identify the real data contract, invariant, or
  failure mode before changing code.
- Fix root causes. Do not add bandaid patches that only hide symptoms.
- Keep changes scoped to the problem being solved, and remove dead code when a
  change makes it obsolete.
- Reuse existing shared helpers when they make the code clearer, and deduplicate
  repeated logic when the shared version stays easy to read.
- Keep data flow explicit. Avoid hidden global state, surprising side effects, and clever indirection.
- Favor small functions with clear names over broad abstractions.
- Do not extract helpers just to avoid repeating one or two obvious lines.
  Simple local code is better than docstring-heavy indirection.
- A simplification must reduce cognitive load at the call site and the helper
  definition together. If it only moves simple code elsewhere, leave it inline.
- When tradeoffs exist, choose the path that a scientist can inspect and trust.
- When splitting one module into related helper files, create a package
  directory instead of sibling files with the same basename.
- Do not add backwards-compatibility shims. Prefer one clear current API and
  update callers/tests/docs in the same change.

## Performance

- Write computationally efficient code. Treat CPU time, memory use, and disk I/O
  as real constraints because imaging data can be large.
- Avoid unnecessary copies of large arrays. Prefer views, streaming, chunked I/O,
  and explicit data shapes when they keep behavior simple and auditable.
- Use vectorized NumPy/SciPy operations where they make the logic clearer and
  faster than Python loops.
- Use GPU acceleration when the workload is a good fit, the dependency cost is
  justified, and there is a CPU fallback or a clear hardware requirement.
- Benchmark or profile before adding complex optimization code.

## GUI

- Use napari for the user interface.
- Keep core modules GUI-independent. Session discovery, MATLAB loading, data conversion, and analysis code must not import napari.
- Keep napari code thin: load recordings, display movies, collect or select ROIs, call documented analysis functions, and display responses.
- Continue building napari support in this repo for now, but structure it so a
  future napari plugin can wrap it. Put reusable GUI workflow code in typed
  `twopy.napari` helpers; a future plugin should be menus/widgets only.
- Use magicgui for small napari control panels until a custom QWidget or plugin
  is justified by real workflow complexity.
- Do not hide analysis decisions inside napari callbacks. Put scientific logic in typed, tested modules.
- Use napari image layers for recordings and Labels layers for ROI drawing or
  editing. ROI analysis operates on pixel masks, so Labels are the primary ROI
  UI contract.

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

## Timing And Synchronization

- Imaging and stimulus presentation run on separate computers with separate
  clocks and different frame rates.
- The imaging computer records the movie at relatively low frame rate and also
  records photodiode signals.
- The stimulus computer presents at relatively high frame rate and flashes the
  photodiode at key events: start, trial transitions, and end.
- Different photodiode flash patterns or durations identify event types.
- Response analysis must align stimulus events to imaging frames through the
  photodiode signal. Do not assign trials from nominal frame rates alone.
- Prefer `highResPd.mat` for precise photodiode event detection and
  `imagingResPd.mat` for frame-resolution mapping.
- Keep photodiode thresholding and stimulus-bounds rules in one shared timing
  helper. The default event threshold is the lab convention
  `min + 0.3 * (max - min)`, with explicit thresholds only for audits.
- Ambiguous stimulus end flashes must fail loudly. Do not silently choose one
  when more than two long photodiode flashes are present.

## Converted Data

- Read microscope source data through a MATLAB file layer.
- Never perform analysis directly on source MATLAB or raw TIFF files.
- Always convert source recording data into twopy-owned files first; all
  analysis and processing must operate on those converted files.
- Persist converted data as HDF5 files with gzip compression.
- Store the aligned movie in a separate converted HDF5 file because it usually
  dominates file size.
- During initial conversion, generate a mean image of the aligned movie. Default
  to the full movie and allow callers to choose a frame range.
- Store synchronization metadata in converted HDF5 files so response-analysis
  code can see the timing contract.
- Keep HDF5 groups and datasets named in plain language so files remain inspectable outside twopy.

## Code Ownership Boundaries

- `twopy.conversion` owns reading source microscope outputs such as MAT files,
  raw TIFF files, stimulus backups, and text metadata.
- `twopy.analysis` owns scientific calculations over converted twopy objects
  and HDF5 files. Analysis code must not read source MAT/TIFF files directly.
- `twopy.parity` owns read-only comparison helpers for prior `savedAnalysis/`
  outputs. Normal conversion, analysis, and napari code should not import
  parity helpers.
- If a feature needs source data, add it to conversion and persist the needed
  plain twopy object first; then analyze the converted object.

## Environment

- Keep the environment reproducible.
- When adding, removing, or changing dependencies, update `environment.yml` or a requirements file in the same change.
- Prefer `environment.yml` for the micromamba development environment.
- Keep one install path. Do not split dependencies into dev or GUI extras.
- Install pre-commit hooks with
  `micromamba run -n twopy pre-commit install` after creating the environment.

## Configuration

- Keep machine-local variables in `config.yml`.
- Never track `config.yml`; track `config.example.yml` as the documented
  template for creating it.
- Load config through typed Python helpers instead of reading YAML ad hoc.
- Initial config keys: `database_path`, `data_path`, `database_access`, and
  `analysis_output`.
- Default `database_access` to `copy` unless a task explicitly needs direct
  mounted DB reads. This mode exists because database queries over the network
  can be slow, while transferring the DB file locally is usually fast.

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
micromamba run -n twopy python -m ty check
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
micromamba run -n twopy pre-commit run --all-files
```

The pre-commit hook runs the same gate before each commit.
