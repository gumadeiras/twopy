# AGENTS.MD

## Git

- Commit with `scripts/committer "<subject>" -- <path>...`; it stages only listed paths. Use `--body` or `--body-file` for commit bodies.

twopy is a two-photon imaging analysis tool.

## Working Principles

- Prefer simple, auditable code.
- Use Occam's razor: add complexity only when a real observed need requires it.
- Start from first principles: identify the real data contract, invariant, or failure mode before changing code.
- Fix root causes. Prefer making the bad state unrepresentable or unreachable over patching each symptom after it appears.
- Keep changes scoped to the problem being solved, and remove dead code when a change makes it obsolete.
- Reuse existing shared helpers when they make the code clearer, and deduplicate repeated logic when the shared version stays easy to read.
- Keep data flow explicit. Avoid hidden global state, surprising side effects, and clever indirection.
- Favor small functions with clear names over broad abstractions.
- Do not extract helpers just to avoid repeating one or two obvious lines, or to return a simple expression. Simple local code is better than docstring-heavy indirection.
- A simplification must reduce cognitive load at the call site and the helper definition together. If it only moves simple code elsewhere, leave it inline.
- When tradeoffs exist, choose the path that a scientist can inspect and trust.
- When splitting one module into related helper files, create a package directory instead of sibling files with the same basename.
- Do not add backwards-compatibility shims. Prefer one clear current API and update callers/tests/docs in the same change.

## Performance

- Write computationally efficient code. Treat CPU time, memory use, and disk I/O as real constraints because imaging data can be large.
- Avoid unnecessary copies of large arrays. Prefer views, streaming, chunked I/O, and explicit data shapes when they keep behavior simple and auditable.
- Use vectorized NumPy/SciPy operations where they make the logic clearer and faster than Python loops.
- Use GPU acceleration when the workload is a good fit, the dependency cost is justified, and there is a CPU fallback or a clear hardware requirement.
- Benchmark or profile before adding complex optimization code.

## GUI

- Use napari for the user interface.
- Keep core modules GUI-independent. Session discovery, MATLAB loading, data conversion, and analysis code must not import napari.
- Keep napari code thin: load recordings, display movies, collect or select ROIs, call documented analysis functions, and display responses.
- Continue building napari support in this repo for now, but structure it so a future napari plugin can wrap it. Put reusable GUI workflow code in typed `twopy.napari` helpers; a future plugin should be menus/widgets only.
- Design the napari workflow around the planned loop: query the lab database, choose a recording, load converted twopy data, draw or edit Labels ROIs, run analysis, and inspect responses. Keep each step as a small helper so widgets stay thin.
- Use magicgui for small napari control panels until a custom QWidget or plugin is justified by real workflow complexity.
- Do not hide analysis decisions inside napari callbacks. Put scientific logic in typed, tested modules.
- Use napari image layers for recordings and Labels layers for ROI drawing or editing. ROI analysis operates on pixel masks, so Labels are the primary ROI UI contract.

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

- Imaging and stimulus presentation run on separate computers with separate clocks and different frame rates.
- The imaging computer records the movie at relatively low frame rate and also records photodiode signals.
- The stimulus computer presents at relatively high frame rate and flashes the photodiode at key events: start, trial transitions, and end.
- Different photodiode flash patterns or durations identify event types.
- Response analysis must align stimulus events to imaging frames through the photodiode signal. Do not assign trials from nominal frame rates alone.
- Prefer `highResPd.mat` for precise photodiode event detection and `imagingResPd.mat` for frame-resolution mapping.
- Keep photodiode thresholding and stimulus-bounds rules in one shared timing helper. The default event threshold is the lab convention `min + 0.3 * (max - min)`, with explicit thresholds only for audits.
- Ambiguous stimulus end flashes must fail loudly. Do not silently choose one when more than two long photodiode flashes are present.

## Converted Data

- Read microscope source data through a MATLAB file layer.
- Never perform analysis directly on source MATLAB or raw TIFF files.
- Always convert source recording data into twopy-owned files first; all analysis and processing must operate on those converted files.
- Persist converted data as HDF5 files with gzip compression.
- Store the aligned movie in a separate converted HDF5 file because it usually dominates file size.
- During initial conversion, generate a mean image of the aligned movie. Default to the full movie and allow callers to choose a frame range.
- Store synchronization metadata in converted HDF5 files so response-analysis code can see the timing contract.
- Keep HDF5 groups and datasets named in plain language so files remain inspectable outside twopy.

## Code Ownership Boundaries

- `twopy.conversion` owns reading source microscope outputs such as MAT files, raw TIFF files, stimulus backups, and text metadata.
- `twopy.analysis` owns scientific calculations over converted twopy objects and HDF5 files. Analysis code must not read source MAT/TIFF files directly.
- GUI selects analysis parameters; workflow owns execution; processing modules own math; persistence owns the audit trail.
- `twopy.parity` owns read-only comparison helpers for prior `savedAnalysis/` outputs and external analysis stacks. Normal conversion, analysis, and napari code should not import parity helpers.
- When implementing behavior inspired by psycho5, make it a native twopy feature with twopy names, twopy defaults, and twopy docs. Do not add psycho5 compatibility APIs, wrappers, aliases, flags, or defaults to normal twopy modules.
- Code whose only purpose is to reproduce or compare psycho5 behavior belongs under `twopy.parity` and must stay out of conversion, analysis, persistence, and napari runtime paths.
- If a feature needs source data, add it to conversion and persist the needed plain twopy object first; then analyze the converted object.

## Environment

- Keep the environment reproducible.
- When adding, removing, or changing dependencies, update `environment.yml` (the canonical micromamba environment file) in the same change.
- Keep one install path. Do not split dependencies into dev or GUI extras.
- Install pre-commit hooks with `micromamba run -n twopy pre-commit install` after creating the environment.

## Configuration

- Keep machine-local variables in `config.yml`.
- Never track `config.yml`; track `config.example.yml` as the documented template for creating it.
- Load config through typed Python helpers instead of reading YAML ad hoc.
- Initial config keys: `database_path`, `data_path`, `database_access`, and `analysis_output`.
- Default `database_access` to `copy` unless a task explicitly needs direct mounted DB reads. This mode exists because database queries over the network can be slow, while transferring the DB file locally is usually fast.

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
- Keep `CHANGELOG.md` updated for user-facing changes. If a commit adds a feature, fix, behavior change, CLI change, GUI change, output-format change, or other user-visible change, add or update an entry under the top `Unreleased` section in the same commit.
- Never edit released changelog sections for current work. Corrections, renames, and behavior changes after a release must be recorded only under the top `Unreleased` section unless Gustavo explicitly asks for release-history repair.
- Follow the changelog structure from `RELEASE.md`: use `Features`, `Fixes`, and `Changes` sections when they apply, omit empty sections, and write user-facing entries instead of repository chore notes.
- Do not hard-wrap Markdown prose or other text docs. Keep each paragraph or list item on one source line, and use manual line breaks only for headings, lists, tables, code fences, and intentional structure.
- Every file must start with a plain-language docstring explaining its purpose.
- Every public function, class, and method must document:
  - inputs
  - outputs
  - what it does
  - why it exists in the analysis workflow
- Always use ASD-STE100 Simplified Technical English in docs, docstrings, comments, and user-facing messages. Use short, direct sentences and simple approved words. Stay accurate, and use a technical term only when it is the subject.
- Write documentation and comments from the reader's concrete object first: say what the user sees, what data value is used, and why it stays correct; avoid dense phrases that combine implementation jargon such as "factory", "derive", "identity", or "contract" unless those words are the subject being explained.
- Key analysis code must include inline comments that explain the reasoning behind the logic.
- Comments should clarify scientific or data-processing intent. Do not restate obvious syntax.

### User-facing docs site (`docs/`)

The user-facing docs live under `docs/` and are published with Sphinx + MyST + Furo to `twopy.gumadeiras.com`. Treat the code (and the actual GUI behavior it produces) as the source of truth, not the previous doc. When the code changes a button label, a tab order, a parameter default, or a public API name, update the affected page in the same change.

**Audience and file layout.** Each page has one audience and one job. Do not merge audiences.

- `docs/index.md` — short landing page that routes readers to the right track (GUI user, Python scripter, workflow author, reference). Keep it brief; link out.
- `docs/getting_started.md` — install → configure → launch → first save, for any new user. About ten minutes end-to-end.
- `docs/gui/*.md` — one page per user task in the napari app (`loading.md`, `rois.md`, `plots.md`, `custom_tab.md`, `group_matching.md`, `saving.md`). Do **not** recreate a monolithic `gui.md`. New GUI surface gets a new task page or extends an existing one.
- `docs/python_api.md` — task-oriented Python scripting guide. Organize by what the user wants to do (find recordings, convert, extract traces, save, …), not by twopy module.
- `docs/writing_custom_workflows.md` — developer guide for people adding workflows. Usage of the Custom tab belongs in `docs/gui/custom_tab.md`; cross-link both directions.
- `docs/development.md` — for people working on twopy itself only.
- `docs/input_data_spec.md` — short high-level contract.
- `docs/recording_file_schema.md` — long developer / auditor reference.

When in doubt, split. A page that mixes "click here" with "import this" with "the converted HDF5 layout" is doing three jobs.

**Page shape.** Lead with a one-sentence hook stating what the page is for, then jump straight into action-oriented H2 sections. No long preambles, no "background" sections at the top, no restating what the project is on every page. Reference details (parameter tables, file layouts, edge cases) go at the bottom of the page they belong to, never as the first thing the reader sees.

**Walls of text are bugs.** Break dense prose with subheadings, bulleted lists, or tables. If a paragraph runs past four or five sentences, split it or convert it. The user should be able to scan the page in seconds and find what they need.

**Voice and language.** The general docs-and-comments rules above still apply (human language, reader's concrete object first, no implementation jargon). On top of those, for the docs site:

- Second person, present tense, imperative: "Click **Save ROIs + analysis**", not "the user clicks" or "the button is clicked".
- Prefer concrete user verbs to engineering verbs: "saves" over "persists", "writes" over "emits", "uses" over "consumes", "settings" over "options surface".
- Describe the current behavior. Do not document deprecated paths, transition shims, or "this used to be" notes.

**Verify before writing.**

- Quote GUI labels exactly as they appear in code. Grep the napari source (`src/twopy/napari/`) for `QPushButton("...")`, `addTab(..., "...")`, and `QLabel("...")` to confirm wording. Wrong button names rot the doc and break user trust faster than missing detail.
- Confirm tab order from the actual `addTab` calls, not memory or the prior doc.
- Confirm Python API names against `twopy/__init__.py`'s `__all__` and the function's actual signature before writing an example. Run examples mentally against the real types.
- Confirm config keys against `config.example.yml`.

**Cross-link generously.** Every GUI task page that has a Python equivalent should link to `python_api.md` and vice versa. The Custom-tab usage page and the writing-workflows page must link to each other. Reference pages link back up to their high-level intros.

**Sphinx / MyST hygiene.**

- `docs/conf.py` sets `myst_heading_anchors = 3`, so use plain `[link text](page.md#heading-slug)` for in-page jumps; do not hand-roll anchors.
- Each top-level grouping in `index.md` gets its own `{toctree}` with a `:caption:`. Pages inside a subdirectory (`docs/gui/`) need their own `index.md` with a nested toctree so the navigation tree stays usable.
- The site must build cleanly with `sphinx-build -W -b html docs build/docs`. Treat warnings as errors locally before shipping.

**Do not create new top-level Markdown docs unless they earn their place.** If new material fits inside an existing page, extend that page. New files mean new navigation surface for every reader.

## Typing

- All Python functions and methods must have type annotations for inputs and return values.
- Use Python 3.13 for development and verification.
- Avoid `Any` unless there is no honest narrower type.
- Treat `typing.cast` as a last-resort boundary marker, not validation. Before adding one, prefer runtime checks, typed helper functions, Protocols, or narrowing control flow that proves the value's contract.
- Casts at external boundaries such as HDF5, JSON, YAML, MATLAB, Qt, napari, or NumPy must sit next to the validation that makes the target type true. For persisted literals and nested decoded structures, validate allowed values and inner shapes before narrowing.
- Do not use `cast(Any, ...)` to silence the checker. Parse the value explicitly or define the smallest useful Protocol/helper for the operation being used.
- When a cast remains necessary, keep it narrow and local. Avoid passing a casted object deeper into the code than the exact operation that needs it.
- Use standard library types where possible.
- Type checking is part of the gate. Run `ty` only through the micromamba command below. Do not use a raw interpreter path such as `/Users/.../envs/twopy/bin/python -m ty check`; `ty` can then resolve Homebrew/global site-packages instead of the micromamba environment and report bogus unresolved imports for installed dependencies.

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
- Avoid star-importing shared test helpers. Prefer explicit imports so `ty` can resolve the names in split or generated test modules.
- Use the standard-library unittest runner for direct test runs. This repo does not depend on pytest. Example:

```sh
micromamba run -n twopy python -m unittest tests.test_stimulus
```

## Verification

Run the touched-surface gate before handoff:

```sh
micromamba run -n twopy pre-commit run --all-files
```

The pre-commit hook runs the same gate before each commit.
