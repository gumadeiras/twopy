# Roadmap

twopy is a simple, auditable two-photon imaging analysis tool with a napari
interface. This file tracks only features or project facts we have explicitly
decided.

## Built

- Private GitHub repository and local project skeleton.
- Python 3.13 micromamba environment named `twopy`.
- Package metadata with napari as the GUI extra.
- Strict ruff and mypy configuration.
- Repository rules in `AGENTS.md`.
- Session path discovery for required microscope output files.
- Mini input data spec in `docs/input_data_spec.md`.
- YAML config file and typed loader for `database_path` and `data_path`.
- Schema-agnostic SQLite database module for discovering DB files and
  filtering experiment rows.
- Modeled database query for `stimulusPresentation` joined with `fly` metadata.
- Script-friendly `find_recordings` API with filters for date parts, genotype,
  stimulus, sensor, cell type, hemisphere, and person.
- DB query access modes: direct mounted DB query or local cached DB copies with
  metadata and SHA-256 change checks.
- Configured database access mode with `copy` as the default.
- MATLAB file inspection and loading layer for older MAT files and HDF5-backed
  MAT files.
- Recording file inspector for MATLAB, TIFF, text, CSV, ZIP, and other files.
- Real example recording inspected successfully: 24 files, 13 MATLAB files, raw
  TIFF shape `(8334, 127, 256)`.
- Real database query matched the example recording in both `experimentLog.db`
  and `experimentInitLog.db` with `stimulusPresentationId=20005` and
  `fly=10923`.
- Real filtered API query matched the example recording with date `2023-10-17`,
  genotype `gh146`, stimulus `combo_stim_singles=3s_blank=3s_intensity=20`,
  sensor `g6f`, cell type `ALPN`, hemisphere `right`, and person `Harsh`.
- Tests for config loading, MATLAB inspection/loading, recording inspection,
  database filtering/copy-cache behavior, valid sessions, optional
  `savedAnalysis/`, missing files, and ambiguous raw TIFF movies.

## Next

- Load a recording in napari.
- Interactively draw or select ROIs in the movie.
- Inspect ROI responses.
- Divide responses by trials.
- Divide responses by other metadata associated with the recording.
- Treat recording metadata as always available.
- Convert MATLAB-derived data into Python objects that can be loaded later.

## Decisions

- GUI: napari.
- Python: 3.13.
- Environment: micromamba env named `twopy`.
- Core modules stay GUI-independent; napari code calls core modules.
- Converted data format: HDF5 with gzip compression.
- Code style: simple, typed, documented, and tested.
- `savedAnalysis/`: optional; only present after prior MATLAB analysis.
- Roadmap maintenance: update with completed progress and explicit decisions;
  keep it high signal and low noise.
