# Roadmap

twopy is a simple, auditable two-photon imaging analysis tool with a napari
interface. This file tracks only features or project facts we have explicitly
decided.

## Built

- Private GitHub repository and local project skeleton.
- Python 3.13 micromamba environment named `twopy`.
- Package metadata with one required install path for core, GUI, linting, and
  typing tools.
- Strict ruff and mypy configuration.
- Pre-commit hooks for ruff, mypy, and unit tests.
- Repository rules in `AGENTS.md`.
- Session path discovery for required microscope output files.
- Mini input data spec in `docs/input_data_spec.md`.
- Recording file reference schema in `docs/recording_file_schema.md`.
- YAML config file and typed loader for `database_path` and `data_path`.
- Tracked `config.example.yml` with instructions for creating private local
  `config.yml`.
- Schema-agnostic SQLite database module for discovering DB files and
  filtering experiment rows.
- Database code split into focused modules for catalog discovery, generic table
  search, modeled recording search, and read-only connections.
- Modeled database query for `stimulusPresentation` joined with `fly` metadata.
- Script-friendly `find_recordings` API with filters for date parts, genotype,
  stimulus, sensor, cell type, hemisphere, and person.
- DB query access modes: direct mounted DB query or local cached DB copies with
  metadata and SHA-256 change checks.
- Configured database access mode with `copy` as the default.
- Configured analysis output routing with source-folder and mirrored-output-root
  modes.
- MATLAB file inspection and loading layer for older MAT files and HDF5-backed
  MAT files.
- MATLAB SciPy loading code simplified so inspect and load paths share one
  visible-variable reader.
- Source-to-twopy HDF5 conversion for acquisition metadata, stimulus
  parameters, stimulus timeline, photodiode signals, mean image, and a separate
  aligned movie file.
- Conversion code split into focused modules for public workflow, source
  loading, HDF5 writing, MATLAB value conversion, frame ranges, and typed data
  objects.
- Acquisition metadata fields named in one explicit constant for easier audit.
- Converted HDF5 synchronization metadata documenting that imaging and stimulus
  clocks are aligned through photodiode events.
- Recording file inspector for MATLAB, TIFF, text, CSV, ZIP, and other files.
- TIFF metadata extraction for Python access plus optional CSV output with
  selected TIFF tags and ScanImage recording fields.
- Documented `imageDescription.mat` as the primary source for recording
  acquisition metadata.
- Real example recording inspected successfully: 24 files, 13 MATLAB files, raw
  TIFF shape `(8334, 127, 256)`.
- Real example recording converted successfully to `twopy/recording_data.h5`
  and `twopy/aligned_movie.h5`; aligned movie shape `(4168, 256, 127)`, mean
  image shape `(256, 127)`, stimulus timeline shape `(18021, 35)`.
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
- Recording acquisition metadata should be loaded from `imageDescription.mat`;
  raw TIFF metadata is for audit or raw interleaved frame access.
- Converted data format: HDF5 with gzip compression.
- Converted aligned movie is stored in its own HDF5 file because it usually
  dominates file size.
- Source MATLAB and raw TIFF files are conversion inputs only. Analysis and
  processing operate on converted twopy HDF5 files.
- Imaging and stimulus presentation run on separate clocks. Response analysis
  must align stimulus events to imaging frames through photodiode decoding, not
  nominal frame rates alone.
- Initial conversion generates a mean image. The default is the full aligned
  movie, with optional frame-range selection.
- Code style: simple, typed, documented, and tested.
- `savedAnalysis/`: optional; only present after prior MATLAB analysis.
- Roadmap maintenance: update with completed progress and explicit decisions;
  keep it high signal and low noise.
