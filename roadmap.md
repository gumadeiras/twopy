# Roadmap

twopy is a simple, auditable two-photon imaging analysis tool with a napari
interface. This file tracks only features or project facts we have explicitly
decided.

## Built

- Private GitHub repository and local project skeleton.
- Python 3.13 micromamba environment named `twopy`.
- Package metadata with one required install path for core, GUI, linting, and
  typing tools.
- Strict ruff and ty configuration.
- Pre-commit hooks for ruff, ty, and unit tests.
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
  parameters, stimulus data, photodiode signals, mean image, and a separate
  aligned movie file.
- Conversion uses configured `analysis_output` by default unless a caller passes
  an explicit output directory.
- Conversion code split into focused modules for public workflow, source
  loading, HDF5 writing, MATLAB value conversion, frame ranges, and typed data
  objects.
- Acquisition metadata fields named in one explicit constant for easier audit.
- Converted HDF5 files include a `frame_counts` audit group for aligned movie,
  imaging-resolution photodiode, and acquisition metadata frame counts.
- Documented the observed ScanImage one-frame acquisition metadata offset:
  `imagingResPd` matches aligned movie frames, while `acq.numberOfFrames` may be
  one less.
- Converted HDF5 synchronization metadata documenting that imaging and stimulus
  clocks are aligned through photodiode events.
- Recording file inspector for MATLAB, TIFF, text, CSV, ZIP, and other files.
- TIFF metadata extraction for Python access plus optional CSV output with
  selected TIFF tags and ScanImage recording fields.
- Documented `imageDescription.mat` as the primary source for recording
  acquisition metadata.
- Typed loader for converted `recording_data.h5` and `aligned_movie.h5` that
  keeps movie frames lazy and validates the frame-resolution photodiode contract.
- GUI-independent ROI mask storage in HDF5 with labels.
- ROI fluorescence trace extraction from converted aligned movies using chunked
  reads.
- GUI-independent `twopy.analysis` package for analysis-time modules.
- ROI background-corrected trace extraction with raw, background, and corrected
  arrays kept side by side for audit.
- Implemented three ROI background modes: no correction, global movie
  percentile subtraction, and local y-neighborhood percentile subtraction.
- Trial/frame-window response helpers live in `twopy.analysis.trials`.
- Stimulus epoch runs from `stimulus/data` can be mapped onto photodiode-aligned
  imaging-frame windows when their counts agree.
- ROI-level dF/F computation uses corrected ROI fluorescence, gray interleave
  windows, last-second baseline samples, one shared exponential tau, and one
  amplitude per ROI.
- ROI-level dF/F supports a default robust exponential fit mode plus
  `source_bounds` mode for original source-bound audit comparisons.
- Photodiode event segmentation for converted `high_res_pd` and
  `imaging_res_pd` signals.
- Order-based pairing of high-resolution photodiode events to imaging-frame
  photodiode events, with mismatched event counts treated as an error.
- Photodiode event classification into stimulus start, trial transition, and
  stimulus end events.
- Classified stimulus timing cross-checks paired photodiode events against
  `stimulus/data` `photodiode_flash` segments and positive epoch runs before
  producing frame windows.
- Frame-window response objects for splitting ROI traces by explicit imaging
  frame boundaries.
- Conversion now stores stimulus-run metadata from `runDetails.mat` with
  snake_case twopy field names such as `rig_name`.
- Conversion stores one code-derived label per stimulus data column using the
  backed-up MATLAB writer as source of truth: time, stimulus frame, epoch, ten
  closed-loop slots, twenty stimulus-specific slots, photodiode flash, and the
  trailing empty field.
- Documented how stimulus-specific data slots are decoded from
  `stimtype` through `filebackup/paramfiles/stimulus_lookup.txt` and the
  backed-up stimulus functions.
- Conversion reads `stimulusData/filebackup.zip` and stores per-`stimtype`
  stimulus-specific slot metadata in `recording_data.h5`.
- Public stimulus helper maps stable `stimulus_specific_*` column names to
  per-`stimtype` source expressions and readable names.
- `movie/mean_image` is stored uncompressed because it is a single small image.
- Real example recording inspected successfully: 24 files, 13 MATLAB files, raw
  TIFF shape `(8334, 127, 256)`.
- Real example recording converted successfully to `twopy/recording_data.h5`
  and `twopy/aligned_movie.h5`; aligned movie shape `(4168, 256, 127)`, mean
  image shape `(256, 127)`, stimulus data shape `(18021, 35)`.
- Real example converted recording loaded successfully with current schema; ROI
  trace smoke test produced shape `(5, 1)`, photodiode detection found 101
  high-resolution and 101 imaging-resolution events, and event pairing produced
  100 frame windows.
- Real example photodiode classification found 1 stimulus start event, 99 trial
  transition events, 1 stimulus end event, and 100 classified stimulus windows.
- Real database query matched the example recording in both `experimentLog.db`
  and `experimentInitLog.db` with `stimulusPresentationId=20005` and
  `fly=10923`.
- Real filtered API query matched the example recording with date `2023-10-17`,
  genotype `gh146`, stimulus `combo_stim_singles=3s_blank=3s_intensity=20`,
  sensor `g6f`, cell type `ALPN`, hemisphere `right`, and person `Harsh`.
- Tests for config loading, MATLAB inspection/loading, recording inspection,
  database filtering/copy-cache behavior, conversion, converted-recording
  loading, ROI storage/extraction, photodiode synchronization, response windows,
  valid sessions, optional `savedAnalysis/`, missing files, and ambiguous raw
  TIFF movies.

## Next

- Add protocol-specific photodiode classifiers for recordings with extra
  within-epoch alignment flashes.
- Group response outputs directly from classified stimulus windows.
- Compare twopy ROI-level dF/F against MATLAB `savedAnalysis/` on a recording
  that already has matching saved ROI masks.
- Load a recording in napari.
- Interactively draw or select ROIs in the movie.
- Save napari-drawn ROIs through the core ROI module.
- Inspect ROI responses.
- Divide responses by trials.
- Divide responses by other metadata associated with the recording.

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
- Background correction must be explicit and auditable. Store raw traces,
  background estimates, corrected traces, and method metadata rather than only
  the final corrected values.
- Code style: simple, typed, documented, and tested.
- `savedAnalysis/`: optional; only present after prior MATLAB analysis.
- Roadmap maintenance: update with completed progress and explicit decisions;
  keep it high signal and low noise.
