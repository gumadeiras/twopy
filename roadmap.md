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
- Conversion computes and stores `movie/alignment_valid_crop` from
  stimulus-bounded alignment offsets while preserving the aligned movie as
  full-frame data.
- Conversion stores per-frame alignment shift magnitudes and a motion artifact
  mask so high-motion frames can be marked after dF/F.
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
- Shared photodiode timing helpers for event segmentation, lab-style
  range-fraction thresholding, stimulus frame bounds, and ambiguous end-flash
  errors.
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
- Global movie percentile background correction defaults to the saved
  alignment-valid crop, validates full-frame ROI masks against that crop, and
  streams only the selected crop from HDF5 for lower I/O.
- Analysis trace extraction with `method="none"` also honors the selected
  spatial domain; lower-level ROI trace extraction remains the full-frame raw
  primitive.
- Trial/frame-window response helpers live in `twopy.analysis.trials`.
- Stimulus epoch runs from `stimulus/data` can be mapped onto photodiode-aligned
  imaging-frame windows when their counts agree.
- Stimulus epoch values can also be interpolated onto stimulus-bounded imaging
  frames from converted `stimulus/data`, without requiring equal high-res and
  imaging-res photodiode event counts.
- ROI-level dF/F computation uses corrected ROI fluorescence, gray interleave
  windows, last-second baseline samples, one shared exponential tau, and one
  amplitude per ROI.
- ROI-level dF/F supports a default robust exponential fit mode plus
  `source_bounds` mode for original source-bound audit comparisons.
- Post-dF/F motion artifact masking marks converted high-motion frames as NaN
  while keeping the fitted fluorescence and baseline auditable.
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
- ROI dF/F responses can be grouped by epoch and trial, with one response
  matrix per trial and compact per-ROI summary rows.
- Analysis outputs can be persisted to one inspectable HDF5 file containing ROI
  masks, fluorescence/background traces, dF/F arrays, epoch windows, and grouped
  responses, with optional CSV response summaries.
- Script-facing response workflow loads a converted recording and ROI set, runs
  trace extraction, dF/F, response grouping, and persistence in one call.
- Persisted analysis HDF5 files can be reloaded into typed twopy analysis
  objects.
- Small committed real-data regression fixture derived from the example
  converted recording, with loader and response-workflow tests over real pixel
  values.
- Core ROI helpers convert between integer label images and ROI masks, so
  napari label edits and scripts use the same ROI HDF5 format.
- Thin napari adapter loads a converted recording mean image, optional bounded
  movie preview, editable ROI Labels layer, and saves edited label images
  through the core ROI module.
- Initial magicgui dock panel for napari with a Save ROIs control that writes
  the current Labels layer to twopy ROI HDF5.
- Magicgui dock panel can start from an empty napari window, load a converted
  recording automatically after folder selection, and then save ROIs beside
  that recording by default.
- Napari Load Recording control accepts a converted output folder or source
  recording folder, auto-detects `recording_data.h5`, `aligned_movie.h5`, and
  `rois.h5`, and loads the full movie by default.
- Napari controls use intuitive ROI file labels and a compact movie frame range
  slider.
- Napari has an initial response plotting dock with tabs for epoch response
  plots and plot options; it can reload persisted analysis outputs or update
  plots from the current Labels ROIs through the core analysis workflow. Plots
  now use saved per-trial response time vectors, match ROI trace colors to the
  Labels layer, share axes, preserve two seconds before stimulus onset, extend
  two seconds after stimulus offset when gray interleave frames exist, hide gray
  epochs by default, sort epochs by epoch number, draw thicker dashed
  zero-reference lines, and expose
  ROI/epoch visibility plus manual axis controls in scrollable option panels;
  the ROI selection list shows each ROI's plot color beside its name.
- Napari recording controls show the loaded path tail instead of `default`
  after a selection, and the Save ROIs control lives in a left-side dock instead
  of duplicating ROI save fields in the recording-load panel.
- Napari response plotting code is scoped under `twopy.napari.plotting` with
  separate modules for plot data, option controls, drawing widgets, and tabs.
- Napari ROI Labels layers open with 50 percent opacity and additive blending so
  ROI masks remain visible over the mean image and movie.
- Launch script for opening napari from a converted output directory, a source
  recording directory, or an explicit `recording_data.h5` path. When a source
  recording has no converted twopy files yet, napari runs conversion first and
  then opens the converted HDF5 files.
- `twopy` terminal command registered as the application launcher for napari.
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
- Read-only `twopy.parity` loader for selected `savedAnalysis/*.mat`
  `lastRoi` fields needed by parity checks: ROI masks, epoch lists, epoch
  windows, and saved ROI traces.
- Parity adapter converts saved ROI label images into full-frame twopy
  `RoiSet` objects for audit comparisons.
- Parity dF/F comparator runs the normal twopy analysis path over converted
  data, using saved ROI masks and saved interleave windows, then reports
  numeric error against saved trace matrices.
- Real example recording inspected successfully: 24 files, 13 MATLAB files, raw
  TIFF shape `(8334, 127, 256)`.
- Real example recording converted successfully to `twopy/recording_data.h5`
  and `twopy/aligned_movie.h5`; aligned movie shape `(4168, 256, 127)`, mean
  image shape `(256, 127)`, stimulus data shape `(18021, 35)`.
- Real example conversion produced alignment-valid crop
  `axis0=[6, 250)`, `axis1=[9, 118)`, shape `(244, 109)`.
- Real example converted recording loaded successfully with current schema; ROI
  trace smoke test produced shape `(5, 1)`, photodiode detection found 101
  high-resolution and 101 imaging-resolution events, and event pairing produced
  100 frame windows.
- Real example shared photodiode timing used threshold `6.54609375`, found 101
  high-resolution events, and interpolated stimulus epochs onto frames
  `[54, 3963)` with 100 epoch windows.
- Real example converted motion artifact mask contains 0 high-motion frames
  under the current threshold.
- Real example `savedAnalysis/` loader read saved ROI mask shape `(244, 109)`
  and saved trace shape `(3909, 1221)` from a prior analysis file.
- Real example dF/F parity comparison against
  `WatershedRoiExtraction_28_12_10_17_10_23.mat` matched 1,221 saved ROI
  traces over 3,909 frames with mean absolute error `1.84e-10` and median ROI
  correlation `1.0`.
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
- Interactively draw or select ROIs in the movie.
- Inspect ROI responses.
- Load persisted analysis outputs in napari.
- Query the database from napari, choose a recording, load converted twopy
  data, draw or edit Labels ROIs, run analysis, and inspect responses.
- Divide responses by trials.
- Divide responses by other metadata associated with the recording.

## Decisions

- GUI: napari.
- Napari implementation stays inside this repo for now, structured as reusable
  `twopy.napari` helpers that a future plugin can wrap with menus/widgets.
- Napari control panels start with magicgui; move to custom widgets or plugin
  packaging only when the workflow needs it.
- Long-term napari workflow: database query, recording selection, converted
  data load, Labels ROI editing, analysis, and response inspection.
- Primary ROI interaction in napari uses Labels layers because they directly
  represent the pixel masks used by analysis.
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
- Source-output ownership is explicit: `twopy.conversion` may read microscope
  MAT/TIFF/text/ZIP files, `twopy.analysis` uses converted twopy data only, and
  `twopy.parity` is isolated read-only comparison code for prior
  `savedAnalysis/` outputs.
- Imaging and stimulus presentation run on separate clocks. Response analysis
  must align stimulus events to imaging frames through photodiode decoding, not
  nominal frame rates alone.
- Default photodiode event thresholding uses the lab convention
  `min + 0.3 * (max - min)`. Explicit thresholds remain available for audits.
- More than two long photodiode flashes is ambiguous and should fail loudly
  rather than silently choosing one.
- Initial conversion generates a mean image. The default is the full aligned
  movie, with optional frame-range selection.
- The converted aligned movie and ROI masks stay full-frame. Crop-domain
  analysis uses saved crop metadata instead of duplicating cropped movie or mean
  image datasets.
- Background correction must be explicit and auditable. Store raw traces,
  background estimates, corrected traces, and method metadata rather than only
  the final corrected values.
- MATLAB-parity background subtraction requires the same spatial crop/mask
  contract as the source analysis path, not only the same percentile rule.
- dF/F defaults to the robust exponential fit mode. It keeps the shared tau
  bound but does not use the source log-amplitude upper bound because that
  bound depends on the first baseline sample and can become invalid or overly
  restrictive when that sample is small, nonpositive, or noisy. Use
  `source_bounds` only for audit comparisons.
- Code style: simple, typed, documented, and tested.
- `savedAnalysis/`: optional; only present after prior MATLAB analysis.
- Roadmap maintenance: update with completed progress and explicit decisions;
  keep it high signal and low noise.
