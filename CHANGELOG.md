# Changelog

## Unreleased

## 0.1.4 - 2026-05-08

### Features

- Added napari Plot-tab dF/F controls for background subtraction, gray interleave baseline sampling, fit mode, and motion masking.
- Added napari Plot-tab response-window controls with automatic gray-interleave context by default and manual pre/post seconds that feed both previews and Save Analysis.
- Added shared and ROI y-stripe percentile background subtraction for broad row-dependent background in dense process recordings.
- Renamed Plot-tab labels and dropdown entries to concise readable names and populated the dF/F baseline selector from recording epoch names, defaulting to names that look like gray, grey, or interleave baselines.
- Added shared helpers for default baseline epoch selection, baseline frame-window resolution, and converted recording frame-rate parsing.
- Added descriptive dF/F fit modes for direct bounded-tau, direct bounded-tau-and-amplitude, and log-linear tau fitting.
- Added parity helpers for psycho5-default response comparisons without adding psycho5-specific APIs to normal twopy analysis.

### Fixes

- Centered live napari response plot titles above their plots.
- Trimmed unused vertical space below live napari response plots.
- Kept live napari Plot-tab recomputes on the same post-stimulus x-axis window used when loading saved response plots.
- Rejected malformed grouped response time, frame, and ROI axes consistently before processing, loading, or exporting response summaries.
- Reused one shared frame-range validator for conversion, converted movie reads, ROI trace extraction, and background-corrected trace extraction.
- Moved the show SEM toggle into the napari Plot subsection and aligned two-decimal Plot-tab numeric controls.
- Standardized napari Plot-tab control widths across dropdowns, spin boxes, and checkboxes for more consistent option-panel layout.
- Added the package version to the napari window title.
- Restored the saved napari response-window Auto checkbox state when reloading analysis outputs.
- Restored saved napari dF/F baseline epoch selections and persisted resolved baseline windows for analysis audits.

## 0.1.3 - 2026-05-07

### Fixes

- Clamped stale napari movie preview ranges when loading a shorter recording.
- Ignored macOS AppleDouble sidecar files when discovering source recordings so napari can auto-convert recordings on mounted volumes.
- Reported source-recording validation errors during napari loading instead of replacing them with a missing converted-file message.
- Renamed the napari window title to `twopy`.

## 0.1.2 - 2026-05-06

### Features

- Added `twopy --version` and `twopy -v` so users can check the installed package version from the command line.

## 0.1.1 - 2026-05-06

### Features

- Added response processing controls for smoothing, low-pass filtering, and correlation filtering in script and napari workflows.
- Saved response processing settings and QC scores in analysis output so results can be audited and reloaded later.
- Restored saved response processing options when reopening analysis output in napari.

### Fixes

- Avoided SEM warnings when plotting sparse response data.
- Stopped live response previews from repeatedly recomputing after one update.

### Changes

- Improved napari response plotting controls and layout.

## 0.1.0 - 2026-05-06

Initial PyPI release.

### Features

- Installable Python 3.13 package with the `twopy` command-line entrypoint.
- Napari interface for loading two-photon recordings, converted recordings, and existing ROI files.
- Napari workflow for viewing mean images and aligned movies, drawing or editing ROI Labels layers, saving ROIs, computing responses, plotting responses, and reloading saved analysis output.
- Source recording discovery for the Clark Lab two-photon output layout.
- YAML configuration for database paths, data paths, database access mode, and analysis output routing.
- Schema-aware read-only SQLite database helpers for finding recordings by date, genotype, stimulus, sensor, cell type, hemisphere, and person.
- Source-to-twopy conversion into inspectable HDF5 files: `recording_data.h5` for metadata, stimulus tables, photodiode signals, mean image, synchronization metadata, and frame-count audit data, plus `aligned_movie.h5` for the aligned movie.
- Configurable converted-output routing to the source recording folder or a mirrored output root.
- MATLAB, TIFF, text, CSV, ZIP, and general recording file inspection helpers.
- Lazy loading of converted recordings and aligned movies.
- Photodiode event detection, event pairing between high-resolution and imaging-resolution traces, and classification of stimulus start, trial transition, and stimulus end events.
- Stimulus epoch mapping to imaging-frame windows through photodiode-aligned timing instead of nominal frame-rate assumptions.
- GUI-independent ROI mask objects, label-image conversion, ROI HDF5 save/load, and chunked ROI trace extraction.
- Background-corrected ROI trace extraction with no correction, global movie percentile subtraction, and ROI y-stripe percentile subtraction.
- Alignment-valid crop support for analysis-time trace extraction and background correction.
- ROI dF/F calculation with gray-interleave baseline fitting, robust fitting by default, and source-bound fitting for audit comparisons.
- Motion artifact masking after dF/F using converted high-motion frame metadata.
- Response grouping by epoch and trial with per-trial response time vectors.
- Analysis output persistence to `analysis_outputs.h5`, including ROI masks, fluorescence/background traces, dF/F arrays, epoch windows, grouped responses, response processing data, and audit metadata.
- CSV export of trial-level and epoch-grouped response time series under `exports/csvs/`.
- Public script-facing APIs for recording lookup, conversion, ROI handling, converted recording loading, photodiode alignment, response analysis, persistence, CSV export, and napari launch helpers.
