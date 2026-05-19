# Changelog

## Unreleased

### Features

- Added a napari Custom tab for built-in and local Python workflows.
- Added a native Direction selectivity workflow with recording epoch dropdowns, metric window controls, optional response rectification, an absolute-DSI plot threshold, and a direct ROI DSI table preview.
- Added versioned workflow metadata for custom outputs, including workflow id, version, source path, source hash, twopy version, parameters, and recording path.
- Added a stable `twopy.custom` API for workflow parameters, recording metadata, ROI selection, response metrics, tables, plots, output paths, and response plot data.
- Added a reference custom workflow example that shows every supported parameter and result type.
- Added all/visible epoch selection to the native Response kernels Custom-tab workflow.
- Added native random-noise response kernel fitting with a stimulus-mode dropdown for olfactory antenna activation or visual contrast, automatic default stimulus-column selection, metadata-driven hemisphere mapping for olfaction, one kernel per unique epoch name, modality-specific CSV outputs, and a packaged napari Custom-tab workflow.
- Added database search favorites in the napari Load-tab search window, with Save favorite, Use, and Remove actions backed by a machine-local YAML file.
- Added a thin epoch-span marker to napari response plots and exported response figures.

### Changes

- Show converted recording hemisphere in the napari Metadata tab.
- Added a Load-tab Reconvert selected action that confirms and rebuilds the selected recording's converted HDF5 files in place.
- Routed napari manual, CSV-list, and database recording loads through one shared Load-tab workflow with consistent load-failure wording.
- Unified napari response preview, live update, and Save Analysis handling around one ROI analysis request so plotted previews and saved outputs use the same ROI masks and Plot-tab settings.
- Let custom line plots set their y-axis label, and label native response-kernel plots as `Weight`.
- Moved manual Group Matching ROI match-table mutation rules into GUI-independent analysis helpers while keeping the napari view focused on rendering and interaction wiring.
- Routed native response timing through one audited recording-timing boundary that prefers classified photodiode boundary evidence and keeps interpolation as the non-boundary-flash fallback.
- Replaced the single configured data root with ordered `data_paths`, using the first available recording path for database loads and matching output/cache mirroring to the selected root.
- Split napari response-map display scaling coverage into a pure helper test module and reused shared converted-HDF5 fixture writers across more analysis tests.

### Fixes

- Used interpolated timing for older flash-train recordings whose `photodiode_flash` pulses are not one-per-epoch-boundary, while still rejecting incomplete boundary-flash evidence, restoring response plots and response-watershed ROI generation for older recordings.
- Restored saved-analysis reload for source recordings routed through the stable external analysis cache.
- Allowed source recording conversion to load older `chosenparams.mat` stimulus epoch parameters when `stimParams.mat` is absent.
- Deleted local cached PDF and PNG export figures after they successfully sync to the configured analysis output destination.
- Prevented database search favorites from changing the dialog state or overwriting a corrupt favorites file when local YAML persistence fails.
- Kept custom workflow ROI visibility filters, including the Direction selectivity threshold, from removing nonpassing ROIs from the ROIs tab.
- Kept custom workflow epoch-window defaults from being overwritten by recording duration caps.
- Made DSI threshold and epoch-window controls show three decimals, and showed synced custom table outputs at their publish paths instead of cache paths.
- Matched custom workflow epoch dropdown defaults from numeric selectors, full labels, and epoch names, including unnamed `Epoch N` labels.
- Made response-kernel CSV columns encode lag seconds, kept per-epoch filenames collision-proof, and reported fitted sample counts after kernel-window filtering.
- Kept Epochs-tab visibility toggles from rebuilding cached response heatmap images.
- Persisted database `fly.eye` metadata as converted recording `hemisphere`/`eye` run fields during conversion, with database lookup for older converted files that predate the field.
- Made the Export-tab Save ROIs + analysis button show the same save and sync status message as the Metadata tab.

## 0.1.9 - 2026-05-17

### Features

- Added a Load-tab Save loaded list action that exports the current loaded recordings to CSV, plus a separate Load CSV list action for reopening those recording lists.
- Allowed napari manual loading and database search loading to select multiple recording folders or result rows in one action.
- Added a Load-tab button that opens a Group Matching window, plus core CSV helpers for manually saving FOV grouping and cross-recording ROI match decisions.
- Added movie-level response heatmaps to the napari response dock, independent of ROI existence, with one signed inhibition-to-activation map per stimulus epoch.
- Added response heatmap computation options for pixel dF/F maps with Gaussian smoothing and square-window dF/F maps with preset or custom size and stride.
- Added robust 95th-percentile signed heatmap color limits, shared-limit display control across epochs, blue-black-orange colorbars, and neutral background rendering over the mean image.
- Added recording-level `response_heatmaps.h5` persistence plus Export-tab PDF and PNG heatmap saving through the existing analysis-cache sync path.
- Added script-facing response heatmap APIs for computing, saving, and loading `ResponseMapData`.

### Changes

- Made Load CSV list automatically use a sibling `fov_groups.csv` when opening Group Matching from a saved loaded-recordings CSV.
- Changed Group Matching FOV notes from one shared field to per-recording card notes so `fov_groups.csv` can carry row-specific audit notes.
- Added a Back to FOV assignment action in the Group Matching ROI assignment view so FOV groups can be revised without closing the popup.
- Made the Group Matching ROI assignment view visual and self-contained, with low-opacity numbered context overlays for all ROIs, selected ROI overlays colored to match each recording trace, a compact saved-groups table that restores selected ROIs, FOV/ROI note fields, ROI CSV load/save-path controls, separate Add new group and Overwrite selected group actions, selected-group removal, clickable response trace visibility chips, optional epoch-peak normalization, and translucent overlaid response traces inside the popup instead of relying on the active main napari selection.
- Made the Group Matching popup open at the full available screen height while preserving its compact width.
- Split Group Matching FOV CSV controls into separate load and save-path browse actions.
- Documented response heatmap math, pixel/window methods, display scaling, persistence, and Python API usage in the GUI and Python API guides.

### Fixes

- Made the Group Matching ROI assignment view scroll as one page and capped the popup height to the available display height.
- Kept napari Loaded Recordings and saved loaded-list CSVs on source session paths instead of leaking local analysis-cache paths, while still reopening existing cached data when the source path is unavailable.
- Made the Group Matching FOV and ROI load buttons open explicit existing-file CSV pickers instead of a macOS folder chooser that could gray out selectable paths.
- Cleared the Group Matching ROI note field after adding, overwriting, or removing a group so notes do not carry into later actions accidentally.
- Omitted gray/grey/interleave epochs from Group Matching ROI response previews, matching the main napari response plot defaults.
- Prevented repeated napari loads of the same recording from adding duplicate Loaded Recordings entries.
- Made every napari sidebar tab scrollable so the Export tab no longer sets a tall minimum sidebar height.
- Made the Search database date filter accept `YYYY-MM-DD`-style input with common separator variants and normalize it before querying.
- Made response heatmaps reserve colorbar space inside the Plot-tab Size value so heatmap panels scale like response plots.
- Made the napari Plot-tab correlation filter default its window stop to the shortest plotted epoch duration instead of `0 s`.
- Made response heatmaps skip gray/interleave and first-frame epochs that lack pre-epoch baseline context instead of disabling all heatmaps for that recording.
- Moved response heatmap option recomputation off the Qt thread so changing heatmap settings does not block napari while movie maps are rebuilt.
- Prevented Group Matching's Save and close action from overwriting a newly added or overwritten ROI group with an empty note unless the selected group was edited after loading.
- Rejected blank manual group CSV `recording_path` fields during load instead of treating them as the current directory.

## 0.1.8 - 2026-05-14

### Features

- Added a compact napari trial timeline rail and viewer HUD that follow the movie frame slider using photodiode-aligned stimulus epoch windows.
- Added GUI-independent grid and watershed ROI extraction helpers with optional region-mask restriction for script workflows.
- Added response-watershed ROI extraction that segments repeatable stimulus-locked pixel responses using amplitude, local response coherence, and split-half reliability score maps, including selection from the napari ROIs tab.
- Added a dF/F `no_baseline_epoch` baseline mode for recordings without a distinct baseline epoch, exposed in the napari Plot tab as `no baseline epoch` with a `First epoch` selector.
- Added parity-only psycho5 `noTrueInterleave` response helpers for future native-to-historical analysis comparisons.
- Added parity-only psycho5 grid and watershed-label helpers for comparing native twopy ROI discovery against historical MATLAB ROI extraction.
- Added a tracked dated CSV-backed pixel-size calibration resolver and micron-sized grid ROI helper for calibrated physical ROI templates.
- Added ROIs-tab ROI mode controls in napari with manual drawing as the default plus generated grid, watershed, and response-watershed ROI modes.
- Added metadata-driven calibration profile preselection for ROIs-tab micron grids, backed by tracked ScanImage config mappings and conservative manual fallback when the profile is incomplete or unmeasured.

### Changes

- Rendered gray, grey, and interleave epochs as neutral gray in the napari trial timeline.
- Simplified ROIs-tab grid controls so pixel and micron settings show only for the selected unit, extrapolation is enabled by default, calibration modes include known ScanImage config names, and micron-grid status reports only the estimated pixel size.
- Made response-watershed ROI masks split disconnected score islands, fill component holes by default, and expose an opt-in closing radius for tiny same-basin gaps.
- Renamed the napari Update tab to Metadata, grouped Metadata into Recording, Microscope, and Outputs sections, made Search Database and Load manually the first Load-tab actions, moved Reload saved analysis to Load, moved Save ROIs + analysis to Export, and removed the Recompute preview button.

### Fixes

- Kept unresolved ROIs-tab calibration fields on explicit selection placeholders instead of silently using the first dropdown option.
- Made script and napari dF/F defaults use the same gray/grey/interleave baseline epoch selection.
- Made napari live ROI plot updates more responsive by cancelling stale background recomputes, reusing unchanged ROI traces, reusing epoch plot widgets, and caching Labels colormap bases for visibility toggles.
- Made napari correlation filtering uncheck and dim ROIs that do not pass QC in the ROIs tab and Labels layer.
- Made Reload saved analysis restore the saved ROI file into the editable Labels layer.

## 0.1.7 - 2026-05-11

### Features

- Added a Load-tab `unload all` button and renamed the existing unload action to `unload selected`.
- Added a napari Plot-tab Normalization section that can scale all grouped responses by each ROI's selected-epoch peak, persisting the selected epoch and per-ROI scale factors with saved analysis outputs.
- Added local analysis caching for napari recording work so converted recordings, ROI edits, and response analysis run from local storage while saved analysis outputs sync back to the configured publish location in the background.

### Fixes

- Synced napari Export-tab figure files from the local analysis cache back to the configured publish/source `twopy/exports` folder.
- Made cached analysis file copies safer and repeatable by using atomic target-file replacement and refreshing stale localized converted files when the selected converted output is newer.
- Made the napari Update tab scroll so long recording summaries and output paths stay visible when labels wrap.
- Preserved noncontiguous ROI label numbers when reopening ROIs in napari so visibility toggles, colors, and deletion target the correct ROI.

## 0.1.6 - 2026-05-08

### Features

- Added a Load-tab database search window with user, cell type, sensor, stimulus, and date filters plus a collapsible user, cell type, sensor, stimulus, date, and source-folder time result tree that can load any selected subtree.
- Tightened the database search layout with side-by-side filters and results, placeholder filter hints, a wider experiment column, a compact count column, and deduplicated source-recording counts across database files.
- Improved database search loading feedback so Load Selected closes the search window and failed paths appear in a separate scrollable error dialog.
- Added a ROIs-tab action for deleting the selected ROIs from the active Labels layer.

### Fixes

- Kept the Load-tab Recording path synchronized with the active loaded recording after database loads, list selection changes, and unloads.
- Reported database-search setup failures and per-recording database load failures without hiding valid recordings from the same selection.
- Reduced the live ROI plot update debounce so response plots refresh sooner after drawing finishes.

## 0.1.5 - 2026-05-08

### Changes

- Replaced the separate right-side napari twopy docks with one tabbed `twopy` dock containing Load, Update, Plot, ROIs, Epochs, and Export tabs.
- Reduced the default napari response plot size to 300 px.
- Removed nested scrollbars from napari ROI and Epoch visibility lists so those tabs use the main tab scrollbar.
- Set a compact 345 px opening height for the top napari response-plot dock.
- Tightened the napari Load tab with shorter labels and compact Browse buttons.
- Removed napari Load-tab frame controls; recording selection now always loads the full movie.
- Tightened Plot-tab subsection spacing so display, response-window, dF/F, and processing controls use one compact rhythm.
- Added an explicit minimum width for the twopy napari sidebar so controls do not collapse into clipped labels on smaller windows.

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
