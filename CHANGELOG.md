# Changelog

## Known Problems

- When multiple recordings with different movie lengths are loaded in napari, the main frame slider is bounded by the longest loaded movie because napari computes dimensions from all layers, including hidden ones; switching the selected recording updates twopy's colored trial timeline but does not make the main slider selected-recording-specific.

## Unreleased

### Changes

- Replaced napari's generic empty-viewer welcome text with a versioned twopy Getting Started message that points users to the Load tab before a recording is open.
- Made Group Matching ROI assignment allow multiple separate ROIs from the same recording in one saved cell group, with one CSV row and one response trace per selected ROI. The manual ROI match helper APIs now take `(recording_path, roi_label)` pairs instead of a recording-to-ROI mapping.
- Made Group Matching selected-ROI chips use a vertically centered red remove marker with normal ROI label text.
- Made same-recording ROI response traces reuse that recording's assigned hue.
- Capped the Group Matching Selected ROIs trace-chip area at five visible rows with local scrolling for larger groups.

## 0.3.2 - 2026-06-17

### Changes

- Made the napari recording-load progress dialog show a dense scrollable queue, source recording identity, the converted folder currently being read, and the read route so large cached batches are easier to monitor.
- Made ROI response normalization use each ROI's strongest selected-epoch response, whether positive or negative, so inhibitory or mixed responses are not exaggerated by a small positive peak.
- Added `twopy config setup` and first-launch config-template creation so installed users can create an editable config file in their user folder without a source checkout. `twopy`, including `twopy /path/to/recording`, now creates the template and stops before opening napari when config is missing, and validates an existing config before continuing. Python napari controls now require config-backed setup; pass `add_controls=False` to `open_recording_in_napari(...)` when scripts only need raw napari layers.
- Made Direction selectivity run on visible ROIs by default and start with a `0.1` DSI show threshold.
- Made napari manual-load output routing explicit for Save ROIs + analysis and every figure export button: source recordings outside `data_paths` copy files to the source `twopy/` folder, converted folders outside `data_paths` save where they were selected, Metadata shows both local and final output paths, and queued background copies no longer cancel earlier saves.
- Added a 33 GB default limit for the local analysis cache. twopy now keeps a small rebuildable cache inventory and removes old cache entries after cache-growing writes only when their files already exist in the final output folder, while preserving active, unsynced, stale, or source-unavailable entries.
- Made napari save ROI-generation mode and mode-specific settings in `rois.h5`, mark generated masks that were edited by hand, and restore those ROIs-tab controls with Reload saved analysis.

### Fixes

- Added spacing between the napari recording-load queue rows and scrollbar, and tightened the queue column so large batches waste less space.
- Kept old saved analyses with positive-peak normalization loadable by disabling the old normalization setting and showing a clear reload warning.
- Made napari watershed ROI generation apply **Min pixels** inside the displayed alignment-valid crop, preventing edge ROIs that are large full-frame but undersized in the editable ROIs tab.

## 0.3.1 - 2026-06-08

### Changes

- Made `convert_recording_to_twopy(...)` only write converted HDF5 files to its resolved output directory; napari loading now owns publishing cached converted files to `analysis_output`.
- Added an `area (px)` column to the napari ROIs tab so editable ROI rows show their current mask size.
- Made top-level `import twopy` leave napari, Qt, and matplotlib unloaded; import GUI helpers from `twopy.napari` instead.
- Moved response plot data objects and builders into `twopy.analysis.response_plotting`; scripts and custom workflows should import them from analysis instead of `twopy.napari.plotting.data`.
- Made napari select a loaded recording's editable `rois` Labels layer when you select that recording in the Loaded Recordings pane.
- Made napari set the selected `rois` paint label to the first unused ROI number when a loaded recording is selected.
- Made Load-tab saved recording CSVs reload valid `recording_data_path` HDF5 files directly before falling back to source recording paths.
- Made Search database favorites reorderable by dragging rows in the favorites list.
- Renamed Search database's `Save favorite...` action to `Save as favorite...` and added favorite editing for saved names and filters.

### Fixes

- Kept direct Load-tab recording calls on the asynchronous progress-dialog path and preserved failed manual path text for correction.
- Kept Group Matching FOV and ROI CSV save targets beside a loaded recording-list CSV even when those matching CSVs do not exist yet.

## 0.3.0 - 2026-06-01

### Changes

- Added a shared programmatic custom-workflow runner so scripts and the napari Custom tab use the same execution, validation, and provenance path.
- Added a ROIs-tab Merge Selected action that combines checked ROI labels before live response plots recompute from the merged mask.
- Made napari Load-tab recording loads prepare recordings in a background worker with a modeless progress dialog, while applying napari layers on the Qt thread.
- Published converted `recording_data.h5` and `aligned_movie.h5` files to the configured `analysis_output` location whenever cached conversion or save/export sync runs.
- Made Load-tab saved recording CSVs write `recording_data_path` as the published `analysis_output` HDF5 path instead of the local analysis-cache file when publish routing applies.
- Tightened the developer function-inventory direct-test attribution so calls through `twopy` package exports and local test helper re-exports count toward the defining function.
- Changed converted movie and ROI storage to Python image order matching MATLAB display at conversion time and removed napari display-axis transposition.
- Made response heatmap files record the Python-image-order spatial marker on saved spatial arrays and reject heatmap files missing that marker.
- Made analysis output files record the Python-image-order spatial marker on embedded ROI masks and reject files missing that marker.

### Fixes

- Kept ROIs-tab checkboxes in sync when a Custom-tab workflow such as Direction selectivity changes visible ROI selection.
- Kept y-stripe background subtraction aligned with displayed rows by storing converted movies and mean images in Python image order.

## 0.2.1 - 2026-05-21

### Features

- Added a GitHub Pages documentation site for `twopy.gumadeiras.com`.
- Added stable custom-workflow access to converted recording metadata through `ctx.recording_metadata()`, including typed helpers for run and acquisition fields.

### Changes

- Added a GitHub repository footer icon and copyright notice to the documentation site.
- Overhauled the documentation site for end users: rewrote the landing page, added a Getting Started walkthrough, split the GUI guide into per-task pages (Loading, ROIs, Plots, Custom tab, Group Matching, Saving) under `docs/gui/`, rewrote the Python API and custom-workflow writing guides in plainer language, and tightened the input-data and recording-file reference intros.
- Switched the documentation site to the Furo theme.
- Made the napari Plot tab's Show SEM toggle start unchecked.
- Made Custom-tab workflow results show one selectable compact output folder and avoid horizontal scrolling from long output paths.
- Changed native Response kernels defaults to 100 past stimulus samples and 10 future stimulus samples for both olfaction and vision.
- Changed Group Matching FOV and ROI card overlays and ROI trace chips to show recording labels as `YYYY.MM.DD HH:MM`.
- Added a developer function-inventory script that reports action-oriented per-function size, docstring size, direct call counts, direct test counts, API surface, domain, git churn, risk score, and simple complexity.
- Made the native Response kernels Stimulus control default to olfaction for `OdorRig` recordings and vision for other rigs.
- Reworked the napari Group Matching FOV and ROI assignment views with larger image previews, clearer staged headings, grouped file/action/preview sections, theme-aware colors, more legible primary and destructive actions, and a compact two-column FOV assignment layout with image-overlaid recording labels, vertical contrast sliders, a vertical-only scrollable left control column, and compact FOV ID step buttons.
- Made the Group Matching popup expand to the current display without entering macOS fullscreen, removed the maximum-height cap, added hover color feedback to matching buttons, and added a compact four-row current-FOV table that reselects assigned recording cards for edits.
- Moved the Group Matching ROI controls into a compact left sidebar, placed a selected-ROI chip panel above separate ROI-response and combined-response plot sections, and added numeric FOV IDs, numeric ROI dropdown labels, selectable ROI load-status paths, inline ROI color chips, resizable saved-group columns, epoch visibility controls, ROI-response plot toggles, and resize-safe response previews with size-aware plot text.
- Made Group Matching ROI cards select ROIs directly from mean-image overlay clicks, with a second click on the selected ROI returning the card to `No ROI`.
- Made Group Matching and live ROI response previews reuse selected ROI traces, response contexts, and compact epoch plot widgets more aggressively, while initial response heatmaps now compute in the existing background worker on recording load.
- Moved napari Group Matching into a package with shared card, table, image, response-preview, and style helpers, and made FOV cards use the same fixed-size responsive grid behavior as ROI cards.

### Fixes

- Made ROI y-stripe background correction stop before dF/F when it over-subtracts any ROI baseline, with guidance to use shared y-stripe/global background or redraw ROIs with dim local background pixels.
- Made Group Matching Selected ROI chips sit directly next to each other while still wrapping to new rows when the window is narrow.
- Kept the Group Matching ROI assignment header and selected-ROI summary visible by placing ROI responses and combined responses side by side in horizontal-only scroll areas that reserve full plot height, plus a vertical-only ROI-card scroll area.
- Tightened the Group Matching popup's outer frame padding so both FOV assignment and ROI assignment use more of the available window.
- Made the Group Matching ROI plot-setting smoothing and normalization dropdowns show all choices without scrolling.
- Kept napari response-plot scrolling responsive for long recordings and many visible ROIs by caching epoch plot rasters and using pixel-density-aware trace drawing.
- Made Return in the napari database-search window search from filter fields and load selected result rows from the result tree.
- Kept the napari Metadata and Export tabs from widening the sidebar with a horizontal scrollbar when status paths are long, and made Metadata text selectable for copying.
- Made missing database-loaded recordings report every configured `data_paths` candidate instead of only the first cache fallback path.
- Kept Direction selectivity's metric window cap tied to the selected preferred/null stimulus epochs instead of unrelated shorter epochs such as gray interleaves.
- Cleared stale Custom-tab workflow outputs when loading, selecting, or clearing a recording.
- Kept native response-kernel mean plot legends focused on mean traces by omitting separate SEM band legend entries.
- Cleared stale Custom-tab workflow outputs after failed DSI or response-kernel runs and when a run returns no displayable outputs.
- Kept Group Matching's auto-managed ROI match CSV path following the current loaded-recordings CSV folder after unloading all recordings and loading a new CSV list, while preserving explicitly chosen ROI CSV paths.
- Kept Group Matching ROI-view selection changes from moving focus back to the main napari window, from flashing the response preview rows on each update, and from carrying hidden response chips into a newly selected saved group.
- Kept the Group Matching Selected ROIs panel showing a current trace-visibility status for single, all-visible, partially hidden, and fully hidden selections.
- Kept napari Load manually and Load CSV list dialog folders independent so choosing one no longer overwrites the other's remembered location.
- Kept napari response plots from carrying stale epoch visibility into newly loaded recordings, so gray/interleave epochs are hidden by default after loading a CSV list.

## 0.2.0 - 2026-05-19

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
- Made the native Response kernels baseline epoch use the recording epoch dropdown and gray/interleave default instead of a numeric spinbox.
- Moved manual Group Matching ROI match-table mutation rules into GUI-independent analysis helpers while keeping the napari view focused on rendering and interaction wiring.
- Added Group Matching FOV select-all controls, shared italic placeholder styling, and ROI preview plot-size, smoothing, and mean-plus-SEM response rows.
- Added a shared `finite_mean_and_sem` API so response plots, CSV exports, custom workflows, and group-matching previews use the same sample-SEM convention.
- Made native response-kernel workflows plot mean +/- SEM summary traces with filled SEM bands using the shared SEM helper.
- Made custom ROI-labeled line plots, including native response-kernel traces, use the current napari ROI colors automatically.
- Routed native response timing through one audited recording-timing boundary that prefers classified photodiode boundary evidence and keeps interpolation as the non-boundary-flash fallback.
- Replaced the single configured data root with ordered `data_paths`, using the first available recording path for database loads and matching output/cache mirroring to the selected root.
- Split napari response-map display scaling coverage into a pure helper test module and reused shared converted-HDF5 fixture writers across more analysis tests.

### Fixes

- Widened the napari Plot-tab normalization checkbox so `Normalize to epoch peak` is no longer cropped.
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
- Made response-kernel fitting skip irregular stimulus-clock segments and report fitted/skipped segment counts instead of failing the whole workflow.
- Kept pooled response-kernel epoch groups from using lag samples across stimulus-segment boundaries, and kept invalid database hemisphere metadata from being hidden as missing metadata.
- Kept malformed `config.yml` errors visible during explicit-output conversion and old converted-file hemisphere lookup, while still allowing missing config when explicit conversion output is provided.
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
