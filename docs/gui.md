# GUI Guide

Use the `twopy` command to open napari with the twopy docks.

```sh
twopy
```

Use Search database on the Load tab to find recordings from `config.yml`, or click Load manually to choose one or more source recording folders. Direct `recording_data.h5` paths can still be opened from the command line or typed into the loader field. With the default `analysis_caching: true`, twopy converts or copies converted files into the local `analysis_cache_dir` first, then all plotting and ROI analysis reads from that local cache. If no recording is selected, napari still opens.

Open a recording directly:

```sh
twopy /path/to/source/recording
twopy /path/to/recording_data.h5
```

## Workflow

1. Load a source recording folder, converted `recording_data.h5`, or database search result.
2. Draw or edit ROIs in the `rois` Labels layer.
3. Update response plots from the current Labels layer.
4. Click Save ROIs + analysis when the plots look right.

By default the launcher opens the mean image, full movie, editable `rois` Labels layer, a compact top response-plot dock, and one tabbed right-side `twopy` dock. The first tab is Load, followed by Metadata, Plot, ROIs, Epochs, and Export. The Metadata tab groups Recording, Microscope, and Outputs details. Recording selections from the Load tab always open the full movie. Actions that require loaded recordings sit below the Loaded Recordings pane: Reload saved analysis and Open Group Matching stay disabled until at least one recording is loaded. Open Group Matching opens the separate group-analysis setup window. The loaded-recordings list can unload only the selected recording or Unload all loaded recordings at once. Loading an already loaded recording selects the existing row instead of adding a duplicate entry. Load manually accepts multiple selected folders and loads each one into the same viewer. Search database opens a separate side-by-side window with user, cell type, sensor, stimulus, and date filters on the left and results on the right; the date field accepts `YYYY-MM-DD`-style input with separators such as `/`, `\`, `.`, `_`, or `-` and normalizes it to `YYYY-MM-DD`. Results browse as user, cell type, sensor, stimulus, date, then source-folder experiment time; loading any hierarchy node or selected set of result rows loads all recordings under those rows, closes the search window, and reports failed paths in a separate scrollable error dialog when needed. Search setup or database-query failures appear in a separate error dialog instead of looking like empty results.

The Group Matching window starts with FOV Assignment only. It shows one mean-image card per loaded recording, each with its own contrast slider so fields can be compared under useful display ranges. Select cards that share a field of view, use New FOV from selected or Assign selected to FOV, and use Select none to clear the current card selection without changing saved assignments. Save FOV groups writes `fov_groups.csv` and keeps the FOV view open. Save and continue to ROI assignment writes the same FOV table, then switches to the ROI Assignment view, where the FOV filter contains only assigned FOV ids and defaults to the first FOV. The ROI view is visual and self-contained: every loaded recording card has its own ROI selector, selected-ROI overlay on the mean image, compact response plots for that ROI, Add to group, and Save unmatched actions, so ROI matching does not require changing the active recording or selected label in the main napari window. After adding ROIs from at least two recordings, Save as same cell appends rows to `roi_matches.csv` with one `group_cell_id` shared by the selected ROIs. Save unmatched saves a reviewed singleton with `status=unmatched`. Both tables are plain CSV so later group-analysis code can audit human-authored FOV and ROI assignments without depending on napari.

When photodiode-aligned stimulus epochs are available, twopy adds a compact bottom trial timeline beside the movie frame workflow. Blocks show epoch windows across the full recording, gray/grey/interleave epochs render as neutral gray, a thin cursor follows the current movie frame, clicking or dragging the rail seeks the movie stack, and the viewer HUD reports the current trial and epoch without adding another large panel.

## ROI Editing

Napari ROI editing is crop-native. twopy displays the alignment-valid crop, then converts drawn Labels ROIs back to full-frame movie coordinates before saving or analysis. Response updates and saved analysis require the active Labels layer to match the displayed crop. The ROIs tab can hide ROIs from plots and remove the currently selected ROI rows from the editable Labels layer. Grid ROIs can be created in pixels or microns; micron mode prefills known calibration fields from converted metadata, but still leaves unavailable measured fields unselected instead of falling back to a different mode. Watershed mode segments bright structures from the converted mean image, while response watershed segments repeatable stimulus-locked pixel responses from photodiode-aligned epoch windows. Response watershed fills holes by default and exposes an opt-in closing radius for tiny same-basin gaps.

ROI masks are saved in twopy HDF5 format and stay independent from napari. Existing ROIs reload automatically when `rois.h5` is beside the converted recording.

## Response Plots

The response dock can reload existing `analysis_outputs.h5` plus the saved `rois.h5`, or update plots from the current Labels layer. The dock has response-trace and heatmap tabs. Plots share one y-axis across epochs and, when gray interleave frames are available, show two seconds before stimulus onset and two seconds after offset. Saved trials include `time_seconds`, so plots use recorded response times instead of inferred array indices.

Heatmaps are movie-level response summaries and do not depend on any ROI existing. For each photodiode-aligned epoch window, twopy averages a local pre-epoch baseline image and the response frames inside that epoch, then computes signed dF/F as `(response - baseline) / max(abs(baseline), foreground_floor)`. The foreground floor is derived from the mean-image foreground percentile and also masks dim background pixels before response scoring, so low-signal background cannot dominate by division through a near-zero baseline. Motion-artifact frames are masked before trial averaging. Repeated trials for the same epoch are averaged into one epoch map, then all epoch maps are persisted in `response_heatmaps.h5` after one shared absolute normalization; `response_scale` in that file records the original dF/F magnitude represented by persisted value `1`.

The Plot tab exposes two heatmap computation modes. Pixel mode computes one dF/F value per foreground pixel and defaults to Gaussian sigma `2`, using NaN-aware smoothing so masked pixels do not behave like zero responses. Window mode averages baseline and response values inside square windows before dF/F, then paints each window response back over its covered pixels; overlapping windows are averaged, and window starts include the trailing crop edge so valid pixels are covered even when stride does not divide the crop size. The quick presets are `3x3 stride 1`, `4x4 stride 2`, and `5x5 stride 2`, with custom size and stride capped to valid crop dimensions.

Heatmap display uses a signed blue-black-orange overlay on the mean image, with black at zero, blue for inhibition, and orange for activation. Masked/background response pixels render as zero-response overlay so the mean image is not interrupted by holes. Color limits are display-only: twopy uses the 95th percentile of finite absolute response values to ignore outlier pixels, and the Shared limits checkbox chooses whether that percentile is pooled across all epochs or computed per visible epoch. Toggling Shared limits repaints cached heatmaps without recomputing movie dF/F maps. Save ROIs + analysis also writes the current movie-level heatmap data to `response_heatmaps.h5` beside `analysis_outputs.h5`; when analysis caching is enabled, that HDF5 file syncs with the same cache publish logic as ROI and analysis outputs. The Export tab can write the visible heatmaps as PDF and PNG under `exports/response_heatmaps/`.

Response options include response-window, dF/F, normalization, smoothing, low-pass filtering, and correlation QC settings. The Response window section defaults to Auto, which keeps the current two-second pre-stimulus context and adds post-stimulus context when gray interleave frames are available. Turning Auto off enables manual pre and post seconds; when a gray/interleave epoch duration is known, those manual values are capped at that duration. Save ROIs + analysis uses the same response-window values as the live preview. The dF/F section controls background correction, baseline mode, the selected epoch from the recording's epoch names, baseline span, fit mode, and motion masking. Its visible menu labels use concise analysis terms such as global percentile, shared y-stripe P%, ROI y-stripe P%, baseline epoch, no baseline epoch, direct bounded tau, log-linear, and direct bounded tau and amplitude while preserving the stored analysis values in saved outputs. The baseline mode defaults to `baseline epoch`; switching to `no baseline epoch` relabels the epoch selector to `First epoch` and uses that epoch as the start of a continuous no-baseline analysis span. The background menu renders `%` as a subscript in the P% labels. Shared y-stripe P% divides each frame into row stripes, takes a dim percentile in each stripe, and subtracts that frame's stripe background from ROIs by position. ROI y-stripe P% takes rows near each ROI center, excludes all ROI pixels, keeps dim unlabeled pixels by percentile, averages those pixels over time, and subtracts that trace from that ROI only. ROI y-stripe P% needs unlabeled local background pixels, so dense grid ROIs that cover the analysis crop should use shared y-stripe P% instead or leave background gaps. The baseline selector defaults to the first epoch name containing `gray`, `grey`, or `interleave`, and falls back to epoch 1 only when names give no baseline hint. The Normalization section can divide all grouped responses by each ROI's peak mean response in a selected epoch, so that selected epoch peaks at 1 for every ROI; it defaults to the first non-baseline epoch when recording names make that clear. Smoothing and low-pass filters run on continuous dF/F before trial grouping. Normalization runs after trial grouping and before correlation QC, and saved outputs store the selected epoch plus the per-ROI scale factors used. Correlation filtering scores grouped trials, stores the selected settings plus QC scores in the analysis HDF5 output, and unchecks/dims ROIs that do not pass the filter in the ROIs tab and Labels layer. Its optional window stop defaults to the shortest plotted epoch duration, so enabling it starts with an epoch-bounded QC window. Processing menus use readable labels such as moving average, epoch mean, and epoch peak. Smoothing supports moving-average and Savitzky-Golay methods; Savitzky-Golay defaults to a seven-frame window with polynomial order two.

## Saved Outputs

Save ROIs + analysis writes `rois.h5` beside the current local recording plus:

- `analysis_outputs.h5`
- `response_heatmaps.h5`
- `exports/csvs/response_summary_trials.csv`
- `exports/csvs/response_summary_grouped.csv`

Export actions write recording views, ROI views, overlays, per-epoch response plots, response heatmaps, and paired ROI-overlay/response figures under `exports/`.

When `analysis_caching: true`, Save ROIs + analysis saves locally first and then syncs the changed ROI, analysis HDF5, response heatmap HDF5, and response-summary CSV files to the configured `analysis_output` destination in a background worker. The Metadata tab reports local save, sync success, or sync failure.

## Architecture Note

Napari is a thin adapter over converted twopy files. It displays the mean image, optional movie preview, editable ROI Labels layer, a top response-plot dock, and one tabbed right-side `twopy` dock for loading recordings, selecting loaded recordings, opening manual group matching, setting response options, and saving ROIs with analysis outputs. Database search uses the core typed database helpers for querying, grouping, and source-path resolution; the Qt dialog only renders filters and results. ROI saving uses the core ROI HDF5 helpers; manual group matching writes core CSV tables; response plotting calls the core analysis workflow. Napari code does not read source MATLAB/TIFF files or own analysis decisions.
