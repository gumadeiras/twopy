# GUI Guide

Use the `twopy` command to open napari with the twopy docks.

```sh
twopy
```

Use Search Database on the Load tab to find recordings from `config.yml`, or click Load manually to choose a source recording folder or `recording_data.h5`. With the default `analysis_caching: true`, twopy converts or copies converted files into the local `analysis_cache_dir` first, then all plotting and ROI analysis reads from that local cache. If no recording is selected, napari still opens.

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

By default the launcher opens the mean image, full movie, editable `rois` Labels layer, a compact top response-plot dock, and one tabbed right-side `twopy` dock. The first tab is Load, followed by Metadata, Plot, ROIs, Epochs, and Export. Recording selections from the Load tab always open the full movie, and the Load tab also exposes Reload saved analysis. The loaded-recordings list can unload only the selected recording or unload all loaded recordings at once. Search Database opens a separate side-by-side window with user, cell type, sensor, stimulus, and date filters on the left and results on the right. Results browse as user, cell type, sensor, stimulus, date, then source-folder experiment time; loading any hierarchy node loads all recordings under that node, closes the search window, and reports failed paths in a separate scrollable error dialog when needed. Search setup or database-query failures appear in a separate error dialog instead of looking like empty results.

When photodiode-aligned stimulus epochs are available, twopy adds a compact bottom trial timeline beside the movie frame workflow. Blocks show epoch windows across the full recording, gray/grey/interleave epochs render as neutral gray, a thin cursor follows the current movie frame, clicking or dragging the rail seeks the movie stack, and the viewer HUD reports the current trial and epoch without adding another large panel.

## ROI Editing

Napari ROI editing is crop-native. twopy displays the alignment-valid crop, then converts drawn Labels ROIs back to full-frame movie coordinates before saving or analysis. Response updates and saved analysis require the active Labels layer to match the displayed crop. The ROIs tab can hide ROIs from plots and remove the currently selected ROI rows from the editable Labels layer. Grid ROIs can be created in pixels or microns; micron mode prefills known calibration fields from converted metadata, but still leaves unavailable measured fields unselected instead of falling back to a different mode. Watershed mode segments bright structures from the converted mean image, while response watershed segments repeatable stimulus-locked pixel responses from photodiode-aligned epoch windows. Response watershed fills holes by default and exposes an opt-in closing radius for tiny same-basin gaps.

ROI masks are saved in twopy HDF5 format and stay independent from napari. Existing ROIs reload automatically when `rois.h5` is beside the converted recording.

## Response Plots

The response dock can reload existing `analysis_outputs.h5` or update plots from the current Labels layer. Plots share one y-axis across epochs and, when gray interleave frames are available, show two seconds before stimulus onset and two seconds after offset. Saved trials include `time_seconds`, so plots use recorded response times instead of inferred array indices.

Response options include response-window, dF/F, normalization, smoothing, low-pass filtering, and correlation QC settings. The Response window section defaults to Auto, which keeps the current two-second pre-stimulus context and adds post-stimulus context when gray interleave frames are available. Turning Auto off enables manual pre and post seconds; when a gray/interleave epoch duration is known, those manual values are capped at that duration. Save Analysis uses the same response-window values as the live preview. The dF/F section controls background correction, the baseline epoch selected from the recording's epoch names, baseline span, fit mode, and motion masking. Its visible menu labels use concise analysis terms such as global percentile, shared y-stripe P%, ROI y-stripe P%, direct bounded tau, log-linear, and direct bounded tau and amplitude while preserving the stored analysis values in saved outputs. The background menu renders `%` as a subscript in the P% labels. Shared y-stripe P% divides each frame into row stripes, takes a dim percentile in each stripe, and subtracts that frame's stripe background from ROIs by position. ROI y-stripe P% takes rows near each ROI center, excludes all ROI pixels, keeps dim unlabeled pixels by percentile, averages those pixels over time, and subtracts that trace from that ROI only. ROI y-stripe P% needs unlabeled local background pixels, so dense grid ROIs that cover the analysis crop should use shared y-stripe P% instead or leave background gaps. The baseline selector defaults to the first epoch name containing `gray`, `grey`, or `interleave`, and falls back to epoch 1 only when names give no baseline hint. The Normalization section can divide all grouped responses by each ROI's peak mean response in a selected epoch, so that selected epoch peaks at 1 for every ROI; it defaults to the first non-baseline epoch when recording names make that clear. Smoothing and low-pass filters run on continuous dF/F before trial grouping. Normalization runs after trial grouping and before correlation QC, and saved outputs store the selected epoch plus the per-ROI scale factors used. Correlation filtering scores grouped trials and stores the selected settings plus QC scores in the analysis HDF5 output. Processing menus use readable labels such as moving average, epoch mean, and epoch peak. Smoothing supports moving-average and Savitzky-Golay methods; Savitzky-Golay defaults to a seven-frame window with polynomial order two.

## Saved Outputs

Save ROIs + analysis writes `rois.h5` beside the current local recording plus:

- `analysis_outputs.h5`
- `exports/csvs/response_summary_trials.csv`
- `exports/csvs/response_summary_grouped.csv`

Export actions write recording views, ROI views, overlays, per-epoch response plots, and paired ROI-overlay/response figures under `exports/`.

When `analysis_caching: true`, Save ROIs + analysis saves locally first and then syncs the changed ROI, analysis HDF5, and response-summary CSV files to the configured `analysis_output` destination in a background worker. The Metadata tab reports local save, sync success, or sync failure.

## Architecture Note

Napari is a thin adapter over converted twopy files. It displays the mean image, optional movie preview, editable ROI Labels layer, a top response-plot dock, and one tabbed right-side `twopy` dock for loading recordings, selecting loaded recordings, setting response options, and saving ROIs with analysis outputs. Database search uses the core typed database helpers for querying, grouping, and source-path resolution; the Qt dialog only renders filters and results. ROI saving uses the core ROI HDF5 helpers; response plotting calls the core analysis workflow. Napari code does not read source MATLAB/TIFF files or own analysis decisions.
