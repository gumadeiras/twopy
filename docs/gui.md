# GUI Guide

Use the `twopy` command to open napari with the twopy docks.

```sh
twopy
```

Choose a source recording folder or `recording_data.h5` from the `twopy` dock. If the selected source folder has not been converted yet, twopy converts it first. If no recording is selected, napari still opens.

Open a recording directly:

```sh
twopy /path/to/source/recording
twopy /path/to/recording_data.h5
```

## Workflow

1. Load a source recording folder or converted `recording_data.h5`.
2. Draw or edit ROIs in the `rois` Labels layer.
3. Click Save ROIs to write `rois.h5`.
4. Update response plots from the current Labels layer.
5. Save analysis and exports when the plots look right.

By default the launcher opens the mean image, full movie, editable `rois` Labels layer, loading dock, Save ROIs dock, and response-plot dock. Use `--no-movie` to skip the movie preview, or `--movie-start` and `--movie-end` to choose a preview range.

## ROI Editing

Napari ROI editing is crop-native. twopy displays the alignment-valid crop, then converts drawn Labels ROIs back to full-frame movie coordinates before saving or analysis. Save ROIs and Update Responses require the active Labels layer to match the displayed crop.

ROI masks are saved in twopy HDF5 format and stay independent from napari. Existing ROIs reload automatically when `rois.h5` is beside the converted recording.

## Response Plots

The response dock can reload existing `analysis_outputs.h5` or update plots from the current Labels layer. Plots share one y-axis across epochs and, when gray interleave frames are available, show two seconds before stimulus onset and two seconds after offset. Saved trials include `time_seconds`, so plots use recorded response times instead of inferred array indices.

Response options include dF/F, smoothing, low-pass filtering, and correlation QC settings. The dF/F section controls background correction, the baseline epoch selected from the recording's epoch names, interleave span, fit mode, and motion masking. Its visible menu labels use concise analysis terms such as global percentile, shared y-stripe P%, ROI y-stripe P%, robust, and log-amplitude bounded while preserving the stored analysis values in saved outputs. The background menu renders `%` as a subscript in the P% labels. Shared y-stripe P% divides each frame into row stripes, takes a dim percentile in each stripe, and subtracts that frame's stripe background from ROIs by position. ROI y-stripe P% takes rows near each ROI center, excludes ROI pixels, keeps dim pixels by percentile, averages those pixels over time, and subtracts that trace from that ROI only. The baseline selector defaults to the first epoch name containing `gray`, `grey`, or `interleave`, and falls back to epoch 1 only when names give no baseline hint. Smoothing and low-pass filters run on continuous dF/F before trial grouping. Correlation filtering scores grouped trials and stores the selected settings plus QC scores in the analysis HDF5 output. Processing menus use readable labels such as moving average, epoch mean, and epoch peak. Smoothing supports moving-average and Savitzky-Golay methods; Savitzky-Golay defaults to a seven-frame window with polynomial order two.

## Saved Outputs

Save ROIs writes `rois.h5` beside the current recording. Saving analysis also writes:

- `analysis_outputs.h5`
- `exports/csvs/response_summary_trials.csv`
- `exports/csvs/response_summary_grouped.csv`

Export actions write recording views, ROI views, overlays, per-epoch response plots, and paired ROI-overlay/response figures under `exports/`.

## Architecture Note

Napari is a thin adapter over converted twopy files. It displays the mean image, optional movie preview, editable ROI Labels layer, and small docks for loading, saving ROIs, and plotting responses. ROI saving uses the core ROI HDF5 helpers; response plotting calls the core analysis workflow. Napari code does not read source MATLAB/TIFF files or own analysis decisions.
