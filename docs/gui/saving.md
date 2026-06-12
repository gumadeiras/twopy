# Saving and exporting

The **Export** tab on the right `twopy` sidebar holds every save and export action.

## Save ROIs + analysis

Click **Save ROIs + analysis** to save the active recording's ROIs and the full response analysis. twopy writes the following files beside the current local converted recording:

- `rois.h5` — the ROI masks.
- `analysis_outputs.h5` — the ROI masks used for the analysis, dF/F traces, grouped responses, the response window / dF/F / smoothing / normalization / QC settings, per-ROI scale factors, and QC scores.
- `response_heatmaps.h5` — the heatmap data shown in the Heatmaps tab.
- `exports/csvs/response_summary_trials.csv` — per-trial flat table.
- `exports/csvs/response_summary_grouped.csv` — per-epoch grouped table.

The save uses the same response window, smoothing, and normalization values you see in the [Plot tab](plots.md).

### Cache sync

With `analysis_caching: true` (the default), twopy saves locally first and then copies the converted HDF5 files and changed analysis files to the final output folder in a background worker. The final output folder is normally `analysis_output`; if you manually load a source recording outside `data_paths`, twopy copies files to that recording's `twopy/` folder, and if you manually load a converted folder outside `data_paths`, twopy saves directly in that selected converted folder. The **Metadata** tab shows the full local and final output paths plus whether the copy succeeded or failed. When export PDFs and PNGs are copied, twopy deletes the local cache copies after the destination copy succeeds.

## Image and plot exports

| Button | Writes to | What it saves |
| --- | --- | --- |
| **Save recording view** | `exports/` | The current recording view as an image |
| **Save ROI view** | `exports/` | The ROI Labels layer view |
| **Save recording ROI overlay** | `exports/plots_with_rois/` | Recording with the ROI overlay |
| **Save plots** | `exports/plots/` | Per-epoch response plot figures |
| **Save plots with ROIs** | `exports/plots_with_rois/` | Paired ROI-overlay + response figures |
| **Save heatmaps** | `exports/response_heatmaps/` | The visible heatmaps as PDF + PNG |

All export folders live beside the converted recording. Export PDFs and PNGs follow the same local-save-then-copy flow as the saved analysis files when caching is on.

## Reload after editing elsewhere

If a teammate updated `analysis_outputs.h5` on the network, you can pull it back into the open view with **Reload saved analysis** on the [Load tab](loading.md). It also rereads the saved ROIs.
