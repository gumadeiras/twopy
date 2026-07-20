# Saving and exporting

The **Export** tab on the right `twopy` sidebar holds every save and export action.

## Save ROIs + analysis

Click **Save ROIs + analysis** to save the active recording's ROIs and the full response analysis. twopy writes the following files beside the current local converted recording:

- `rois.h5` — the ROI masks.
- `analysis_outputs.h5` — the ROI masks, dF/F traces, grouped responses, per-ROI scale factors, and QC scores. It also contains the response window, dF/F, smoothing, normalization, and QC settings.
- `response_heatmaps.h5` — the heatmap data shown in the Heatmaps tab.
- `exports/csvs/response_summary_trials.csv` — per-trial flat table.
- `exports/csvs/response_summary_grouped.csv` — per-epoch grouped table.

The save uses the same response window, smoothing, and normalization values you see in the [Plot tab](plots.md).

### Cache sync

With `analysis_caching: true`, twopy saves locally first. This is the default setting. A background worker copies changed analysis files and converted HDF5 files to the final output folder. This folder is usually `analysis_output`. A manual source load outside `data_paths` uses the recording `twopy/` folder. A manual converted load outside `data_paths` uses the selected folder. The **Metadata** tab shows the local path, final path, and copy result. After a successful PDF or PNG copy, twopy deletes the local cache copy.

## Image and plot exports

| Button | Writes to | What it saves |
| --- | --- | --- |
| **Save recording view** | `exports/` | The current recording view as an image |
| **Save ROI view** | `exports/` | The ROI Labels layer view |
| **Save recording ROI overlay** | `exports/recording_roi_overlay/` | Recording with the ROI overlay |
| **Save plots** | `exports/plots/` | Per-epoch response plot figures |
| **Save plots with ROIs** | `exports/plots_with_rois/` | Paired ROI-overlay + response figures |
| **Save heatmaps** | `exports/response_heatmaps/` | The visible heatmaps as PDF + PNG |

All export folders live beside the converted recording. Export PDFs and PNGs follow the same local-save-then-copy flow as the saved analysis files when caching is on.

## Reload after editing elsewhere

If a teammate updated `analysis_outputs.h5` on the network, you can pull it back into the open view with **Reload saved analysis** on the [Load tab](loading.md). It also rereads the saved ROIs.
