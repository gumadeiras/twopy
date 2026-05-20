# Response plots and heatmaps

The top **twopy responses** dock has two inner tabs: **Responses** (ROI traces) and **Heatmaps** (movie-level pixel responses). All the controls that feed them live in the right **Plot** tab.

If the responses dock is collapsed, drag the `...` handle above the viewer down to give the plots room.

## Trace plots (Responses tab)

The trace tab shows one panel per epoch and one trace per active ROI. Plots share one y-axis across epochs. When gray / interleave frames are available, each panel includes two seconds before stimulus onset and two seconds after offset. A thin top rail marks the photodiode-aligned epoch span without changing the dF/F scale. Saved trials carry `time_seconds`, so plots use recorded times instead of inferred indices.

### Response window

In the **Plot** tab → **Response window** section:

- **Auto** (default) keeps the two-second pre-stimulus context and adds matching post-stimulus context when gray interleave frames exist.
- Turn Auto off to set **pre** and **post seconds** manually. When the gray / interleave epoch duration is known, manual values are capped at that duration.

The same window values are used by **Save ROIs + analysis** so the saved plots match what you see live.

### dF/F

In **Plot** tab → **dF/F** section:

- **Background correction**: `none`, `global percentile`, `shared y-stripe P%`, or `ROI y-stripe P%`. The y-stripe variants are designed for dense axon / dendrite fields. ROI y-stripe needs unlabeled background near each ROI, so dense grids should use **shared y-stripe P%** or leave background gaps.
- **Baseline mode**: defaults to `baseline epoch`. Switching to `no baseline epoch` relabels the epoch selector to **First epoch** and fits a continuous span starting from that epoch.
- **Baseline epoch**: defaults to the first epoch name containing `gray`, `grey`, or `interleave`. Falls back to epoch 1 when no name hints at a baseline.
- **Fit mode**: `direct bounded tau`, `log-linear`, or `direct bounded tau and amplitude`. Direct bounded tau is the default.
- **Motion masking** — masks motion-artifact frames out of trial averaging.

The labels you see are the friendly names; the saved analysis file stores the canonical analysis values.

### Smoothing

- **Method**: `none`, `moving average`, or `savgol`.
- For `savgol`, set **Window** to an odd frame count (default 7) and **Order** to the polynomial order (default 2).

Smoothing runs on continuous dF/F before trials are grouped.

### Low-pass filter

Optional Butterworth low-pass. Runs alongside smoothing on continuous dF/F before grouping.

### Normalization

Optional epoch-peak normalization. After trial grouping, every grouped response is divided by the per-ROI peak mean response in the selected epoch — so that epoch peaks at 1 for every ROI. Defaults to the first non-baseline epoch when names make that clear. The selected epoch and per-ROI scale factors are saved with the analysis.

### Correlation QC

Scores grouped trials and stores the settings plus per-ROI scores in the analysis HDF5. ROIs that fail the filter are unchecked / dimmed in the ROIs tab and Labels layer. The optional window stop defaults to the shortest plotted epoch duration, so enabling it starts within a sensible epoch window.

## Heatmaps tab

The Heatmaps tab paints a signed blue-black-orange response overlay on the mean image: black is no response, blue is suppression, orange is activation. Heatmaps are movie-level — they do not need any ROIs.

For each photodiode-aligned epoch window, twopy averages a local pre-epoch baseline image and the response frames, then computes signed dF/F. The mean-image foreground percentile both masks dim background and sets the dF/F denominator floor, so low-signal background pixels cannot blow up by dividing through near-zero baseline. Motion-artifact frames are masked first. Repeated trials of the same epoch are averaged into one epoch map; all maps are saved with one shared absolute normalization, and `response_scale` in `response_heatmaps.h5` records the original dF/F magnitude that maps to value 1.

### Heatmap modes (Plot tab → Response Map options)

- **pixel** — one dF/F value per foreground pixel. **Pixel smoothing** is a Gaussian sigma applied with NaN-aware smoothing so masked pixels do not behave like zero responses. Default sigma is 2.
- **window** — averages baseline and response inside square windows before dF/F, then paints each window value back over its pixels. Overlapping windows are averaged. Window starts include the trailing crop edge so valid pixels are covered even when the stride does not divide the crop size.
- **Presets** for window mode: `3x3 stride 1`, `4x4 stride 2`, `5x5 stride 2`, or **custom**. Custom **size** and **stride** are capped to valid crop dimensions.

### Color limits

Color limits are display-only. twopy uses the 95th percentile of finite absolute responses to ignore outlier pixels. The **Shared limits** checkbox controls whether that percentile is pooled across all epochs or computed per visible epoch; toggling it repaints cached heatmaps without recomputing.

Heatmap data is saved by **Save ROIs + analysis** into `response_heatmaps.h5`. The Export tab can write the visible heatmaps as PDF and PNG under `exports/response_heatmaps/`.
