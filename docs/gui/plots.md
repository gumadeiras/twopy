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

- **Background correction**: `none`, `global percentile`, `shared y-stripe P%`, or `ROI y-stripe P%`. Use the y-stripe methods for dense axon or dendrite fields. ROI y-stripe needs dim, unlabeled background near each ROI. A band with bright structures can subtract more than the ROI baseline. If this occurs, twopy stops before dF/F. A zero or negative corrected baseline can cause very large artificial responses. Use **shared y-stripe P%**, **global percentile**, or dim background gaps around each ROI.
- **Baseline mode**: defaults to `baseline epoch`. Switching to `no baseline epoch` relabels the epoch selector to **First epoch** and fits a continuous span starting from that epoch.
- **Baseline epoch**: defaults to the first epoch name containing `gray`, `grey`, or `interleave`. Falls back to epoch 1 when no name hints at a baseline.
- **Fit mode**: `direct bounded tau`, `log-linear`, or `direct bounded tau and amplitude`. Direct bounded tau is the default.
- **Motion masking** — masks motion-artifact frames out of trial averaging.

The GUI shows readable labels. The saved analysis file stores the canonical analysis values.

### Smoothing

- **Method**: `none`, `moving average`, or `savgol`.
- For `savgol`, set **Window** to an odd frame count (default 7) and **Order** to the polynomial order (default 2).

Smoothing runs on continuous dF/F before trials are grouped.

### Low-pass filter

Optional Butterworth low-pass. Runs alongside smoothing on continuous dF/F before grouping.

### Normalization

Optional response-size normalization. After trial grouping, twopy finds each ROI's strongest average response in the selected epoch, whether it is positive or negative. Every grouped response for that ROI is divided by that response size, so the strongest selected-epoch response has size 1. Defaults to the first non-baseline epoch when names make that clear. The selected epoch and per-ROI scale factors are saved with the analysis.

### Correlation QC

Scores grouped trials and stores the settings plus per-ROI scores in the analysis HDF5. ROIs that fail the filter are unchecked / dimmed in the ROIs tab and Labels layer. The optional window stop defaults to the shortest plotted epoch duration, so enabling it starts within a sensible epoch window.

## Heatmaps tab

The Heatmaps tab paints a signed blue-black-orange response overlay on the mean image: black is no response, blue is suppression, orange is activation. Heatmaps are movie-level — they do not need any ROIs.

For each photodiode-aligned epoch window, twopy averages a local baseline image and the response frames. Then, it calculates signed dF/F. The mean-image foreground percentile masks dim background and sets the minimum dF/F denominator. Thus, a near-zero baseline cannot make low-signal pixels very large. twopy masks motion-artifact frames first. It averages repeated trials of the same epoch into one map. All maps use one shared absolute normalization. In `response_heatmaps.h5`, `response_scale` records the original dF/F magnitude that maps to 1.

### Heatmap modes (Plot tab → Response Map options)

- **pixel** — one dF/F value per foreground pixel. **Pixel smoothing** is a Gaussian sigma applied with NaN-aware smoothing so masked pixels do not behave like zero responses. Default sigma is 2.
- **window** — averages baseline and response inside square windows before dF/F, then paints each window value back over its pixels. Overlapping windows are averaged. Window starts include the trailing crop edge so valid pixels are covered even when the stride does not divide the crop size.
- **Presets** for window mode: `3x3 stride 1`, `4x4 stride 2`, `5x5 stride 2`, or **custom**. Custom **size** and **stride** are capped to valid crop dimensions.

### Color limits

Color limits are display-only. twopy uses the 95th percentile of finite absolute responses to ignore outlier pixels. **Shared limits** calculates that percentile across all epochs or for each visible epoch. A change repaints cached heatmaps without a new calculation.

Heatmap data is saved by **Save ROIs + analysis** into `response_heatmaps.h5`. The Export tab can write the visible heatmaps as PDF and PNG under `exports/response_heatmaps/`.
