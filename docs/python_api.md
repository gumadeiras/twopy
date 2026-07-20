# Python API guide

Scripting twopy from Python is a good fit when you want batch analyses, custom plots, or to feed twopy outputs into other code.

Two rules to keep in mind:

1. **Analysis starts from converted HDF5 files.** Source MATLAB/TIFF folders are conversion inputs only. The conversion step turns one into the other.
2. **ROI masks are always full-frame.** Trace extraction handles the alignment-valid crop for you.

Everything below imports from the top-level `twopy` package. All names are re-exported from `__all__` so the imports stay short.

## A quick end-to-end script

```python
from pathlib import Path

import numpy as np

from twopy import (
    analyze_recording_responses,
    convert_recording_to_twopy,
    find_recordings,
    grid_roi_set,
    load_converted_recording,
)

experiments = find_recordings(year=2023, month=10, day=17, genotype="gh146")
converted = convert_recording_to_twopy(experiments[0].session_dir)

recording = load_converted_recording(converted.path)
rois = grid_roi_set(recording.movie.shape[1:], grid_size_pixels=12)
run = analyze_recording_responses(recording, rois)
print(run.grouped_responses.epoch_names)
```

`analyze_recording_responses` runs the same pipeline the app runs when you click **Save ROIs + analysis** — dF/F with sensible defaults, trial grouping, summaries.

## Find recordings

```python
from twopy import find_recordings

experiments = find_recordings(
    year=2023,
    month=10,
    day=17,
    genotype="gh146",
    stimulus="combo_stim",
    sensor="g6f",
    cell_type="ALPN",
    hemisphere="right",
    person="Gustavo",
)
```

Every filter is optional. The function loads twopy config automatically so you do not need to pass database paths. Run `twopy config setup` once before using config-backed APIs, or set `TWOPY_CONFIG` when a script needs a specific file.

`config.yml` controls whether queries read the shared database file directly or copy it locally. The default is `database_access: copy` because network DB queries can be slow, while copying the file is usually fast. Pass `database_access="direct"` to override.

## Convert a recording

```python
from pathlib import Path

from twopy import convert_recording_to_twopy

converted = convert_recording_to_twopy(Path("/path/to/source/recording"))
print(converted.path)         # recording_data.h5
print(converted.movie_path)   # aligned_movie.h5
```

Conversion writes `recording_data.h5` for metadata, stimulus tables, photodiode signals, and the mean image. It writes the large movie array to `aligned_movie.h5`. By default, the mean image uses the full movie. Set `mean_start_frame` and `mean_stop_frame` to use a frame range.

By default, twopy gets the output folder from `config.yml`. With `analysis_caching: true`, conversion writes files to the local `analysis_cache_dir` mirror. The mirror uses the matched `data_paths` root. Paths outside `data_paths` use `_external`. `analysis_cache_max_gb` sets a default cache limit of 33 GB. After writes and copies, twopy removes an old entry only if its files exist in the final folder. With `analysis_caching: false`, conversion writes directly to `analysis_output`. The napari load workflow copies cached converted files to `analysis_output`. Python conversion returns the `recording_data.h5` and `aligned_movie.h5` paths. It can also update the cache inventory and remove eligible old entries. Use `output_dir=...` only when one call must use a different folder. This explicit path also permits conversion without config output rules.

## Load a converted recording

```python
from pathlib import Path

from twopy import load_converted_recording, recording_frame_rate_hz, recording_hemisphere

recording = load_converted_recording(Path("/path/to/recording_data.h5"))
print(recording.movie.shape)
print(recording_frame_rate_hz(recording))
print(recording_hemisphere(recording))
```

`recording.movie` streams from `aligned_movie.h5`. twopy loads `recording.mean_image`, `recording.stimulus_table`, and `recording.photodiode` immediately.

## Make ROIs

You can build ROIs from a hand-built mask, load them from disk, or generate them from the mean image.

### From a mask

```python
import numpy as np

from twopy import make_roi_set

mask_array = np.zeros((1, *recording.movie.shape[1:]), dtype=bool)
mask_array[0, :10, :10] = True
rois = make_roi_set(mask_array)
```

`mask_array` is shaped `(n_rois, height, width)` in the same Python image order as the converted movie, with `True` where each ROI's pixels are.

### From a saved file

```python
from pathlib import Path

from twopy import load_roi_set, save_roi_set

rois = load_roi_set(Path("/path/to/rois.h5"))
# ...edit / replace...
save_roi_set(rois, Path("/path/to/rois.h5"))
```

### Generate from the mean image

```python
from twopy import (
    extract_response_watershed_rois,
    grid_roi_set,
    grid_roi_set_microns,
    load_pixel_calibrations,
    resolve_pixel_size_um,
    watershed_roi_set,
)
from twopy.config import load_config

# Pixel grid
grid = grid_roi_set(recording.movie.shape[1:], grid_size_pixels=12)

# Micron grid (uses pixel calibration registry)
config = load_config()
calibrations = load_pixel_calibrations(config.pixel_calibration_path)
pixel_size = resolve_pixel_size_um(calibrations, rig="day", mode=2, scanner="galvo", zoom=10)
micron_grid = grid_roi_set_microns(
    recording.movie.shape[1:],
    micron_grid_size=10,
    pixel_size_um=pixel_size.pixel_size_um,
)

# Watershed segmentation of the mean image
watershed = watershed_roi_set(recording.mean_image, min_pixels=5)

# Response-driven watershed (uses photodiode-aligned epoch windows)
timing = resolve_recording_timing(recording)
response_watershed = extract_response_watershed_rois(
    recording,
    timing.epoch_windows,
    epoch_numbers=(2, 3),
    min_pixels=20,
    fill_holes=True,
)
```

`grid_roi_set_microns` changes a physical width to pixels with `floor(micron_grid_size / pixel_size_um)`. Pixel-size calibration comes from a dated CSV registry. twopy first uses an exact rig, mode, scanner, and zoom match. If necessary, it interpolates inside the measured range of the same group. It extrapolates only when you permit it.

`extract_response_watershed_rois` makes response-amplitude, local response-coherence, and split-half reliability maps for each pixel. It combines them into a score image. Then, it makes watershed segments and trims them into full-frame ROI masks. It returns the `RoiSet` and audit score images. Use `response_watershed_roi_set` when you only need the masks.

## Extract traces and dF/F

```python
from twopy import (
    compute_roi_delta_f_over_f,
    extract_background_corrected_roi_traces,
    recording_frame_rate_hz,
    resolve_recording_timing,
    select_baseline_frame_windows,
)

traces = extract_background_corrected_roi_traces(
    recording,
    rois,
    method="movie_global_percentile",
)
timing = resolve_recording_timing(recording)
baseline_windows = select_baseline_frame_windows(timing.epoch_windows, epoch_name="Gray Interleave")
dff = compute_roi_delta_f_over_f(
    traces,
    baseline_windows,
    data_rate_hz=recording_frame_rate_hz(recording),
    fit_mode="direct_bounded_tau",
)
```

A few details worth knowing:

- Trace extraction streams the movie in chunks and uses the saved alignment-valid crop by default. Set `spatial_domain="full_frame"` only when you need full-frame extraction. `extract_roi_traces` is the low-level function for raw full-frame extraction.
- For dense axon or dendrite fields, `method="movie_y_stripe_percentile"` estimates a low-percentile background for each frame and y-stripe. Then, it subtracts the background by position. `method="roi_y_stripe_percentile"` uses displayed rows near each ROI center and excludes ROI pixels. It needs dim, unlabeled local background pixels. Bright cells, processes, or stimulus bleedthrough can make it subtract more than the ROI baseline. If this occurs, twopy stops before dF/F. A zero or negative corrected baseline can cause very large artificial responses. Use `movie_y_stripe_percentile`, `movie_global_percentile`, or dim local background gaps around each ROI.
- The dF/F fit defaults to `direct_bounded_tau`. Use `log_linear` for a log-space linear fit, or `direct_bounded_tau_and_log_amplitude` when both tau and log-amplitude should be bounded.
- When you do not pass `baseline_windows`, twopy picks the first epoch name containing `gray`, `grey`, or `interleave`, falling back to epoch 1. For stimuli with no distinct baseline epoch, pass `baseline_mode="no_baseline_epoch"` plus the first epoch number to include and twopy fits one continuous span.

## Timing comes from the photodiode

```python
timing = resolve_recording_timing(recording)
print(timing.source)
print(timing.epoch_windows)
```

Native timing uses classified photodiode boundaries when evidence is complete. The stimulus table must have one active `photodiode_flash` segment for each boundary, start, and end. twopy uses interpolation for recordings without flash rows or with old flash-train rows. It rejects incomplete boundary evidence. `timing.source` and `timing.metadata` record the selected method for audits.

## Save analysis outputs

```python
from pathlib import Path

from twopy import analyze_recording_responses, save_analysis_outputs

run = analyze_recording_responses(recording, rois)
save_analysis_outputs(Path("/path/to/analysis_outputs.h5"), run)
```

`analyze_recording_responses` is the single-call equivalent of **Save ROIs + analysis**. It runs background correction, dF/F, trial grouping, and optional processing. Use `ResponseProcessingOptions(...)` to set smoothing, low-pass filtering, response-size normalization, or correlation QC. The saved HDF5 file contains the processing settings.

For finer control, `compute_recording_responses(recording, rois, options=...)` returns the same computation object without writing it.

A few invariants you can rely on:

- Smoothing and low-pass filters run on continuous dF/F **before** trial grouping.
- Response-size normalization runs **after** trial grouping. The selected epoch and per-ROI scale factors are saved with the outputs.
- Correlation filtering scores grouped trials and saves the settings plus per-ROI scores.
- Use `finite_mean_and_sem(values, axis=...)` for the same finite-sample mean / sample-SEM convention used by twopy's response plots and CSV exports.
- Call `validate_grouped_roi_responses(...)` when you build grouped response objects manually. Processing, persistence, and CSV exports use the same check before they use the time, frame, and ROI axes.

## Run a custom workflow from Python

Use the same custom workflows from scripts when you want the analysis from the **Custom** tab without opening napari. Custom workflow APIs live under `twopy.custom` because workflow files use that package too.

```python
from pathlib import Path

from twopy import load_converted_recording, load_roi_set
from twopy.custom import discover_custom_workflows, run_custom_workflow

recording = load_converted_recording(Path("/path/to/recording_data.h5"))
rois = load_roi_set(Path("/path/to/rois.h5"))
workflow = next(
    workflow
    for workflow in discover_custom_workflows((Path("/path/to/workflows"),)).workflows
    if workflow.id == "direction-selectivity"
)
run = run_custom_workflow(
    workflow,
    recording,
    roi_set=rois,
    params={
        "preferred_epoch": "2: Right",
        "null_epoch": "3: Left",
        "metric": "peak",
    },
)
print(run.result.message)
print(run.output_dir)
```

The runner builds the same `CustomRunContext` that the GUI uses. It fills omitted parameters with workflow defaults and validates the returned `CustomResult`. It writes workflow source information for returned files. Then, it returns tables, plots, ROI updates, and response-plot data to your script.

## Movie-level response heatmaps

Heatmaps do not require ROIs. They compute one response image per epoch:

```python
from pathlib import Path

from twopy import (
    ResponseMapOptions,
    compute_recording_response_maps,
    load_response_map_data,
    save_response_map_data,
)

pixel_maps = compute_recording_response_maps(
    recording,
    epoch_windows=timing.epoch_windows,
    options=ResponseMapOptions(mode="pixel", pixel_smoothing_sigma=2.0),
)
save_response_map_data(Path("/path/to/response_heatmaps.h5"), pixel_maps)
reloaded = load_response_map_data(Path("/path/to/response_heatmaps.h5"))

window_maps = compute_recording_response_maps(
    recording,
    epoch_windows=timing.epoch_windows,
    options=ResponseMapOptions(mode="window", window_size_pixels=4, window_stride_pixels=2),
)
```

How the data is built:

- **Pixel mode** computes one signed dF/F value per foreground pixel with optional NaN-aware Gaussian smoothing.
- **Window mode** averages baseline and response intensity in each square window before dF/F. It puts that value on the covered pixels. Then, it averages overlapping windows.
- The mean-image foreground percentile both masks dim background and sets the dF/F denominator floor, so near-zero baseline pixels cannot create artificial hot spots.
- Saved epoch maps are scaled by the largest absolute finite response across all epochs. Multiply `epoch.response_values` by `map_data.response_scale` to recover original dF/F units.
- Saved heatmap images use the same Python image order as converted movies and ROI masks.
- Display (napari and exports) uses a separate robust 95th-percentile color limit, optionally shared across epochs, without changing the saved values.

## Random-noise temporal kernels

The kernel fitter uses the same computed-response object. It keeps complete, regular stimulus streams for selected non-baseline epochs. It groups the streams by unique name. It uses photodiode-aligned frame windows for sparse ROI response times. The recording hemisphere maps raw left and right kernels to ipsi and contra.

```python
from twopy import (
    StimulusKernelOptions,
    compute_recording_responses,
    fit_recording_stimulus_kernels,
    recording_hemisphere,
)

computation = compute_recording_responses(recording, rois)
kernels = fit_recording_stimulus_kernels(
    computation,
    StimulusKernelOptions(stimulus_modality="olfaction", num_stim_past=100, num_stim_future=10, method="ols"),
)
print(kernels.time_seconds)
print(kernels.epoch_names)
print(kernels.ipsilateral.shape)        # (epoch_names, rois, lags)
```

Modality-specific notes:

- `stimulus_modality` selects the default `stimulus_specific_05` column. For `olfaction`, it matches the converted `antenna_stim` value of the random-noise workflow. LED recordings use `0=left`, `1=both`, `2=right`, and `3=blank`. twopy makes raw left and right streams before ipsi/contra mapping. For `vision`, the same column contains signed visual contrast, such as `-0.2/+0.2`. twopy stores it in `kernels.contrast` without hemisphere mapping.
- Pass `stimulus_column="..."` only to override the modality default. Pass `hemisphere="left"` or `hemisphere="right"` only to override audited olfactory metadata.
- `kernels.fitted_stimulus_segment_counts` and `kernels.skipped_irregular_stimulus_segment_counts` report how many selected segments were used or skipped because their sample times were too irregular for fixed-lag fitting.
- Negative kernel times are future stimulus samples. You can use them for timing QC. Native fitting aligns with frames. Converted HDF5 files do not contain per-ROI line timing maps.

## Manual FOV and ROI match tables

The manual FOV groups and cross-recording ROI matches the napari Group Matching window writes are plain CSV. Scripts can read and extend them through the same helpers:

```python
from pathlib import Path

from twopy import (
    append_manual_roi_match_rows,
    load_manual_fov_group_rows,
    load_manual_roi_match_rows,
    make_manual_fov_group_rows,
    make_manual_roi_match_rows,
    next_group_cell_id,
    save_manual_fov_group_rows,
)

fov_path = Path("/path/to/fov_groups.csv")
fov_rows = make_manual_fov_group_rows(
    {
        Path("/recordings/first"): "fov_1",
        Path("/recordings/second"): "fov_1",
    },
)
save_manual_fov_group_rows(fov_rows, fov_path)
loaded_fov_rows = load_manual_fov_group_rows(fov_path)

match_path = Path("/path/to/roi_matches.csv")
existing_rows = load_manual_roi_match_rows(match_path) if match_path.exists() else ()
rows = make_manual_roi_match_rows(
    (
        (Path("/recordings/first"), "roi_0004"),
        (Path("/recordings/second"), "roi_0012"),
    ),
    group_cell_id=next_group_cell_id(existing_rows),
    fov_group_id="fov_1",
)
append_manual_roi_match_rows(match_path, rows)
```

Rows sharing a `group_cell_id` with `status="matched"` are the same visually assigned cell across recordings. A recording can appear more than once in the same group when you select multiple separate ROIs from that recording. Save reviewed singletons with `status="unmatched"` so downstream analysis can tell unreviewed ROIs apart from ROIs you intentionally left unmatched.

## Open napari from Python

```python
from pathlib import Path

from twopy.napari import (
    launch_napari,
    open_recording_in_napari,
    roi_label_image_from_layer,
    save_napari_label_rois,
)

launch_napari(Path("/path/to/recording_data.h5"))

view = open_recording_in_napari(
    Path("/path/to/recording_data.h5"),
    movie_frame_range=(0, 200),
)

# After drawing or editing the rois Labels layer:
label_image = roi_label_image_from_layer(view.roi_labels_layer)
rois = save_napari_label_rois(label_image, Path("/path/to/rois.h5"))
```

Pass `roi_set=Path("/path/to/rois.h5")` to `open_recording_in_napari` to reopen existing ROIs with the recording.

Run `twopy config setup` before opening napari controls from Python. The controls use `config.yml` for output routing, pixel calibration, custom workflow folders, and database-backed loading. If you only need raw napari layers for a converted recording, pass `add_controls=False` to `open_recording_in_napari(...)`.

## psycho5 parity helpers

For audits against historical psycho5 ROI extraction, use `twopy.parity` helpers such as `psycho5_grid_roi_label_image` and `psycho5_watershed_image_from_preseg`. They preserve psycho5-specific label ordering and watershed border-fill behavior. Normal twopy analysis should use the native helpers above.
