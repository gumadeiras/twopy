# Python API Guide

twopy scripts should work from converted HDF5 files. Source microscope folders are conversion inputs; analysis starts from twopy-owned outputs.

## Find Recordings

```python
from twopy import find_recordings

recordings = find_recordings(
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

`config.yml` controls whether DB queries use mounted files directly or cached local copies. The default is `database_access: copy` because database searches over the network can be slow, while copying the DB file locally is usually fast.

## Convert Recording

```python
from pathlib import Path

from twopy import convert_recording_to_twopy

recording = Path("/path/to/recording")

converted = convert_recording_to_twopy(recording)
print(converted.path)
print(converted.movie_path)
```

Conversion writes `recording_data.h5` for metadata, stimulus tables, photodiode signals, and the mean image. The large aligned movie is written separately to `aligned_movie.h5`. By default the mean image uses the full movie; pass `mean_start_frame` and `mean_stop_frame` to use a frame range.

By default, conversion writes to the configured analysis work directory. With `analysis_caching: true` this is the local `analysis_cache_dir`, mirrored by recording path under the matched ordered `data_paths` root for normal lab recordings and placed under `_external` for recordings outside `data_paths`; with `analysis_caching: false` it is the configured `analysis_output`. Use `analysis_output: source` to publish saved analysis outputs into `recording/twopy`; use a path to mirror the recording directory structure under that output root. Pass `output_dir` only when overriding routing for a specific call.

## Analyze Converted Data

```python
from pathlib import Path

import numpy as np

from twopy import (
    compute_roi_delta_f_over_f,
    extract_background_corrected_roi_traces,
    load_converted_recording,
    make_roi_set,
    recording_frame_rate_hz,
    resolve_recording_timing,
    select_baseline_frame_windows,
)

recording = load_converted_recording(Path("/path/to/recording_data.h5"))
mask_array = np.zeros((1, *recording.movie.shape[1:]), dtype=bool)
mask_array[0, :10, :10] = True
roi_set = make_roi_set(mask_array)
traces = extract_background_corrected_roi_traces(
    recording,
    roi_set,
    method="movie_global_percentile",
)
timing = resolve_recording_timing(recording)
epoch_windows = timing.epoch_windows
baseline_windows = select_baseline_frame_windows(
    epoch_windows,
    epoch_name="Gray Interleave",
)
dff = compute_roi_delta_f_over_f(
    traces,
    baseline_windows,
    data_rate_hz=recording_frame_rate_hz(recording),
    fit_mode="direct_bounded_tau",
)
```

ROI masks are GUI-independent and full-frame. Trace extraction streams movie chunks and uses the saved alignment-valid crop by default. Pass `spatial_domain="full_frame"` only when you need explicit full-frame extraction. The lower-level `extract_roi_traces` helper is the full-frame raw primitive. For dense axon/dendrite process fields, `method="movie_y_stripe_percentile"` estimates a low-percentile background separately for each frame and y-stripe, then subtracts the stripe background from ROIs by position. `method="roi_y_stripe_percentile"` takes rows near each ROI center, excludes all ROI pixels, keeps dim unlabeled pixels by percentile, averages those pixels over time, and subtracts that trace from that ROI only. It needs unlabeled local background pixels, so dense grid ROIs that cover the analysis crop should use `method="movie_y_stripe_percentile"` instead or leave background gaps.

Use native ROI extraction helpers when scripts should create masks from a mean image instead of drawing or loading them:

```python
from twopy import (
    grid_roi_set,
    grid_roi_set_microns,
    load_pixel_calibrations,
    resolve_pixel_size_um,
    extract_response_watershed_rois,
    watershed_roi_set,
)
from twopy.config import load_config

accepted_region = np.zeros(recording.movie.shape[1:], dtype=bool)
crop = recording.alignment_valid_crop
accepted_region[crop.axis0_start : crop.axis0_stop, crop.axis1_start : crop.axis1_stop] = True
grid_rois = grid_roi_set(recording.movie.shape[1:], grid_size_pixels=12)
config = load_config()
calibrations = load_pixel_calibrations(config.pixel_calibration_path)
pixel_size = resolve_pixel_size_um(
    calibrations,
    rig="day",
    mode=2,
    scanner="galvo",
    zoom=10,
)
physical_grid_rois = grid_roi_set_microns(
    recording.movie.shape[1:],
    micron_grid_size=10,
    pixel_size_um=pixel_size.pixel_size_um,
)
watershed_rois = watershed_roi_set(
    recording.mean_image,
    region_mask=accepted_region,
    min_pixels=5,
)
response_watershed = extract_response_watershed_rois(
    recording,
    epoch_windows,
    epoch_numbers=(2, 3),
    min_pixels=20,
    fill_holes=True,
    closing_radius=0,
)
```

`grid_roi_set` makes deterministic square template ROIs. `grid_roi_set_microns` uses calibrated microns per pixel and converts a physical grid width with `floor(micron_grid_size / pixel_size_um)`. Pixel-size calibration is loaded from a dated CSV registry and resolved by exact rig/mode/scanner/zoom match when available, interpolation within the same rig/mode/scanner group when the requested zoom is inside the measured range, and extrapolation only when explicitly allowed. `watershed_roi_set` segments bright structures from a two-dimensional summary image. `extract_response_watershed_rois` segments selected stimulus windows instead: it builds per-pixel response-amplitude, local response-coherence, and split-half reliability maps, combines them into a score image, then watersheds and trims that score image into full-frame ROI masks. Response-watershed masks are split into connected components after trimming, fill holes by default, and can apply opt-in conservative binary closing for tiny same-basin gaps. It returns both the `RoiSet` and audit score images; use `response_watershed_roi_set` when scripts only need the masks. These ROI helpers accept optional spatial or region restrictions depending on the method, but extraction itself is GUI-independent.

For comparison against historical psycho5 ROI extraction, use `twopy.parity` helpers such as `psycho5_grid_roi_label_image` and `psycho5_watershed_image_from_preseg`. Those helpers preserve psycho5-specific label ordering and watershed border-fill behavior for audits; normal twopy analysis should use the native ROI extraction helpers above.

Movie-level response heatmaps use converted movies directly and do not require ROIs. They use the same photodiode-aligned epoch windows as ROI response analysis, compute local baseline-vs-response dF/F images, and persist normalized signed maps plus the original dF/F divisor for audit:

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
    epoch_windows=epoch_windows,
    options=ResponseMapOptions(
        mode="pixel",
        pixel_smoothing_sigma=2.0,
    ),
)
save_response_map_data(Path("/path/to/response_heatmaps.h5"), pixel_maps)
reloaded_maps = load_response_map_data(Path("/path/to/response_heatmaps.h5"))

window_maps = compute_recording_response_maps(
    recording,
    epoch_windows=epoch_windows,
    options=ResponseMapOptions(
        mode="window",
        window_size_pixels=4,
        window_stride_pixels=2,
    ),
)
```

Response maps use the following data flow:

- Pixel mode computes one signed dF/F value per foreground pixel, with optional NaN-aware Gaussian smoothing.
- Window mode averages baseline and response intensity inside each square window before dF/F, paints that value back over the covered pixels, and averages overlapping windows.
- The mean-image foreground percentile both masks dim background pixels and sets the dF/F denominator floor, preventing near-zero baseline pixels from creating artificial hot spots.
- Saved epoch maps are scaled by the largest absolute finite response across all epochs; multiply `epoch.response_values` by `map_data.response_scale` to recover original dF/F units.
- Napari display and exports use a separate robust 95th-percentile color limit, optionally shared across epochs, without changing saved heatmap values.

Random-noise temporal kernels can be fit from the same computed response object. The fitter keeps complete regular stimulus streams for selected non-baseline epochs, groups selected epochs by unique epoch name, uses photodiode-aligned frame windows for sparse ROI response times, and maps raw left/right kernels to ipsi/contra from the recording hemisphere stored in converted metadata:

```python
from twopy import (
    StimulusKernelOptions,
    fit_recording_stimulus_kernels,
    recording_hemisphere,
)

computation = compute_recording_responses(recording, roi_set)
print(recording_hemisphere(recording))
kernels = fit_recording_stimulus_kernels(
    computation,
    StimulusKernelOptions(
        stimulus_modality="olfaction",
        num_stim_past=150,
        num_stim_future=25,
        method="ols",
    ),
)
print(kernels.time_seconds)
print(kernels.epoch_names)
print(kernels.ipsilateral.shape)
```

The kernel arrays are shaped `(epoch_names, rois, lags)`. The default `stimulus_specific_05` column is selected from `stimulus_modality`. For `stimulus_modality="olfaction"`, it matches the random-noise workflow's `antenna_stim` value after conversion. LED recordings encode `0=left`, `1=both`, `2=right`, and `3=blank`; twopy derives raw left and raw right streams from that single activation column before mapping them to ipsi/contra. For `stimulus_modality="vision"`, the same default column is interpreted as signed visual contrast, such as the Matulis full-field contrast-flicker workflow's `-0.2/+0.2` and `-0.9/+0.9` values, and the result is stored in `kernels.contrast` without hemisphere mapping. Pass `stimulus_column="..."` only to override the modality default, and pass `hemisphere="left"` or `hemisphere="right"` only to override audited olfactory metadata. `kernels.fitted_stimulus_segment_counts` and `kernels.skipped_irregular_stimulus_segment_counts` report how many selected segments were used or skipped because their stimulus sample times were too irregular for fixed-lag fitting. Negative kernel times are future stimulus samples and are useful for timing QC. Current native fitting is frame-aligned; converted files do not yet store per-ROI line timing maps.

Stimulus epoch windows come from `resolve_recording_timing(...)`, not nominal frame-rate assumptions. Native timing prefers classified photodiode boundary evidence when the converted stimulus table has one active `photodiode_flash` segment for each epoch boundary plus stimulus start and end, uses interpolation for recordings without flash rows or with older flash-train rows, and rejects incomplete boundary-flash evidence. `timing.source` and `timing.metadata` keep the chosen path auditable. ROI dF/F uses corrected fluorescence plus baseline windows to fit one shared exponential tau and one amplitude per ROI. When scripts do not pass explicit baseline windows or a baseline selector, twopy defaults to the first epoch name containing `gray`, `grey`, or `interleave`, then falls back to epoch 1. For stimuli without a distinct baseline epoch, pass `baseline_mode="no_baseline_epoch"` plus the first epoch number to include in the baseline fit; twopy fits over one continuous span from that epoch through later epochs. The default dF/F fit mode is `direct_bounded_tau`; use `log_linear` for a log-space linear fit, or `direct_bounded_tau_and_log_amplitude` when both tau and log-amplitude should be bounded.

Scripts and napari can pass `ResponseProcessingOptions` for post-dF/F response processing:

- Smoothing and low-pass filters run on continuous dF/F before trial grouping.
- Epoch-peak normalization runs after trial grouping and divides every grouped response by each ROI's peak mean response in the selected epoch. The selected epoch and per-ROI scale factors are saved in `analysis_outputs.h5`.
- Correlation filtering scores grouped trials and saves the selected settings plus QC scores in `analysis_outputs.h5`.
- Use `validate_grouped_roi_responses` when a script constructs grouped response objects directly; processing, persistence, and CSV exports call the same validator before trusting time, frame, and ROI axes.

Manual FOV groups and cross-recording ROI matches are stored as plain CSV tables. Napari's Group Matching window writes the same formats, and scripts can read or extend them through the core API:

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
    {
        Path("/recordings/first"): "roi_0004",
        Path("/recordings/second"): "roi_0012",
    },
    group_cell_id=next_group_cell_id(existing_rows),
    fov_group_id="fov_1",
)
append_manual_roi_match_rows(match_path, rows)
```

Rows that share a `group_cell_id` and `status="matched"` are the same visually assigned cell across recordings. Reviewed singletons can be saved with `status="unmatched"` so downstream analysis can distinguish unreviewed ROIs from ROIs the user intentionally left unmatched.

## Open Napari From Python

```python
from pathlib import Path

from twopy import (
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
roi_set = save_napari_label_rois(label_image, Path("/path/to/rois.h5"))
```

Pass `roi_set=Path("/path/to/rois.h5")` to reopen existing ROIs.

The napari ROIs tab can also create editable grid ROIs directly. Pixel grids need only a pixel width. Micron grids use the configured pixel calibration registry, auto-fill zoom from converted `acq.zoomFactor` metadata when present, and auto-select rig/mode/scanner only when converted metadata plus the tracked ScanImage config mapping identify one measured calibration group. Pixel and micron settings are shown only for the selected unit; micron grids allow calibration extrapolation by default.
