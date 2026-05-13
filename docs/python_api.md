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

By default, conversion writes to the configured analysis work directory. With `analysis_caching: true` this is the local `analysis_cache_dir`, mirrored by recording path under `data_path` for normal lab recordings and placed under `_external` for recordings outside `data_path`; with `analysis_caching: false` it is the configured `analysis_output`. Use `analysis_output: source` to publish saved analysis outputs into `recording/twopy`; use a path to mirror the recording directory structure under that output root. Pass `output_dir` only when overriding routing for a specific call.

## Analyze Converted Data

```python
from pathlib import Path

import numpy as np

from twopy import (
    classify_recording_photodiode_events,
    compute_roi_delta_f_over_f,
    detect_recording_photodiode_events,
    extract_background_corrected_roi_traces,
    load_converted_recording,
    make_roi_set,
    map_stimulus_epochs_to_frame_windows,
    recording_frame_rate_hz,
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
alignment = detect_recording_photodiode_events(recording)
timing = classify_recording_photodiode_events(recording, alignment)
epoch_windows = map_stimulus_epochs_to_frame_windows(recording, alignment)
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

ROI masks are GUI-independent and full-frame. Trace extraction streams movie chunks and uses the saved alignment-valid crop by default. Pass `spatial_domain="full_frame"` only when you need explicit full-frame extraction. The lower-level `extract_roi_traces` helper is the full-frame raw primitive. For dense axon/dendrite process fields, `method="movie_y_stripe_percentile"` estimates a low-percentile background separately for each frame and y-stripe, then subtracts the stripe background from ROIs by position. `method="roi_y_stripe_percentile"` takes rows near each ROI center, excludes ROI pixels, keeps dim pixels by percentile, averages those pixels over time, and subtracts that trace from that ROI only.

Use native ROI extraction helpers when scripts should create masks from a mean image instead of drawing or loading them:

```python
from twopy import (
    grid_roi_set,
    grid_roi_set_microns,
    load_pixel_calibrations,
    resolve_pixel_size_um,
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
```

`grid_roi_set` makes deterministic square template ROIs. `grid_roi_set_microns` uses calibrated microns per pixel and converts a physical grid width with `floor(micron_grid_size / pixel_size_um)`. Pixel-size calibration is loaded from a dated CSV registry and resolved by exact rig/mode/scanner/zoom match when available, interpolation within the same rig/mode/scanner group when the requested zoom is inside the measured range, and extrapolation only when explicitly allowed. `watershed_roi_set` segments bright structures from a two-dimensional summary image. These ROI helpers accept an optional boolean region mask and keep ROIs whose center of mass falls inside that region; UI code may collect that mask, but extraction itself is GUI-independent.

For comparison against historical psycho5 ROI extraction, use `twopy.parity` helpers such as `psycho5_grid_roi_label_image` and `psycho5_watershed_image_from_preseg`. Those helpers preserve psycho5-specific label ordering and watershed border-fill behavior for audits; normal twopy analysis should use the native ROI extraction helpers above.

Stimulus epoch windows come from classified photodiode events, not nominal frame-rate assumptions. `timing.events` keeps the start, transition, and end classifications auditable. ROI dF/F uses corrected fluorescence plus baseline windows to fit one shared exponential tau and one amplitude per ROI. The default dF/F fit mode is `direct_bounded_tau`; use `log_linear` for a log-space linear fit, or `direct_bounded_tau_and_log_amplitude` when both tau and log-amplitude should be bounded.

Scripts and napari can pass `ResponseProcessingOptions` for post-dF/F response processing. Smoothing and low-pass filters run on continuous dF/F before trial grouping. Epoch-peak normalization runs after trial grouping and divides every grouped response by each ROI's peak mean response in the selected epoch, with the selected normalization epoch and per-ROI scale factors saved in the analysis HDF5 output. Correlation filtering scores grouped trials and stores the selected settings plus QC scores in the analysis HDF5 output. Use `validate_grouped_roi_responses` when a script constructs grouped response objects directly; processing, persistence, and CSV exports call the same validator before trusting time, frame, and ROI axes.

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
