# twopy

Two-photon imaging analysis tool with a napari interface.

## Setup

```sh
micromamba env create -f environment.yml
micromamba activate twopy
micromamba run -n twopy pre-commit install
cp config.example.yml config.yml
```

The environment installs twopy as an editable package, so the `twopy` terminal
command is available after activating the environment. If the environment
already existed before the command was added, refresh the editable install:

```sh
micromamba run -n twopy python -m pip install -e .
```

Edit `config.yml` after copying it. The example file describes every field in
plain language. `config.yml` stays local to your machine and is not tracked by
git.

## Check

```sh
micromamba run -n twopy pre-commit run --all-files
```

The installed pre-commit hook runs ruff, ty, and the unit tests before each
commit.

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

`config.yml` controls whether DB queries use mounted files directly or cached
local copies. The default is `database_access: copy` because database searches
over the network can be slow, while copying the DB file locally is usually fast.

## Convert Recording

```python
from pathlib import Path

from twopy import convert_recording_to_twopy

recording = Path("/path/to/recording")

converted = convert_recording_to_twopy(recording)
print(converted.path)
print(converted.movie_path)
```

Conversion writes `recording_data.h5` for metadata, stimulus tables,
photodiode signals, and the mean image. The large aligned movie is written
separately to `aligned_movie.h5`. By default the mean image uses the full movie;
pass `mean_start_frame` and `mean_stop_frame` to use a frame range. By default,
conversion writes to the location configured by `analysis_output`; pass
`output_dir` only when you want to override that for a specific call.

`config.yml` also controls analysis output routing. `analysis_output: source`
writes into `recording/twopy`; a path mirrors the recording directory structure
under that output root.

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
    select_epoch_frame_windows,
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
interleave_windows = select_epoch_frame_windows(
    epoch_windows,
    epoch_name="Gray Interleave",
)
dff = compute_roi_delta_f_over_f(
    traces,
    interleave_windows,
    data_rate_hz=float(recording.acquisition_metadata["acq.frameRate"]),
    fit_mode="robust",
)
```

Analysis starts from converted HDF5 files. ROI masks are GUI-independent, trace
extraction streams movie chunks, background correction stays explicit, and
stimulus epoch windows come from classified photodiode events instead of
nominal frame-rate assumptions. `timing.events` keeps the start, transition,
and end classifications auditable. Analysis trace extraction uses the saved
alignment-valid spatial crop by default, including `method="none"` when you
want uncorrected traces through the analysis path. ROI masks remain full-frame;
pass `spatial_domain="full_frame"` only when that is the intended audit path.
The lower-level `extract_roi_traces` helper is a full-frame raw primitive. ROI
dF/F uses corrected ROI fluorescence plus gray interleave windows to fit one
shared exponential tau and one amplitude per ROI. The default dF/F fit mode is
`robust`; pass `fit_mode="source_bounds"` when you need original source-bound
behavior for audit comparisons.

## Open In Napari

From a converted output directory that contains `recording_data.h5`, or from a
source recording directory that contains `twopy/recording_data.h5`:

```sh
twopy
```

Or pass the converted recording explicitly:

```sh
twopy /path/to/recording_data.h5
```

By default the launcher opens the mean image, the first 200 movie frames, an
editable `rois` Labels layer, and the `twopy` dock panel. Use `--no-movie` to
skip the movie preview, or `--movie-start` and `--movie-stop` to choose a
different preview range.

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

Napari code is a thin adapter. It loads converted twopy files, displays the
mean image, optionally displays a bounded movie preview, creates an editable
ROI Labels layer, and adds a small magicgui `twopy` dock panel with a
`Save ROIs` button. The button saves the current Labels layer through the core
ROI HDF5 helpers. Pass `roi_set=Path("/path/to/rois.h5")` when reopening
existing ROIs. It does not read source MATLAB/TIFF files or own analysis
decisions.
