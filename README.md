# twopy

Two-photon imaging analysis tool for the Clark Lab output format.

## Getting Started

twopy lets you open two-photon recordings, draw ROIs, plot responses in real time,
process and analyze them, and save them.

When you first load a recording, twopy converts it to a standardized HDF5 format.
The converted format includes the aligned movie, mean image, stimulus tables,
photodiode signals, and  recording metadata. Analysis and the GUI both work from
the converted files, so the original source files remain separate from twopy's outputs.

### Install

Examples use micromamba, but any conda-compatible environment manager should work.

Preferred setup is a fresh Python 3.13 environment with the published PyPI
package:

```sh
micromamba create -n twopy -c conda-forge python=3.13 pip -y && micromamba run -n twopy python -m pip install twopy
```

Verify the install:

```sh
micromamba run -n twopy python -c 'import twopy; print(twopy.__name__)'
```

If you are working from a source checkout, copy `config.example.yml` to
`config.yml`, then edit `config.yml` so the paths match your computer. The
example file explains each setting in plain language. `config.yml` stays local
to your machine and is not tracked by git.

### Start The GUI

Start napari from the twopy environment:

```sh
micromamba activate twopy
twopy
```

Or run it without activating the environment first:

```sh
micromamba run -n twopy twopy
```

You can also open a recording directly:

```sh
twopy /path/to/source/recording
```

Or direct path to converted HDF5 files:

```sh
twopy /path/to/recording_data.h5
```

Inside napari, use the `twopy` panel to choose a recording folder or a
`recording_data.h5` file. If a source recording has not been converted yet,
twopy converts it first, then opens the converted files.

Basic GUI flow:

1. Start twopy.
2. Choose a recording.
3. Draw or edit ROIs in the `rois` Labels layer.
4. Click Save ROIs.
5. Use the response plot panel to update plots from the current ROIs.

## Setup Details

For development, install from the repository so local code edits are used:

```sh
micromamba env create -f environment.yml
micromamba activate twopy
micromamba run -n twopy pre-commit install
```

The development environment installs twopy as an editable package, so the
`twopy` terminal command is available after activating the environment. If the
environment already existed before the command was added, refresh the editable
install:

```sh
micromamba run -n twopy python -m pip install -e .
```

## Check

```sh
micromamba run -n twopy pre-commit run --all-files
```

The installed pre-commit hook runs ruff, ty, and the unit tests before each
commit.

## Release

PyPI publishing uses GitHub Actions Trusted Publishing. No PyPI API token secret
is required.

One-time setup in PyPI:

1. Add a trusted publisher for the `twopy` project.
2. Use owner `gumadeiras`, repository `twopy`, workflow
   `publish-to-pypi.yml`, and environment `pypi`.
3. In GitHub, create the `pypi` environment and require manual approval.

Release flow:

1. Update `project.version` in `pyproject.toml`.
2. Run `micromamba run -n twopy pre-commit run --all-files`.
3. Commit the version change.
4. Create a GitHub release whose tag is the same version, with or without a
   leading `v`.
5. Publish the release.

The release workflow checks that the tag matches `pyproject.toml`, builds the
wheel and source distribution, checks package metadata, and publishes to PyPI
after the `pypi` environment approval.

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

The script and napari workflows also accept `ResponseProcessingOptions` for
post-dF/F response processing. Smoothing and low-pass filters are applied to
continuous dF/F before trial grouping, while correlation filtering scores
grouped trial responses and stores the selected settings plus QC scores in the
analysis HDF5 output. Smoothing supports moving-average and Savitzky-Golay
methods; Savitzky-Golay defaults to a seven-frame window with polynomial order
two.

## Open In Napari

From a converted output directory that contains `recording_data.h5`, or from a
source recording directory:

```sh
twopy
```

If converted files are missing and the selected folder has the expected source
recording files, twopy runs conversion first and then opens the converted HDF5
files. If no recording is found, `twopy` still opens napari. Choose a recording
folder or `recording_data.h5` in the `twopy` dock panel; twopy loads it after
selection.

Or pass a source folder or converted recording explicitly:

```sh
twopy /path/to/source/recording
twopy /path/to/recording_data.h5
```

By default the launcher opens the mean image, the full movie, an editable
`rois` Labels layer, a top response-plot dock, a `twopy` loading dock, and a left
Save ROIs dock. Use `--no-movie` to skip the movie preview, or `--movie-start`
and `--movie-end` to choose a different preview range. Save ROIs writes
`rois.h5` beside the current recording by default. The response dock can reload
existing `analysis_outputs.h5` or update plots from the current Labels layer.
Saving analysis writes `rois.h5`, `analysis_outputs.h5`,
`exports/csvs/response_summary_trials.csv`, and
`exports/csvs/response_summary_grouped.csv` beside the converted recording.
Response plots share one y-axis across epochs and show two seconds before
stimulus onset and two seconds after stimulus offset by default, when gray
interleave frames are available in the grouped responses. Epoch plots are laid
out horizontally. Each saved response trial includes its own `time_seconds`
vector, so plots use direct response time values rather than inferring time from
array indices.

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
ROI Labels layer, and adds small dock widgets for loading folders, saving ROIs,
and plotting responses. ROI saving writes the current Labels layer through the
core ROI HDF5 helpers. Response plotting calls the core analysis workflow when
updating from current ROIs. Pass `roi_set=Path("/path/to/rois.h5")` when
reopening existing ROIs. Napari code does not read source MATLAB/TIFF files or
own analysis decisions.
