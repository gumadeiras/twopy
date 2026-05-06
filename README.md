# twopy

Two-photon imaging analysis tool with a napari interface.

## Setup

```sh
micromamba env create -f environment.yml
micromamba run -n twopy pre-commit install
cp config.example.yml config.yml
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
    detect_recording_photodiode_events,
    extract_roi_traces,
    frame_windows_from_photodiode_alignment,
    load_converted_recording,
    make_roi_set,
    split_traces_by_frame_windows,
)

recording = load_converted_recording(Path("/path/to/recording_data.h5"))
mask_array = np.zeros((1, *recording.movie.shape[1:]), dtype=bool)
mask_array[0, :10, :10] = True
roi_set = make_roi_set(mask_array)
traces = extract_roi_traces(recording, roi_set)
alignment = detect_recording_photodiode_events(recording)
windows = frame_windows_from_photodiode_alignment(
    alignment,
    frame_count=recording.movie.shape[0],
)
responses = split_traces_by_frame_windows(traces, windows)
```

Analysis starts from converted HDF5 files. ROI masks are GUI-independent, trace
extraction streams movie chunks, and response windows come from photodiode
events instead of nominal frame-rate assumptions.
