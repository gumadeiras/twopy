# twopy

Two-photon imaging analysis tool with a napari interface.

## Setup

```sh
micromamba env create -f environment.yml
micromamba run -n twopy python -m pip install -e ".[dev,gui]"
cp config.example.yml config.yml
```

Edit `config.yml` after copying it. The example file describes every field in
plain language. `config.yml` stays local to your machine and is not tracked by
git.

## Check

```sh
micromamba run -n twopy python -m ruff check .
micromamba run -n twopy python -m ruff format --check .
micromamba run -n twopy python -m mypy
micromamba run -n twopy python -m unittest discover -s tests
```

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
output_dir = recording / "twopy"

converted = convert_recording_to_twopy(recording, output_dir)
print(converted.path)
print(converted.movie_path)
```

Conversion writes `twopy_recording.h5` for metadata, stimulus tables,
photodiode signals, and the mean image. The large aligned movie is written
separately to `aligned_movie.h5`. By default the mean image uses the full movie;
pass `mean_start_frame` and `mean_stop_frame` to use a frame range.

`config.yml` also controls analysis output routing. `analysis_output: source`
writes into `recording/twopy`; a path mirrors the recording directory structure
under that output root.
