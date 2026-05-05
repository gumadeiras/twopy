# twopy

Two-photon imaging analysis tool with a napari interface.

## Setup

```sh
micromamba env create -f environment.yml
micromamba run -n twopy python -m pip install -e ".[dev,gui]"
```

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
local copies. The default is `database_access: copy`.

## Convert Recording

```python
from pathlib import Path

from twopy import convert_recording_to_twopy

recording = Path("/path/to/recording")
output_dir = recording / "twopy"

converted = convert_recording_to_twopy(recording, output_dir)
print(converted.path)
```

Conversion writes `twopy_recording.h5`, including the aligned movie, acquisition
metadata, stimulus tables, photodiode signals, and a mean image. By default the
mean image uses the full movie; pass `mean_start_frame` and `mean_stop_frame` to
use a frame range.

`config.yml` also controls analysis output routing. `analysis_output: source`
writes into `recording/twopy`; a path mirrors the recording directory structure
under that output root.
