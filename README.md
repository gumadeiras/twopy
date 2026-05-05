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
    database_access="copy",
)
```
