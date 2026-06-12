# twopy

Two-photon imaging analysis tool for the Clark Lab output format.

![twopy napari interface](https://raw.githubusercontent.com/gumadeiras/twopy/main/docs/assets/twopy-napari.png)

## Getting Started

twopy lets you open two-photon recordings, draw ROIs, plot responses in real time, process and analyze them, and save them.

When you first load a recording, twopy converts it to HDF5 files that it owns. Those files include the aligned movie, mean image, stimulus tables, photodiode signals, and recording metadata. Analysis and the GUI both work from the converted files, so the original source files remain separate from twopy's outputs. By default, converted recordings and interactive analysis outputs are cached locally under `~/.cache/twopy/recordings` for speed; saving ROIs and analysis copies the changed output files back to the configured `analysis_output` location in the background.

## Install

Examples use micromamba, but any conda-compatible environment manager should work.


```sh
micromamba create -n twopy -c conda-forge python=3.13 pip -y
micromamba run -n twopy python -m pip install twopy
```

Verify the install:

```sh
micromamba run -n twopy python -c 'import twopy; print(twopy.__name__)'
```

Create the config template in your user folder:

```sh
twopy config setup
```

twopy prints the config path it wrote. On macOS and Linux the default is `~/.config/twopy/config.yml`; on Windows it is under `%APPDATA%\twopy\config.yml`. Edit that file with your database, data, and output paths before launching twopy.

Validate the active config and show its path and file contents:

```sh
twopy config
```

## Start

```sh
micromamba activate twopy
twopy
```

When a config file exists, `twopy` validates it before opening napari. It does not require lab data folders or network drives to be mounted at launch.

Open a source recording or converted recording directly:

```sh
twopy /path/to/source/recording
twopy /path/to/recording_data.h5
```

Basic GUI flow:

1. Search the database or load a recording manually.
2. Draw or edit ROIs in the `rois` Labels layer.
3. Update response plots from the current ROIs.
4. Click Save ROIs + analysis in Export when the plots look right.

When a source recording has not been converted yet, twopy converts it first. Converted data includes the aligned movie, mean image, stimulus tables, photodiode signals, and recording metadata.

## Docs

- [Full documentation](https://twopy.gumadeiras.com)
- [Getting started](docs/getting_started.md): install, configure, launch, and save your first analysis.
- [GUI guide](docs/gui/index.md): one page per task — loading, ROIs, plots, custom workflows, group matching, saving.
- [Python API guide](docs/python_api.md): find recordings, convert data, extract traces, fit kernels, and open napari from scripts.
- [Writing custom workflows](docs/writing_custom_workflows.md): add your own analysis to the napari Custom tab.
- [Development guide](docs/development.md): source setup and local checks.
- [Input data spec](docs/input_data_spec.md): short recording folder contract.
- [Recording file reference](docs/recording_file_schema.md): detailed source and converted file schema.

## Check

```sh
micromamba run -n twopy pre-commit run --all-files
```
