# Getting started

This page takes you from a blank machine to a saved analysis. It should take about ten minutes.

## 1. Install twopy

The examples use [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html). You can use any conda-compatible tool.

```sh
micromamba create -n twopy -c conda-forge python=3.13 pip -y
micromamba run -n twopy python -m pip install twopy
```

Check that the package imported cleanly:

```sh
micromamba run -n twopy python -c 'import twopy; print(twopy.__version__)'
```

## 2. Create your config file

Create an editable config template:

```sh
twopy config setup
```

twopy prints the path that it wrote. The default path on macOS and Linux is `~/.config/twopy/config.yml`. On Windows, the path is under `%APPDATA%\twopy\config.yml`. If the config does not exist, twopy creates the template and prints its path. Then, twopy stops so you can edit it.

Check which config twopy will use and validate it:

```sh
twopy config
```

The main keys you usually edit:

- `database_path` — folder holding the lab SQLite database files.
- `data_paths` — list of folders that contain microscope recordings. twopy checks the folders in order. It uses the first folder that contains the recording.
- `analysis_output` — where saved analyses end up. Use `source` to write into each recording's own `twopy/` folder, or a folder path to mirror the recording tree under it.
- `analysis_caching` — keep `true` for normal use. twopy converts and writes locally first, then copies converted HDF5 files and saved analysis files to `analysis_output`.
- `analysis_cache_max_gb` — maximum local analysis-cache size. The default is `33`. After writes, twopy removes old entries only if their files are in the final output folder.

`config.yml` is private to your machine and is never committed. For source checkouts, a local `./config.yml` still works and takes precedence over the user config file.

## 3. Launch the app

```sh
micromamba activate twopy
twopy
```

`twopy` checks that the config is valid before opening napari. This check does not require lab data folders or network drives to be mounted. If you launch `twopy` before a config exists, it creates the template, prints the path, and stops so you can edit it.

Or open a recording directly:

```sh
twopy /path/to/source/recording
twopy /path/to/recording_data.h5
```

You can pass a raw microscope folder, an already-converted folder, or a `recording_data.h5` file. If the recording has not been converted yet, twopy converts it first.

## 4. Open your first recording

The right side of napari has a tabbed dock called **twopy**. The first tab, **Load**, has three ways to pick a recording:

- **Search database** — opens a side-by-side filter window (user, cell type, sensor, stimulus, date) over your lab database. See [Loading recordings](gui/loading.md).
- **Load manually** — pick one or more source folders, converted folders, or `recording_data.h5` files.
- **Load CSV list** — reopen a list you saved earlier with **Save loaded list**.

When the recording loads, napari shows the mean image, an optional aligned movie, and an editable `rois` Labels layer.

## 5. Draw an ROI and watch the response

1. Click the `rois` layer in the layer list on the left.
2. Pick a label number and use the paint tool to draw a cell.
3. The top **twopy responses** dock updates as soon as your edit is committed.

To generate ROIs, use the **ROIs** tab. It can make a pixel grid, a micron grid, or watershed segments from the mean image. See [Draw and generate ROIs](gui/rois.md).

## 6. Save the analysis

Switch to the **Export** tab and click **Save ROIs + analysis**. twopy writes:

- `rois.h5` — the ROI masks.
- `analysis_outputs.h5` — dF/F traces, grouped responses, QC scores, and your current processing settings.
- `response_heatmaps.h5` — movie-level response heatmaps.
- `exports/csvs/response_summary_trials.csv` and `response_summary_grouped.csv` — flat per-trial and per-epoch tables.

With `analysis_caching: true`, these files and the converted HDF5 files sync to `analysis_output` in the background. The **Metadata** tab shows whether the sync succeeded.

## Where to go next

- [GUI guide](gui/index.md) — every tab and window in the app, with what each button does.
- [Python API guide](python_api.md) — script the same workflow from Python.
- [Writing custom workflows](writing_custom_workflows.md) — add your own analysis to the **Custom** tab.
