# Loading recordings

The **Load** tab is the first tab on the right `twopy` sidebar. You can also give a path on the command line. Load actions prepare the selected recordings in a background worker. The Qt thread adds napari layers when each recording is ready. A progress dialog shows the load queue and a batch progress bar. It also shows the active source root, genotype, stimulus, date, and time. **Reading from** shows the converted folder and its route. The route is `local cache`, `source output`, `analysis output`, or `selected output`. **Cancel pending** stops queued recordings after the current recording is complete.

## From the command line

```sh
twopy                                  # open empty, then pick from the Load tab
twopy /path/to/source/recording        # source folder (twopy converts it first)
twopy /path/to/converted/folder        # already-converted folder
twopy /path/to/recording_data.h5       # direct converted file
```

Useful flags:

- `--roi-file-to-load /path/to/rois.h5` — reopen an existing ROI file with the recording.
- `--movie-start N`, `--movie-end N` — preview only a frame range.
- `--no-movie` — skip the movie layer and load only the mean image.

## Search database

Opens a side-by-side window: filters on the left, results on the right.

Filters: **user**, **cell type**, **sensor**, **stimulus**, **date**. The date field accepts `YYYY-MM-DD` with `/`, `\`, `.`, `_`, or `-` as separators and normalizes the value before searching.

Buttons: **Search** runs the query. **Load selected** loads everything under the selected rows. Results are grouped by user, cell type, sensor, stimulus, date, then source-folder experiment time — selecting a parent loads all children. Failed paths show up in a separate scrollable error dialog so a few bad rows do not block the rest.

**Favorites** (lower left): Click **Save as favorite...** to save the current filters. Drag saved favorites into the required order. Click **Use** to apply one. Click **Edit...** to change its name or filters. Click **Remove** to delete the selected favorite. twopy keeps favorites in a local YAML file.

## Load manually

Pick one or more source folders, converted folders, or `recording_data.h5` files. They all open into the same viewer. Loading a folder you already have open just selects the existing row instead of duplicating it.

## Load CSV list

Reopen a session list that you wrote with **Save loaded list**. If `recording_data_path` is valid, twopy opens that converted HDF5 file. If it is not valid, twopy uses the source path in `recording_path`. [Group Matching](group_matching.md) uses the CSV folder for its default `fov_groups.csv` and `roi_matches.csv` files. If these files exist, twopy loads their rows. This picker remembers its last folder separately from **Load manually**. Source recordings and saved CSV lists usually use different folders.

## The Loaded Recordings pane

Each loaded recording is in a row below the load buttons. Click a row to make that recording active. The plot dock and **ROIs** tab use the active recording. twopy also selects its `rois` Labels layer. The paint label changes to the first unused ROI number. Thus, manual edits make a new ROI on the same recording.

The **Metadata** tab updates for the active recording. Its **Motion** section shows crop size, removed border pixels, total movement, and high-motion frame count. It also shows movement over time. New converted recordings show left-right movement in blue and up-down movement in yellow. Older files show only total movement. The tab tells you to reconvert them for separate x/y pixel movement.

Buttons under the list (disabled until at least one recording is loaded):

- **Save loaded list** — Write a CSV of source paths for the loaded recordings. When possible, `recording_data_path` points to the final HDF5 file in `analysis_output`.
- **Reload saved analysis** — reread `analysis_outputs.h5` and `rois.h5` for the active recording.
- **Reconvert selected** — Confirm the action. twopy repeats conversion in the same folder and replaces `recording_data.h5` and `aligned_movie.h5`. If local caching is on, twopy copies these files to the final output folder. Then, it reloads the row. It does not change ROI or analysis output files.
- **Open Group Matching** — opens the separate [Group Matching](group_matching.md) window.
- **Unload selected** — remove the selected row from the viewer.
- **Unload all** — clear every loaded recording at once.

## Caching and where files end up

With `analysis_caching: true`, twopy converts and reads data in the local `analysis_cache_dir` mirror. This is the default setting. All plots and analysis use the local cache. For recordings under `data_paths`, twopy copies converted files and saved outputs to `analysis_output` in the background. Manual source loads use the source `twopy/` folder. Manual converted loads use the selected folder. `analysis_cache_max_gb` sets the cache limit to 33 GB by default. After writes, twopy removes an old entry only if its final files and source recording are available.

If a source path is temporarily unavailable, twopy can open its cache entry. The **Save loaded list** CSV always records the source path. Thus, the list works when the network volume is available again. If a final output path exists, `recording_data_path` does not contain a local cache path. CSV reload uses a valid `recording_data_path` first. Thus, converted files open without another source-path search.
