# Loading recordings

The **Load** tab is the first tab on the right `twopy` sidebar. You can also pass a path on the command line. Load-tab actions prepare selected recordings in a background worker, show a modeless progress dialog, and apply napari layers on the Qt thread as each recording is ready. **Cancel pending** stops queued recordings after the current recording finishes.

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

**Favorites** (lower left): save the current filter set with **Save as favorite...**, drag saved favorites into the order you want, restore one later with **Use**, change its name or filters with **Edit...**, or **Remove** the selected favorite. Favorites are kept in a local YAML file on your machine.

## Load manually

Pick one or more source folders, converted folders, or `recording_data.h5` files. They all open into the same viewer. Loading a folder you already have open just selects the existing row instead of duplicating it.

## Load CSV list

Reopens a session list you previously wrote with **Save loaded list**. When the CSV has a valid `recording_data_path`, twopy opens that converted HDF5 file directly; if that file is missing or invalid, twopy falls back to the source path in `recording_path`. Twopy points [Group Matching](group_matching.md) at the same folder as the loaded CSV, using `fov_groups.csv` and `roi_matches.csv` there as the default save targets; if either file already exists, twopy loads its saved rows. The picker remembers the last folder you used here independently from **Load manually**, because source recordings and saved CSV lists usually live in different places.

## The Loaded Recordings pane

Each loaded recording shows as a row below the load buttons. Click a row to make it the active recording (the one the plot dock and ROIs tab read from); twopy also selects that recording's `rois` Labels layer in napari and sets the paint label to the first unused ROI number so hand edits go to a new ROI on the same recording.

Buttons under the list (disabled until at least one recording is loaded):

- **Save loaded list** — write a CSV of source paths for the currently loaded recordings, with `recording_data_path` pointing at the published `analysis_output` HDF5 when that path can be resolved.
- **Reload saved analysis** — reread `analysis_outputs.h5` and `rois.h5` for the active recording.
- **Reconvert selected** — confirms, reruns conversion into the same converted folder, overwrites `recording_data.h5` and `aligned_movie.h5`, syncs those converted files to `analysis_output` when cache publishing applies, then reloads the row in place. ROI and analysis output files are not touched.
- **Open Group Matching** — opens the separate [Group Matching](group_matching.md) window.
- **Unload selected** — remove the selected row from the viewer.
- **Unload all** — clear every loaded recording at once.

## Caching and where files end up

With `analysis_caching: true` (the default), twopy converts and reads through a local cache directory mirrored under `analysis_cache_dir`. All plotting and analysis runs against the local cache for speed. Converted HDF5 files and saved outputs publish back to `analysis_output` in the background.

If a source path is temporarily unavailable but its cache entry exists, twopy reopens from the cache. The **Save loaded list** CSV always records the source path so the file remains usable when the network volume is back, and it keeps local cache paths out of `recording_data_path` when a published output path exists. Reloading that CSV uses a valid `recording_data_path` first so published converted files reopen without another source-path lookup.
