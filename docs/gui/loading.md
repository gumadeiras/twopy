# Loading recordings

The **Load** tab is the first tab on the right `twopy` sidebar. You can also pass a path on the command line.

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

**Favorites** (lower left): save the current filter set under a name, restore it later with **Use**, or **Remove** the selected favorite. Favorites are kept in a local YAML file on your machine.

## Load manually

Pick one or more source folders, converted folders, or `recording_data.h5` files. They all open into the same viewer. Loading a folder you already have open just selects the existing row instead of duplicating it.

## Load CSV list

Reopens a session list you previously wrote with **Save loaded list**. If a `fov_groups.csv` sits next to the CSV file, twopy automatically loads it into [Group Matching](group_matching.md). The picker remembers the last folder you used here independently from **Load manually**, because source recordings and saved CSV lists usually live in different places.

## The Loaded Recordings pane

Each loaded recording shows as a row below the load buttons. Click a row to make it the active recording (the one the plot dock and ROIs tab read from).

Buttons under the list (disabled until at least one recording is loaded):

- **Save loaded list** — write a CSV of source paths for the currently loaded recordings.
- **Reload saved analysis** — reread `analysis_outputs.h5` and `rois.h5` for the active recording.
- **Reconvert selected** — confirms, reruns conversion into the same converted folder, overwrites `recording_data.h5` and `aligned_movie.h5`, then reloads the row in place. ROI and analysis output files are not touched.
- **Open Group Matching** — opens the separate [Group Matching](group_matching.md) window.
- **Unload selected** — remove the selected row from the viewer.
- **Unload all** — clear every loaded recording at once.

## Caching and where files end up

With `analysis_caching: true` (the default), twopy converts and reads through a local cache directory mirrored under `analysis_cache_dir`. All plotting and analysis runs against the local cache for speed. Saving publishes the changed files back to `analysis_output` in the background.

If a source path is temporarily unavailable but its cache entry exists, twopy reopens from the cache. The **Save loaded list** CSV always records the source path so the file remains usable when the network volume is back.
