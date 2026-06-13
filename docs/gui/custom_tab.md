# Running custom workflows

The **Custom** tab runs Python analyses against the active recording's converted data and current ROIs. twopy ships two built-in workflows and can also load workflows from folders you list in `config.yml`.

If you want to *write* a workflow, see [Writing custom workflows](../writing_custom_workflows.md).

## Using the tab

1. Pick a workflow from the **dropdown** at the top.
2. The **version**, **source file**, and **description** appear below it.
3. Set the workflow's parameters in the **Parameters** group. Controls reflect the workflow's parameter types (numbers, dropdowns, paths, checkboxes).
4. Click **Run**. The workflow uses the loaded recording and the current ROI Labels layer.
5. Results show in the **Result** group: a message, files, tables, line plots, replacement ROIs, or updates to visible ROI selection.

If you change workflow files on disk, click **Reload workflows** to rescan without restarting twopy.

twopy writes workflow metadata next to non-HDF5 outputs (`*.twopy-workflow.yml` sidecars) and inside HDF5 outputs (a `twopy_workflow` group). With analysis caching on, custom outputs sync to `analysis_output` through the same path **Save ROIs + analysis** uses.

Invalid workflows (missing metadata, bad type annotations, unknown roles, …) are listed in the tab's status text but kept out of the dropdown so you can fix them without restarting.

## Built-in workflows

### Direction selectivity

Computes a direction-selectivity index (DSI) for the visible ROIs by default.

- **Preferred epoch** / **Null epoch** — pick from the loaded recording's epoch names.
- **Metric** — `mean`, `peak`, or `minimum` over the configured window.
- **ROIs** — `visible ROIs` by default, or `all ROIs` when you want every current ROI included.
- **Window start (s)** / **Window end (s)** — epoch-relative metric window. Capped to the shorter of the preferred and null epoch durations.
- **Rectify** option zeroes negative responses before computing DSI.

The result is a three-decimal per-ROI DSI table. Rows at or above the **DSI show threshold** are highlighted, and only the passing ROIs stay selected in the response plot and ROIs tab — without replacing the plot data. The threshold defaults to `0.1`.

### Response kernels

Fits temporal kernels from random-noise stimulus epochs.

- **Stimulus** — `olfaction` or `vision`. Defaults from converted rig metadata: `OdorRig` selects olfaction, anything else selects vision.
- **Baseline epoch** — gray / interleave epoch excluded from the fit.
- Selected non-baseline epochs are grouped by unique name; only complete regular stimulus streams are kept. Segments with irregular sample times are skipped (and counted) so display-timing hitches stay visible.

For olfaction, the workflow writes one CSV per kernel stream:

- `response_kernels_group_<index>_epochs_<numbers>_<epoch_name>_ipsi.csv`
- `response_kernels_group_<index>_epochs_<numbers>_<epoch_name>_contra.csv`

Raw left and right LED-activation streams are derived from the stimulus column (`0=left`, `1=both`, `2=right`, `3=blank`) and mapped to ipsi / contra using the recording's hemisphere metadata. A manual override is available for audits.

For vision, the workflow writes one CSV per kernel stream interpreted as signed contrast:

- `response_kernels_group_<index>_epochs_<numbers>_<epoch_name>_contrast.csv`

Both modes also write `response_kernels_summary.csv` reporting fitted and skipped segment counts. Matrix columns are labeled `lag_s_<seconds>` so the lag axis stays inspectable without opening the plot. Negative kernel times are future stimulus samples — useful as a timing diagnostic.

## Adding your own workflows

Edit your private `config.yml`:

```yaml
custom_workflow_paths:
  - ~/git/twopy-workflows
  - /Volumes/magic/clarklab/shared/twopy_workflows
```

Folders are scanned for top-level `.py` files. Helper modules can sit beside workflow files; prefix their filenames with `_` to keep them out of the dropdown.

To learn the file shape, see [Writing custom workflows](../writing_custom_workflows.md) and the reference example at `examples/custom_workflows/reference_showcase.py`.
