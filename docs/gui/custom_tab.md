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

The status text lists invalid workflows, such as workflows with missing metadata, incorrect type annotations, or unknown roles. The dropdown does not show them. Thus, you can correct them without a restart.

## Built-in workflows

### Direction selectivity

Computes a direction-selectivity index (DSI) for the visible ROIs by default.

- **Preferred epoch** / **Null epoch** — pick from the loaded recording's epoch names.
- **Metric** — `mean`, `peak`, or `minimum` over the configured window.
- **ROIs** — `visible ROIs` by default, or `all ROIs` when you want every current ROI included.
- **Window start (s)** / **Window end (s)** — epoch-relative metric window. Capped to the shorter of the preferred and null epoch durations.
- **Rectify** option zeroes negative responses before computing DSI.

The result is a per-ROI DSI table with three decimal places. Rows at or above **DSI show threshold** are highlighted. Only these ROIs stay selected in the response plot and the ROIs tab. The plot data does not change. The default threshold is `0.1`.

### Response kernels

Fits temporal kernels from random-noise stimulus epochs.

- **Stimulus** — `olfaction` or `vision`. Defaults from converted rig metadata: `OdorRig` selects olfaction, anything else selects vision.
- **Baseline epoch** — gray / interleave epoch excluded from the fit.
- twopy groups selected non-baseline epochs by unique name. It keeps only complete, regular stimulus streams. It skips and counts segments with irregular sample times. Thus, display-timing problems stay visible.

For olfaction, the workflow writes one CSV per kernel stream:

- `response_kernels_group_<index>_epochs_<numbers>_<epoch_name>_ipsi.csv`
- `response_kernels_group_<index>_epochs_<numbers>_<epoch_name>_contra.csv`

twopy gets the raw left and right LED-activation streams from the stimulus column (`0=left`, `1=both`, `2=right`, `3=blank`). It uses the recording hemisphere to map the streams to ipsi and contra. You can override this mapping for an audit.

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

twopy scans folders for top-level `.py` files. Helper modules can be next to workflow files. Start a helper file name with `_` to keep it out of the dropdown.

To learn the file shape, see [Writing custom workflows](../writing_custom_workflows.md) and the reference example at `examples/custom_workflows/reference_showcase.py`.
