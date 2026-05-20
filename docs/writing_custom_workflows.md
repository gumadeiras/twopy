# Writing custom workflows

A custom workflow is a Python file twopy can load into the **Custom** tab. It runs against the active recording's converted data and current ROIs, and it can write files, return tables, draw line plots, replace the ROI set, or update the response-plot visibility.

If you only want to *use* the Custom tab, see [Running custom workflows](gui/custom_tab.md).

A reference example showing every supported parameter type and result type lives at `examples/custom_workflows/reference_showcase.py`.

## Pointing twopy at your workflow folder

Add the folders to your private `config.yml`:

```yaml
custom_workflow_paths:
  - ~/git/twopy-workflows
  - /Volumes/magic/clarklab/shared/twopy_workflows
```

Each folder is scanned for top-level `.py` files. Helper modules can sit beside workflow files; prefix helper filenames with `_` to keep them out of the dropdown. Invalid workflows are listed in the Custom tab status text and excluded from the dropdown.

## The minimum a workflow needs

Every workflow imports its building blocks from `twopy.custom` and provides a `@workflow(...)` decorated function plus a frozen params dataclass. Do **not** import from `twopy.analysis`, `twopy.napari`, `twopy.stimulus`, or `twopy.roi` — go through `CustomRunContext` instead so workflows keep working when twopy internals move.

```python
from dataclasses import dataclass, field

import numpy as np

from twopy.custom import CustomResult, CustomRunContext, CustomTable, workflow


@dataclass(frozen=True)
class DirectionSelectivityParams:
    preferred_epoch: str = field(default="right", metadata={"label": "Preferred epoch", "twopy_role": "epoch"})
    null_epoch: str = field(default="left", metadata={"label": "Null epoch", "twopy_role": "epoch"})
    metric: str = field(default="peak", metadata={"label": "Metric", "twopy_role": "response_metric"})
    window_start_seconds: float = field(default=0.0, metadata={"label": "Window start (s)", "twopy_role": "epoch_window_start"})
    window_stop_seconds: float = field(default=3.0, metadata={"label": "Window end (s)", "twopy_role": "epoch_window_stop"})


@workflow(
    id="direction-selectivity",
    name="Direction selectivity",
    version="1.0",
    description="Computes a direction-selectivity index for each current ROI.",
    params=DirectionSelectivityParams,
)
def run(ctx: CustomRunContext, params: DirectionSelectivityParams) -> CustomResult:
    rois = ctx.current_rois()
    computation = ctx.compute_standard_responses(rois)
    window = ctx.epoch_window(params.window_start_seconds, params.window_stop_seconds)
    preferred = ctx.epoch_metric(
        computation.grouped_responses, params.preferred_epoch, params.metric, window_seconds=window,
    )
    null = ctx.epoch_metric(
        computation.grouped_responses, params.null_epoch, params.metric, window_seconds=window,
    )
    dsi = np.divide(
        preferred - null,
        preferred + null,
        out=np.full_like(preferred, np.nan),
        where=np.abs(preferred + null) > 1e-12,
    )
    csv_path = ctx.output_path("direction_selectivity.csv")
    ctx.write_roi_table(csv_path, {"preferred": preferred, "null": null, "dsi": dsi})
    return CustomResult(
        message=f"Computed DSI for {len(rois.labels)} ROIs.",
        tables=(CustomTable("DSI", csv_path),),
    )
```

Required at the decorator: `id`, `name`, `version`, `description`, `params`. Required on the function: an exact `CustomRunContext` first argument, the params dataclass as second, and `CustomResult` as the return type. Versions must use `X.Y` (`1.0`, `2.3`, ...).

twopy rejects workflows that are missing any of those, that use a non-frozen params dataclass, that have parameters without defaults, or that use unsupported parameter types or unknown roles.

## CustomRunContext

`CustomRunContext` is the public API for workflow code. Use it instead of importing twopy internals.

ROIs and responses:

- `ctx.current_rois()` — current ROI masks, or a clear error if none.
- `ctx.rois_for_selector(selector)` — pass the value of a `roi_selector` parameter (`"all_rois"` or `"visible_rois"`).
- `ctx.roi_indices_for_selector(selector)` — matching zero-based ROI rows from the full set, useful when a workflow computes on a subset but needs to update the existing Plot-tab visibility without replacing plot data.
- `ctx.compute_standard_responses(rois)` — runs the same dF/F and trial grouping the Plot tab uses.

Recording metadata:

- `ctx.recording_metadata()` — a stable metadata snapshot. Read fields with `metadata.text("run", "rig_name", default="")`, `metadata.float("acquisition", "acq.zoomFactor")`, `metadata.int("run", "run_number")`, or raw mappings such as `metadata.run` and `metadata.stimulus_parameters`.

Epochs:

- `ctx.epoch_names()` — `{epoch_number: epoch_name}` for the loaded recording.
- `ctx.epoch_choices()` — `CustomEpoch` objects with `number`, `name`, `label`, `selector`, and `duration_seconds`.
- `ctx.epoch_durations_seconds()` — shortest observed duration per epoch number.
- `ctx.min_epoch_duration_seconds()` — shortest valid epoch duration in the recording.
- `ctx.epoch_window(start, stop)` — validates an epoch-relative metric window before passing it to a response helper.
- `ctx.response_window(start, stop)` — validates a response-plot-relative window. Negative starts are allowed because response plots can include pre-epoch baseline.
- `ctx.epoch_metric(grouped, epoch, metric, window_seconds=(start, stop))` — mean, peak, or minimum per ROI over one epoch and optional epoch-relative window. `epoch` accepts a `CustomEpoch`, an epoch number, an epoch name, or a GUI dropdown label such as `2: Odor`.

Plots:

- `ctx.response_plot_data(grouped, source_path=..., max_rois=..., roi_indices=...)` — build response plot data for `CustomResult.response_plot_data`.
- `ctx.roi_colors_for_labels(labels)` — `#RRGGBB` colors matching current ROI plot colors. The Custom tab auto-colors `CustomLinePlot` outputs whose labels exactly match current ROI labels when `colors` is omitted.

Outputs:

- `ctx.output_path("name.csv")` — build a path below the workflow output folder. Always use this instead of building paths by hand.
- `ctx.write_roi_table(path, {...})` — write a per-ROI table CSV and attach workflow metadata.
- `ctx.write_matrix_csv(path, matrix, row_labels=..., column_labels=...)` — write a matrix CSV. Pass `column_labels` when columns have scientific coordinates like lag seconds.

Helpers from `twopy.custom`:

- `finite_mean_and_sem(values, axis=...)` — same finite-sample mean / sample-SEM convention used by twopy's response plots and CSV exports.

## Parameters and their controls

The Custom tab renders dataclass fields as form controls. Supported field types are `bool`, `int`, `float`, `str`, `Path`, `Literal[...]`, and `Enum`. Field metadata can set `label`, `description`, `min`, `max`, `step`, and `decimals`.

`twopy_role` declares standard controls and validation. Unknown roles reject the workflow before it appears in the GUI; mismatched role/type pairs also reject (e.g. `twopy_role="epoch"` on a `float` field).

| `twopy_role` | Field type | GUI behavior |
| --- | --- | --- |
| `baseline_epoch` | `str` | Loaded-recording epoch dropdown, defaults to the same gray/grey/interleave baseline the dF/F controls use. |
| `comparison_epoch` | `str` | Loaded-recording epoch dropdown for a comparison epoch. |
| `epoch` | `str` | Loaded-recording epoch dropdown. Passes the selected epoch label back to the workflow. |
| `epoch_window_start` | `float` | Epoch-relative start time, min 0, step 0.1s, three decimals, recording-based max when timing is available. |
| `epoch_window_stop` | `float` | Epoch-relative stop time, min 0, step 0.1s, three decimals, recording-based max when timing is available. Workflow default is preserved. |
| `output_name` | `str` or `Path` | Marks a relative workflow output filename. Use with `ctx.output_path(...)`. |
| `response_metric` | `str` | `mean`, `peak`, or `minimum` dropdown. |
| `response_window_start` | `float` | Response-plot start time. Min = current pre-epoch window. Step 0.1s, three decimals. |
| `response_window_stop` | `float` | Response-plot stop time, recording-based max when timing is available. Step 0.1s, three decimals. |
| `roi_limit` | `int` | Positive integer for caps such as maximum plotted ROIs. |
| `roi_selector` | `str` | Dropdown with `all ROIs` and `visible ROIs`. Passes stable values `all_rois` / `visible_rois`. |
| `stimulus_column` | `str` | Dropdown of converted stimulus columns (excluding clock and epoch columns when other columns exist). |
| `table_highlight_threshold` | `float` | Non-negative threshold control with 0.05 step and three displayed decimals. Use the value with `CustomTable(..., highlighted_rows=(...))`. |

Roles are tied to declared `metadata`, not field names. A field named `start`, `window_start_seconds`, or `foo` gets the standard control when its metadata declares the role.

## What the workflow returns

`CustomResult` can include:

- `message` — short status string.
- `files=(path, ...)` — non-HDF5 files to attach metadata to.
- `tables=(CustomTable(name, path, highlighted_rows=(...)), ...)` — CSV/TSV tables previewed in the Custom tab. `highlighted_rows` marks zero-based data rows.
- `plots=(CustomLinePlot(name, x, y, labels, y_label=..., colors=..., bands=(CustomLineBand(...),)), ...)` — line plots. `colors` accepts `#RRGGBB` strings; `CustomLineBand` draws filled uncertainty bands behind a series.
- `rois=RoiSet(...)` — replace the current Labels layer ROI set.
- `visible_roi_indices=(...)` — change the Plot-tab visibility without replacing the plot data.
- `response_plot_data=ctx.response_plot_data(...)` — replacement response plot data.

All file and table outputs must live below `ctx.output_dir`; using `ctx.output_path("name.csv")` enforces this.

## Provenance

twopy records what ran:

- For CSV, PDF, PNG, and other non-HDF5 outputs, twopy writes a sidecar like `direction_selectivity.twopy-workflow.yml`.
- For HDF5 outputs, twopy writes a `twopy_workflow` metadata group inside the file.

Both record the workflow id, name, version, source file path, source hash, twopy version, run time, parameters, and recording path. Files written via `ctx.write_roi_table()` and `ctx.write_matrix_csv()` get metadata automatically; files returned in `CustomResult.files` or `CustomResult.tables` get metadata after the workflow returns.

With analysis caching on, custom outputs and their metadata sync to `analysis_output` through the same path **Save ROIs + analysis** uses.

## A local plotting workflow

Keep experimental or long-running code inside your workflow file (or imported lab modules). Keep what you hand back to twopy small and explicit.

```python
import numpy as np

from twopy.custom import (
    CustomLineBand,
    CustomLinePlot,
    CustomResult,
    CustomRunContext,
    finite_mean_and_sem,
    workflow,
)

@workflow(
    id="response-kernels",
    name="Response kernels",
    version="0.1",
    description="Fits and plots one response kernel per ROI.",
    params=KernelParams,
)
def run(ctx: CustomRunContext, params: KernelParams) -> CustomResult:
    rois = ctx.current_rois()
    computation = ctx.compute_standard_responses(rois)
    kernels = fit_kernels(computation.grouped_responses, params)
    csv_path = ctx.output_path("kernels.csv")
    ctx.write_matrix_csv(csv_path, kernels.values, row_labels=rois.labels)
    mean, sem = finite_mean_and_sem(kernels.values, axis=0)
    return CustomResult(
        message=f"Fit kernels for {len(rois.labels)} ROIs.",
        plots=(
            CustomLinePlot(
                "Mean +/- SEM kernels",
                kernels.time_seconds,
                mean,
                ("mean",),
                y_label="Weight",
                bands=(
                    CustomLineBand(series_index=0, lower=mean - sem, upper=mean + sem, label="SEM"),
                ),
            ),
        ),
        files=(csv_path,),
    )
```
