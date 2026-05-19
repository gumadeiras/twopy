# Custom Workflow Guide

Custom workflows are trusted Python files shown in the napari Custom tab. twopy ships Direction selectivity by default and can also load lab-local workflows from `config.yml`. Use a local workflow when an analysis should run on converted twopy data but should live outside the shared repo.

Direction selectivity uses the loaded recording's epoch names for its preferred and null epoch dropdowns. It computes DSI per ROI, shows the table in the Custom tab, highlights rows that pass the absolute DSI threshold, and sends only passing ROIs to the response plot.

Response kernels is a separate workflow. It fits random-noise temporal kernels for the current ROIs, supports olfactory antenna activation and visual contrast with a Stimulus dropdown, writes one CSV matrix per unique epoch name and kernel stream, and shows per-ROI plus mean kernel plots.

## Configure

Native workflows are available without config. Add extra Python files or folders to your private `config.yml`:

```yaml
custom_workflow_paths:
  - ~/git/twopy-workflows
  - /Volumes/magic/clarklab/shared/twopy_workflows
```

Folders are scanned for top-level `.py` files. Helper modules can live beside workflow files. Prefix helper filenames with `_` when they should not appear as workflows. Invalid workflows are listed in the Custom tab status text and are not added to the dropdown.

The reference example at `examples/custom_workflows/reference_showcase.py` shows every supported parameter type and result type.

## Required Workflow Shape

Workflow files should import twopy workflow objects from `twopy.custom`. Do not import `twopy.analysis`, `twopy.napari`, `twopy.stimulus`, `twopy.roi`, or other internal modules. Use `CustomRunContext` helpers instead so workflows keep working when twopy internals move.

Each workflow must provide metadata, a version, a description, supported parameters, exact type annotations, and a `CustomResult` return value:

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
    preferred = ctx.epoch_metric(computation.grouped_responses, params.preferred_epoch, params.metric, window_seconds=window)
    null = ctx.epoch_metric(computation.grouped_responses, params.null_epoch, params.metric, window_seconds=window)
    dsi = np.divide(preferred - null, preferred + null, out=np.full_like(preferred, np.nan), where=np.abs(preferred + null) > 1e-12)
    csv_path = ctx.output_path("direction_selectivity.csv")
    ctx.write_roi_table(csv_path, {"preferred": preferred, "null": null, "dsi": dsi})
    return CustomResult(message=f"Computed DSI for {len(rois.labels)} ROIs.", tables=(CustomTable("DSI", csv_path),))
```

twopy rejects workflows that are missing `id`, `name`, `version`, `description`, exact annotations, a frozen params dataclass, defaults for every parameter, or supported parameter types. Workflow versions must use `X.Y`, such as `1.0` or `2.3`.

## Stable Context API

`CustomRunContext` is the public API for workflow code. It exposes the active recording through small helpers:

- `ctx.current_rois()` returns the current ROI masks or raises a clear error.
- `ctx.rois_for_selector(selector)` returns all current ROIs or the visible response-plot ROI subset for a standard `roi_selector` parameter.
- `ctx.compute_standard_responses()` runs the same dF/F and response grouping used by the Plot tab.
- `ctx.epoch_names()` returns `{epoch_number: epoch_name}` for the loaded recording.
- `ctx.epoch_choices()` returns `CustomEpoch` objects with `number`, `name`, `label`, `selector`, and `duration_seconds`.
- `ctx.epoch_durations_seconds()` returns the shortest observed duration per epoch number.
- `ctx.min_epoch_duration_seconds()` returns the shortest valid epoch duration.
- `ctx.epoch_window(start, stop)` validates an epoch-relative metric window before it is passed to response helpers.
- `ctx.response_window(start, stop)` validates a response-plot-relative window. Negative starts are allowed because response plots can include pre-epoch baseline.
- `ctx.epoch_metric(grouped, epoch, metric, window_seconds=(start, stop))` computes mean, peak, or minimum per ROI over one epoch and optional epoch-relative time window. The `epoch` value can be a `CustomEpoch`, epoch number, epoch name, or GUI dropdown label such as `2: Odor`.
- `ctx.response_plot_data(grouped, source_path=..., max_rois=..., roi_indices=...)` creates response plot data for `CustomResult.response_plot_data`.
- `ctx.output_path(...)`, `ctx.write_roi_table(...)`, and `ctx.write_matrix_csv(...)` write outputs below the workflow output folder and attach workflow metadata.

## Parameters

The Custom tab renders dataclass fields as controls. Supported field types are `bool`, `int`, `float`, `str`, `Path`, `Literal[...]`, and `Enum`. Field metadata can set `label`, `description`, `min`, `max`, `step`, and `decimals`.

`twopy_role` is a validated metadata field for standard controls. Unknown roles reject the workflow before it appears in the GUI.

| `twopy_role` | Field type | GUI behavior |
| --- | --- | --- |
| `baseline_epoch` | `str` | Shows loaded-recording epochs and defaults to the same gray/grey/interleave baseline used by dF/F controls. |
| `comparison_epoch` | `str` | Shows loaded-recording epochs for a comparison epoch. |
| `epoch` | `str` | Shows a dropdown of loaded-recording epochs and passes the selected epoch label back to the workflow. |
| `epoch_window_start` | `float` | Shows an epoch-relative start time with zero minimum, 0.1-second step, three decimals, and a recording-based maximum when timing is available. |
| `epoch_window_stop` | `float` | Shows an epoch-relative stop time with zero minimum, 0.1-second step, three decimals, and a recording-based maximum when timing is available. The workflow default is preserved. |
| `output_name` | `str` or `Path` | Marks a relative workflow output filename. Use with `ctx.output_path(...)` so outputs stay below the workflow output folder. |
| `response_metric` | `str` | Shows the standard metric dropdown: `mean`, `peak`, or `minimum`. |
| `response_window_start` | `float` | Shows a response-plot start time with the current pre-epoch window as the minimum, 0.1-second step, and three decimals. |
| `response_window_stop` | `float` | Shows a response-plot stop time with a recording-based maximum when timing is available, 0.1-second step, and three decimals. |
| `roi_limit` | `int` | Shows a positive integer control for caps such as maximum plotted ROIs. |
| `roi_selector` | `str` | Shows a dropdown with `all ROIs` and `visible ROIs` while passing stable values `all_rois` and `visible_rois` to `ctx.rois_for_selector(...)`. |
| `stimulus_column` | `str` | Shows a dropdown of converted stimulus columns from the loaded recording, excluding clock and epoch columns when other columns are available. |
| `table_highlight_threshold` | `float` | Shows a non-negative threshold control with 0.05 step and three displayed decimals. Use the value to build `CustomTable(..., highlighted_rows=(...))`. |

Roles are not tied to field names. A field named `start`, `window_start_seconds`, or `foo` gets the standard control when its metadata declares the role. Role/type pairs are validated during workflow discovery; for example, `twopy_role="epoch"` on a `float` field rejects the workflow before it appears in the GUI.

## Outputs And Provenance

Workflows return `CustomResult` objects. A result can list files, tables, line plots, replacement ROIs, or response plot data. File and table outputs must be below `ctx.output_dir`; use `ctx.output_path("name.csv")` instead of building paths by hand. Returned CSV and TSV tables are previewed in the Custom tab, `CustomTable(..., highlighted_rows=(...))` marks zero-based data rows, and `CustomLinePlot(..., y_label="Weight")` sets the plot y-axis label when the default `Value` is too generic. Files written through `ctx.write_roi_table()` and `ctx.write_matrix_csv()` automatically receive workflow metadata. Files returned in `CustomResult.files` or `CustomResult.tables` receive metadata after the workflow returns.

For CSV, PDF, PNG, and other non-HDF5 outputs, twopy writes a sidecar like `direction_selectivity.twopy-workflow.yml`. For HDF5 outputs, twopy writes a `twopy_workflow` metadata group. The metadata records the workflow id, name, version, source path, source hash, twopy version, run time, parameters, and recording path. When analysis caching is enabled, custom outputs and their metadata sync to the configured analysis output folder.

## Kernel Workflow

twopy ships a native Response kernels workflow. It runs through the same Custom tab as local workflows, uses the current Plot-tab dF/F and processing settings, keeps complete regular stimulus streams for selected non-baseline epochs, samples ROI responses from photodiode-aligned frame windows, and fits temporal kernels per ROI for each unique selected epoch name. The Stimulus dropdown selects how the default converted stimulus column is interpreted.

For `olfaction`, the workflow fits one raw-left and raw-right antenna kernel per ROI and epoch name, maps those raw kernels to ipsi and contra from the recording hemisphere stored in converted metadata, with a manual override available for audits, then writes:

- `response_kernels_<epoch_name>_ipsi.csv`
- `response_kernels_<epoch_name>_contra.csv`
- `response_kernels_summary.csv`

For `vision`, the workflow fits one signed-contrast kernel per ROI and epoch name and writes:

- `response_kernels_<epoch_name>_contrast.csv`
- `response_kernels_summary.csv`

Both current native modes default to `stimulus_specific_05`, but the meaning differs by modality. In olfactory LED recordings it matches the random-noise workflow's `antenna_stim` value after twopy's stable column naming; LED recordings encode `0=left`, `1=both`, `2=right`, and `3=blank`. In Matulis-style full-field visual contrast recordings it is signed contrast, such as `-0.2/+0.2` and `-0.9/+0.9`. Negative kernel times are future stimulus samples and are useful as a timing diagnostic. Current native fitting is frame-aligned; per-ROI line timing maps are not yet part of converted HDF5 files, so line-precise psycho5 parity remains a separate timing-extension task.

A local kernel workflow can still return custom line plots if it needs experimental fitting code:

```python
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
    return CustomResult(message=f"Fit kernels for {len(rois.labels)} ROIs.", plots=(CustomLinePlot("Kernels", kernels.time_seconds, kernels.values, rois.labels, y_label="Weight"),), files=(csv_path,))
```

Keep long-running or experimental code inside the workflow file or imported lab modules. Keep the values passed to twopy small and explicit.
