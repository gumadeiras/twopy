# GUI guide

This section walks through the napari app one panel at a time. Start with [What you see on launch](#what-you-see-on-launch), then jump to the task you need.

```{toctree}
:maxdepth: 1
:caption: Tasks

loading
rois
plots
custom_tab
group_matching
saving
```

## What you see on launch

Run `twopy` (optionally with a recording path) and napari opens with four docks around the viewer:

- **Top — `twopy responses`**: response trace plots and a Heatmaps tab. Reads from the current ROIs and processing settings.
- **Right — `twopy`**: the tabbed control sidebar. Tabs in order: **Load**, **Metadata**, **Plot**, **ROIs**, **Epochs**, **Custom**, **Export**.
- **Center — viewer layers**: Before you load data, twopy shows a short Load-tab prompt. After loading, this area shows `mean image`, optional `aligned movie`, and an editable `rois` Labels layer.
- **Bottom — `twopy trial timeline`**: Shows stimulus epochs across the recording. Click or drag the rail to move through the movie. Gray and interleave epochs use neutral gray. The viewer HUD shows the current trial and epoch.

If no recording is loaded yet, the docks still appear and the center viewer shows the twopy prompt. When PyPI has a newer twopy release, this prompt also shows `python -m pip install -U twopy`. Pick a recording from the **Load** tab to populate the viewer.

## Where each task lives

| I want to... | Tab or window |
| --- | --- |
| Find and open recordings | [**Load** tab](loading.md) |
| Look at acquisition info, save paths, and update notices | **Metadata** tab |
| Draw ROIs or generate them automatically | [**ROIs** tab](rois.md) |
| Change response window, dF/F, smoothing, heatmaps | [**Plot** tab](plots.md) |
| Hide stimulus epochs from plots | **Epochs** tab |
| Run Direction selectivity or your own workflow | [**Custom** tab](custom_tab.md) |
| Match the same cell across recordings | [**Open Group Matching** button](group_matching.md) |
| Save analysis or export figures | [**Export** tab](saving.md) |

## The two-photon timing contract

Imaging and stimulus presentation run on separate computers. twopy uses the recorded photodiode signal to align stimulus events with imaging frames. twopy does this alignment for you. This process explains why ROI responses use photodiode-aligned epoch windows instead of nominal frame rates. See [Input data spec](../input_data_spec.md) for the data requirements.
