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
- **Center — viewer layers**: before you load data, twopy shows a short Load-tab prompt; after loading, this area shows `mean image`, optional `aligned movie`, and an editable `rois` Labels layer.
- **Bottom — `twopy trial timeline`**: shows stimulus epochs across the recording. Click or drag the rail to seek the movie; gray / interleave epochs render as neutral gray. The viewer HUD reports the current trial and epoch.

If no recording is loaded yet, the docks still appear and the center viewer shows the twopy prompt. Pick one from the **Load** tab to populate them.

## Where each task lives

| I want to... | Tab or window |
| --- | --- |
| Find and open recordings | [**Load** tab](loading.md) |
| Look at acquisition info and save paths | **Metadata** tab |
| Draw ROIs or generate them automatically | [**ROIs** tab](rois.md) |
| Change response window, dF/F, smoothing, heatmaps | [**Plot** tab](plots.md) |
| Hide stimulus epochs from plots | **Epochs** tab |
| Run Direction selectivity or your own workflow | [**Custom** tab](custom_tab.md) |
| Match the same cell across recordings | [**Open Group Matching** button](group_matching.md) |
| Save analysis or export figures | [**Export** tab](saving.md) |

## The two-photon timing contract

Imaging and stimulus presentation run on separate computers. The photodiode signal in the recording is the bridge: twopy uses it to align stimulus events to imaging frames. You can ignore this when using the app — twopy aligns the timing for you — but it explains why ROI responses use photodiode-aligned epoch windows rather than nominal frame rates. See [Input data spec](../input_data_spec.md) for the contract.
