# twopy

twopy is a two-photon imaging analysis tool for Clark Lab recordings. It opens converted recordings in napari, keeps source microscope files separate from twopy-owned HDF5 outputs, and gives scientists a GUI and Python API for ROI editing, response plots, custom workflows, and saved analysis outputs.

![twopy napari interface](assets/twopy-napari.png)

## Start Here

- [GUI guide](gui.md): napari loading, ROI editing, plotting, custom workflows, group matching, and saved outputs.
- [Python API guide](python_api.md): find recordings, convert data, run analysis, create ROIs, and compute response heatmaps from scripts.
- [Custom workflow guide](custom_workflows.md): add versioned local analyses to the napari Custom tab.
- [Development guide](development.md): source setup and local checks.
- [Input data spec](input_data_spec.md): short recording folder contract.
- [Recording file reference](recording_file_schema.md): detailed source and converted file schema.

```{toctree}
:maxdepth: 2
:caption: Documentation

gui
python_api
custom_workflows
development
input_data_spec
recording_file_schema
```
