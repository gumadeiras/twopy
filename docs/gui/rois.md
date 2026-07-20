# Drawing and generating ROIs

ROIs live in the `rois` Labels layer of the viewer. You can paint them by hand or let twopy generate them.

twopy draws ROIs on the *alignment-valid crop*. This is the part of each frame that stays in the field after motion correction. When you save or run analysis, twopy changes the ROIs to full-frame coordinates. Response updates work only when the active Labels layer matches the displayed crop.

## Draw by hand

1. Select the `rois` layer in the layer list.
2. Pick the paint tool in the napari layer controls (top-left).
3. Set the `label` number before painting each cell so each ROI gets its own number.
4. Edit existing ROIs with the same tool. Release the mouse to finish the edit and update the response plots.

## Generate from the ROIs tab

The **ROIs** tab has a **Create ROIs** section with a **Mode** dropdown:

- **manual** — leaves the layer alone for hand-drawing.
- **grid** — square template ROIs that cover the crop. Select **Units = pixels** for an exact pixel width. Select **Units = microns** to use a physical width and the pixel calibration registry. Micron mode fills known rig, mode, scanner, and zoom fields from converted metadata. If a field is missing, it stays unselected.
- **watershed** — segments bright structures from the displayed mean-image crop. Tune **Min pixels** and **Smoothing** (sigma). **Min pixels** uses the same displayed crop pixels shown in the **area (px)** column.
- **response watershed** — makes segments from stimulus-locked pixel responses in photodiode-aligned epoch windows. Adjust **Response min pixels** and **Response smoothing**. **Fill response holes** is on by default. **Response closing** can close small gaps in the same basin.

Click **Create ROIs** to replace the contents of the `rois` Labels layer with the generated mask set.

## Hide or remove ROIs

The ROIs tab also shows a per-ROI table:

- **area (px)** shows how many displayed crop pixels belong to each ROI.
- **Hide** a row to keep its mask but exclude it from response plots and saved analysis.
- **Merge Selected** combines the checked rows into the first checked ROI label, then updates response plots from the combined mask.
- **Remove Selected** drops the checked rows from the Labels layer outright.

For dense grids, use **shared y-stripe P%** background correction in the [Plot tab](plots.md). **ROI y-stripe P%** needs dim, unlabeled local background pixels. It stops before dF/F if local pixels are brighter than the ROI baseline. In this condition, the selected band is not valid additive background.

## Persistence

ROI masks are saved in twopy's own HDF5 format independently from napari. If a `rois.h5` file sits beside the converted recording, twopy loads it automatically when you open the recording.

When you save from the app, `rois.h5` also records the ROI mode and generator settings for the current masks. Generated modes save only the settings that apply to that mode, such as watershed **Min pixels** and **Smoothing** or grid size and micron-calibration fields. If you edit generated masks by hand before saving, the file marks that the saved masks changed after generation. **Reload saved analysis** reloads the masks, restores those ROIs-tab settings, and shows that edit marker. Older ROI files without this metadata still load their masks normally and leave the ROI mode at the default manual setting.
