# Drawing and generating ROIs

ROIs live in the `rois` Labels layer of the viewer. You can paint them by hand or let twopy generate them.

twopy draws and shows ROIs on the *alignment-valid crop* — the part of every frame that stays in-field after motion correction. When you save or run analysis, the ROIs are converted back to full-frame coordinates automatically. You can ignore this; it just means response updates only work while the active Labels layer matches the displayed crop.

## Draw by hand

1. Select the `rois` layer in the layer list.
2. Pick the paint tool in the napari layer controls (top-left).
3. Set the `label` number before painting each cell so each ROI gets its own number.
4. Edit existing ROIs with the same tool; commit the edit (release the mouse) and the response plots update.

## Generate from the ROIs tab

The **ROIs** tab has a **Create ROIs** section with a **Mode** dropdown:

- **manual** — leaves the layer alone for hand-drawing.
- **grid** — square template ROIs covering the crop. Choose **Units = pixels** for an exact pixel width, or **Units = microns** to convert from a physical width using the pixel calibration registry. Micron mode prefills known calibration fields (rig, mode, scanner, zoom) from converted metadata; if a field is missing it is left unselected rather than guessed.
- **watershed** — segments bright structures from the mean image. Tune **Min pixels** and **Smoothing** (sigma).
- **response watershed** — segments stimulus-locked pixel responses from photodiode-aligned epoch windows. Tune **Response min pixels** and **Response smoothing**. **Fill response holes** is on by default; **Response closing** adds an opt-in conservative binary close for tiny same-basin gaps.

Click **Create ROIs** to replace the contents of the `rois` Labels layer with the generated mask set.

## Hide or remove ROIs

The ROIs tab also shows a per-ROI table:

- **Hide** a row to keep its mask but exclude it from response plots and saved analysis.
- **Remove** drops the selected rows from the Labels layer outright.

For dense grids that cover the whole crop, prefer **shared y-stripe P%** background correction (in the [Plot tab](plots.md)) over **ROI y-stripe P%** — the latter needs unlabeled local background pixels and cannot work when there are none.

## Persistence

ROI masks are saved in twopy's own HDF5 format independently from napari. If a `rois.h5` file sits beside the converted recording, twopy loads it automatically when you open the recording.
