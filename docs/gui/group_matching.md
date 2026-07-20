# Matching cells across recordings

Group Matching is a separate window that links recordings imaging the same field of view, then matches the same cell across those recordings. Open it from the Load tab with **Open Group Matching** (only enabled once recordings are loaded).

The window has two stages: **FOV assignment** first, then **ROI assignment**. You can move between them at any time. Both stages write plain CSV tables so downstream code can audit the assignments without depending on napari.

## Stage 1 — FOV assignment

Goal: group recordings that share a field of view.

The left column has a compact vertical list of controls. The right side has one large mean-image card for each loaded recording. Each card shows the recording ID and current FOV ID on the image. It has a vertical contrast slider and an audit **Note** field.

Buttons in the left column:

- **Assign FOV ID** — assign the currently chosen FOV id to selected cards.
- **Assign new FOV ID** — assign the next unused FOV id.
- **Remove assigned FOV** — clear the FOV assignment from selected cards.
- **Select all** / **Select none** — change card selection without touching saved assignments.
- **Load FOV CSV** — load an existing `fov_groups.csv`. Assignments and per-recording notes appear on the matching cards.
- **Browse save path** — choose where the next save writes.
- **Save FOV groups** — write `fov_groups.csv` and stay on this view.
- **Save and continue to ROI assignment** — save and switch to stage 2.

The **Current FOV groups** table lists each assigned ID and shows four rows at a time. Its headers align to the left. Click a row to select its cards and edit the group. Click the row again, or click **Select none**, to clear the selection.

## Stage 2 — ROI assignment

Goal: identify the same cell across recordings that share a FOV.

Stage 2 uses the same popup layout as stage 1. The left column contains file controls, the FOV filter, response settings, and finish actions. The fixed right workspace contains two side-by-side response previews. The previews scroll horizontally. The ROI cards scroll vertically.

The **FOV** dropdown shows compact numeric FOV ids while preserving the saved `fov_#` values internally.

Each ROI card has a colored ROI marker and a numeric ROI dropdown above the mean image. The image shows the recording ID and FOV ID. Click an ROI in the image to add it to the card. Click the selected ROI again to remove it. The dropdown gives the same selection action. Cards wrap to use the available window width.

Above the cards:

- **Selected ROIs** — wrapping clickable chips labeled `recording - ROI #`. When a saved table row is selected, the section title also shows the active group id.
- **Separate ROI responses** and **Combined response** — checkboxes that toggle the response previews above the cards.

Buttons (left column):

- **Load ROI CSV** — load an existing match table.
- **Browse save path** — choose where new match edits go.
- **Add new group** — append the currently selected non-empty ROIs as a new `group_cell_id` in `roi_matches.csv`. A recording can contribute more than one separate ROI row. Recordings left with no selected ROI are omitted from the group.
- **Overwrite selected group** — replace the selected saved row with the current selection and note.
- **Remove selected group** — delete that row from the CSV.
- **Clear ROI selection** — reset card selectors without deleting saved rows.
- **Back to FOV assignment** — return to stage 1 without closing.
- **Save and close** — save edits to the selected group and close.

The **Saved groups** table lists matched groups for the current FOV. It shows Group ID, ROIs, Note, and five rows at a time. Select a row to restore its ROI selections and note. Click the selected row again to clear the selection.

**Plot settings** in the left column hold response-row toggles, plot size, epoch visibility (gray / grey / interleave rows hidden by default), smoothing, and normalization. Click the colored ROI trace chips above the plots to toggle each selected ROI trace. The chip area shows up to five rows at a time, then scrolls.

## Notes and outputs

Notes on FOV cards write to the CSV `note` column for that recording row. The ROI note writes to every row in the selected group. Both notes are audit-only free text and are not used by matching logic.

CSV files written:

- `fov_groups.csv` — recording → FOV id, plus per-recording notes.
- `roi_matches.csv` — one row for each selected ROI in a `group_cell_id`. Each row has the FOV ID, recording path, ROI number, status, and note. Rows with `status="matched"` identify the same visual cell across recordings. `status="unmatched"` identifies reviewed single ROIs. Thus, later code can separate them from unreviewed ROIs.

### A macOS Qt message you can ignore

On macOS you may see `Cell requested for row 0 is out of bounds for table with 8 rows! Resizing table model.` while the ROI Assignment view refreshes the Saved groups table. This comes from Qt's Cocoa accessibility cache rebuilding after Hover Text, VoiceOver, or another accessibility client asks for a table cell during the refresh. twopy's data and rows are not out of bounds.
