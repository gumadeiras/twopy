# Matching cells across recordings

Group Matching is a separate window that links recordings imaging the same field of view, then matches the same cell across those recordings. Open it from the Load tab with **Open Group Matching** (only enabled once recordings are loaded).

The window has two stages: **FOV assignment** first, then **ROI assignment**. You can move between them at any time. Both stages write plain CSV tables so downstream code can audit the assignments without depending on napari.

## Stage 1 — FOV assignment

Goal: group recordings that share a field of view.

The left column has a compact, vertical-only scroll of controls; the right side is filled with large mean-image cards (one per loaded recording). Each card overlays the recording id and current FOV id on the image, has a vertical contrast slider beside it, and a free-text **Note** below it for audit annotations.

Buttons in the left column:

- **Assign FOV ID** — assign the currently chosen FOV id to selected cards.
- **Assign new FOV ID** — assign the next unused FOV id.
- **Remove assigned FOV** — clear the FOV assignment from selected cards.
- **Select all** / **Select none** — change card selection without touching saved assignments.
- **Load FOV CSV** — load an existing `fov_groups.csv`. Assignments and per-recording notes appear on the matching cards.
- **Browse save path** — choose where the next save writes.
- **Save FOV groups** — write `fov_groups.csv` and stay on this view.
- **Save and continue to ROI assignment** — save and switch to stage 2.

The **Current FOV groups** table lists every assigned id with left-aligned headers and shows four rows at a time. Clicking a row re-selects those cards so you can edit the group; clicking the selected row again (or **Select none**) clears the table and card selection.

## Stage 2 — ROI assignment

Goal: identify the same cell across recordings that share a FOV.

The layout mirrors stage 1 inside the same tight popup frame: a vertical-only left column for file controls, FOV filter, response settings, and finish actions; a fixed right workspace where the two response previews sit side by side, scroll sideways without vertical clipping, and leave the ROI cards to scroll vertically.

The **FOV** dropdown shows compact numeric FOV ids while preserving the saved `fov_#` values internally.

Each ROI card stacks a colored ROI chip plus a numeric ROI dropdown above the mean image, which carries the same recording-id and FOV-id overlay used by FOV cards. Click an ROI in the mean-image overlay to select it; click the selected ROI again to return that card to **No ROI**. The dropdown stays available for precise selection. Cards wrap to fit as many per row as the window allows.

Above the cards:

- **Selected ROIs** — wrapping clickable chips labeled `recording - ROI #`. When a saved table row is selected, the section title also shows the active group id.
- **Separate ROI responses** and **Combined response** — checkboxes that toggle the response previews above the cards.

Buttons (left column):

- **Load ROI CSV** — load an existing match table.
- **Browse save path** — choose where new match edits go.
- **Add new group** — append the currently selected non-empty ROIs as a new `group_cell_id` in `roi_matches.csv`. Recordings left at *No ROI* are omitted from the group.
- **Overwrite selected group** — replace the selected saved row with the current selection and note.
- **Remove selected group** — delete that row from the CSV.
- **Clear ROI selection** — reset card selectors without deleting saved rows.
- **Back to FOV assignment** — return to stage 1 without closing.
- **Save and close** — save edits to the selected group and close.

The **Saved groups** table lists matched groups for the current FOV (Group ID, ROIs, Note) and shows five rows at a time. Selecting a row restores its ROI selections and note; clicking the selected row again clears the table.

**Plot settings** in the left column hold response-row toggles, plot size, epoch visibility (gray / grey / interleave rows hidden by default), smoothing, and normalization. Click the colored recording chips above the plots to toggle that recording's trace.

## Notes and outputs

Notes on FOV cards write to the CSV `note` column for that recording row. The ROI note writes to every row in the selected group. Both notes are audit-only free text and are not used by matching logic.

CSV files written:

- `fov_groups.csv` — recording → FOV id, plus per-recording notes.
- `roi_matches.csv` — one row per recording in a `group_cell_id`, with FOV id, ROI number, status, and note. Rows with `status="matched"` are the same visually assigned cell across recordings; `status="unmatched"` records reviewed singletons so downstream code can distinguish them from unreviewed ROIs.

### A macOS Qt message you can ignore

On macOS you may see `Cell requested for row 0 is out of bounds for table with 8 rows! Resizing table model.` while the ROI Assignment view refreshes the Saved groups table. This comes from Qt's Cocoa accessibility cache rebuilding after Hover Text, VoiceOver, or another accessibility client asks for a table cell during the refresh. twopy's data and rows are not out of bounds.
