# Input data spec

This is the short input contract for one twopy recording. It is intentionally high-level. The canonical file-by-file reference — observed example details, MATLAB / TIFF metadata, conversion load set, and converted HDF5 schema — lives in [Recording file reference](recording_file_schema.md).

## Recording folder contract

A recording starts as one timestamped microscope output folder. Stimulus names, raw movie stems, dates, and timestamps vary between experiments, so twopy detects variable files by stable patterns rather than hard-coded full names.

### Required

- `stimulusData/`
- `alignedMovie.mat`
- one raw `*.tif` movie
- one `*_alignment.txt` alignment file
- `defaultAlignChannel.txt`
- `highResPd.mat`
- `imageDescription.mat`
- `imagingResPd.mat`

### Optional

- `savedAnalysis/` — present only when the recording was already analyzed with the lab MATLAB package. Useful for inspecting prior results, but new twopy analysis must not require it.

## Source-of-truth summary

- `imageDescription.mat` — primary acquisition metadata.
- `alignedMovie.mat` — primary movie for ROI drawing and trace extraction.
- `stimulusData/stimParams.mat` (or older `stimulusData/chosenparams.mat`) — primary structured stimulus epoch metadata.
- `stimulusData/stimdata.mat` — primary structured stimulus time series.
- `stimulusData/filebackup.zip` — exact stimulus / runtime code provenance and source for stimulus-specific column meanings.
- `highResPd.mat` — preferred photodiode signal for precise event detection.
- `imagingResPd.mat` — frame-resolution photodiode vector for mapping events to imaging frames.
- Raw `*.tif` — raw interleaved frame access and metadata audit path.
- `savedAnalysis/` — optional prior MATLAB analysis output.

Do not treat TIFF `XResolution` / `YResolution` as microscope pixel size. Physical pixel size needs scanner / objective calibration beyond the observed recording metadata.

## Timing contract

Imaging and stimulus presentation use separate computers, clocks, and frame rates. The imaging computer records the movie and photodiode signals. The stimulus computer presents stimuli. It flashes the photodiode at stimulus start, trial transitions, and stimulus end.

Response analysis must align stimulus events to imaging frames through the photodiode signal. **Nominal frame rates are not enough.**

The expected workflow:

1. Load converted stimulus data and photodiode signals.
2. Detect photodiode events, using `highResPd.mat` when precise timing is needed.
3. Pair those events with `imagingResPd.mat` frames.
4. Classify stimulus windows against `stimulusData/stimdata.mat`.
5. Split ROI traces by explicit imaging-frame windows.

## Conversion contract

Analysis runs on twopy-owned converted HDF5 files, not directly on source MAT / TIFF files. Source files stay read-only conversion inputs.

Conversion writes:

- `recording_data.h5` — acquisition metadata, frame-count audit data, and run metadata. It also contains stimulus data, labels, parameters, function lookup data, and column metadata. It contains photodiode signals, synchronization metadata, the mean image in Python image order, and alignment-valid crop bounds.
- `aligned_movie.h5` — the aligned movie in `movie/aligned`, stored in Python image order matching MATLAB display and kept separate because the movie dominates file size.
- ROI HDF5 files — twopy-owned ROI masks and labels in the same Python image order as the converted movie, independent from napari.

Large compressible arrays use gzip compression. Small direct-access datasets, such as the mean image, stay uncompressed. The mean image defaults to the full aligned movie and can be computed over a requested frame range.

## Detail index

See [Recording file reference](recording_file_schema.md) for:

- observed example files and non-contract acquisition / transfer artifacts
- MATLAB loader goals and observed MAT file formats
- per-file variables, shapes, dtypes, and usage rules
- TIFF tag and ScanImage field details
- `stimulusData/stimdata.mat` column definitions
- conversion load set versus full folder contents
- converted HDF5 group / dataset layout
- analysis output routing
- frame-count audit rules
