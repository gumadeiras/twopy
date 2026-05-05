# Recording File Reference Schema

This document describes the files twopy expects in a two-photon microscope
recording folder and what each file should be used for. It is based on the
example recording inspected during development. Some values are example-specific
and will vary with stimulus design, recording settings, rig configuration, and
prior MATLAB analysis.

## Source-Of-Truth Rules

- Use `imageDescription.mat` as the primary source for recording acquisition
  metadata.
- Use the raw TIFF only when we need raw interleaved frame data or need to audit
  that its embedded `ImageDescription` matches `imageDescription.mat`.
- Do not treat TIFF `XResolution` or `YResolution` as microscope pixel size.
  In the inspected example they look like display DPI metadata.
- Use `alignedMovie.mat` as the primary movie source for ROI drawing and trace
  extraction.
- Use `stimulusData/stimParams.mat` and `stimulusData/stimdata.mat` as the
  primary stimulus metadata/timeline sources.
- Use `imagingResPd.mat` for frame-resolution stimulus alignment and
  `highResPd.mat` when higher timing precision is needed.
- Treat imaging frames and stimulus frames as independent clocks until the
  photodiode signal aligns them.
- Use `savedAnalysis/` only as optional prior MATLAB analysis output. New
  recordings may not have it.

## Timing And Synchronization

Imaging and stimulus presentation happen on different computers.

The imaging computer records the two-photon movie at a relatively low frame
rate. It also records a photodiode signal. In the converted data,
`photodiode/imaging_res_pd` has one sample per aligned imaging frame, and
`photodiode/high_res_pd` contains a higher-rate photodiode trace for more
precise event detection.

The stimulus computer presents stimuli at a relatively high frame rate. It also
flashes the photodiode at key timepoints, including stimulus start, trial
transitions, and stimulus end. Different flash patterns or flash durations mark
different event types.

Response analysis must use the photodiode to align the two clocks. Do not assign
imaging frames to stimulus trials by assuming nominal frame rates are enough.
The correct workflow is:

1. Load the converted stimulus timeline and photodiode signals.
2. Decode photodiode events from the high-resolution signal when precise timing
   is needed.
3. Map decoded stimulus events onto imaging frames.
4. Extract ROI responses by trial or epoch from that aligned frame map.

twopy now represents this in four GUI-independent layers:

- `load_converted_recording(...)` loads `recording_data.h5` and keeps
  `aligned_movie.h5` lazy.
- ROI masks are saved as twopy-owned HDF5 files and are independent from
  napari.
- `detect_recording_photodiode_events(...)` segments photodiode flashes in
  `high_res_pd` and `imaging_res_pd`, then pairs matching events by order.
- Frame windows are made from paired photodiode event frames and used to split
  ROI traces. Stimulus-specific flash-pattern classification is still separate
  future work.

## Session Folder

One recording is one timestamped microscope output folder.

Example path:

```text
.../combo_stim_singles=3s_blank=3s_intensity=20/2023/10_17/10_02_49
```

The genotype, stimulus name, date, and timestamp vary between experiments.

## Required Top-Level Files

### `stimulusData/`

Directory containing the stimulus program output and stimulus metadata. File
names inside this directory can vary with stimulus design, but the observed
example includes the files described below.

### `alignedMovie.mat`

Aligned imaging movie produced by the lab MATLAB pipeline.

Observed example contents:

- MATLAB v7.3/HDF5 file.
- Dataset: `imgFrames_ch1`.
- Shape: `(4168, 256, 127)`.
- Dtype: `float64`.
- Compression: gzip.
- Chunking: `(1, 64, 127)`.

Use this for ROI drawing and fluorescence trace extraction. Load lazily with
HDF5/chunked reads; do not load the full movie into memory unless the caller
explicitly requests it.

### `*.tif`

Raw microscope TIFF movie. The filename includes stimulus/run details and will
vary. There should be exactly one top-level raw TIFF in a session.

Observed example contents:

- Shape: `(8334, 127, 256)`.
- Dtype: `uint16`.
- Two channels are interleaved: the actual imaging recording and a photodiode
  channel.
- Each page contains a ScanImage `ImageDescription` text block with `state.*`
  assignments.

Use this for raw interleaved channel access. For ordinary recording metadata,
prefer `imageDescription.mat` because it contains the same ScanImage state in a
MATLAB struct that is easier to parse reliably.

### `*_alignment.txt`

Per-frame alignment output from the lab MATLAB pipeline. The filename includes
stimulus/run/channel details and will vary.

Observed example contents:

- 4168 rows.
- First rows look like tab-delimited numeric alignment shifts and scores.
- Row count matches `alignedMovie.mat/imgFrames_ch1`.

Use this to audit or reproduce motion correction decisions.

### `defaultAlignChannel.txt`

Text file identifying the default alignment channel.

Observed example contents:

- `1`

Use this when deciding which channel/movie the MATLAB pipeline aligned.

### `highResPd.mat`

High-resolution photodiode/synchronization signal.

Observed example contents:

- MATLAB variable: `highResPd`.
- Shape: `(529336,)`.
- Dtype: `float64`.

Use this for precise stimulus timing or onset detection when frame-resolution
alignment is not enough. This is the preferred signal for decoding photodiode
flash patterns and event durations.

### `imageDescription.mat`

Primary recording acquisition metadata.

Observed example contents:

- MATLAB variable: `state`.
- ScanImage state struct.
- Important observed fields:
  - `configName`: `256x128_0.5ms_fastAcquisition`
  - `software.version`: `3.8`
  - `acq.linesPerFrame`: `128`
  - `acq.pixelsPerLine`: `256`
  - `acq.numberOfFrames`: `4167`
  - `acq.numberOfChannelsSave`: `2`
  - `acq.frameRate`: `13.0208333333333`
  - `acq.zoomFactor`: `9`
  - `acq.pixelTime`: `1.6e-06`
  - `acq.msPerLine`: `0.6`
  - `acq.zStepSize`: `1`
  - `acq.scanAngleMultiplierFast`: `1`
  - `acq.scanAngleMultiplierSlow`: `0.57744`
  - `motor.absXPosition`: `199006.1`
  - `motor.absYPosition`: `-4694.3`
  - `motor.absZPosition`: `-8503.4`

Use this as the canonical source for acquisition settings. Values vary with the
recording. Physical pixel size is not directly established from the inspected
fields; it likely needs scanner/objective calibration.

### `imagingResPd.mat`

Photodiode/synchronization signal sampled at imaging-frame resolution.

Observed example contents:

- MATLAB variable: `imagingResPd`.
- Shape: `(4168,)`.
- Dtype: `float64`.
- Row count matches `alignedMovie.mat/imgFrames_ch1`.

Use this to align each imaging frame to stimulus timing after photodiode events
have been decoded.

## Optional Top-Level Directory

### `savedAnalysis/`

Prior MATLAB analysis output. This directory exists only if somebody already
analyzed the recording with the lab MATLAB package.

Observed example files:

- `WatershedRegionRestrictedRoiExtraction_33_20_10_17_10_23.mat`
- `WatershedRegionRestrictedRoiExtraction_42_18_10_17_10_23.mat`
- `WatershedRoiExtraction_28_12_10_17_10_23.mat`
- `WatershedRoiExtraction_35_15_10_17_10_23.mat`

Observed common contents:

- `lastRoi/roiMaskInitial`: prior ROI mask image.
- `lastRoi/timeByRoisInitial`: object reference to ROI traces.
- `lastRoi/epochList`: epoch assignment by frame/time index.
- `lastRoi/epochStartTimes`: per-epoch start frames or time indices.
- `lastRoi/epochDurations`: per-epoch durations.
- `lastRoi/params`: stimulus parameters copied into the analysis output.
- `lastRoi/stim`: stimulus table copied into the analysis output.
- `lastRoi/runDetails`: run metadata copied into the analysis output.

Observed trace matrix shapes vary by analysis output:

- `(44, 3909)`
- `(30, 3909)`
- `(1221, 3909)`

Use this for fast inspection of prior ROI results. Do not require it for new
analysis.

## Observed `stimulusData/` Files

### `stimulusData/metadata.txt`

Text metadata emitted by the stimulus system.

Observed example contents include:

- Run date/time-like folder: `2023\10_17\10_03_15`
- Stimulus parameter file path.
- Stimulus lookup file path.
- View locations file path.
- Event log lines.

Use this as human-readable run provenance. Treat format as stimulus-system
output, not as the primary structured stimulus table.

### `stimulusData/<stimulus_name>.txt`

Stimulus parameter text file. The name varies with the stimulus.

Observed example file:

```text
combo_stim_singles=3s_blank=3s_intensity=20.txt
```

Observed example contents:

- `PARAMS 8`
- `EPOCHS 11`
- Epoch names:
  - `Gray Interleave`
  - `LR20`
  - `RL20`
  - `LR20Single`
  - `RL20Single`
  - `LR20Both`
  - `Empty`
- Stimulus fields such as `stimtype`, `antenna`, `intensity`, `duration`, and
  `ordertype`.

Use this as a human-readable stimulus parameter source. Prefer
`stimParams.mat` for structured loading.

### `stimulusData/textStimData.csv`

CSV stimulus timeline.

Observed example contents:

- Rows: `18021`.
- Columns: `34`.
- Time range: `0` to `300.160252` seconds.
- Median time step: about `0.016666` seconds.
- Epoch counts:
  - epoch `0`: `21`
  - epoch `1`: `9000`
  - epoch `2`: `1440`
  - epoch `3`: `1620`
  - epoch `4`: `1620`
  - epoch `5`: `1440`
  - epoch `6`: `1440`
  - epoch `7`: `1440`

Use this for stimulus timing and per-row epoch labels when a CSV workflow is
more convenient than MATLAB loading.

### `stimulusData/stimdata.mat`

MATLAB stimulus timeline.

Observed example contents:

- MATLAB variable: `stimData`.
- Shape: `(18021, 35)`.
- Dtype: `float64`.
- Column 1: time in seconds.
- Column 2: stimulus frame number.
- Column 3: epoch number.
- Other columns contain closed-loop/stimulus/flash values.

Use this as the primary structured stimulus timeline.

### `stimulusData/stimParams.mat`

MATLAB stimulus epoch parameter structs.

Observed example contents:

- MATLAB variable: `stimParams`.
- Seven epoch parameter structs.
- Common fields include:
  - `epochName`
  - `stimtype`
  - `antenna`
  - `intensity`
  - `duration`
  - `ordertype`
  - `repeats`
  - `StartFrame`
  - `framerate`

Use this as the primary structured description of stimulus epochs.

### `stimulusData/chosenparams.mat`

MATLAB selected stimulus parameters.

Observed example contents:

- MATLAB variable: `params`.
- Seven structs matching the observed stimulus epochs.
- Includes fields similar to `stimParams.mat`, plus example-specific fields
  such as `totalTime`.

Use this to audit the exact parameters chosen for the run.

### `stimulusData/runDetails.mat`

Run metadata.

Observed example contents:

- `runNumber`: `1`
- `genotype`: `gh146gal4_w;uasg6f_+;orcolexa,lexaopcschrimsontdtomato_+`
- `flyId`: `10845513951022698826`
- `rigName`: `OdorRig`
- `rigTemperature`: `0`

Use this for experiment provenance.

### `stimulusData/seedState.mat`

Random seed and random-generator state.

Observed example contents:

- `Type`: `twister`
- `Seed`: `2015372430`
- `State`: vector of 625 `uint32` values

Use this for stimulus reproducibility.

### `stimulusData/filebackup.zip`

Archive of stimulus/runtime code files.

Use this as provenance for the exact stimulus code state. Do not extract or
modify it during normal analysis.

### `stimulusData/fileinfo.txt`

Text listing backed-up MATLAB/stimulus files and timestamps.

Use this as provenance. It is not required for twopy conversion.

## Conversion Load Set

Minimum source-file load set for conversion:

- `alignedMovie.mat`
- `highResPd.mat`
- `imageDescription.mat`
- `stimulusData/stimParams.mat`
- `stimulusData/stimdata.mat`
- `imagingResPd.mat`

twopy reads this set with `load_source_conversion_inputs(recording_dir)` and
writes converted HDF5 files with `convert_recording_to_twopy(...)`. The source
files are read-only conversion inputs. Response analysis should use the
converted HDF5 files rather than reading MATLAB files directly.

The converted `recording_data.h5` file contains:

- `movie`: attributes pointing to the separate aligned movie file and dataset.
- `movie/mean_image`: mean image generated during conversion.
- `metadata`: selected acquisition fields as HDF5 attributes.
- `stimulus/timeline`: numeric stimulus timeline.
- `stimulus/parameters_json`: stimulus epoch parameters.
- `photodiode`: synchronization metadata explaining the two-computer timing
  model.
- `photodiode/imaging_res_pd`: frame-resolution photodiode vector, one sample
  per aligned imaging frame.
- `photodiode/high_res_pd`: high-resolution photodiode vector for precise
  photodiode event detection.

The converted `aligned_movie.h5` file contains:

- `movie/aligned`: copied aligned movie.

The twopy ROI HDF5 file contains:

- `masks`: boolean ROI masks with shape `(rois, x, y)` in aligned movie
  coordinates.
- `labels`: one human-readable label per ROI.

ROI trace extraction reads `movie/aligned` in chunks and writes frame-by-ROI
arrays in memory for the requested frame range. Frame-window response splitting
uses explicit imaging-frame boundaries, usually from paired photodiode events.

By default, the mean image uses the entire aligned movie. Callers can pass
`mean_start_frame` and `mean_stop_frame` to compute it over a frame range.

Optional prior-analysis shortcut:

- `savedAnalysis/*/lastRoi/timeByRoisInitial`
- `savedAnalysis/*/lastRoi/roiMaskInitial`
- `savedAnalysis/*/lastRoi/epochList`
- `savedAnalysis/*/lastRoi/epochStartTimes`
- `savedAnalysis/*/lastRoi/epochDurations`
- `savedAnalysis/*/lastRoi/params`

Raw-data fallback/audit:

- top-level raw `*.tif`

## Analysis Output Routing

`config.yml` contains `analysis_output`.

- `analysis_output: source` writes twopy outputs into a `twopy/` folder inside
  the recording folder.
- `analysis_output: /some/output/root` mirrors the recording directory structure
  relative to `data_path` under that output root.
- `convert_recording_to_twopy(recording)` uses this configured output routing by
  default. Passing `output_dir` to that function is an explicit one-call
  override.

Example:

```text
data_path: /Volumes/magic/clarklab/2p_microscope_data
analysis_output: /Volumes/magic/clarklab/twopy_outputs
recording: /Volumes/magic/clarklab/2p_microscope_data/fly/stim/2023/10_17/10_02_49
output: /Volumes/magic/clarklab/twopy_outputs/fly/stim/2023/10_17/10_02_49
```

## Frame Count Audit

Conversion writes a `frame_counts` group into `recording_data.h5`.

This group exists because ScanImage acquisition metadata uses a slightly
different frame-count convention from the aligned movie and frame-resolution
photodiode vector. Do not "fix" this by forcing every count to be identical.

Random sampling on 2026-05-05 checked 20 candidate recordings from the mounted
lab data paths. Among recordings whose `alignedMovie.mat` could be opened as
HDF5, the pattern was consistent:

- `aligned_movie_frames == imaging_res_pd_samples`
- `acq.numberOfFrames == aligned_movie_frames - 1`

Two sampled Dropbox recordings had `alignedMovie.mat` files that were not
HDF5-readable by the current loader, so they were skipped for this specific
count comparison.

Interpretation:

- `aligned_movie_frames` is the frame count twopy uses for ROI extraction.
- `imaging_res_pd_samples` must match `aligned_movie_frames` exactly because it
  is the frame-resolution photodiode vector used to map imaging frames to
  stimulus timing.
- `acq.numberOfFrames` may be equal to `aligned_movie_frames` or exactly one
  less. The one-frame difference is treated as a known ScanImage metadata
  convention, not as a dropped imaging frame.

twopy stores all counts and deltas so response-analysis code can audit the frame
contract before assigning trials. Conversion fails if `imaging_res_pd_samples`
does not match `aligned_movie_frames`, or if `acq.numberOfFrames` differs by
anything other than `0` or `-1`.
