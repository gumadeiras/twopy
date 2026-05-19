# Recording File Reference Schema

This document describes the files twopy expects in a two-photon microscope recording folder and what each file should be used for. It is based on the example recording inspected during development. Some values are example-specific and will vary with stimulus design, recording settings, rig configuration, and prior MATLAB analysis.

## Source-Of-Truth Rules

- Use `imageDescription.mat` as the primary source for recording acquisition metadata.
- Use the raw TIFF only when we need raw interleaved frame data or need to audit that its embedded `ImageDescription` matches `imageDescription.mat`.
- Do not treat TIFF `XResolution` or `YResolution` as microscope pixel size. In the inspected example they look like display DPI metadata.
- Use `alignedMovie.mat` as the primary movie source for ROI drawing and trace extraction.
- Use `stimulusData/stimParams.mat` and `stimulusData/stimdata.mat` as the primary stimulus metadata/data sources.
- Use `imagingResPd.mat` for frame-resolution stimulus alignment and `highResPd.mat` when higher timing precision is needed.
- Treat imaging frames and stimulus frames as independent clocks until the photodiode signal aligns them.
- Use `savedAnalysis/` only as optional prior MATLAB analysis output. New recordings may not have it.

## MATLAB File Layer

twopy needs a MATLAB file layer because microscope source data arrives as a mix of older MAT files and HDF5-backed MAT files.

The MATLAB layer should:

- Inspect `.mat` files without requiring analysis code to know MATLAB details.
- Report variable names, shapes, dtypes, and Python types.
- Support older MAT files through SciPy.
- Support HDF5-backed MAT files through h5py.

Observed formats in the inspected example:

- `alignedMovie.mat`: HDF5-backed MAT file.
- `savedAnalysis/*.mat`: HDF5-backed prior MATLAB analysis files.
- `highResPd.mat`, `imageDescription.mat`, `imagingResPd.mat`, `stimulusData/chosenparams.mat`, `stimulusData/runDetails.mat`, `stimulusData/seedState.mat`, `stimulusData/stimParams.mat`, and `stimulusData/stimdata.mat`: older MAT files.

## Timing And Synchronization

Imaging and stimulus presentation happen on different computers.

The imaging computer records the two-photon movie at a relatively low frame rate. It also records a photodiode signal. In the converted data, `photodiode/imaging_res_pd` has one sample per aligned imaging frame, and `photodiode/high_res_pd` contains a higher-rate photodiode trace for more precise event detection.

The stimulus computer presents stimuli at a relatively high frame rate. It also flashes the photodiode at key timepoints, including stimulus start, trial transitions, and stimulus end. Different flash patterns or flash durations mark different event types.

Response analysis must use the photodiode to align the two clocks. Do not assign imaging frames to stimulus trials by assuming nominal frame rates are enough. The correct workflow is:

1. Load the converted stimulus data and photodiode signals.
2. Decode photodiode events from the high-resolution signal when precise timing is needed.
3. Map decoded stimulus events onto imaging frames.
4. Extract ROI responses by trial or epoch from that aligned frame map.

twopy now represents this in GUI-independent layers:

- `load_converted_recording(...)` loads `recording_data.h5` and keeps `aligned_movie.h5` lazy.
- ROI masks are saved as twopy-owned HDF5 files and are independent from napari.
- `detect_recording_photodiode_events(...)` segments photodiode flashes in `high_res_pd` and `imaging_res_pd`, then pairs matching events by order.
- `classify_recording_photodiode_events(...)` cross-checks the paired events against `stimulus/data` `photodiode_flash` segments, classifies stimulus start, trial-transition, and stimulus-end events, then returns classified stimulus windows with epoch metadata.
- Frame windows are made from classified photodiode event frames and used to split ROI traces.

## Session Folder

One recording is one timestamped microscope output folder.

Example path:

```text
.../combo_stim_singles=3s_blank=3s_intensity=20/2023/10_17/10_02_49
```

The genotype, stimulus name, date, and timestamp vary between experiments.

Observed non-contract top-level files in the inspected example:

- `.DS_Store`
- `sftpTransferComands.batch`
- `transferComplete.txt`

Treat these as acquisition or transfer artifacts, not twopy input contracts.

## Required Top-Level Files

### `stimulusData/`

Directory containing the stimulus program output and stimulus metadata. File names inside this directory can vary with stimulus design, but the observed example includes the files described below.

### `alignedMovie.mat`

Aligned imaging movie produced by the lab MATLAB pipeline.

Observed example contents:

- MATLAB v7.3/HDF5 file.
- Dataset: `imgFrames_ch1`.
- Shape: `(4168, 256, 127)`.
- Dtype: `float64`.
- Compression: gzip.
- Chunking: `(1, 64, 127)`.

Use this for ROI drawing and fluorescence trace extraction. Load lazily with HDF5/chunked reads; do not load the full movie into memory unless the caller explicitly requests it.

### `*.tif`

Raw microscope TIFF movie. The filename includes stimulus/run details and will vary. There should be exactly one top-level raw TIFF in a session.

Observed example contents:

- One TIFF series.
- Shape: `(8334, 127, 256)`.
- Page count: `8334`.
- First-page shape: `(127, 256)`.
- Dtype: `uint16`.
- Two channels are interleaved: the actual imaging recording and a photodiode channel.
- TIFF tags include `ImageWidth`, `ImageLength`, `BitsPerSample`, `SamplesPerPixel`, `XResolution`, `YResolution`, `ResolutionUnit`, and `ImageDescription`.
- Each page contains a ScanImage `ImageDescription` text block with `state.*` assignments.

Use this for raw interleaved channel access. For ordinary recording metadata, prefer `imageDescription.mat` because it contains the same ScanImage state in a MATLAB struct that is easier to parse reliably.

### `*_alignment.txt`

Per-frame alignment output from the lab MATLAB pipeline. The filename includes stimulus/run/channel details and will vary.

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

Use this for precise stimulus timing or onset detection when frame-resolution alignment is not enough. This is the preferred signal for decoding photodiode flash patterns and event durations.

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
  - `acq.scanRotation`
  - `acq.scanShiftFast`
  - `acq.scanShiftSlow`
  - `acq.xstep`
  - `acq.ystep`
  - `motor.absXPosition`: `199006.1`
  - `motor.absYPosition`: `-4694.3`
  - `motor.absZPosition`: `-8503.4`

Use this as the canonical source for acquisition settings. Values vary with the recording. Physical pixel size is not directly established from the inspected fields; it needs scanner/objective calibration. twopy can use converted fields such as `configName`, `acq.zoomFactor`, scanner labels, timing, and run-level rig names to preselect a pixel-calibration profile when that evidence identifies one measured calibration group; measured pixel size still comes from the tracked calibration registry.

### `imagingResPd.mat`

Photodiode/synchronization signal sampled at imaging-frame resolution.

Observed example contents:

- MATLAB variable: `imagingResPd`.
- Shape: `(4168,)`.
- Dtype: `float64`.
- Row count matches `alignedMovie.mat/imgFrames_ch1`.

Use this to align each imaging frame to stimulus timing after photodiode events have been decoded.

## Optional Top-Level Directory

### `savedAnalysis/`

Prior MATLAB analysis output. This directory exists only if somebody already analyzed the recording with the lab MATLAB package.

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

Use this for fast inspection of prior ROI results. Do not require it for new analysis.

## Observed `stimulusData/` Files

### `stimulusData/metadata.txt`

Text metadata emitted by the stimulus system.

Observed example contents include:

- Run date/time-like folder: `2023\10_17\10_03_15`
- Stimulus parameter file path.
- Stimulus lookup file path.
- View locations file path.
- Event log lines.

Use this as human-readable run provenance. Treat format as stimulus-system output, not as the primary structured stimulus table.

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
- Stimulus fields such as `stimtype`, `antenna`, `intensity`, `duration`, and `ordertype`.

Use this as a human-readable stimulus parameter source. Prefer `stimParams.mat` for structured loading.

### `stimulusData/textStimData.csv`

CSV stimulus data.

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

Use this for stimulus timing and per-row epoch labels when a CSV workflow is more convenient than MATLAB loading.

### `stimulusData/stimdata.mat`

MATLAB stimulus data.

Observed example contents:

- MATLAB variable: `stimData`.
- Shape: `(18021, 35)`.
- Dtype: `float64`.
- Column 1, `time_seconds`: stimulus-computer time in seconds, written as `Q.timing.flipt - Q.timing.t0`.
- Column 2, `stimulus_frame_number`: stimulus-computer frame number, written as `Q.timing.framenumber`.
- Column 3, `epoch_number`: current stimulus epoch number, written as `Q.stims.currStimNum`.
- Columns 4-13, `closed_loop_01` through `closed_loop_10`: closed-loop slots written from `Q.stims.stimData.cl(1:10)`. They are initialized as zeros in `SetupLEDStimulus.m`; whether they are meaningful depends on the stimulus code used during the recording.
- Columns 14-33, `stimulus_specific_01` through `stimulus_specific_20`: stimulus-specific slots written from `Q.stims.stimData.mat(1:20)`. Their meaning is defined by the MATLAB stimulus function used for that epoch.
- Column 34, `photodiode_flash`: photodiode flash value written from `Q.stims.stimData.flash`. In `RunLEDStimulus.m`, this is set high while `framesSinceEpochChange < 11`, and the stimulus function can pass it to the LabJack photodiode output.
- Column 35, `trailing_empty`: trailing empty CSV field caused by `WriteStimData.m` writing a final comma after `photodiode_flash`. Do not interpret this as stimulus data.

Use this as the primary structured stimulus data. twopy also stores these code-derived labels in `stimulus/data_column_names`.

The source of truth for the stable column order is the backed-up MATLAB code in each recording:

- `stimulusData/filebackup/utils/ledUtils/RunLEDStimulus.m` initializes flash events and calls the selected stimulus function.
- `stimulusData/filebackup/utils/ledUtils/SetupLEDStimulus.m` initializes `Q.stims.stimData.cl` with 10 slots and `Q.stims.stimData.mat` with 20 slots.
- `stimulusData/filebackup/utils/ledUtils/WriteStimData.m` writes the stable column order above.

Stimulus-specific columns are decoded from the same backed-up code:

1. Read `stimulusData/chosenparams.mat` or `stimulusData/stimParams.mat`.
2. Use each epoch's `stimtype` number.
3. Map that number through `stimulusData/filebackup/paramfiles/stimulus_lookup.txt`.
4. Read the matching MATLAB file in `stimulusData/filebackup/stimfunctions/`.
5. Inspect assignments to `stimData.mat(...)`.

In the inspected example, `stimtype` values map as follows:

- `62002`: `LEDMovingBars`
- `62005`: `LEDMovingBarsSingleAntenna`
- `62006`: `LEDMovingBarsBothAntenna`

Those three stimulus functions assign the same first seven stimulus-specific slots:

- `stimulus_specific_01`: configured epoch antenna, `p.antenna`.
- `stimulus_specific_02`: configured LED intensity, `p.intensity`.
- `stimulus_specific_03`: configured epoch duration, `p.duration`.
- `stimulus_specific_04`: antenna value active on this stimulus frame.
- `stimulus_specific_05`: LabJack-read left LED value, `stimRead.LEFT`.
- `stimulus_specific_06`: LabJack-read right LED value, `stimRead.RIGHT`.
- `stimulus_specific_07`: LabJack-read photodiode value, `stimRead.PD`.
- `stimulus_specific_08` through `stimulus_specific_20`: unused/zero in these example stimulus functions.

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
- Includes fields similar to `stimParams.mat`, plus example-specific fields such as `totalTime`.

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

Use this as provenance for the exact stimulus code state. During conversion, twopy reads `paramfiles/stimulus_lookup.txt` and the stimulus functions used by the recording to decode experiment-specific `stimulus_specific_*` slot metadata. Do not extract or modify it during normal analysis.

### `stimulusData/fileinfo.txt`

Text listing backed-up MATLAB/stimulus files and timestamps.

Use this as provenance. It is not required for twopy conversion.

## Conversion Load Set

Minimum source-file load set for conversion:

- `alignedMovie.mat`
- `highResPd.mat`
- `imageDescription.mat`
- `stimulusData/runDetails.mat`
- `stimulusData/stimParams.mat` or older `stimulusData/chosenparams.mat`
- `stimulusData/stimdata.mat`
- `stimulusData/filebackup.zip`
- `imagingResPd.mat`
- one `*_alignment.txt` file

twopy reads this set with `load_source_conversion_inputs(recording_dir)` and writes converted HDF5 files with `convert_recording_to_twopy(...)`. The source files are read-only conversion inputs. Response analysis should use the converted HDF5 files rather than reading MATLAB files directly.

This load set is smaller than the full required top-level folder list. The raw `*.tif` movie and `defaultAlignChannel.txt` are still required in the source recording because they support raw-data access, metadata audits, and alignment review, but normal conversion does not need to read them.

The converted `recording_data.h5` file contains:

- `movie`: attributes pointing to the separate aligned movie file and dataset.
- `movie/mean_image`: uncompressed mean image generated during conversion. This is one image, so compression is unnecessary complexity.
- `movie/alignment_valid_crop`: half-open spatial crop bounds computed from stimulus-bounded alignment offsets. The converted aligned movie stays full-frame; analysis code uses this crop when it should ignore invalid motion-border pixels.
- `metadata`: selected acquisition fields as HDF5 attributes.
- `run`: stimulus-run metadata from `runDetails.mat`, converted to snake_case twopy field names such as `rig_name`, `run_number`, `fly_id`, and `rig_temperature`.
- `stimulus/data`: numeric stimulus data.
- `stimulus/data_column_names`: one label per data column.
- `stimulus/parameters_json`: stimulus epoch parameters.
- `stimulus/function_lookup_json`: `stimtype` to backed-up MATLAB stimulus function names used by this recording.
- `stimulus/stimulus_specific_columns_json`: per-`stimtype` assignments from `stimData.mat(N)` slots to the source MATLAB expression and line number. Use `map_stimulus_specific_column(...)` to map a stable column name such as `stimulus_specific_04` to the per-`stimtype` meaning for scripts.
- `photodiode`: synchronization metadata explaining the two-computer timing model.
- `photodiode/imaging_res_pd`: frame-resolution photodiode vector, one sample per aligned imaging frame.
- `photodiode/high_res_pd`: high-resolution photodiode vector for precise photodiode event detection.

The converted `aligned_movie.h5` file contains:

- `movie/aligned`: copied aligned movie.

The twopy ROI HDF5 file contains:

- `masks`: boolean ROI masks with shape `(rois, x, y)` in aligned movie coordinates.
- `labels`: one human-readable label per ROI.

Trace extraction uses these rules:

- `extract_roi_traces` reads `movie/aligned` in chunks and returns full-frame ROI traces for the requested frame range.
- `extract_background_corrected_roi_traces` reads only the selected crop from the aligned movie. With the default `alignment_valid_crop`, background pixels and ROI pixels come from valid motion-aligned pixels instead of invalid border pixels.
- ROI masks stay full-frame. Crop-domain analysis checks that every ROI pixel lies inside the selected crop before extracting traces.
- Response splitting uses imaging-frame boundaries, usually from paired photodiode events.

By default, the mean image uses the entire aligned movie. Callers can pass `mean_start_frame` and `mean_stop_frame` to compute it over a frame range.

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

`config.yml` contains `analysis_caching`, `analysis_cache_dir`, and `analysis_output`.

- `analysis_caching: true` keeps converted recordings and interactive analysis work under `analysis_cache_dir`, mirrored relative to `data_path` for normal lab recordings. Recordings outside `data_path` use a stable `_external` cache folder.
- `analysis_caching: false` uses `analysis_output` directly as the work directory.
- `analysis_output: source` publishes saved twopy outputs into a `twopy/` folder inside the recording folder.
- `analysis_output: /some/output/root` mirrors the recording directory structure relative to `data_path` under that output root.
- `convert_recording_to_twopy(recording)` uses the configured work directory by default. Passing `output_dir` to that function is an explicit one-call override.

Example:

```text
data_path: /Volumes/magic/clarklab/2p_microscope_data
analysis_caching: true
analysis_cache_dir: ~/.cache/twopy/recordings
analysis_output: /Volumes/magic/clarklab/twopy_outputs
recording: /Volumes/magic/clarklab/2p_microscope_data/fly/stim/2023/10_17/10_02_49
work: ~/.cache/twopy/recordings/fly/stim/2023/10_17/10_02_49
publish: /Volumes/magic/clarklab/twopy_outputs/fly/stim/2023/10_17/10_02_49
```

## Frame Count Audit

Conversion writes a `frame_counts` group into `recording_data.h5`.

This group exists because ScanImage acquisition metadata uses a slightly different frame-count convention from the aligned movie and frame-resolution photodiode vector. Do not "fix" this by forcing every count to be identical.

Random sampling on 2026-05-05 checked 20 candidate recordings from the mounted lab data paths. Among recordings whose `alignedMovie.mat` could be opened as HDF5, the pattern was consistent:

- `aligned_movie_frames == imaging_res_pd_samples`
- `acq.numberOfFrames == aligned_movie_frames - 1`

Two sampled Dropbox recordings had `alignedMovie.mat` files that were not HDF5-readable by the current loader, so they were skipped for this specific count comparison.

Interpretation:

- `aligned_movie_frames` is the frame count twopy uses for ROI extraction.
- `imaging_res_pd_samples` must match `aligned_movie_frames` exactly because it is the frame-resolution photodiode vector used to map imaging frames to stimulus timing.
- `acq.numberOfFrames` may be equal to `aligned_movie_frames` or exactly one less. The one-frame difference is treated as a known ScanImage metadata convention, not as a dropped imaging frame.

twopy stores all counts and deltas so response-analysis code can audit the frame contract before assigning trials. Conversion fails if `imaging_res_pd_samples` does not match `aligned_movie_frames`, or if `acq.numberOfFrames` differs by anything other than `0` or `-1`.
