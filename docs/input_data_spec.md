# Input Data Spec

This is the current observed input contract for one twopy recording. It is based
on the example two-photon microscope session Gustavo provided.

For detailed file-by-file contents, source-of-truth choices, and response
analysis load recommendations, see `docs/recording_file_schema.md`.

## Session Folder

A recording starts from one timestamped microscope output folder. The stimulus
name and raw movie stem can change.

Required top-level contents:

- `stimulusData/`
- `alignedMovie.mat`
- one `*.tif` raw movie
- one `*_alignment.txt` alignment file
- `defaultAlignChannel.txt`
- `highResPd.mat`
- `imageDescription.mat`
- `imagingResPd.mat`

Optional top-level contents:

- `savedAnalysis/`

`savedAnalysis/` exists only when the recording was analyzed before with the lab
MATLAB package.

## Timing Model

The imaging computer and stimulus computer are separate machines with separate
clocks and different frame rates. The imaging computer records the movie at a
relatively low frame rate and records photodiode signals. The stimulus computer
presents stimuli at a relatively high frame rate and flashes the photodiode at
key timepoints: start, trial transitions, and end.

Different photodiode flash patterns or durations identify different event
types. twopy must use these photodiode signals to align stimulus events to
imaging frames before trial-level response analysis. Nominal frame rates are not
enough by themselves.

## Observed Example Files

Top-level files observed in the example session:

- `.DS_Store`
- `alignedMovie.mat`
- `combo_stim_singles=3s_blank=3s_intensity=20_-8503.4down004.tif`
- `combo_stim_singles=3s_blank=3s_intensity=20_-8503.4down004_ch1_disinterleaved_alignment.txt`
- `defaultAlignChannel.txt`
- `highResPd.mat`
- `imageDescription.mat`
- `imagingResPd.mat`
- `sftpTransferComands.batch`
- `transferComplete.txt`

Files observed under `stimulusData/`:

- `chosenparams.mat`
- `combo_stim_singles=3s_blank=3s_intensity=20.txt`
- `filebackup.zip`
- `fileinfo.txt`
- `metadata.txt`
- `runDetails.mat`
- `seedState.mat`
- `stimParams.mat`
- `stimdata.mat`
- `textStimData.csv`

Files observed under `savedAnalysis/`:

- MATLAB `.mat` files produced by prior lab analysis.

## MATLAB Layer

twopy needs a MATLAB layer because microscope data arrives as MATLAB files.

Current layer goals:

- Inspect `.mat` files without requiring analysis code to know MATLAB details.
- Report variable names, shapes, dtypes, and Python types.
- Support older MAT files through SciPy.
- Support HDF5-backed MAT files through h5py.

Observed MATLAB variables in the example session:

- `alignedMovie.mat`: HDF5-backed, variable `imgFrames_ch1`
- `highResPd.mat`: older MAT file, variable `highResPd`
- `imageDescription.mat`: older MAT file, variable `state`
- `imagingResPd.mat`: older MAT file, variable `imagingResPd`
- `stimulusData/chosenparams.mat`: older MAT file, variable `params`
- `stimulusData/runDetails.mat`: older MAT file, variables `flyId`,
  `genotype`, `rigName`, `rigTemperature`
- `stimulusData/seedState.mat`: older MAT file, variables `Seed`, `State`,
  `Type`
- `stimulusData/stimParams.mat`: older MAT file, variable `stimParams`
- `stimulusData/stimdata.mat`: older MAT file, variable `stimData`
- `savedAnalysis/*.mat`: HDF5-backed prior MATLAB analysis files

Observed TIFF metadata in the example session:

- one raw movie with one TIFF series
- shape `(8334, 127, 256)`
- page count `8334`
- first-page shape `(127, 256)`
- pixel dtype `uint16`
- TIFF tags include `ImageWidth`, `ImageLength`, `BitsPerSample`,
  `SamplesPerPixel`, `XResolution`, `YResolution`, `ResolutionUnit`, and
  `ImageDescription`
- `ImageDescription` contains ScanImage `state.*` fields, including
  `configName`, `software.version`, `acq.linesPerFrame`, `acq.pixelsPerLine`,
  `acq.numberOfFrames`, `acq.numberOfChannelsSave`, `acq.frameRate`,
  `acq.zoomFactor`, `acq.pixelTime`, `acq.msPerLine`, `acq.zStepSize`,
  `acq.scanAngleMultiplierFast`, `acq.scanAngleMultiplierSlow`,
  `acq.scanRotation`, `acq.scanShiftFast`, `acq.scanShiftSlow`, `acq.xstep`,
  `acq.ystep`, and motor absolute positions
- `XResolution` and `YResolution` appear to be display DPI metadata, not
  physical microscope pixel size

The same ScanImage state is available as the MATLAB struct in
`imageDescription.mat`, which should be the primary source for recording
metadata. The raw TIFF metadata path is mainly for audit or raw-frame access.

## Converted Data

twopy will convert MATLAB-derived source data into twopy-owned HDF5 files before
any analysis or processing. Large compressible arrays use gzip compression;
small direct-access datasets such as the mean image stay uncompressed.

The conversion writes the aligned movie to a separate `aligned_movie.h5` file
because it usually dominates file size. It writes acquisition metadata, run
metadata from `stimulusData/runDetails.mat`, stimulus parameters, stimulus
timeline plus column labels, photodiode signals, synchronization metadata, and a
mean image to `recording_data.h5`. The mean image defaults to the full movie and
can be computed over a requested frame range.
