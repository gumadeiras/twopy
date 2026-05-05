"""Typed data objects used by twopy conversion.

Inputs: MATLAB/HDF5 source metadata and loaded NumPy arrays.
Outputs: immutable Python objects passed between conversion steps.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt

from twopy.conversion.frame_ranges import normalize_frame_range
from twopy.session import TwoPhotonSessionFiles


@dataclass(frozen=True)
class AlignedMovieSource:
    """Lazy source for the aligned imaging movie before conversion.

    Inputs: HDF5-backed MATLAB file path and dataset metadata.
    Outputs: movie shape/dtype metadata plus methods for reading frames.

    This object is only for conversion. Downstream analysis should read the
    copied movie from the twopy aligned-movie HDF5 file.
    """

    path: Path
    dataset_name: str
    shape: tuple[int, ...]
    dtype: str
    chunks: tuple[int, ...] | None
    compression: str | None

    def read_frames(
        self,
        start: int | None = None,
        stop: int | None = None,
    ) -> npt.NDArray[np.float64]:
        """Read a frame range from the aligned source movie.

        Args:
            start: First frame index to read. ``None`` starts at frame zero.
            stop: Exclusive stop frame index. ``None`` reads through the end.

        Returns:
            NumPy array with shape ``(frames, x, y)`` for the requested range.

        This method exists for conversion checks and small previews, not for
        long-lived analysis workflows.
        """
        frame_slice = slice(start, stop)
        with h5py.File(self.path, "r") as mat_file:
            dataset = mat_file[self.dataset_name]
            return cast(npt.NDArray[np.float64], dataset[frame_slice, :, :])

    def mean_image(
        self,
        *,
        start_frame: int | None = None,
        stop_frame: int | None = None,
        chunk_frames: int = 128,
    ) -> npt.NDArray[np.float64]:
        """Compute a mean image over a frame range without loading all frames.

        Args:
            start_frame: First frame included in the mean. ``None`` uses zero.
            stop_frame: Exclusive stop frame. ``None`` uses the movie end.
            chunk_frames: Number of frames read per chunk.

        Returns:
            Float64 mean image with shape ``(x, y)``.

        The computation streams the movie in chunks, keeping memory use bounded
        by ``chunk_frames`` rather than the full movie length.
        """
        start, stop = normalize_frame_range(
            frame_count=self.shape[0],
            start_frame=start_frame,
            stop_frame=stop_frame,
        )
        if chunk_frames < 1:
            msg = f"chunk_frames must be at least 1; got {chunk_frames}"
            raise ValueError(msg)

        spatial_shape = self.shape[1:]
        accumulator = np.zeros(spatial_shape, dtype=np.float64)
        with h5py.File(self.path, "r") as mat_file:
            dataset = mat_file[self.dataset_name]
            for chunk_start in range(start, stop, chunk_frames):
                chunk_stop = min(chunk_start + chunk_frames, stop)
                block = np.asarray(dataset[chunk_start:chunk_stop, :, :])
                accumulator += block.sum(axis=0, dtype=np.float64)

        return accumulator / float(stop - start)


@dataclass(frozen=True)
class AcquisitionMetadata:
    """Recording acquisition metadata from ``imageDescription.mat``.

    Inputs: MATLAB ScanImage ``state`` struct.
    Outputs: selected scalar fields in a dictionary.

    The raw MATLAB struct is not exposed because it is source-format data. The
    converted HDF5 file stores this dictionary as HDF5 attributes.
    """

    path: Path
    fields: dict[str, object]


@dataclass(frozen=True)
class StimulusParameters:
    """Stimulus epoch definitions from ``stimParams.mat``.

    Inputs: MATLAB ``stimParams`` structs.
    Outputs: one plain dictionary per epoch.

    Field names and values vary with the stimulus design, so dictionaries keep
    the conversion auditable without inventing a premature fixed schema.
    """

    path: Path
    epochs: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class RunMetadata:
    """Stimulus-run metadata from ``stimulusData/runDetails.mat``.

    Inputs: MATLAB run details produced by the stimulus computer.
    Outputs: plain scalar fields such as rig name, genotype, and run number.

    This is separate from acquisition metadata because it describes the
    stimulus run and rig identity, not ScanImage acquisition settings.
    """

    path: Path
    fields: dict[str, object]


@dataclass(frozen=True)
class StimulusData:
    """Per-stimulus-frame data from ``stimdata.mat``.

    Inputs: MATLAB ``stimData`` table.
    Outputs: the full numeric table, column labels, and common columns as
    direct arrays.

    The stimulus data comes from the stimulus presentation computer. Its
    clock is independent from the imaging computer clock, so response analysis
    must use photodiode synchronization before assigning imaging frames to
    stimulus trials.

    The first observed columns are time, stimulus frame number, and epoch.
    Later columns are protocol-specific closed-loop/stimulus/flash values, so
    twopy stores auditable column labels with the full table.
    """

    path: Path
    data: npt.NDArray[np.float64]
    column_names: tuple[str, ...]
    time_seconds: npt.NDArray[np.float64]
    frame_numbers: npt.NDArray[np.int64]
    epoch_numbers: npt.NDArray[np.int64]


@dataclass(frozen=True)
class PhotodiodeSignals:
    """Photodiode signals used to align imaging frames and stimulus frames.

    Inputs: frame-resolution and high-resolution MATLAB photodiode files.
    Outputs: NumPy arrays ready to write into the twopy HDF5 file.

    Imaging and stimulus presentation run on different computers with different
    frame rates. The stimulus computer flashes the photodiode at key timepoints
    such as start, trial transitions, and end. These signals are the bridge
    between clocks and must be decoded before trial-level response analysis.
    """

    imaging_res_path: Path
    high_res_path: Path
    imaging_res_pd: npt.NDArray[np.float64]
    high_res_pd: npt.NDArray[np.float64]


@dataclass(frozen=True)
class FrameCountAudit:
    """Frame-count relationship checked during source conversion.

    Inputs: aligned movie frames, imaging-resolution photodiode samples, and
    ScanImage acquisition frame metadata.
    Outputs: explicit counts and deltas for converted HDF5 audit metadata.

    The aligned movie and imaging-resolution photodiode should describe the
    same imaging frames. In sampled real recordings, ScanImage
    ``acq.numberOfFrames`` was usually one less than the aligned movie frame
    count. twopy treats that as a known metadata convention, not as a dropped
    frame, and records the offset so response analysis can audit it.
    """

    aligned_movie_frames: int
    imaging_res_pd_samples: int
    acquisition_number_of_frames: int
    imaging_res_pd_minus_movie: int
    acquisition_minus_movie: int


@dataclass(frozen=True)
class SourceConversionInputs:
    """All source inputs needed to create a twopy converted recording.

    Inputs: one recording session directory.
    Outputs: validated file paths and Python objects used only for conversion.

    This object is the boundary between microscope source data and twopy-owned
    analysis files.
    """

    session_files: TwoPhotonSessionFiles
    aligned_movie: AlignedMovieSource
    acquisition: AcquisitionMetadata
    run: RunMetadata
    stimulus_parameters: StimulusParameters
    stimulus_data: StimulusData
    photodiode: PhotodiodeSignals
    frame_counts: FrameCountAudit


@dataclass(frozen=True)
class ConvertedRecording:
    """Summary of converted twopy recording files.

    Inputs: conversion output path and source recording folder.
    Outputs: HDF5 paths plus mean-image frame range and movie shape.

    Later analysis modules should accept these converted HDF5 paths, not source
    MATLAB file paths. ``path`` points at the small recording data file;
    ``movie_path`` points at the large aligned movie file.
    """

    path: Path
    movie_path: Path
    source_session_dir: Path
    movie_shape: tuple[int, ...]
    mean_image_start_frame: int
    mean_image_stop_frame: int
