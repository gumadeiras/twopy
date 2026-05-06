"""Shared constants for the twopy napari adapter.

Inputs: none.
Outputs: small GUI defaults used by launcher and controls.

Keeping defaults in one file prevents launcher and widget behavior from
drifting as the napari surface grows.
"""

DEFAULT_MOVIE_CHUNK_FRAMES = 128
DEFAULT_PATH_TEXT = "default"
DEFAULT_ROI_FILENAME = "rois.h5"
RECORDING_DATA_FILENAME = "recording_data.h5"
ALIGNED_MOVIE_FILENAME = "aligned_movie.h5"
