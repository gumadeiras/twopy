"""Stable signatures for napari ROI masks.

Inputs: boolean ROI masks from Labels-derived ROI sets.
Outputs: compact hashes used by interactive caches to detect pixel edits.
"""

from hashlib import blake2b

import numpy as np
import numpy.typing as npt

__all__ = ["roi_mask_signature"]


def roi_mask_signature(mask: npt.NDArray[np.bool_]) -> str:
    """Return a stable signature for one ROI mask."""
    digest = blake2b(digest_size=16)
    digest.update(np.asarray(mask.shape, dtype=np.int64).tobytes())
    digest.update(np.flatnonzero(mask).astype(np.int64, copy=False).tobytes())
    return digest.hexdigest()
