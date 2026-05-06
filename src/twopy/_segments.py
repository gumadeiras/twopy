"""Small helpers for finding contiguous runs in one-dimensional arrays.

Inputs: boolean or integer NumPy vectors.
Outputs: inclusive-start, exclusive-stop index pairs for contiguous runs.

Timing code uses these helpers to turn threshold masks and epoch vectors into
auditable windows without duplicating low-level indexing logic.
"""

import numpy as np
import numpy.typing as npt

__all__ = ["constant_segments", "true_segments"]


def true_segments(active: npt.NDArray[np.bool_]) -> tuple[tuple[int, int], ...]:
    """Find contiguous true runs in a Boolean vector.

    Args:
        active: Boolean vector.

    Returns:
        Tuple of inclusive-start, exclusive-stop index pairs.
    """
    if active.size == 0:
        return ()

    padded = np.concatenate(([False], active, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return tuple(
        (int(changes[index]), int(changes[index + 1]))
        for index in range(0, len(changes), 2)
    )


def constant_segments(values: npt.NDArray[np.int64]) -> tuple[tuple[int, int], ...]:
    """Find contiguous runs with the same integer value.

    Args:
        values: One-dimensional integer vector.

    Returns:
        Tuple of inclusive-start, exclusive-stop index pairs.
    """
    if values.size == 0:
        return ()

    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.r_[0, changes]
    stops = np.r_[changes, values.size]
    return tuple(
        (int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)
    )
