"""Correlation-based quality control for grouped ROI responses.

Inputs: grouped trial responses with explicit relative time vectors.
Outputs: ROI-level correlation scores and inclusion masks.

The functions here score response consistency; they do not delete raw data.
Callers can use the inclusion mask for plotting or summaries while preserving
the scores as an audit trail.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from twopy.analysis.responses import GroupedRoiResponses, RoiResponseTrial

__all__ = [
    "RoiCorrelationScores",
    "score_roi_correlations",
]


@dataclass(frozen=True)
class RoiCorrelationScores:
    """ROI-level trial-consistency scores.

    Inputs: grouped responses and a minimum accepted correlation.
    Outputs: one score per ROI plus a Boolean inclusion mask.

    Scores are averaged across repeated epoch trials. ROIs with no estimable
    score are excluded when correlation filtering is enabled.
    """

    roi_labels: tuple[str, ...]
    scores: npt.NDArray[np.float64]
    included_mask: npt.NDArray[np.bool_]
    minimum_correlation: float
    reference: str


def score_roi_correlations(
    grouped: GroupedRoiResponses,
    *,
    minimum_correlation: float,
) -> RoiCorrelationScores:
    """Score each ROI by correlation to repeated epoch responses.

    Args:
        grouped: Trial responses grouped by stimulus epoch.
        minimum_correlation: Minimum score required for inclusion.

    Returns:
        ROI correlation scores and inclusion mask.

    Raises:
        ValueError: If grouped response shapes or the threshold are invalid.

    Each trial is compared with the mean response for the same epoch and ROI.
    A leave-one-out mean is used when an epoch has more than one trial, which
    avoids scoring a trial against itself.
    """
    _validate_correlation_inputs(grouped, minimum_correlation)
    correlations: list[list[float]] = [[] for _ in grouped.roi_labels]
    for trials in _trials_by_epoch(grouped).values():
        cube = _aligned_epoch_cube(trials, grouped)
        finite_trial_counts = np.sum(np.isfinite(cube), axis=0)
        for trial_index in range(cube.shape[0]):
            trial_values = cube[trial_index, :, :]
            reference_values = _reference_values_for_trial(
                cube,
                finite_trial_counts,
                trial_index=trial_index,
            )
            for roi_index in range(cube.shape[2]):
                score = _correlation(
                    trial_values[:, roi_index],
                    reference_values[:, roi_index],
                )
                if not np.isnan(score):
                    correlations[roi_index].append(score)

    scores = np.full(len(grouped.roi_labels), np.nan, dtype=np.float64)
    for roi_index, roi_scores in enumerate(correlations):
        if len(roi_scores) > 0:
            scores[roi_index] = float(np.nanmean(roi_scores))
    included = np.isfinite(scores) & (scores >= minimum_correlation)
    return RoiCorrelationScores(
        roi_labels=grouped.roi_labels,
        scores=scores,
        included_mask=included,
        minimum_correlation=float(minimum_correlation),
        reference="epoch_mean",
    )


def _validate_correlation_inputs(
    grouped: GroupedRoiResponses,
    minimum_correlation: float,
) -> None:
    """Validate grouped-response correlation inputs."""
    if not -1.0 <= minimum_correlation <= 1.0:
        msg = f"minimum_correlation must be between -1 and 1; got {minimum_correlation}"
        raise ValueError(msg)
    if grouped.data_rate_hz <= 0:
        msg = f"data_rate_hz must be positive; got {grouped.data_rate_hz}"
        raise ValueError(msg)
    for trial in grouped.trials:
        if trial.values.ndim != 2:
            msg = (
                f"trial values must have shape (frames, rois); got {trial.values.shape}"
            )
            raise ValueError(msg)
        if trial.values.shape[1] != len(grouped.roi_labels):
            msg = (
                "trial ROI width does not match labels: "
                f"{trial.values.shape[1]} values, {len(grouped.roi_labels)} labels"
            )
            raise ValueError(msg)


def _trials_by_epoch(
    grouped: GroupedRoiResponses,
) -> dict[tuple[int, str], list[RoiResponseTrial]]:
    """Group trials by stable epoch identity."""
    by_epoch: dict[tuple[int, str], list[RoiResponseTrial]] = {}
    for trial in grouped.trials:
        by_epoch.setdefault((trial.epoch_number, trial.epoch_name), []).append(trial)
    return by_epoch


def _aligned_epoch_cube(
    trials: list[RoiResponseTrial],
    grouped: GroupedRoiResponses,
) -> npt.NDArray[np.float64]:
    """Return trials aligned onto one relative-time frame axis."""
    first_time = min(float(trial.time_seconds[0]) for trial in trials)
    last_time = max(float(trial.time_seconds[-1]) for trial in trials)
    frame_count = int(round((last_time - first_time) * grouped.data_rate_hz)) + 1
    cube = np.full(
        (len(trials), frame_count, len(grouped.roi_labels)),
        np.nan,
        dtype=np.float64,
    )
    for trial_index, trial in enumerate(trials):
        offsets = np.rint(
            (trial.time_seconds - first_time) * grouped.data_rate_hz
        ).astype(np.int64, copy=False)
        cube[trial_index, offsets, :] = trial.values
    return cube


def _reference_values_for_trial(
    cube: npt.NDArray[np.float64],
    finite_trial_counts: npt.NDArray[np.int64],
    *,
    trial_index: int,
) -> npt.NDArray[np.float64]:
    """Return a leave-one-out epoch mean for one trial when possible."""
    if cube.shape[0] == 1:
        return np.full(cube.shape[1:], np.nan, dtype=np.float64)
    trial_values = cube[trial_index, :, :]
    sums = np.nansum(cube, axis=0) - np.where(
        np.isfinite(trial_values),
        trial_values,
        0.0,
    )
    counts = finite_trial_counts - np.isfinite(trial_values).astype(np.int64)
    reference = np.full(cube.shape[1:], np.nan, dtype=np.float64)
    np.divide(sums, counts, out=reference, where=counts > 0)
    return reference


def _correlation(
    values: npt.NDArray[np.float64],
    reference: npt.NDArray[np.float64],
) -> float:
    """Return finite-sample Pearson correlation for two traces."""
    mask = np.isfinite(values) & np.isfinite(reference)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    centered_values = values[mask] - float(np.nanmean(values[mask]))
    centered_reference = reference[mask] - float(np.nanmean(reference[mask]))
    denominator = float(
        np.sqrt(np.sum(centered_values**2) * np.sum(centered_reference**2))
    )
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(centered_values * centered_reference) / denominator)
