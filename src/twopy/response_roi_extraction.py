"""Extract ROIs from repeatable stimulus-locked pixel responses.

Inputs: converted recordings, photodiode-aligned epoch windows, and explicit
response-scoring parameters.
Outputs: full-frame ``RoiSet`` objects plus auditable response score images.

This module keeps response-driven ROI discovery separate from the existing
mean-image watershed path. The watershed step is only the spatial partitioner;
the image being segmented is built from pixel responses, repeatability, and
local response coherence.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from twopy.analysis.trials import EpochFrameWindow
from twopy.converted import RecordingData, recording_frame_rate_hz
from twopy.roi import RoiSet, make_roi_set
from twopy.roi_extraction import watershed_roi_set
from twopy.roi_mask_cleanup import cleaned_connected_components
from twopy.spatial import SpatialCrop, SpatialDomain, full_frame_crop

__all__ = [
    "ResponseNormalization",
    "ResponsePolarity",
    "ResponseStatistic",
    "ResponseWatershedExtraction",
    "ResponseWatershedScoreImages",
    "extract_response_watershed_rois",
    "response_watershed_roi_set",
]
ResponseNormalization = Literal["baseline_subtracted", "delta_f_over_f"]
ResponsePolarity = Literal["positive", "negative", "absolute"]
ResponseStatistic = Literal["mean", "peak"]

_NEIGHBOR_OFFSETS = tuple(
    (axis0_delta, axis1_delta)
    for axis0_delta in (-1, 0, 1)
    for axis1_delta in (-1, 0, 1)
    if axis0_delta != 0 or axis1_delta != 0
)


@dataclass(frozen=True)
class ResponseWatershedScoreImages:
    """Auditable images used to build response-watershed ROIs.

    Args:
        amplitude_image: Nonnegative response magnitude per pixel.
        reliability_image: Split-half epoch-profile reliability per pixel.
        coherence_image: Mean local neighbor response similarity per pixel.
        score_image: Final nonnegative image passed to watershed segmentation.
        reliability_available: Whether repeatable epoch profiles were available.

    The images are in the extraction crop coordinate system. When extraction is
    run over the alignment-valid crop, the returned ``RoiSet`` is expanded back
    to full-frame movie coordinates but these score maps remain cropped so they
    can be displayed directly over the same extracted pixels.
    """

    amplitude_image: npt.NDArray[np.float64]
    reliability_image: npt.NDArray[np.float64]
    coherence_image: npt.NDArray[np.float64]
    score_image: npt.NDArray[np.float64]
    reliability_available: bool


@dataclass(frozen=True)
class ResponseWatershedExtraction:
    """Response-driven ROI extraction output.

    Args:
        roi_set: Full-frame ROI masks ready for normal twopy analysis.
        score_images: Auditable response score components.
        spatial_crop: Movie region used to compute the score images.
        selected_window_indices: Epoch-window indices used for scoring.
        selected_epoch_numbers: Epoch numbers represented in the score images.
    """

    roi_set: RoiSet
    score_images: ResponseWatershedScoreImages
    spatial_crop: SpatialCrop
    selected_window_indices: tuple[int, ...]
    selected_epoch_numbers: tuple[int, ...]


def response_watershed_roi_set(
    recording: RecordingData,
    epoch_windows: Sequence[EpochFrameWindow],
    *,
    epoch_numbers: Sequence[int] | None = None,
    epoch_names: Sequence[str] | None = None,
    spatial_domain: SpatialDomain = "alignment_valid_crop",
    baseline_sample_seconds: float = 1.0,
    response_window_seconds: tuple[float, float | None] = (0.0, None),
    response_statistic: ResponseStatistic = "peak",
    response_polarity: ResponsePolarity = "positive",
    response_normalization: ResponseNormalization = "baseline_subtracted",
    score_smoothing_sigma: float = 0.0,
    minimum_score_percentile: float = 75.0,
    within_roi_percentile: float = 50.0,
    min_pixels: int = 5,
    max_pixels: int | None = None,
    fill_holes: bool = True,
    closing_radius: int = 0,
    apply_motion_mask: bool = True,
    data_rate_hz: float | None = None,
    label_prefix: str = "response_watershed",
) -> RoiSet:
    """Create ROIs from repeatable stimulus-locked pixel responses.

    Args:
        recording: Loaded converted recording.
        epoch_windows: Photodiode-aligned stimulus windows.
        epoch_numbers: Optional epoch-number filter for identification trials.
        epoch_names: Optional exact epoch-name filter for identification trials.
        spatial_domain: ``alignment_valid_crop`` by default, or full frame.
        baseline_sample_seconds: Seconds before each epoch onset used for the
            pixel baseline.
        response_window_seconds: Half-open response interval relative to epoch
            onset. ``None`` on the stop side uses the epoch window stop.
        response_statistic: Per-trial response summary, mean or peak.
        response_polarity: Positive, negative, or absolute response scoring.
        response_normalization: Baseline subtraction or pixel dF/F.
        score_smoothing_sigma: Optional Gaussian sigma before watershed seeding.
        minimum_score_percentile: Global positive-score percentile retained.
        within_roi_percentile: Per-basin score percentile retained.
        min_pixels: Minimum accepted ROI size after score trimming.
        max_pixels: Optional maximum accepted ROI size after score trimming.
        fill_holes: Whether holes inside each connected ROI component are
            filled after score trimming.
        closing_radius: Radius for conservative binary closing before final
            connected-component splitting. ``0`` disables closing.
        apply_motion_mask: Whether converted motion-artifact frames become NaN.
        data_rate_hz: Optional frame rate override. Defaults to recording metadata.
        label_prefix: Prefix for generated ROI labels.

    Returns:
        ``RoiSet`` with full-frame movie-coordinate masks.

    This convenience wrapper returns only masks, matching other ``*_roi_set``
    helpers. Use ``extract_response_watershed_rois`` when scripts should inspect
    amplitude, reliability, coherence, and score images.
    """
    return extract_response_watershed_rois(
        recording,
        epoch_windows,
        epoch_numbers=epoch_numbers,
        epoch_names=epoch_names,
        spatial_domain=spatial_domain,
        baseline_sample_seconds=baseline_sample_seconds,
        response_window_seconds=response_window_seconds,
        response_statistic=response_statistic,
        response_polarity=response_polarity,
        response_normalization=response_normalization,
        score_smoothing_sigma=score_smoothing_sigma,
        minimum_score_percentile=minimum_score_percentile,
        within_roi_percentile=within_roi_percentile,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        fill_holes=fill_holes,
        closing_radius=closing_radius,
        apply_motion_mask=apply_motion_mask,
        data_rate_hz=data_rate_hz,
        label_prefix=label_prefix,
    ).roi_set


def extract_response_watershed_rois(
    recording: RecordingData,
    epoch_windows: Sequence[EpochFrameWindow],
    *,
    epoch_numbers: Sequence[int] | None = None,
    epoch_names: Sequence[str] | None = None,
    spatial_domain: SpatialDomain = "alignment_valid_crop",
    baseline_sample_seconds: float = 1.0,
    response_window_seconds: tuple[float, float | None] = (0.0, None),
    response_statistic: ResponseStatistic = "peak",
    response_polarity: ResponsePolarity = "positive",
    response_normalization: ResponseNormalization = "baseline_subtracted",
    score_smoothing_sigma: float = 0.0,
    minimum_score_percentile: float = 75.0,
    within_roi_percentile: float = 50.0,
    min_pixels: int = 5,
    max_pixels: int | None = None,
    fill_holes: bool = True,
    closing_radius: int = 0,
    apply_motion_mask: bool = True,
    data_rate_hz: float | None = None,
    label_prefix: str = "response_watershed",
) -> ResponseWatershedExtraction:
    """Extract ROIs by watershedding a stimulus-response score image.

    Args:
        recording: Loaded converted recording.
        epoch_windows: Photodiode-aligned stimulus windows.
        epoch_numbers: Optional epoch-number filter for identification trials.
        epoch_names: Optional exact epoch-name filter for identification trials.
        spatial_domain: ``alignment_valid_crop`` by default, or full frame.
        baseline_sample_seconds: Seconds before each epoch onset used for the
            pixel baseline.
        response_window_seconds: Half-open response interval relative to epoch
            onset. ``None`` on the stop side uses the epoch window stop.
        response_statistic: Per-trial response summary, mean or peak.
        response_polarity: Positive, negative, or absolute response scoring.
        response_normalization: Baseline subtraction or pixel dF/F.
        score_smoothing_sigma: Optional Gaussian sigma before watershed seeding.
        minimum_score_percentile: Global positive-score percentile retained.
        within_roi_percentile: Per-basin score percentile retained.
        min_pixels: Minimum accepted ROI size after score trimming.
        max_pixels: Optional maximum accepted ROI size after score trimming.
        fill_holes: Whether holes inside each connected ROI component are
            filled after score trimming.
        closing_radius: Radius for conservative binary closing before final
            connected-component splitting. ``0`` disables closing.
        apply_motion_mask: Whether converted motion-artifact frames become NaN.
        data_rate_hz: Optional frame rate override. Defaults to recording metadata.
        label_prefix: Prefix for generated ROI labels.

    Returns:
        ``ResponseWatershedExtraction`` with full-frame ROIs and score images.

    The score image is ``amplitude * local_coherence`` multiplied by split-half
    epoch-profile cosine reliability when enough repeated epochs are present.
    This keeps the segmentation tied to repeatable stimulus responses without
    discarding broad responders whose response profile is flat across selected
    epochs.
    """
    frame_rate = (
        recording_frame_rate_hz(recording) if data_rate_hz is None else data_rate_hz
    )
    _validate_common_options(
        frame_rate=frame_rate,
        baseline_sample_seconds=baseline_sample_seconds,
        response_window_seconds=response_window_seconds,
        score_smoothing_sigma=score_smoothing_sigma,
        minimum_score_percentile=minimum_score_percentile,
        within_roi_percentile=within_roi_percentile,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        closing_radius=closing_radius,
        response_statistic=response_statistic,
        response_polarity=response_polarity,
        response_normalization=response_normalization,
    )
    selected_windows = _selected_epoch_windows(
        epoch_windows,
        epoch_numbers=epoch_numbers,
        epoch_names=epoch_names,
    )
    crop = _spatial_crop(recording, spatial_domain)
    trial_responses = _trial_response_images(
        recording,
        selected_windows,
        crop=crop,
        data_rate_hz=frame_rate,
        baseline_sample_seconds=baseline_sample_seconds,
        response_window_seconds=response_window_seconds,
        response_statistic=response_statistic,
        response_polarity=response_polarity,
        response_normalization=response_normalization,
        apply_motion_mask=apply_motion_mask,
    )
    scores = _response_watershed_scores(
        trial_responses,
        epoch_numbers=tuple(window.epoch_number for window in selected_windows),
    )
    cropped_roi_set = _roi_set_from_score_image(
        scores.score_image,
        score_smoothing_sigma=score_smoothing_sigma,
        minimum_score_percentile=minimum_score_percentile,
        within_roi_percentile=within_roi_percentile,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        fill_holes=fill_holes,
        closing_radius=closing_radius,
        label_prefix=label_prefix,
    )
    roi_set = _full_frame_roi_set(cropped_roi_set, crop)
    return ResponseWatershedExtraction(
        roi_set=roi_set,
        score_images=scores,
        spatial_crop=crop,
        selected_window_indices=tuple(
            window.window.index for window in selected_windows
        ),
        selected_epoch_numbers=tuple(
            sorted({window.epoch_number for window in selected_windows})
        ),
    )


def _trial_response_images(
    recording: RecordingData,
    epoch_windows: tuple[EpochFrameWindow, ...],
    *,
    crop: SpatialCrop,
    data_rate_hz: float,
    baseline_sample_seconds: float,
    response_window_seconds: tuple[float, float | None],
    response_statistic: ResponseStatistic,
    response_polarity: ResponsePolarity,
    response_normalization: ResponseNormalization,
    apply_motion_mask: bool,
) -> npt.NDArray[np.float64]:
    """Return one scalar response image per selected trial."""
    baseline_frames = max(1, int(round(baseline_sample_seconds * data_rate_hz)))
    response_start_seconds, response_stop_seconds = response_window_seconds
    response_start_offset = int(round(response_start_seconds * data_rate_hz))
    response_stop_offset = (
        None
        if response_stop_seconds is None
        else int(round(response_stop_seconds * data_rate_hz))
    )
    response_images: list[npt.NDArray[np.float64]] = []
    for epoch_window in epoch_windows:
        window = epoch_window.window
        baseline_stop = window.start_frame
        baseline_start = baseline_stop - baseline_frames
        if baseline_start < 0:
            msg = (
                f"Epoch window {window.index} starts at frame {window.start_frame}, "
                f"leaving fewer than {baseline_frames} baseline frames"
            )
            raise ValueError(msg)
        response_start = window.start_frame + response_start_offset
        response_stop = (
            window.stop_frame
            if response_stop_offset is None
            else window.start_frame + response_stop_offset
        )
        response_stop = min(response_stop, window.stop_frame)
        if response_stop <= response_start:
            msg = (
                f"Epoch window {window.index} has empty response interval "
                f"[{response_start}, {response_stop})"
            )
            raise ValueError(msg)

        baseline_block = recording.movie.read_frames(
            baseline_start,
            baseline_stop,
            spatial_crop=crop,
        )
        response_block = recording.movie.read_frames(
            response_start,
            response_stop,
            spatial_crop=crop,
        )
        if apply_motion_mask:
            baseline_block = _apply_motion_mask(
                baseline_block,
                recording.motion_artifact_mask[baseline_start:baseline_stop],
            )
            response_block = _apply_motion_mask(
                response_block,
                recording.motion_artifact_mask[response_start:response_stop],
            )
        baseline_image = np.nanmean(baseline_block, axis=0)
        response_values = _normalized_response_block(
            response_block,
            baseline_image,
            response_normalization=response_normalization,
        )
        response_images.append(
            _response_image(
                response_values,
                response_statistic=response_statistic,
                response_polarity=response_polarity,
            )
        )
    return np.stack(response_images).astype(np.float64, copy=False)


def _response_watershed_scores(
    trial_responses: npt.NDArray[np.float64],
    *,
    epoch_numbers: tuple[int, ...],
) -> ResponseWatershedScoreImages:
    """Compute amplitude, reliability, coherence, and combined score maps."""
    if trial_responses.ndim != 3:
        msg = (
            "trial responses must have shape (trials, axis0, axis1); "
            f"got {trial_responses.shape}"
        )
        raise ValueError(msg)
    if trial_responses.shape[0] < 2:
        msg = "Response watershed requires at least two selected epoch windows"
        raise ValueError(msg)
    if len(epoch_numbers) != trial_responses.shape[0]:
        msg = "epoch number count must match trial response count"
        raise ValueError(msg)

    responses = np.nan_to_num(trial_responses, nan=0.0, posinf=0.0, neginf=0.0)
    amplitude = np.nanmax(responses, axis=0)
    amplitude = np.maximum(amplitude, 0.0)
    coherence = _local_response_coherence(responses)
    reliability, reliability_available = _split_half_epoch_reliability(
        responses,
        epoch_numbers=epoch_numbers,
    )

    normalized_amplitude = _normalize_nonnegative_image(amplitude)
    normalized_coherence = np.clip(coherence, 0.0, 1.0)
    normalized_reliability = np.clip(reliability, 0.0, 1.0)
    score = normalized_amplitude * normalized_coherence * normalized_reliability
    if not reliability_available:
        score = normalized_amplitude * normalized_coherence
    if not np.any(score > 0):
        msg = "Response watershed score image has no positive pixels"
        raise ValueError(msg)
    return ResponseWatershedScoreImages(
        amplitude_image=amplitude,
        reliability_image=normalized_reliability,
        coherence_image=normalized_coherence,
        score_image=score.astype(np.float64, copy=False),
        reliability_available=reliability_available,
    )


def _local_response_coherence(
    responses: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return mean positive neighbor cosine similarity for each pixel."""
    trial_count, axis0_length, axis1_length = responses.shape
    if trial_count < 2:
        return np.zeros((axis0_length, axis1_length), dtype=np.float64)
    denominator = np.sqrt(np.sum(responses * responses, axis=0))
    coherence_sum = np.zeros((axis0_length, axis1_length), dtype=np.float64)
    coherence_count = np.zeros((axis0_length, axis1_length), dtype=np.float64)
    for axis0_delta, axis1_delta in _NEIGHBOR_OFFSETS:
        center0, neighbor0 = _paired_slices(axis0_delta, axis0_length)
        center1, neighbor1 = _paired_slices(axis1_delta, axis1_length)
        center_values = responses[:, center0, center1]
        neighbor_values = responses[:, neighbor0, neighbor1]
        numerator = np.sum(center_values * neighbor_values, axis=0)
        denom = denominator[center0, center1] * denominator[neighbor0, neighbor1]
        correlation = np.divide(
            numerator,
            denom,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=denom > 0,
        )
        coherence_sum[center0, center1] += np.clip(correlation, 0.0, 1.0)
        coherence_count[center0, center1] += 1.0
    return np.divide(
        coherence_sum,
        coherence_count,
        out=np.zeros_like(coherence_sum, dtype=np.float64),
        where=coherence_count > 0,
    )


def _split_half_epoch_reliability(
    responses: npt.NDArray[np.float64],
    *,
    epoch_numbers: tuple[int, ...],
) -> tuple[npt.NDArray[np.float64], bool]:
    """Return split-half cosine reliability across repeated epoch profiles."""
    epoch_values = tuple(sorted(set(epoch_numbers)))
    first_half_profiles: list[npt.NDArray[np.float64]] = []
    second_half_profiles: list[npt.NDArray[np.float64]] = []
    epoch_numbers_array = np.asarray(epoch_numbers, dtype=np.int64)
    for epoch_number in epoch_values:
        indices = np.flatnonzero(epoch_numbers_array == epoch_number)
        if indices.size < 2:
            continue
        first_half_profiles.append(np.mean(responses[indices[::2]], axis=0))
        second_half_profiles.append(np.mean(responses[indices[1::2]], axis=0))
    if len(first_half_profiles) < 2:
        return np.ones(responses.shape[1:], dtype=np.float64), False

    first = np.stack(first_half_profiles)
    second = np.stack(second_half_profiles)
    return _pixelwise_profile_cosine_similarity(first, second), True


def _pixelwise_profile_cosine_similarity(
    first: npt.NDArray[np.float64],
    second: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return per-pixel cosine similarity between two epoch profiles."""
    numerator = np.sum(first * second, axis=0)
    denominator = np.sqrt(
        np.sum(first * first, axis=0) * np.sum(second * second, axis=0)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )


def _roi_set_from_score_image(
    score_image: npt.NDArray[np.float64],
    *,
    score_smoothing_sigma: float,
    minimum_score_percentile: float,
    within_roi_percentile: float,
    min_pixels: int,
    max_pixels: int | None,
    fill_holes: bool,
    closing_radius: int,
    label_prefix: str,
) -> RoiSet:
    """Create a ROI set by watershedding and trimming a response score image."""
    watershed_rois = watershed_roi_set(
        score_image,
        min_pixels=1,
        smoothing_sigma=score_smoothing_sigma,
        label_prefix=label_prefix,
    )
    positive_scores = score_image[score_image > 0]
    if positive_scores.size == 0:
        msg = "Response watershed score image has no positive pixels"
        raise ValueError(msg)
    global_threshold = float(
        np.percentile(positive_scores, minimum_score_percentile),
    )
    global_keep = score_image >= global_threshold
    masks: list[npt.NDArray[np.bool_]] = []
    for mask in watershed_rois.masks:
        candidate = mask & global_keep
        if not np.any(candidate):
            continue
        candidate_scores = score_image[candidate]
        within_threshold = float(np.percentile(candidate_scores, within_roi_percentile))
        trimmed = candidate & (score_image >= within_threshold)
        components = cleaned_connected_components(
            trimmed,
            domain_mask=mask,
            fill_holes=fill_holes,
            closing_radius=closing_radius,
        )
        for component in components:
            pixel_count = int(np.sum(component))
            if pixel_count < min_pixels:
                continue
            if max_pixels is not None and pixel_count > max_pixels:
                continue
            masks.append(component)
    if len(masks) == 0:
        msg = "Response watershed produced no ROIs after score and size filtering"
        raise ValueError(msg)
    return make_roi_set(
        np.stack(masks),
        labels=tuple(f"{label_prefix}_{index + 1:04d}" for index in range(len(masks))),
    )


def _selected_epoch_windows(
    epoch_windows: Sequence[EpochFrameWindow],
    *,
    epoch_numbers: Sequence[int] | None,
    epoch_names: Sequence[str] | None,
) -> tuple[EpochFrameWindow, ...]:
    """Return epoch windows selected for response-driven ROI discovery."""
    number_filter = (
        None if epoch_numbers is None else {int(value) for value in epoch_numbers}
    )
    name_filter = None if epoch_names is None else set(epoch_names)
    selected = tuple(
        window
        for window in epoch_windows
        if (number_filter is None or window.epoch_number in number_filter)
        and (name_filter is None or window.epoch_name in name_filter)
    )
    if len(selected) == 0:
        msg = "Response watershed selected no epoch windows"
        raise ValueError(msg)
    return selected


def _normalized_response_block(
    response_block: npt.NDArray[np.float64],
    baseline_image: npt.NDArray[np.float64],
    *,
    response_normalization: ResponseNormalization,
) -> npt.NDArray[np.float64]:
    """Return baseline-subtracted or dF/F pixel response frames."""
    delta = response_block - baseline_image
    if response_normalization == "baseline_subtracted":
        return delta
    if response_normalization == "delta_f_over_f":
        denominator = np.maximum(np.abs(baseline_image), np.finfo(np.float64).eps)
        return delta / denominator
    msg = f"Unknown response normalization: {response_normalization!r}"
    raise ValueError(msg)


def _response_image(
    values: npt.NDArray[np.float64],
    *,
    response_statistic: ResponseStatistic,
    response_polarity: ResponsePolarity,
) -> npt.NDArray[np.float64]:
    """Collapse one trial response block into one nonnegative image."""
    if response_polarity == "positive":
        polarized = values
    elif response_polarity == "negative":
        polarized = -values
    elif response_polarity == "absolute":
        polarized = np.abs(values)
    else:
        msg = f"Unknown response polarity: {response_polarity!r}"
        raise ValueError(msg)

    if response_statistic == "peak":
        image = np.nanmax(polarized, axis=0)
    elif response_statistic == "mean":
        image = np.nanmean(polarized, axis=0)
    else:
        msg = f"Unknown response statistic: {response_statistic!r}"
        raise ValueError(msg)
    return np.maximum(image, 0.0)


def _apply_motion_mask(
    block: npt.NDArray[np.float64],
    motion_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Return a copy with high-motion frames marked NaN."""
    if not np.any(motion_mask):
        return block
    masked = block.copy()
    masked[motion_mask, :, :] = np.nan
    return masked


def _full_frame_roi_set(cropped_roi_set: RoiSet, crop: SpatialCrop) -> RoiSet:
    """Embed cropped ROI masks in full-frame movie coordinates."""
    if crop.is_full_frame:
        return cropped_roi_set
    full_masks = np.zeros(
        (cropped_roi_set.masks.shape[0], *crop.original_shape),
        dtype=np.bool_,
    )
    full_masks[
        :,
        crop.axis0_start : crop.axis0_stop,
        crop.axis1_start : crop.axis1_stop,
    ] = cropped_roi_set.masks
    return make_roi_set(full_masks, labels=cropped_roi_set.labels)


def _spatial_crop(
    recording: RecordingData, spatial_domain: SpatialDomain
) -> SpatialCrop:
    """Return the spatial crop used for response-watershed scoring."""
    if spatial_domain == "alignment_valid_crop":
        return recording.alignment_valid_crop
    if spatial_domain == "full_frame":
        return full_frame_crop(recording.movie.shape[1:])
    msg = f"Unknown spatial domain: {spatial_domain!r}"
    raise ValueError(msg)


def _normalize_nonnegative_image(
    image: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Scale a nonnegative image to ``[0, 1]`` when possible."""
    clipped = np.maximum(np.nan_to_num(image, nan=0.0), 0.0)
    maximum = float(np.max(clipped))
    if maximum <= 0:
        return np.zeros_like(clipped, dtype=np.float64)
    return clipped / maximum


def _paired_slices(delta: int, length: int) -> tuple[slice, slice]:
    """Return center and neighbor slices for one axis offset."""
    if delta < 0:
        return slice(-delta, length), slice(0, length + delta)
    if delta > 0:
        return slice(0, length - delta), slice(delta, length)
    return slice(0, length), slice(0, length)


def _validate_common_options(
    *,
    frame_rate: float,
    baseline_sample_seconds: float,
    response_window_seconds: tuple[float, float | None],
    score_smoothing_sigma: float,
    minimum_score_percentile: float,
    within_roi_percentile: float,
    min_pixels: int,
    max_pixels: int | None,
    closing_radius: int,
    response_statistic: ResponseStatistic,
    response_polarity: ResponsePolarity,
    response_normalization: ResponseNormalization,
) -> None:
    """Validate response-watershed scalar options."""
    if not np.isfinite(frame_rate) or frame_rate <= 0:
        msg = f"data_rate_hz must be a finite positive number; got {frame_rate}"
        raise ValueError(msg)
    if not np.isfinite(baseline_sample_seconds) or baseline_sample_seconds <= 0:
        msg = (
            "baseline_sample_seconds must be a finite positive number; "
            f"got {baseline_sample_seconds}"
        )
        raise ValueError(msg)
    response_start, response_stop = response_window_seconds
    if not np.isfinite(response_start) or response_start < 0:
        msg = (
            "response window start must be finite and nonnegative; "
            f"got {response_start}"
        )
        raise ValueError(msg)
    if response_stop is not None and (
        not np.isfinite(response_stop) or response_stop <= response_start
    ):
        msg = (
            "response window stop must be greater than start or None; "
            f"got {response_window_seconds}"
        )
        raise ValueError(msg)
    if score_smoothing_sigma < 0:
        msg = f"score_smoothing_sigma cannot be negative; got {score_smoothing_sigma}"
        raise ValueError(msg)
    for name, value in (
        ("minimum_score_percentile", minimum_score_percentile),
        ("within_roi_percentile", within_roi_percentile),
    ):
        if not 0.0 <= value <= 100.0:
            msg = f"{name} must be between 0 and 100; got {value}"
            raise ValueError(msg)
    if min_pixels <= 0:
        msg = f"min_pixels must be positive; got {min_pixels}"
        raise ValueError(msg)
    if max_pixels is not None and max_pixels < min_pixels:
        msg = f"max_pixels must be at least min_pixels; got {max_pixels}"
        raise ValueError(msg)
    if closing_radius < 0:
        msg = f"closing_radius cannot be negative; got {closing_radius}"
        raise ValueError(msg)
    if response_statistic not in {"mean", "peak"}:
        msg = f"Unknown response statistic: {response_statistic!r}"
        raise ValueError(msg)
    if response_polarity not in {"positive", "negative", "absolute"}:
        msg = f"Unknown response polarity: {response_polarity!r}"
        raise ValueError(msg)
    if response_normalization not in {"baseline_subtracted", "delta_f_over_f"}:
        msg = f"Unknown response normalization: {response_normalization!r}"
        raise ValueError(msg)
