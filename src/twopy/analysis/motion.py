"""Motion-artifact masking for ROI response traces.

Inputs: converted per-frame motion masks and computed dF/F traces.
Outputs: dF/F traces with high-motion frames marked as NaN.

Motion masking happens after dF/F because the mask describes frames that should
not contribute to downstream response statistics. It does not change ROI masks,
background correction, or baseline fitting.
"""

import numpy as np

from twopy.analysis.dff import RoiDeltaFOverF
from twopy.converted import RecordingData

__all__ = ["apply_motion_artifact_mask_to_delta_f_over_f"]


def apply_motion_artifact_mask_to_delta_f_over_f(
    dff: RoiDeltaFOverF,
    recording: RecordingData,
) -> RoiDeltaFOverF:
    """Mask high-motion dF/F frames with NaN.

    Args:
        dff: ROI dF/F result to mask.
        recording: Loaded converted recording with ``motion_artifact_mask``.

    Returns:
        A new ``RoiDeltaFOverF`` with ``values`` copied and high-motion frames
        set to ``NaN``.

    Raises:
        ValueError: If the dF/F frame range is outside the converted recording
            mask.

    The mask is sliced by absolute frame coordinates, so cropped trace results
    and full-recording results use the same converted per-frame evidence.
    """
    mask = recording.motion_artifact_mask
    if dff.start_frame < 0 or dff.stop_frame > mask.shape[0]:
        msg = (
            "dF/F frame range is outside motion artifact mask: "
            f"[{dff.start_frame}, {dff.stop_frame}) for {mask.shape[0]} frames"
        )
        raise ValueError(msg)

    local_mask = mask[dff.start_frame : dff.stop_frame]
    masked_values = dff.values.copy()
    masked_values[local_mask, :] = np.nan
    metadata = dict(dff.metadata)
    metadata["motion_artifact_masked_frame_count"] = int(np.count_nonzero(local_mask))

    return RoiDeltaFOverF(
        fluorescence=dff.fluorescence,
        baseline=dff.baseline,
        values=masked_values,
        labels=dff.labels,
        start_frame=dff.start_frame,
        stop_frame=dff.stop_frame,
        tau=dff.tau,
        amplitudes=dff.amplitudes,
        interleave_frame_numbers=dff.interleave_frame_numbers,
        interleave_fluorescence=dff.interleave_fluorescence,
        metadata=metadata,
    )
