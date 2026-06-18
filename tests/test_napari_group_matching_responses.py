"""Group Matching response-cache tests.

Inputs: tiny converted recordings, selected ROI label images, and patched trace
caches.
Outputs: assertions that Group Matching reuses selected response previews and
keeps trace caches scoped by recording.
"""

from tests.napari_support import (
    Any,
    Path,
    ResponseProcessingOptions,
    SmoothingOptions,
    _tiny_response_plot_data,
    _write_converted_recording,
    cast,
    load_converted_recording,
    np,
    patch,
    temporary_directory,
    unittest,
)
from twopy.napari.group_matching.responses import SelectedRoiResponseCache


class GroupMatchingResponseCacheTest(unittest.TestCase):
    """Tests for selected-ROI Group Matching response caching."""

    def test_selected_response_cache_reuses_unchanged_preview(self) -> None:
        """Confirm unchanged selected ROI previews do not recompute."""
        with temporary_directory() as temp_dir:
            recording = load_converted_recording(
                _write_converted_recording(Path(temp_dir)),
            )
            label_image = np.array([[1, 0], [0, 2]], dtype=np.int64)
            fake_trace_cache = _FakeTraceCache()

            with patch(
                "twopy.napari.group_matching.responses._new_trace_cache",
                return_value=fake_trace_cache,
            ) as new_trace_cache:
                cache = SelectedRoiResponseCache()
                first = cache.compute_selected_response(
                    recording,
                    label_image,
                    roi_label="roi_0001",
                    label_value=1,
                    source_path=recording.source_session_dir,
                    response_processing_options=ResponseProcessingOptions(),
                )
                second = cache.compute_selected_response(
                    recording,
                    label_image,
                    roi_label="roi_0001",
                    label_value=1,
                    source_path=recording.source_session_dir,
                    response_processing_options=ResponseProcessingOptions(),
                )
                cache.compute_selected_response(
                    recording,
                    label_image,
                    roi_label="roi_0002",
                    label_value=2,
                    source_path=recording.source_session_dir,
                    response_processing_options=ResponseProcessingOptions(),
                )

            self.assertIs(first, second)
            self.assertEqual(
                fake_trace_cache.roi_labels,
                [("roi_0001",), ("roi_0002",)],
            )
            new_trace_cache.assert_called_once()

    def test_selected_response_cache_invalidates_on_mask_or_options_change(
        self,
    ) -> None:
        """Confirm mask edits and processing edits recompute plot data."""
        with temporary_directory() as temp_dir:
            recording = load_converted_recording(
                _write_converted_recording(Path(temp_dir)),
            )
            first_labels = np.array([[1, 0], [0, 0]], dtype=np.int64)
            edited_labels = np.array([[1, 0], [1, 0]], dtype=np.int64)
            fake_trace_cache = _FakeTraceCache()

            with patch(
                "twopy.napari.group_matching.responses._new_trace_cache",
                return_value=fake_trace_cache,
            ):
                cache = SelectedRoiResponseCache()
                cache.compute_selected_response(
                    recording,
                    first_labels,
                    roi_label="roi_0001",
                    label_value=1,
                    source_path=recording.source_session_dir,
                    response_processing_options=ResponseProcessingOptions(),
                )
                cache.compute_selected_response(
                    recording,
                    edited_labels,
                    roi_label="roi_0001",
                    label_value=1,
                    source_path=recording.source_session_dir,
                    response_processing_options=ResponseProcessingOptions(),
                )
                cache.compute_selected_response(
                    recording,
                    edited_labels,
                    roi_label="roi_0001",
                    label_value=1,
                    source_path=recording.source_session_dir,
                    response_processing_options=ResponseProcessingOptions(
                        smoothing=SmoothingOptions(method="moving_average"),
                    ),
                )

            self.assertEqual(
                fake_trace_cache.roi_labels,
                [("roi_0001",), ("roi_0001",), ("roi_0001",)],
            )


class _FakeTraceCache:
    """Trace-cache stand-in that records request ROI labels."""

    def __init__(self) -> None:
        """Create an empty fake trace cache."""
        self.roi_labels: list[tuple[str, ...]] = []

    def compute_response_preview(self, request: object) -> object:
        """Record the request ROI labels and return tiny plot data."""
        self.roi_labels.append(cast(Any, request).roi_set.labels)
        return _tiny_response_plot_data()


if __name__ == "__main__":
    unittest.main()
