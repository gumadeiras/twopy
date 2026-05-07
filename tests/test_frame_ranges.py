"""Tests for shared half-open movie frame range validation.

Inputs: frame counts and optional start/stop frame bounds.
Outputs: normalized frame ranges or clear validation errors.
"""

import unittest

from twopy.frame_ranges import normalize_frame_range


class FrameRangeTest(unittest.TestCase):
    """Tests shared frame-range normalization."""

    def test_defaults_to_full_frame_range(self) -> None:
        """Confirm missing bounds select the whole movie.

        Inputs: five available frames and no explicit start or stop.
        Outputs: the half-open full range.
        """
        self.assertEqual(
            normalize_frame_range(
                frame_count=5,
                start_frame=None,
                stop_frame=None,
            ),
            (0, 5),
        )

    def test_preserves_requested_valid_range(self) -> None:
        """Confirm valid explicit bounds pass through unchanged.

        Inputs: five available frames and a middle two-frame range.
        Outputs: the requested half-open range.
        """
        self.assertEqual(
            normalize_frame_range(
                frame_count=5,
                start_frame=1,
                stop_frame=3,
            ),
            (1, 3),
        )

    def test_rejects_ranges_outside_or_empty_movie_bounds(self) -> None:
        """Confirm invalid frame ranges fail at the shared boundary.

        Inputs: negative starts, stops beyond the movie, and empty ranges.
        Outputs: clear validation errors that include the caller context.
        """
        cases = (
            (-1, 3),
            (0, 6),
            (2, 2),
            (4, 2),
        )
        for start_frame, stop_frame in cases:
            with (
                self.subTest(start_frame=start_frame, stop_frame=stop_frame),
                self.assertRaisesRegex(ValueError, "preview frame range"),
            ):
                normalize_frame_range(
                    frame_count=5,
                    start_frame=start_frame,
                    stop_frame=stop_frame,
                    context="preview frame range",
                )


if __name__ == "__main__":
    unittest.main()
