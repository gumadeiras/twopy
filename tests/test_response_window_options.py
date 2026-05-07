"""Tests for response-window option resolution.

Inputs: automatic/manual response-window options and recording-derived limits.
Outputs: concrete pre/post seconds used for response grouping.
"""

import unittest

from twopy.analysis.response_window_options import (
    ResponseWindowOptions,
    resolve_response_window_seconds,
)


class ResponseWindowOptionsTest(unittest.TestCase):
    """Verify response-window option resolution remains explicit."""

    def test_auto_uses_automatic_pre_and_post_seconds(self) -> None:
        """Confirm Auto preserves the caller-provided plot policy.

        Inputs: Auto options with automatic pre and post durations.
        Outputs: resolved seconds match the automatic values.
        """
        self.assertEqual(
            resolve_response_window_seconds(
                ResponseWindowOptions(auto=True),
                automatic_pre_window_seconds=2.0,
                automatic_post_window_seconds=0.0,
            ),
            (2.0, 0.0),
        )

    def test_manual_windows_are_capped_by_interleave_duration(self) -> None:
        """Confirm manual pre/post values cannot exceed the recording cap.

        Inputs: manual response-window options larger than the known interleave
        duration.
        Outputs: both resolved windows clamp to the cap.
        """
        self.assertEqual(
            resolve_response_window_seconds(
                ResponseWindowOptions(
                    auto=False,
                    pre_window_seconds=3.0,
                    post_window_seconds=4.0,
                ),
                automatic_pre_window_seconds=2.0,
                automatic_post_window_seconds=2.0,
                max_window_seconds=1.5,
            ),
            (1.5, 1.5),
        )

    def test_negative_manual_window_is_rejected(self) -> None:
        """Confirm invalid manual windows fail before analysis starts.

        Inputs: manual response-window options with a negative pre duration.
        Outputs: a clear validation error.
        """
        with self.assertRaisesRegex(ValueError, "pre_window_seconds"):
            resolve_response_window_seconds(
                ResponseWindowOptions(
                    auto=False,
                    pre_window_seconds=-0.1,
                    post_window_seconds=1.0,
                ),
                automatic_pre_window_seconds=2.0,
                automatic_post_window_seconds=2.0,
            )


if __name__ == "__main__":
    unittest.main()
