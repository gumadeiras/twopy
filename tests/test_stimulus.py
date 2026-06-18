"""Tests for script-friendly stimulus metadata helpers.

Inputs: small ``RecordingData`` objects with decoded stimulus-code metadata.
Outputs: assertions that stable column names map to per-``stimtype`` meanings.
"""

import unittest

import numpy as np

from tests.recording_data import minimal_recording_data
from twopy.converted import RecordingData
from twopy.stimulus import map_stimulus_specific_column


class StimulusMetadataTest(unittest.TestCase):
    """Tests helpers that interpret stimulus-specific data slots."""

    def test_maps_stable_column_to_all_stimtype_specific_meanings(self) -> None:
        """Confirm one stable slot can map to multiple stimulus functions.

        Inputs: a loaded recording with two ``stimtype`` metadata entries.
        Outputs: mappings preserving the backed-up MATLAB source assignment.
        """
        recording = self._recording()

        mappings = map_stimulus_specific_column(
            recording,
            "stimulus_specific_04",
        )

        self.assertEqual([mapping.stimtype for mapping in mappings], ["62002", "62005"])
        self.assertEqual(mappings[0].specific_name, "antenna being activated")
        self.assertEqual(mappings[0].source_expression, "antenna")
        self.assertEqual(mappings[0].stimulus_function, "LEDMovingBars")
        self.assertEqual(mappings[1].specific_name, "single antenna active")

    def test_maps_stable_column_for_one_stimtype(self) -> None:
        """Confirm callers can request a specific ``stimtype`` mapping.

        Inputs: a loaded recording and one requested stimulus type.
        Outputs: only the matching mapping.
        """
        recording = self._recording()

        mappings = map_stimulus_specific_column(
            recording,
            "stimulus_specific_04",
            stimtype=62005,
        )

        self.assertEqual(len(mappings), 1)
        self.assertEqual(mappings[0].stimtype, "62005")
        self.assertEqual(mappings[0].source_path, "stimfunctions/Single.m")

    def test_unused_stimulus_specific_slot_returns_no_mappings(self) -> None:
        """Confirm unused stimulus-specific slots are represented explicitly.

        Inputs: a loaded recording where slot 20 was never assigned.
        Outputs: an empty tuple rather than a fabricated meaning.
        """
        recording = self._recording()

        self.assertEqual(
            map_stimulus_specific_column(recording, "stimulus_specific_20"),
            (),
        )

    def test_rejects_non_stimulus_specific_column(self) -> None:
        """Confirm stable non-specific columns are not treated as decoded slots.

        Inputs: a loaded recording and the stable time column.
        Outputs: a clear error.
        """
        recording = self._recording()

        with self.assertRaisesRegex(ValueError, "stimulus_specific"):
            map_stimulus_specific_column(recording, "time_seconds")

    def test_rejects_malformed_stimulus_column_metadata(self) -> None:
        """Confirm stimulus column metadata must be a string-key dictionary.

        Inputs: decoded metadata whose columns list contains a scalar.
        Outputs: a clear malformed-metadata error.
        """
        recording = self._recording()
        recording.stimulus_specific_columns["62002"]["columns"] = [1]

        with self.assertRaisesRegex(ValueError, "malformed"):
            map_stimulus_specific_column(recording, "stimulus_specific_04")

    def _recording(
        self,
        *,
        stimulus_parameters: tuple[dict[str, object], ...] = (),
    ) -> RecordingData:
        """Create a minimal loaded recording with stimulus-code metadata.

        Args:
            stimulus_parameters: Optional decoded stimulus epoch metadata.

        Returns:
            ``RecordingData`` object with two decoded stimulus functions.
        """
        return minimal_recording_data(
            frame_count=3,
            stimulus_data=np.zeros((3, 35), dtype=np.float64),
            stimulus_parameters=stimulus_parameters,
            stimulus_function_lookup={
                "62002": "LEDMovingBars",
                "62005": "LEDMovingBarsSingleAntenna",
            },
            stimulus_specific_columns={
                "62002": {
                    "stimtype": "62002",
                    "function": "LEDMovingBars",
                    "source_path": "stimfunctions/LEDMovingBars.m",
                    "columns": [
                        {
                            "mat_slot": 4,
                            "column_name": "stimulus_specific_04",
                            "source_expression": "antenna",
                            "source_comment": "antenna being activated",
                            "source_line": 66,
                        },
                    ],
                },
                "62005": {
                    "stimtype": "62005",
                    "function": "LEDMovingBarsSingleAntenna",
                    "source_path": "stimfunctions/Single.m",
                    "columns": [
                        {
                            "mat_slot": 4,
                            "column_name": "stimulus_specific_04",
                            "source_expression": "antenna",
                            "source_comment": "single antenna active",
                            "source_line": 65,
                        },
                    ],
                },
            },
            acquisition_frame_count=2,
        )


if __name__ == "__main__":
    unittest.main()
