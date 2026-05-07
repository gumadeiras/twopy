"""Tests for script-friendly stimulus metadata helpers.

Inputs: small ``RecordingData`` objects with decoded stimulus-code metadata.
Outputs: assertions that stable column names map to per-``stimtype`` meanings.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from twopy.conversion.types import FrameCountAudit
from twopy.converted import ConvertedMovie, RecordingData
from twopy.spatial import full_frame_crop
from twopy.stimulus import (
    default_interleave_epoch_number,
    is_interleave_epoch_name,
    map_stimulus_specific_column,
)


class StimulusMetadataTest(unittest.TestCase):
    """Tests helpers that interpret stimulus-specific data slots."""

    def test_default_interleave_epoch_uses_gray_like_name(self) -> None:
        """Confirm baseline epoch selection uses the shared name rule.

        Inputs: epoch names where the gray-like baseline is not first.
        Outputs: the matching epoch number.
        """
        epoch_names = {1: "Odor A", 2: "Grey screen", 3: "Odor B"}

        self.assertEqual(default_interleave_epoch_number(epoch_names), 2)

    def test_default_interleave_epoch_accepts_interleave_name(self) -> None:
        """Confirm interleave text is accepted without gray spelling.

        Inputs: epoch names with an explicit interleave baseline.
        Outputs: the matching epoch number.
        """
        epoch_names = {1: "Odor A", 2: "Baseline Interleave"}

        self.assertEqual(default_interleave_epoch_number(epoch_names), 2)

    def test_default_interleave_epoch_falls_back_to_one(self) -> None:
        """Confirm unknown epoch names preserve the historical default.

        Inputs: epoch names with no gray-like baseline.
        Outputs: epoch one.
        """
        self.assertEqual(default_interleave_epoch_number({2: "Odor A"}), 1)

    def test_interleave_epoch_name_predicate_is_shared(self) -> None:
        """Confirm the gray/interleave string rule stays explicit.

        Inputs: gray, grey, interleave, and non-baseline names.
        Outputs: boolean classification for each name.
        """
        self.assertTrue(is_interleave_epoch_name("Gray Interleave"))
        self.assertTrue(is_interleave_epoch_name("Grey screen"))
        self.assertTrue(is_interleave_epoch_name("Baseline Interleave"))
        self.assertFalse(is_interleave_epoch_name("Odor A"))

    def test_maps_stable_column_to_all_stimtype_specific_meanings(self) -> None:
        """Confirm one stable slot can map to multiple stimulus functions.

        Inputs: a loaded recording with two ``stimtype`` metadata entries.
        Outputs: mappings preserving the backed-up MATLAB source assignment.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = self._recording(Path(temp_dir))

            mappings = map_stimulus_specific_column(
                recording,
                "stimulus_specific_04",
            )

            self.assertEqual(
                [mapping.stimtype for mapping in mappings], ["62002", "62005"]
            )
            self.assertEqual(mappings[0].specific_name, "antenna being activated")
            self.assertEqual(mappings[0].source_expression, "antenna")
            self.assertEqual(mappings[0].stimulus_function, "LEDMovingBars")
            self.assertEqual(mappings[1].specific_name, "single antenna active")

    def test_maps_stable_column_for_one_stimtype(self) -> None:
        """Confirm callers can request a specific ``stimtype`` mapping.

        Inputs: a loaded recording and one requested stimulus type.
        Outputs: only the matching mapping.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = self._recording(Path(temp_dir))

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
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = self._recording(Path(temp_dir))

            self.assertEqual(
                map_stimulus_specific_column(recording, "stimulus_specific_20"),
                (),
            )

    def test_rejects_non_stimulus_specific_column(self) -> None:
        """Confirm stable non-specific columns are not treated as decoded slots.

        Inputs: a loaded recording and the stable time column.
        Outputs: a clear error.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = self._recording(Path(temp_dir))

            with self.assertRaisesRegex(ValueError, "stimulus_specific"):
                map_stimulus_specific_column(recording, "time_seconds")

    def test_rejects_malformed_stimulus_column_metadata(self) -> None:
        """Confirm stimulus column metadata must be a string-key dictionary.

        Inputs: decoded metadata whose columns list contains a scalar.
        Outputs: a clear malformed-metadata error.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = self._recording(Path(temp_dir))
            recording.stimulus_specific_columns["62002"]["columns"] = [1]

            with self.assertRaisesRegex(ValueError, "malformed"):
                map_stimulus_specific_column(recording, "stimulus_specific_04")

    def _recording(self, root: Path) -> RecordingData:
        """Create a minimal loaded recording with stimulus-code metadata.

        Args:
            root: Temporary directory used for placeholder paths.

        Returns:
            ``RecordingData`` object with two decoded stimulus functions.
        """
        return RecordingData(
            path=root / "recording_data.h5",
            movie=ConvertedMovie(
                path=root / "aligned_movie.h5",
                dataset_name="movie/aligned",
                shape=(3, 2, 2),
                dtype="float64",
            ),
            source_session_dir=root / "source",
            acquisition_metadata={},
            run_metadata={},
            synchronization_metadata={},
            stimulus_data=np.zeros((3, 35), dtype=np.float64),
            stimulus_data_column_names=(),
            stimulus_parameters=(),
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
            imaging_res_pd=np.zeros(3, dtype=np.float64),
            high_res_pd=np.zeros(0, dtype=np.float64),
            mean_image=np.zeros((2, 2), dtype=np.float64),
            alignment_valid_crop=full_frame_crop((2, 2)),
            alignment_shift_pixels=np.zeros(3, dtype=np.float64),
            motion_artifact_mask=np.zeros(3, dtype=np.bool_),
            frame_counts=FrameCountAudit(
                aligned_movie_frames=3,
                imaging_res_pd_samples=3,
                acquisition_number_of_frames=2,
                imaging_res_pd_minus_movie=0,
                acquisition_minus_movie=-1,
            ),
        )


if __name__ == "__main__":
    unittest.main()
