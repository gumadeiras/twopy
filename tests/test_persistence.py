"""Tests for saving twopy analysis outputs.

Inputs: small ROI, trace, dF/F, window, and grouped-response objects.
Outputs: inspectable HDF5 files and response time-series CSV files.
"""

import csv
import unittest
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from tests.tempdir import temporary_directory
from twopy.analysis.background_subtraction import BackgroundCorrectedRoiTraces
from twopy.analysis.dff import RoiDeltaFOverF
from twopy.analysis.persistence import (
    ANALYSIS_OUTPUT_FILE_FORMAT,
    load_analysis_outputs,
    save_analysis_outputs,
    write_response_summary_grouped_csv,
)
from twopy.analysis.response_processing import (
    CorrelationFilterOptions,
    LowPassFilterOptions,
    NormalizationOptions,
    ResponseProcessingOptions,
    RoiCorrelationScores,
    RoiNormalizationFactors,
    SmoothingOptions,
)
from twopy.analysis.responses import group_delta_f_over_f_by_epoch
from twopy.analysis.trials import EpochFrameWindow, FrameWindow
from twopy.roi import make_roi_set
from twopy.spatial_orientation import (
    SPATIAL_ORIENTATION_ATTR,
    TWOPY_SPATIAL_ORIENTATION,
)


class PersistenceTest(unittest.TestCase):
    """Tests analysis HDF5 and CSV persistence."""

    def test_saves_analysis_outputs_and_response_summary(self) -> None:
        """Confirm analysis objects are saved in auditable HDF5 groups.

        Inputs: one ROI set, traces, dF/F, epoch window, and grouped response.
        Outputs: one HDF5 file plus trial-level and grouped response CSVs.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            h5_path = root / "analysis_outputs.h5"
            trials_csv_path = root / "response_summary_trials.csv"
            grouped_csv_path = root / "response_summary_grouped.csv"
            roi_set = make_roi_set(
                np.array(
                    [
                        [[True, False], [False, False]],
                        [[False, False], [False, True]],
                    ],
                ),
                labels=("roi_1", "roi_2"),
            )
            traces = self._traces()
            dff = self._dff()
            windows = (EpochFrameWindow(FrameWindow(0, 0, 2, "gray"), 1, "Gray"),)
            grouped = group_delta_f_over_f_by_epoch(
                dff,
                windows,
                data_rate_hz=2.0,
            )
            processing_options = ResponseProcessingOptions(
                smoothing=SmoothingOptions(
                    method="moving_average",
                    window_frames=3,
                ),
                low_pass=LowPassFilterOptions(
                    method="butterworth",
                    cutoff_hz=0.5,
                    order=2,
                ),
                normalization=NormalizationOptions(
                    method="epoch_peak",
                    epoch_number=1,
                    epoch_name="Gray",
                ),
                correlation_filter=CorrelationFilterOptions(
                    reference="epoch_mean",
                    minimum_correlation=0.4,
                    window_seconds=(0.0, 2.0),
                ),
            )
            correlation_scores = RoiCorrelationScores(
                roi_labels=("roi_1", "roi_2"),
                scores=np.array([0.9, 0.1], dtype=np.float64),
                included_mask=np.array([True, False]),
                minimum_correlation=0.4,
                reference="epoch_mean",
                window_seconds=(0.0, 2.0),
            )
            normalization_factors = RoiNormalizationFactors(
                roi_labels=("roi_1", "roi_2"),
                factors=np.array([2.0, 4.0], dtype=np.float64),
                epoch_number=1,
                epoch_name="Gray",
            )

            save_analysis_outputs(
                h5_path,
                roi_set=roi_set,
                traces=traces,
                dff=dff,
                epoch_windows=windows,
                baseline_windows=(windows[0].window,),
                grouped_responses=grouped,
                response_processing_options=processing_options,
                normalization_factors=normalization_factors,
                correlation_scores=correlation_scores,
                response_summary_trials_csv=trials_csv_path,
                response_summary_grouped_csv=grouped_csv_path,
            )

            with h5py.File(h5_path, "r") as h5_file:
                self.assertEqual(
                    h5_file.attrs["twopy_format"],
                    ANALYSIS_OUTPUT_FILE_FORMAT,
                )
                self.assertEqual(h5_file["roi_set/masks"].shape, (2, 2, 2))
                self.assertEqual(
                    h5_file["roi_set"].attrs[SPATIAL_ORIENTATION_ATTR],
                    TWOPY_SPATIAL_ORIENTATION,
                )
                self.assertEqual(
                    h5_file["roi_set/masks"].attrs[SPATIAL_ORIENTATION_ATTR],
                    TWOPY_SPATIAL_ORIENTATION,
                )
                self.assertEqual(h5_file["traces/raw_values"].shape, (3, 2))
                self.assertEqual(h5_file["dff/values"].shape, (3, 2))
                self.assertEqual(h5_file["epoch_windows/start_frame"][0], 0)
                self.assertEqual(h5_file["baseline_windows/start_frame"][0], 0)
                self.assertEqual(
                    h5_file["response_processing/smoothing"].attrs["method"],
                    "moving_average",
                )
                self.assertEqual(
                    h5_file["response_processing/smoothing"].attrs["polynomial_order"],
                    2,
                )
                self.assertEqual(
                    h5_file["response_processing/low_pass"].attrs["cutoff_hz"],
                    0.5,
                )
                self.assertEqual(
                    h5_file["response_processing/normalization"].attrs["method"],
                    "epoch_peak",
                )
                np.testing.assert_allclose(
                    h5_file["response_processing/normalization/factors"][()],
                    np.array([2.0, 4.0], dtype=np.float64),
                )
                np.testing.assert_array_equal(
                    h5_file["response_processing/correlation_scores/included_mask"][()],
                    np.array([True, False]),
                )
                trial_group = h5_file["responses/trials/trial_0001"]
                self.assertEqual(trial_group.attrs["epoch_name"], "Gray")
                np.testing.assert_array_equal(
                    trial_group["values"][()],
                    np.array([[0.0, 1.0], [2.0, 3.0]]),
                )

            with trials_csv_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["epoch_name"], "Gray")
            self.assertEqual(rows[0]["roi_label"], "roi_1")
            self.assertEqual(float(rows[0]["time_s_0.000000"]), 0.0)
            self.assertEqual(float(rows[0]["time_s_0.500000"]), 2.0)

            with grouped_csv_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as csv_file:
                grouped_rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(grouped_rows), 6)
            self.assertEqual(grouped_rows[0]["epoch_name"], "Gray")
            self.assertEqual(grouped_rows[0]["roi_label"], "roi_1")
            self.assertEqual(grouped_rows[0]["statistic"], "mean")
            self.assertEqual(float(grouped_rows[0]["time_s_0.000000"]), 0.0)
            self.assertEqual(float(grouped_rows[0]["time_s_0.500000"]), 2.0)
            self.assertEqual(grouped_rows[1]["statistic"], "sem")
            self.assertEqual(float(grouped_rows[1]["time_s_0.500000"]), 0.0)
            self.assertEqual(grouped_rows[2]["statistic"], "n_trials")
            self.assertEqual(int(grouped_rows[2]["time_s_0.500000"]), 1)

            loaded = load_analysis_outputs(h5_path)

            self.assertIsNotNone(loaded.roi_set)
            self.assertIsNotNone(loaded.traces)
            self.assertIsNotNone(loaded.dff)
            self.assertIsNotNone(loaded.grouped_responses)
            self.assertIsNotNone(loaded.response_processing_options)
            self.assertIsNotNone(loaded.normalization_factors)
            self.assertIsNotNone(loaded.correlation_scores)
            self.assertEqual(len(loaded.epoch_windows), 1)
            self.assertEqual(loaded.baseline_windows, (windows[0].window,))
            if loaded.roi_set is not None:
                self.assertEqual(loaded.roi_set.labels, ("roi_1", "roi_2"))
            if loaded.traces is not None:
                np.testing.assert_array_equal(
                    loaded.traces.raw_values,
                    traces.raw_values,
                )
            if loaded.dff is not None:
                np.testing.assert_array_equal(loaded.dff.values, dff.values)
            if loaded.grouped_responses is not None:
                self.assertEqual(len(loaded.grouped_responses.trials), 1)
            if loaded.response_processing_options is not None:
                self.assertEqual(
                    loaded.response_processing_options.smoothing.method,
                    "moving_average",
                )
                self.assertEqual(
                    loaded.response_processing_options.low_pass.cutoff_hz,
                    0.5,
                )
                self.assertEqual(
                    loaded.response_processing_options.normalization.epoch_name,
                    "Gray",
                )
                self.assertEqual(
                    loaded.response_processing_options.correlation_filter.window_seconds,
                    (0.0, 2.0),
                )
            if loaded.normalization_factors is not None:
                np.testing.assert_allclose(
                    loaded.normalization_factors.factors,
                    np.array([2.0, 4.0], dtype=np.float64),
                )
            if loaded.correlation_scores is not None:
                np.testing.assert_array_equal(
                    loaded.correlation_scores.included_mask,
                    np.array([True, False]),
                )

    def test_csv_requires_grouped_responses(self) -> None:
        """Confirm response CSV is not written without response groups.

        Inputs: a requested CSV path and no grouped response object.
        Outputs: a clear validation error.
        """
        with temporary_directory() as temp_dir:
            h5_path = Path(temp_dir) / "analysis_outputs.h5"
            with self.assertRaisesRegex(ValueError, "grouped_responses"):
                save_analysis_outputs(
                    h5_path,
                    response_summary_trials_csv=Path(temp_dir) / "summary.csv",
                )
            self.assertFalse(h5_path.exists())

    def test_failed_hdf5_save_preserves_existing_analysis_file(self) -> None:
        """Confirm a failed HDF5 rewrite leaves the previous complete file.

        Inputs: An existing analysis HDF5 file and a later save whose writer
            fails after opening its temporary file.
        Outputs: The original HDF5 file remains loadable and no temp file is left.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            h5_path = root / "analysis_outputs.h5"
            save_analysis_outputs(h5_path, traces=self._traces())

            with (
                patch(
                    "twopy.analysis.persistence._write_dff",
                    side_effect=RuntimeError("forced write failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "forced write failure"),
            ):
                save_analysis_outputs(h5_path, dff=self._dff())

            loaded = load_analysis_outputs(h5_path)
            self.assertIsNotNone(loaded.traces)
            self.assertIsNone(loaded.dff)
            self.assertEqual(tuple(root.glob(".analysis_outputs.h5.*.tmp")), ())

    def test_atomic_hdf5_save_preserves_existing_file_mode(self) -> None:
        """Confirm atomic replacement keeps permissions from the old file.

        Inputs: An existing analysis file with an explicit read mode.
        Outputs: The rewritten analysis file keeps the same permission bits.
        """
        with temporary_directory() as temp_dir:
            h5_path = Path(temp_dir) / "analysis_outputs.h5"
            save_analysis_outputs(h5_path, traces=self._traces())
            h5_path.chmod(0o640)

            save_analysis_outputs(h5_path, dff=self._dff())

            self.assertEqual(h5_path.stat().st_mode & 0o777, 0o640)

    def test_failed_grouped_csv_write_preserves_existing_file(self) -> None:
        """Confirm a failed grouped CSV rewrite leaves the previous complete file.

        Inputs: An existing CSV file and a later grouped-response writer that
            fails while streaming rows.
        Outputs: The original CSV text remains unchanged and no temp file is left.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            output_path = root / "response_summary_grouped.csv"
            previous_text = "previous,content\n1,2\n"
            output_path.write_text(previous_text, encoding="utf-8")
            grouped = group_delta_f_over_f_by_epoch(
                self._dff(),
                (EpochFrameWindow(FrameWindow(0, 0, 2, "gray"), 1, "Gray"),),
                data_rate_hz=2.0,
            )

            def failing_rows(
                *_args: object,
                **_kwargs: object,
            ) -> Iterator[dict[str, str | int | float]]:
                yield {
                    "epoch_number": 1,
                    "epoch_name": "Gray",
                    "roi_label": "roi_1",
                    "statistic": "mean",
                    "time_s_0.000000": 0.0,
                    "time_s_0.500000": 2.0,
                }
                raise RuntimeError("forced csv failure")

            with (
                patch(
                    "twopy.analysis.persistence._iter_grouped_response_rows",
                    side_effect=failing_rows,
                ),
                self.assertRaisesRegex(RuntimeError, "forced csv failure"),
            ):
                write_response_summary_grouped_csv(grouped, output_path)

            self.assertEqual(output_path.read_text(encoding="utf-8"), previous_text)
            self.assertEqual(
                tuple(root.glob(".response_summary_grouped.csv.*.tmp")),
                (),
            )

    def test_rejects_persisted_grouped_response_with_misaligned_time_axis(
        self,
    ) -> None:
        """Confirm loaded grouped responses pass the shared shape contract.

        Inputs: Analysis HDF5 with one response trial whose time axis is too
            short for its value matrix.
        Outputs: clear validation error before returning loaded outputs.
        """
        with temporary_directory() as temp_dir:
            h5_path = Path(temp_dir) / "analysis_outputs.h5"
            grouped = group_delta_f_over_f_by_epoch(
                self._dff(),
                (EpochFrameWindow(FrameWindow(0, 0, 2, "gray"), 1, "Gray"),),
                data_rate_hz=2.0,
            )
            save_analysis_outputs(h5_path, grouped_responses=grouped)
            with h5py.File(h5_path, "r+") as h5_file:
                trial_group = h5_file["responses/trials/trial_0001"]
                del trial_group["time_seconds"]
                trial_group.create_dataset(
                    "time_seconds",
                    data=np.array([0.0], dtype=np.float64),
                )

            with self.assertRaisesRegex(ValueError, "time_seconds"):
                load_analysis_outputs(h5_path)

    def test_rejects_roi_set_without_current_spatial_orientation(self) -> None:
        """Confirm embedded ROI masks carry the converted movie orientation.

        Inputs: Analysis HDF5 where the ROI orientation marker is missing.
        Outputs: clear validation error before returning ROI masks.
        """
        for object_path in ("roi_set", "roi_set/masks"):
            with (
                self.subTest(object_path=object_path),
                temporary_directory() as temp_dir,
            ):
                h5_path = Path(temp_dir) / "analysis_outputs.h5"
                save_analysis_outputs(
                    h5_path,
                    roi_set=make_roi_set(np.ones((1, 2, 2), dtype=bool)),
                )
                with h5py.File(h5_path, "r+") as h5_file:
                    del h5_file[object_path].attrs[SPATIAL_ORIENTATION_ATTR]

                with self.assertRaisesRegex(ValueError, object_path):
                    load_analysis_outputs(h5_path)

    def test_grouped_response_csv_keeps_time_series_statistics(self) -> None:
        """Confirm grouped CSV rows contain mean, SEM, and count traces.

        Inputs: two one-frame trials from the same epoch and one ROI.
        Outputs: grouped CSV rows with one relative-time column.
        """
        with temporary_directory() as temp_dir:
            output_path = Path(temp_dir) / "response_summary_grouped.csv"
            dff = RoiDeltaFOverF(
                fluorescence=np.ones((2, 1), dtype=np.float64),
                baseline=np.ones((2, 1), dtype=np.float64),
                values=np.array([[0.0], [2.0]], dtype=np.float64),
                labels=("roi_1",),
                start_frame=0,
                stop_frame=2,
                tau=0.0,
                amplitudes=np.ones(1, dtype=np.float64),
                baseline_frame_numbers=np.array([], dtype=np.float64),
                baseline_fluorescence=np.empty((0, 1), dtype=np.float64),
                metadata={},
            )
            grouped = group_delta_f_over_f_by_epoch(
                dff,
                (
                    EpochFrameWindow(FrameWindow(0, 0, 1, "odor_a"), 2, "Odor"),
                    EpochFrameWindow(FrameWindow(1, 1, 2, "odor_b"), 2, "Odor"),
                ),
                data_rate_hz=2.0,
            )

            write_response_summary_grouped_csv(grouped, output_path)

            with output_path.open("r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(
                [row["statistic"] for row in rows],
                ["mean", "sem", "n_trials"],
            )
            self.assertEqual(float(rows[0]["time_s_0.000000"]), 1.0)
            self.assertEqual(float(rows[1]["time_s_0.000000"]), 1.0)
            self.assertEqual(int(rows[2]["time_s_0.000000"]), 2)

    def test_rejects_unknown_persisted_trace_statistic(self) -> None:
        """Confirm trace metadata is validated when loading analysis output.

        Inputs: HDF5 traces group with an unsupported statistic attribute.
        Outputs: clear validation error before returning typed traces.
        """
        with temporary_directory() as temp_dir:
            h5_path = Path(temp_dir) / "analysis_outputs.h5"
            save_analysis_outputs(h5_path, traces=self._traces())
            with h5py.File(h5_path, "r+") as h5_file:
                h5_file["traces"].attrs["statistic"] = "median"

            with self.assertRaisesRegex(ValueError, "traces statistic"):
                load_analysis_outputs(h5_path)

    def test_rejects_unknown_persisted_background_method(self) -> None:
        """Confirm background-correction metadata is validated on load.

        Inputs: HDF5 traces group with an unsupported method attribute.
        Outputs: clear validation error before returning typed traces.
        """
        with temporary_directory() as temp_dir:
            h5_path = Path(temp_dir) / "analysis_outputs.h5"
            save_analysis_outputs(h5_path, traces=self._traces())
            with h5py.File(h5_path, "r+") as h5_file:
                h5_file["traces"].attrs["method"] = "median_filter"

            with self.assertRaisesRegex(
                ValueError,
                "traces background correction method",
            ):
                load_analysis_outputs(h5_path)

    def test_rejects_persisted_array_with_wrong_dtype(self) -> None:
        """Confirm persisted arrays keep their declared numeric dtype.

        Inputs: Analysis HDF5 where ``dff/values`` was rewritten as integers.
        Outputs: clear validation error before returning a dF/F object.
        """
        with temporary_directory() as temp_dir:
            h5_path = Path(temp_dir) / "analysis_outputs.h5"
            save_analysis_outputs(h5_path, dff=self._dff())
            with h5py.File(h5_path, "r+") as h5_file:
                del h5_file["dff/values"]
                h5_file["dff"].create_dataset(
                    "values",
                    data=np.arange(6, dtype=np.int64).reshape(3, 2),
                )

            with self.assertRaisesRegex(ValueError, "dff/values must have dtype"):
                load_analysis_outputs(h5_path)

    def test_rejects_persisted_array_with_wrong_rank(self) -> None:
        """Confirm persisted arrays keep their expected rank.

        Inputs: Analysis HDF5 where ``traces/raw_values`` is one-dimensional.
        Outputs: clear validation error before returning trace objects.
        """
        with temporary_directory() as temp_dir:
            h5_path = Path(temp_dir) / "analysis_outputs.h5"
            save_analysis_outputs(h5_path, traces=self._traces())
            with h5py.File(h5_path, "r+") as h5_file:
                del h5_file["traces/raw_values"]
                h5_file["traces"].create_dataset(
                    "raw_values",
                    data=np.arange(3, dtype=np.float64),
                )

            with self.assertRaisesRegex(ValueError, "traces/raw_values must have 2"):
                load_analysis_outputs(h5_path)

    def _traces(self) -> BackgroundCorrectedRoiTraces:
        """Create small background-corrected traces.

        Args:
            None.

        Returns:
            ``BackgroundCorrectedRoiTraces`` with three frames and two ROIs.
        """
        raw = np.arange(6, dtype=np.float64).reshape(3, 2) + 10.0
        background = np.ones_like(raw)
        return BackgroundCorrectedRoiTraces(
            raw_values=raw,
            background_values=background,
            corrected_values=raw - background,
            labels=("roi_1", "roi_2"),
            start_frame=0,
            stop_frame=3,
            statistic="mean",
            method="movie_global_percentile",
            metadata={"method": "movie_global_percentile", "percentile": 10.0},
        )

    def _dff(self) -> RoiDeltaFOverF:
        """Create small dF/F output.

        Args:
            None.

        Returns:
            ``RoiDeltaFOverF`` with three frames and two ROIs.
        """
        values = np.arange(6, dtype=np.float64).reshape(3, 2)
        return RoiDeltaFOverF(
            fluorescence=values + 10.0,
            baseline=np.full_like(values, 10.0),
            values=values,
            labels=("roi_1", "roi_2"),
            start_frame=0,
            stop_frame=3,
            tau=0.0,
            amplitudes=np.array([10.0, 10.0]),
            baseline_frame_numbers=np.array([0.5]),
            baseline_fluorescence=np.array([[10.0, 10.0]]),
            metadata={"method": "test", "fit_mode": "direct_bounded_tau"},
        )


if __name__ == "__main__":
    unittest.main()
