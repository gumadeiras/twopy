"""Tests for the thin napari adapter.

Inputs: a tiny converted recording, ROI labels, and a fake viewer.
Outputs: layer data sent to napari-shaped methods and saved ROI HDF5 files.
"""

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from twopy import (
    load_roi_set,
    make_roi_set,
    open_recording_in_napari,
    roi_label_image_from_layer,
    save_napari_label_rois,
    save_roi_set,
)


@dataclass(frozen=True)
class _FakeLayer:
    """Layer object returned by the fake viewer.

    Inputs: layer name and data.
    Outputs: inspectable test object.
    """

    name: str
    data: object


class _FakeViewer:
    """Small napari-shaped viewer for adapter tests."""

    def __init__(self) -> None:
        """Create an empty fake viewer.

        Inputs: none.
        Outputs: viewer with no recorded layers.
        """
        self.images: list[_FakeLayer] = []
        self.labels: list[_FakeLayer] = []

    def add_image(self, data: object, *, name: str, **kwargs: object) -> object:
        """Record an image layer request.

        Args:
            data: Image data sent by the adapter.
            name: Layer name.
            kwargs: Napari image options.

        Returns:
            Fake image layer.
        """
        layer = _FakeLayer(name=name, data=data)
        self.images.append(layer)
        return layer

    def add_labels(self, data: object, *, name: str, **kwargs: object) -> object:
        """Record a labels layer request.

        Args:
            data: Label image sent by the adapter.
            name: Layer name.
            kwargs: Napari label options.

        Returns:
            Fake labels layer.
        """
        layer = _FakeLayer(name=name, data=data)
        self.labels.append(layer)
        return layer


class NapariAdapterTest(unittest.TestCase):
    """Tests napari loading and label saving without starting a GUI."""

    def test_opens_empty_roi_labels_layer_by_default(self) -> None:
        """Confirm new recordings open with an editable empty ROI layer.

        Inputs: tiny converted recording and fake viewer.
        Outputs: one zero-valued labels layer matching the movie frame shape.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            viewer = _FakeViewer()

            opened = open_recording_in_napari(recording_path, viewer=viewer)

            self.assertIsNotNone(opened.roi_labels_layer)
            self.assertEqual(len(viewer.labels), 1)
            np.testing.assert_array_equal(
                roi_label_image_from_layer(viewer.labels[0]),
                np.zeros((2, 2), dtype=np.int64),
            )

    def test_opens_converted_recording_with_roi_labels(self) -> None:
        """Confirm the adapter sends mean image, movie preview, and ROIs.

        Inputs: tiny converted recording, ROI HDF5 file, and fake viewer.
        Outputs: fake napari layers with expected shapes.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            roi_path = root / "rois.h5"
            save_roi_set(
                make_roi_set(
                    np.array(
                        [
                            [[True, False], [False, False]],
                            [[False, False], [False, True]],
                        ],
                    ),
                ),
                roi_path,
            )
            viewer = _FakeViewer()

            opened = open_recording_in_napari(
                recording_path,
                roi_set=roi_path,
                viewer=viewer,
                movie_frame_range=(0, 2),
            )

            self.assertIs(opened.viewer, viewer)
            self.assertEqual(opened.recording.movie.shape, (3, 2, 2))
            self.assertEqual(len(viewer.images), 2)
            self.assertEqual(viewer.images[0].name, "mean image")
            self.assertEqual(np.asarray(viewer.images[0].data).shape, (2, 2))
            self.assertEqual(viewer.images[1].name, "aligned movie")
            self.assertEqual(np.asarray(viewer.images[1].data).shape, (2, 2, 2))
            self.assertEqual(len(viewer.labels), 1)
            np.testing.assert_array_equal(
                np.unique(np.asarray(viewer.labels[0].data)),
                np.array([0, 1, 2]),
            )

    def test_saves_napari_label_image_as_roi_file(self) -> None:
        """Confirm edited napari labels round-trip through core ROI storage.

        Inputs: one label image and a temporary output path.
        Outputs: saved ROI HDF5 file loaded by the normal ROI helper.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "drawn_rois.h5"

            saved = save_napari_label_rois(
                np.array([[0, 1], [2, 2]]),
                output_path,
                label_prefix="drawn",
            )
            loaded = load_roi_set(output_path)

            self.assertEqual(saved.labels, ("drawn_0001", "drawn_0002"))
            self.assertEqual(loaded.labels, saved.labels)
            np.testing.assert_array_equal(loaded.masks, saved.masks)


def _write_converted_recording(root: Path) -> Path:
    """Write a tiny converted recording for adapter tests.

    Args:
        root: Temporary directory receiving HDF5 files.

    Returns:
        Path to ``recording_data.h5``.
    """
    movie_values = np.arange(12, dtype=np.float64).reshape(3, 2, 2)
    movie_path = root / "aligned_movie.h5"
    with h5py.File(movie_path, "w") as h5_file:
        h5_file.attrs["twopy_format"] = "aligned-movie"
        h5_file.create_dataset("movie/aligned", data=movie_values)

    recording_path = root / "recording_data.h5"
    with h5py.File(recording_path, "w") as h5_file:
        h5_file.attrs["twopy_format"] = "converted-recording"
        h5_file.attrs["source_session_dir"] = str(root / "source")
        movie_group = h5_file.create_group("movie")
        movie_group.attrs["aligned_movie_file"] = "aligned_movie.h5"
        movie_group.attrs["aligned_movie_dataset"] = "movie/aligned"
        movie_group.attrs["aligned_movie_shape"] = movie_values.shape
        movie_group.attrs["aligned_movie_dtype"] = "float64"
        movie_group.create_dataset("mean_image", data=movie_values.mean(axis=0))
        crop_group = movie_group.create_group("alignment_valid_crop")
        crop_group.attrs["source"] = "full_frame"
        crop_group.attrs["axis0_start"] = 0
        crop_group.attrs["axis0_stop"] = 2
        crop_group.attrs["axis1_start"] = 0
        crop_group.attrs["axis1_stop"] = 2
        crop_group.attrs["original_shape"] = (2, 2)
        crop_group.create_dataset(
            "alignment_shift_pixels",
            data=np.zeros(3, dtype=np.float64),
        )
        crop_group.create_dataset(
            "motion_artifact_mask",
            data=np.zeros(3, dtype=np.bool_),
        )
        h5_file.create_group("metadata")
        h5_file.create_group("run")
        stimulus_group = h5_file.create_group("stimulus")
        stimulus_group.create_dataset("data", data=np.zeros((1, 2), dtype=np.float64))
        stimulus_group.create_dataset(
            "data_column_names",
            data=np.asarray(
                ("time_seconds", "epoch_number"),
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        stimulus_group.create_dataset("parameters_json", data="[]")
        stimulus_group.create_dataset("function_lookup_json", data="{}")
        stimulus_group.create_dataset("stimulus_specific_columns_json", data="{}")
        photodiode_group = h5_file.create_group("photodiode")
        photodiode_group.create_dataset(
            "imaging_res_pd",
            data=np.zeros(3, dtype=np.float64),
        )
        photodiode_group.create_dataset(
            "high_res_pd",
            data=np.zeros(3, dtype=np.float64),
        )
        frame_counts = h5_file.create_group("frame_counts")
        frame_counts.attrs["aligned_movie_frames"] = 3
        frame_counts.attrs["imaging_res_pd_samples"] = 3
        frame_counts.attrs["acquisition_number_of_frames"] = 3
        frame_counts.attrs["imaging_res_pd_minus_movie"] = 0
        frame_counts.attrs["acquisition_minus_movie"] = 0
    return recording_path


if __name__ == "__main__":
    unittest.main()
