"""Core napari adapter tests.

Inputs: shared fake napari state and tiny converted recordings.
Outputs: assertions for one napari workflow area.
"""

from tests.napari_support import (
    APPLICATION_TITLE,
    TWOPY_SIDEBAR_MINIMUM_WIDTH,
    NapariAdapterTestCase,
    Path,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QWidget,
    __version__,
    _FakeViewer,
    _write_converted_recording,
    cast,
    counted_noun,
    create_viewer,
    load_roi_set,
    make_roi_set,
    np,
    open_recording_in_napari,
    patch,
    roi_label_image_from_layer,
    save_napari_label_rois,
    save_roi_set,
    temporary_directory,
    unittest,
)


class CoreNapariAdapterTest(NapariAdapterTestCase):
    """Core napari adapter tests."""

    def test_counted_noun_formats_status_text(self) -> None:
        """Confirm napari status text uses normal singular and plural words.

        Inputs: singular and plural counts.
        Outputs: phrases with full singular or plural nouns.
        """
        self.assertEqual(counted_noun(1, "file"), "1 file")
        self.assertEqual(counted_noun(2, "file"), "2 files")
        self.assertEqual(counted_noun(1, "ROI", "ROIs"), "1 ROI")
        self.assertEqual(counted_noun(2, "ROI", "ROIs"), "2 ROIs")

    def test_create_viewer_uses_twopy_version_window_title(self) -> None:
        """Confirm the launcher brands the top-level napari window.

        Inputs: patched napari Viewer constructor.
        Outputs: viewer construction receives the twopy name and version.
        """
        viewer = _FakeViewer()

        with patch("napari.Viewer", return_value=viewer) as viewer_constructor:
            created = create_viewer()

        self.assertIs(created, viewer)
        self.assertEqual(APPLICATION_TITLE, f"twopy {__version__}")
        viewer_constructor.assert_called_once_with(title=APPLICATION_TITLE)

    def test_opens_empty_roi_labels_layer_by_default(self) -> None:
        """Confirm new recordings open with an editable empty ROI layer.

        Inputs: tiny converted recording and fake viewer.
        Outputs: one zero-valued labels layer matching the movie frame shape.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            viewer = _FakeViewer()

            opened = open_recording_in_napari(recording_path, viewer=viewer)

            self.assertIsNotNone(opened.roi_labels_layer)
            self.assertIsNotNone(opened.load_widget)
            self.assertIsNotNone(opened.loaded_recordings_widget)
            self.assertIsNotNone(opened.twopy_sidebar_widget)
            self.assertIsNotNone(opened.twopy_sidebar_dock_widget)
            self.assertIsNotNone(opened.response_plot_widget)
            self.assertIsNotNone(opened.response_plot_dock_widget)
            self.assertIsNotNone(opened.response_options_widget)
            self.assertEqual(len(viewer.labels), 1)
            self.assertEqual(viewer.labels[0].options["opacity"], 0.5)
            self.assertEqual(viewer.labels[0].options["blending"], "additive")
            self.assertEqual(viewer.labels[0].brush_size, 6)
            self.assertEqual(viewer.images[0].options["opacity"], 0.5)
            self.assertEqual(viewer.images[0].options["gamma"], 1.3)
            self.assertEqual(viewer.images[0].options["contrast_limits"], (4.3, 7.0))
            self.assertEqual(viewer.images[0].contrast_limits_range, (4.0, 7.0))
            self.assertEqual(len(viewer.window.dock_widgets), 2)
            self.assertEqual(viewer.window.dock_widgets[0].name, "twopy responses")
            self.assertEqual(viewer.window.dock_widgets[0].area, "top")
            self.assertEqual(viewer.window._qt_window.resize_calls[0].sizes, [345])
            self.assertEqual(
                viewer.window._qt_window.resize_calls[0].docks,
                [viewer.window.dock_widgets[0]],
            )
            self.assertEqual(viewer.window.dock_widgets[1].name, "twopy")
            self.assertEqual(viewer.window.dock_widgets[1].area, "right")
            load_panel = cast(QWidget, opened.load_widget)
            load_labels = {label.text() for label in load_panel.findChildren(QLabel)}
            self.assertNotIn("Recording", load_labels)
            self.assertNotIn("ROI file", load_labels)
            browse_buttons = [
                button
                for button in load_panel.findChildren(QPushButton)
                if button.text() == "Browse"
            ]
            self.assertEqual(len(browse_buttons), 0)
            load_buttons = [
                button.text() for button in load_panel.findChildren(QPushButton)
            ]
            self.assertEqual(
                load_buttons,
                ["Search database", "Load manually", "Load CSV list"],
            )
            sidebar_tabs = cast(QTabWidget, opened.twopy_sidebar_widget)
            self.assertIs(sidebar_tabs, opened.response_options_widget)
            self.assertEqual(
                sidebar_tabs.minimumWidth(),
                TWOPY_SIDEBAR_MINIMUM_WIDTH,
            )
            self.assertEqual(sidebar_tabs.currentIndex(), 0)
            options_widget = cast(QTabWidget, opened.response_options_widget)
            self.assertEqual(
                tuple(
                    options_widget.tabText(index)
                    for index in range(options_widget.count())
                ),
                (
                    "Load",
                    "Metadata",
                    "Plot",
                    "ROIs",
                    "Epochs",
                    "Export",
                ),
            )
            for index in range(options_widget.count()):
                self.assertIsInstance(options_widget.widget(index), QScrollArea)
            options_buttons = {
                button.text() for button in options_widget.findChildren(QPushButton)
            }
            self.assertIn("Search database", options_buttons)
            self.assertIn("Load manually", options_buttons)
            self.assertIn("Load CSV list", options_buttons)
            self.assertIn("Open Group Matching", options_buttons)
            self.assertIn("Save loaded list", options_buttons)
            self.assertIn("Save ROIs + analysis", options_buttons)
            self.assertNotIn("Recompute preview now", options_buttons)
            self.assertIn("Reload saved analysis", options_buttons)
            action_buttons = {
                button.text(): button
                for button in options_widget.findChildren(QPushButton)
            }
            self.assertTrue(action_buttons["Open Group Matching"].isEnabled())
            self.assertTrue(action_buttons["Reload saved analysis"].isEnabled())
            self.assertTrue(action_buttons["Save loaded list"].isEnabled())
            group_titles = {
                group.title() for group in options_widget.findChildren(QGroupBox)
            }
            self.assertIn("Recording", group_titles)
            self.assertIn("Microscope", group_titles)
            self.assertIn("Outputs", group_titles)
            self.assertNotIn("Status", group_titles)
            self.assertIn("Plot", group_titles)
            self.assertIn("Smoothing", group_titles)
            np.testing.assert_array_equal(
                roi_label_image_from_layer(viewer.labels[0]),
                np.zeros((2, 2), dtype=np.int64),
            )

    def test_opens_converted_recording_with_roi_labels(self) -> None:
        """Confirm the adapter sends mean image, movie preview, and ROIs.

        Inputs: tiny converted recording, ROI HDF5 file, and fake viewer.
        Outputs: fake napari layers with expected shapes.
        """
        with temporary_directory() as temp_dir:
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
                    labels=("roi_0003", "roi_0004"),
                ),
                roi_path,
            )
            viewer = _FakeViewer()

            opened = open_recording_in_napari(
                recording_path,
                roi_set=roi_path,
                viewer=viewer,
                movie_frame_range=(0, 1),
            )

            self.assertIs(opened.viewer, viewer)
            self.assertEqual(opened.recording.movie.shape, (3, 2, 2))
            self.assertEqual(len(viewer.images), 2)
            self.assertEqual(viewer.images[0].name, "mean image")
            self.assertEqual(viewer.images[0].options["contrast_limits"], (4.3, 7.0))
            self.assertEqual(viewer.images[0].contrast_limits_range, (4.0, 7.0))
            self.assertEqual(np.asarray(viewer.images[0].data).shape, (2, 2))
            self.assertEqual(viewer.images[1].name, "aligned movie")
            self.assertEqual(viewer.images[1].options["blending"], "additive")
            self.assertEqual(
                viewer.images[1].options["contrast_limits"],
                (4.0, 7.0),
            )
            self.assertEqual(viewer.images[1].contrast_limits_range, (0.0, 7.0))
            self.assertEqual(np.asarray(viewer.images[1].data).shape, (2, 2, 2))
            self.assertEqual(len(viewer.labels), 1)
            np.testing.assert_array_equal(
                np.unique(np.asarray(viewer.labels[0].data)),
                np.array([0, 3, 4]),
            )

    def test_saves_napari_label_image_as_roi_file(self) -> None:
        """Confirm edited napari labels round-trip through core ROI storage.

        Inputs: one label image and a temporary output path.
        Outputs: saved ROI HDF5 file loaded by the normal ROI helper.
        """
        with temporary_directory() as temp_dir:
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


if __name__ == "__main__":
    unittest.main()
