"""Napari ROI visibility tests.

Inputs: shared fake napari state and tiny converted recordings.
Outputs: assertions for one napari workflow area.
"""

from tests.napari_support import (
    Any,
    Labels,
    NapariAdapterTestCase,
    QApplication,
    QCheckBox,
    QColor,
    QPushButton,
    QScrollArea,
    _two_roi_response_plot_data,
    _two_roi_response_plot_data_with_correlation_scores,
    apply_roi_visibility_to_labels_layer,
    cast,
    create_response_plot_widget,
    global_value_bounds,
    interior_random_frame_indices,
    np,
    replace,
    resolve_movie_frame_range,
    unittest,
    visibility_options_widget,
)


class NapariRoiVisibilityTest(NapariAdapterTestCase):
    """Napari ROI visibility tests."""

    def test_correlation_filter_hides_excluded_rois_in_roi_tab(self) -> None:
        """Confirm correlation QC initializes ROI visibility checkboxes.

        Inputs: two-ROI plot data whose correlation mask excludes ROI 2.
        Outputs: ROI 2 is unchecked/hidden until unfiltered plot data loads.
        """
        _ = QApplication.instance() or QApplication([])
        response_widget = cast(Any, create_response_plot_widget(None))

        response_widget.set_response_plot_data(
            _two_roi_response_plot_data_with_correlation_scores(),
            reset_axes=True,
        )

        self.assertEqual(response_widget._visible_roi_indices(), (0,))
        self.assertEqual(response_widget._roi_visibility, {0: True, 1: False})

        response_widget.set_response_plot_data(
            _two_roi_response_plot_data(),
            reset_axes=True,
        )

        self.assertEqual(response_widget._visible_roi_indices(), (0, 1))

    def test_initial_visible_roi_indices_do_not_remove_roi_rows(self) -> None:
        """Confirm custom visibility filters uncheck ROIs without deleting rows."""
        _ = QApplication.instance() or QApplication([])
        response_widget = cast(Any, create_response_plot_widget(None))
        plot_data = replace(
            _two_roi_response_plot_data(),
            visible_roi_indices=(0,),
        )

        response_widget.set_response_plot_data(plot_data, reset_axes=True)

        self.assertEqual(response_widget._roi_labels(), ("roi_0001", "roi_0002"))
        self.assertEqual(response_widget._visible_roi_indices(), (0,))
        self.assertEqual(response_widget._roi_visibility, {0: True, 1: False})

    def test_global_value_bounds_accepts_empty_roi_selection(self) -> None:
        """Confirm empty ROI selection uses stable default y-axis bounds.

        Inputs: two-ROI plot data with no visible ROI rows.
        Outputs: default y-axis bounds instead of a NumPy empty-reduction
        error.
        """
        self.assertEqual(
            global_value_bounds(_two_roi_response_plot_data(), ()), (-1.0, 1.0)
        )

    def test_visibility_select_none_uses_one_batch_callback(self) -> None:
        """Confirm bulk visibility changes avoid one redraw per checkbox.

        Inputs: three ROI labels and a Select-none button click.
        Outputs: no per-checkbox callback and one batch visibility update.
        """
        _ = QApplication.instance() or QApplication([])
        single_calls: list[tuple[str, bool]] = []
        batch_calls: list[dict[str, bool]] = []
        widget = visibility_options_widget(
            title="ROIs",
            labels=("roi_1", "roi_2", "roi_3"),
            visibility={"roi_1": True, "roi_2": True, "roi_3": True},
            on_change=lambda label, visible: single_calls.append((label, visible)),
            on_change_batch=lambda visibility: batch_calls.append(visibility),
        )
        none_button = next(
            button
            for button in widget.findChildren(QPushButton)
            if button.text() == "None"
        )

        none_button.click()

        self.assertEqual(single_calls, [])
        self.assertEqual(
            batch_calls,
            [{"roi_1": False, "roi_2": False, "roi_3": False}],
        )

    def test_visibility_options_use_stable_keys_for_duplicate_labels(self) -> None:
        """Confirm duplicate display labels can still toggle distinct items.

        Inputs: two checkboxes with the same visible label but separate keys.
        Outputs: toggling the second checkbox reports the second key.
        """
        _ = QApplication.instance() or QApplication([])
        calls: list[tuple[object, bool]] = []
        widget = visibility_options_widget(
            title="Epochs",
            labels=("same epoch", "same epoch"),
            keys=(0, 1),
            visibility={0: True, 1: True},
            on_change=lambda key, visible: calls.append((key, visible)),
        )
        checkboxes = widget.findChildren(QCheckBox)

        checkboxes[1].setChecked(False)

        self.assertEqual(calls, [(1, False)])

    def test_visibility_options_read_initial_state_from_keys(self) -> None:
        """Confirm keyed rows do not read visibility from duplicate labels.

        Inputs: two rows with the same display label and different keys.
        Outputs: each checkbox uses the matching key state.
        """
        _ = QApplication.instance() or QApplication([])
        widget = visibility_options_widget(
            title="ROIs",
            labels=("same roi", "same roi"),
            keys=(0, 1),
            visibility={0: True, 1: False},
            on_change=lambda _key, _visible: None,
        )
        checkboxes = widget.findChildren(QCheckBox)

        self.assertTrue(checkboxes[0].isChecked())
        self.assertFalse(checkboxes[1].isChecked())

    def test_visibility_options_do_not_add_inner_scroll_area(self) -> None:
        """Confirm ROI/Epoch rows rely on the tab scrollbar.

        Inputs: many visibility labels.
        Outputs: option widget has no nested list scroll area.
        """
        _ = QApplication.instance() or QApplication([])
        widget = visibility_options_widget(
            title="ROIs",
            labels=tuple(f"roi_{index}" for index in range(20)),
            visibility={},
            on_change=lambda _key, _visible: None,
        )

        self.assertEqual(widget.findChildren(QScrollArea), [])

    def test_roi_visibility_can_hide_labels_layer_rois(self) -> None:
        """Confirm plot ROI visibility can hide ROI labels in napari.

        Inputs: one Labels layer with two ROI labels and one hidden ROI.
        Outputs: hidden ROI color is dimmed while label pixels remain.
        """
        label_image = np.array([[0, 1], [2, 0]], dtype=np.int64)
        layer = Labels(label_image)

        apply_roi_visibility_to_labels_layer(
            layer,
            roi_labels=("roi_1", "roi_2"),
            visibility={"roi_1": False, "roi_2": True},
            colors=(QColor("#ff0000"), QColor("#0000ff")),
        )

        np.testing.assert_array_equal(layer.data, label_image)
        self.assertAlmostEqual(layer.get_color(1)[3], 0.2)
        self.assertEqual(layer.get_color(2)[3], 1.0)

    def test_roi_visibility_keeps_new_labels_drawable(self) -> None:
        """Confirm hiding ROIs does not make future Labels values transparent.

        Inputs: Labels layer with one hidden high-numbered ROI label.
        Outputs: that ROI is hidden, but a new unseen label keeps visible color.
        """
        label_image = np.zeros((2, 2), dtype=np.int64)
        layer = Labels(label_image)

        apply_roi_visibility_to_labels_layer(
            layer,
            roi_labels=("roi_0010",),
            visibility={"roi_0010": False},
            colors=(QColor("#ff0000"),),
        )

        self.assertAlmostEqual(layer.get_color(10)[3], 0.2)
        self.assertEqual(layer.get_color(11)[3], 1.0)
        self.assertEqual(layer.get_color(12)[3], 1.0)
        self.assertFalse(np.array_equal(layer.get_color(11), layer.get_color(12)))

    def test_roi_visibility_layer_uses_stable_keys_for_duplicate_names(self) -> None:
        """Confirm duplicate ROI names can hide distinct label values.

        Inputs: two label values with the same displayed ROI name.
        Outputs: hiding the second key dims only the second label.
        """
        layer = Labels(np.array([[1, 2]], dtype=np.int64))

        apply_roi_visibility_to_labels_layer(
            layer,
            roi_labels=("cell", "cell"),
            visibility={0: True, 1: False},
            keys=(0, 1),
            colors=(QColor("#ff0000"), QColor("#0000ff")),
        )

        self.assertEqual(layer.get_color(1)[3], 1.0)
        self.assertAlmostEqual(layer.get_color(2)[3], 0.2)

    def test_roi_visibility_ignores_malformed_colormap_metadata(self) -> None:
        """Confirm malformed persisted label-colormap metadata is rebuilt.

        Inputs: Labels layer with incomplete twopy colormap metadata.
        Outputs: visibility still applies instead of raising from metadata access.
        """
        layer = Labels(np.array([[1]], dtype=np.int64))
        layer.metadata["twopy_base_label_colormap"] = {"num_colors": "invalid"}

        apply_roi_visibility_to_labels_layer(
            layer,
            roi_labels=("roi_1",),
            visibility={"roi_1": False},
            colors=(QColor("#ff0000"),),
        )

        self.assertAlmostEqual(layer.get_color(1)[3], 0.2)

    def test_roi_visibility_reuses_cached_base_colormap(self) -> None:
        """Confirm visibility toggles do not rebuild base label colors.

        Inputs: one Labels layer and repeated visibility applications.
        Outputs: the base color dictionary is cached on the layer metadata and
        reused on later toggles.
        """
        layer = Labels(np.array([[1]], dtype=np.int64))

        apply_roi_visibility_to_labels_layer(
            layer,
            roi_labels=("roi_1",),
            visibility={"roi_1": False},
            colors=(QColor("#ff0000"),),
        )
        cached = layer.metadata["twopy_base_label_color_dict"]
        apply_roi_visibility_to_labels_layer(
            layer,
            roi_labels=("roi_1",),
            visibility={"roi_1": True},
            colors=(QColor("#ff0000"),),
        )

        self.assertIs(layer.metadata["twopy_base_label_color_dict"], cached)
        self.assertAlmostEqual(layer.get_color(1)[3], 1.0)

    def test_movie_frame_range_accepts_last_default(self) -> None:
        """Confirm empty-launch widget defaults request the full movie.

        Inputs: start/end frame zero before a recording is selected.
        Outputs: the actual final frame after frame count is known.
        """
        self.assertEqual(
            resolve_movie_frame_range(start_frame=0, end_frame=0, frame_count=3),
            (0, 0),
        )
        self.assertEqual(
            resolve_movie_frame_range(
                start_frame=0,
                end_frame=0,
                frame_count=3,
                zero_end_means_last=True,
            ),
            (0, 2),
        )

    def test_movie_contrast_sampling_skips_recording_edges(self) -> None:
        """Confirm movie display sampling avoids the first and last 10 percent.

        Inputs: 100-frame recording contract and deterministic random seed.
        Outputs: 10 sampled frames, none from the first or last 10 frames.
        """
        frame_indices = interior_random_frame_indices(
            frame_count=100,
            sample_count=10,
            edge_fraction=0.10,
            random_seed=0,
        )

        np.testing.assert_array_equal(
            frame_indices,
            np.array([11, 13, 15, 23, 29, 33, 47, 55, 70, 75]),
        )


if __name__ == "__main__":
    unittest.main()
