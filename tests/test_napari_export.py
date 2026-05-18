"""Napari response export tests.

Inputs: shared fake napari state and tiny converted recordings.
Outputs: assertions for one napari workflow area.
"""

from tests.napari_support import (
    EpochResponseMap,
    Figure,
    LineCollection,
    NapariAdapterTestCase,
    Path,
    QApplication,
    QLabel,
    QPushButton,
    ResponseExportState,
    ResponseMapData,
    ResponseMapOptions,
    SpatialCrop,
    _FakeViewer,
    _tiny_response_plot_data,
    _write_converted_recording,
    _write_source_recording_shape,
    cast,
    chdir,
    create_response_export_tab,
    draw_epoch_response_plot,
    draw_response_heatmap,
    draw_roi_contours,
    export_epoch_plots,
    export_response_heatmaps,
    labels_for_recording_image,
    load_converted_recording,
    np,
    open_recording_in_napari,
    replace,
    roi_boundary_segments,
    temporary_directory,
    unittest,
)


class NapariExportTest(NapariAdapterTestCase):
    """Napari response export tests."""

    def test_response_plot_export_writes_editable_figure_bundle(self) -> None:
        """Confirm response plots export in the expected file formats.

        Inputs: one tiny plot-ready epoch and a temporary output folder.
        Outputs: PDF and PNG files under a plot-specific export folder.
        """
        plot_data = _tiny_response_plot_data()
        with temporary_directory() as temp_dir:
            written = export_epoch_plots(
                plot_data=plot_data,
                output_dir=Path(temp_dir),
                epoch_indices=(0,),
                roi_indices=(0,),
                roi_colors=("#ff0000",),
                show_sem=True,
                time_bounds=(0.0, 1.0),
                value_bounds=(0.0, 1.0),
            )

            self.assertEqual({path.suffix for path in written}, {".pdf", ".png"})
            self.assertEqual({path.parent.name for path in written}, {"plots"})
            self.assertTrue(all(path.is_file() for path in written))

    def test_response_heatmap_export_writes_editable_figure_bundle(self) -> None:
        """Confirm response heatmaps export in the expected file formats.

        Inputs: one tiny response map and a temporary output folder.
        Outputs: PDF and PNG files under the heatmap-specific export folder.
        """
        map_data = ResponseMapData(
            mean_image=np.ones((3, 4), dtype=np.float64),
            epochs=(
                EpochResponseMap(
                    epoch_name="Odor",
                    epoch_number=2,
                    response_values=np.ones((3, 4), dtype=np.float64),
                    trial_count=1,
                ),
            ),
            options=ResponseMapOptions(),
            spatial_crop=SpatialCrop(0, 3, 0, 4, (3, 4), "test"),
            response_scale=1.0,
        )
        with temporary_directory() as temp_dir:
            written = export_response_heatmaps(
                map_data=map_data,
                output_dir=Path(temp_dir),
                epoch_indices=(0,),
            )

            self.assertEqual({path.suffix for path in written}, {".pdf", ".png"})
            self.assertEqual(
                {path.parent.name for path in written},
                {"response_heatmaps"},
            )
            self.assertTrue(all(path.is_file() for path in written))

    def test_response_heatmap_export_colorbar_ticks_match_display_limit(self) -> None:
        """Confirm heatmap export labels the actual robust response limits.

        Inputs: one heatmap whose robust display limit is below one.
        Outputs: colorbar ticks are placed at ``-limit``, ``0``, and ``+limit``.
        """
        map_data = ResponseMapData(
            mean_image=np.ones((2, 2), dtype=np.float64),
            epochs=(
                EpochResponseMap(
                    epoch_name="Odor",
                    epoch_number=1,
                    response_values=np.array(
                        [[-0.2, 0.0], [0.2, 0.0]],
                        dtype=np.float64,
                    ),
                    trial_count=1,
                ),
            ),
            options=ResponseMapOptions(),
            spatial_crop=SpatialCrop(0, 2, 0, 2, (2, 2), "test"),
            response_scale=1.0,
        )
        fig = Figure()
        ax = fig.add_subplot(111)

        draw_response_heatmap(ax, map_data=map_data, epoch=map_data.epochs[0])

        self.assertEqual(len(fig.axes), 2)
        np.testing.assert_allclose(fig.axes[1].get_yticks(), (-0.2, 0.0, 0.2))

    def test_response_plot_export_draws_epoch_marker(self) -> None:
        """Confirm exported response plots show the coarse epoch top rail.

        Inputs: one epoch with a marked span from zero to one second.
        Outputs: a quiet gold line is drawn at the top of the response axis.
        """
        plot_data = _tiny_response_plot_data()
        epoch = replace(plot_data.epochs[0], epoch_time_spans=((0.0, 1.0),))
        fig = Figure()
        ax = fig.add_subplot(111)

        draw_epoch_response_plot(
            ax,
            epoch=epoch,
            roi_indices=(0,),
            roi_colors=("#ff0000",),
            show_sem=False,
            time_bounds=(-0.5, 1.5),
            value_bounds=(0.0, 1.0),
        )

        active_lines = [line for line in ax.lines if line.get_color() == "#f2c14e"]
        self.assertEqual(len(active_lines), 1)
        np.testing.assert_allclose(
            np.asarray(active_lines[0].get_xdata(), dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
        )
        np.testing.assert_allclose(
            np.asarray(active_lines[0].get_ydata(), dtype=np.float64),
            np.array([0.965, 0.965], dtype=np.float64),
        )

    def test_response_export_tab_syncs_cached_exports_to_publish_root(self) -> None:
        """Confirm cached Export-tab figures publish then leave local cache.

        Inputs: cached converted recording, source publish destination, and the
        recording-view export button.
        Outputs: local export figures are copied to the source ``twopy`` folder
        and deleted from the cache.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            source_dir = data_root / "fly" / "stim" / "2023" / "10_17"
            local_dir = root / "cache" / "fly" / "stim" / "2023" / "10_17"
            publish_dir = source_dir / "twopy"
            local_dir.mkdir(parents=True)
            _write_source_recording_shape(source_dir)
            recording = load_converted_recording(
                _write_converted_recording(local_dir, source_session_dir=source_dir),
            )
            (root / "config.yml").write_text(
                f"database_path: {root / 'db'}\n"
                f"data_path: {data_root.resolve()}\n"
                "database_access: copy\n"
                "analysis_caching: true\n"
                f"analysis_cache_dir: {root / 'cache'}\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )
            state = ResponseExportState(
                viewer=None,
                recording=recording,
                roi_labels_layer=None,
                plot_data=None,
                output_dir=local_dir / "exports",
                roi_label_values=(),
                roi_colors=(),
                epoch_indices=(),
                response_map_epoch_indices=(),
                roi_indices=(),
                show_sem=True,
                time_bounds=(0.0, 1.0),
                value_bounds=(0.0, 1.0),
            )
            original_cwd = Path.cwd()
            try:
                chdir(root)
                tab = create_response_export_tab(
                    lambda: state,
                    save_analysis_button=QPushButton("Save ROIs + analysis"),
                )
                buttons = {
                    button.text(): button for button in tab.findChildren(QPushButton)
                }
                buttons["Save recording view"].click()
                status = tab.findChild(QLabel)
            finally:
                chdir(original_cwd)

            local_pdf = local_dir / "exports" / "recording_view" / "recording_view.pdf"
            local_png = local_dir / "exports" / "recording_view" / "recording_view.png"
            self.assertFalse(local_pdf.exists())
            self.assertFalse(local_png.exists())
            self.assertTrue(
                (
                    publish_dir / "exports" / "recording_view" / "recording_view.pdf"
                ).is_file(),
            )
            self.assertTrue(
                (
                    publish_dir / "exports" / "recording_view" / "recording_view.png"
                ).is_file(),
            )
            self.assertIsNotNone(status)
            self.assertIn("removed local cache copies", status.text())

    def test_export_roi_contours_keep_display_orientation(self) -> None:
        """Confirm exported ROI contours keep the same top-origin view as napari.

        Inputs: one ROI in the top rows of a display-coordinate label image.
        Outputs: contour vertices stay near the top rows, not vertically flipped.
        """
        labels = np.zeros((6, 6), dtype=np.int64)
        labels[0:2, 1:3] = 1
        fig = Figure()
        ax = fig.add_subplot(111)

        draw_roi_contours(
            ax,
            labels=labels,
            roi_label_values=(1,),
            roi_indices=(0,),
            roi_colors=("#ff0000",),
        )
        contour_lines = cast(LineCollection, ax.collections[0])
        vertices = np.asarray(
            contour_lines.get_segments()[0],
            dtype=np.float64,
        )

        self.assertLess(float(np.max(vertices[:, 1])), 2.0)

    def test_export_roi_contours_follow_pixel_edges_at_image_boundary(self) -> None:
        """Confirm edge-touching ROIs export outlines on the image boundary.

        Inputs: one ROI pixel at the top-left image corner.
        Outputs: boundary segments include the top and left external pixel
        edges instead of shifting inward.
        """
        segments = roi_boundary_segments(
            np.array([[True, False], [False, False]], dtype=np.bool_),
        )

        self.assertIn(((-0.5, -0.5), (0.5, -0.5)), segments)
        self.assertIn(((-0.5, -0.5), (-0.5, 0.5)), segments)

    def test_export_crops_full_frame_display_labels_to_recording_view(self) -> None:
        """Confirm stale full-frame Labels data exports only the displayed crop.

        Inputs: converted recording with an alignment-valid crop and one
        full-frame display label image.
        Outputs: labels cropped to the same display shape as the recording
        image, excluding ROI pixels outside the valid crop.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(
                root,
                movie_values=np.arange(27, dtype=np.float64).reshape(3, 3, 3),
                alignment_valid_crop=SpatialCrop(
                    axis0_start=1,
                    axis0_stop=3,
                    axis1_start=0,
                    axis1_stop=2,
                    original_shape=(3, 3),
                    source="alignment_valid_crop",
                ),
            )
            opened = open_recording_in_napari(
                recording_path,
                viewer=_FakeViewer(),
                add_controls=False,
            )
            full_frame_display_labels = np.zeros((3, 3), dtype=np.int64)
            full_frame_display_labels[0, 0] = 1
            full_frame_display_labels[1, 2] = 2

            labels = labels_for_recording_image(
                full_frame_display_labels,
                recording=opened.recording,
            )

            self.assertEqual(labels.shape, (2, 2))
            np.testing.assert_array_equal(np.unique(labels), np.array([0, 2]))


if __name__ == "__main__":
    unittest.main()
