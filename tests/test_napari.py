"""Core napari adapter tests.

Inputs: shared fake napari state and tiny converted recordings.
Outputs: assertions for one napari workflow area.
"""

from tests.napari_support import (
    APPLICATION_TITLE,
    TWOPY_SIDEBAR_MINIMUM_WIDTH,
    NapariAdapterTestCase,
    Path,
    QApplication,
    QColor,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    Qt,
    QTableWidget,
    QTabWidget,
    QWidget,
    __version__,
    _FakeViewer,
    _write_converted_recording,
    cast,
    create_viewer,
    load_roi_set,
    make_roi_set,
    np,
    open_recording_in_napari,
    patch,
    process_qt_events_until,
    roi_label_image_from_layer,
    save_napari_label_rois,
    save_roi_set,
    temporary_directory,
    unittest,
)
from twopy.analysis_cache import AnalysisSyncPlan
from twopy.custom import (
    CustomParameterSpec,
    CustomResult,
    CustomTable,
    CustomWorkflow,
    WorkflowDiscoveryResult,
)
from twopy.napari.controls import NapariControlState
from twopy.napari.custom_tab import CustomWorkflowPanel, _parameter_widget
from twopy.napari.empty_state import (
    EMPTY_VIEWER_MESSAGE,
    empty_viewer_message,
    hide_empty_viewer_message,
    refresh_empty_viewer_message,
)
from twopy.napari.plotting.docks.response_plot_widget import (
    _custom_workflow_display_result,
)
from twopy.napari.session import select_loaded_recording


class CoreNapariAdapterTest(NapariAdapterTestCase):
    """Core napari adapter tests."""

    def test_epoch_window_parameters_show_three_decimals(self) -> None:
        """Confirm role-backed window and threshold controls show three decimals."""
        _ = QApplication.instance() or QApplication([])
        epoch_window_widget = _parameter_widget(
            CustomParameterSpec(
                name="window_stop_seconds",
                label="Window end (s)",
                kind="float",
                default=1.23456,
                description="",
                role="epoch_window_stop",
            )
        )
        response_window_widget = _parameter_widget(
            CustomParameterSpec(
                name="response_window_stop_seconds",
                label="Response window end (s)",
                kind="float",
                default=1.23456,
                description="",
                role="response_window_stop",
            )
        )
        threshold_widget = _parameter_widget(
            CustomParameterSpec(
                name="highlight_threshold",
                label="Highlight threshold",
                kind="float",
                default=1.23456,
                description="",
                role="table_highlight_threshold",
            )
        )
        ordinary_widget = _parameter_widget(
            CustomParameterSpec(
                name="amplitude_scale",
                label="Amplitude scale",
                kind="float",
                default=1.23456,
                description="",
            )
        )

        self.assertIsInstance(epoch_window_widget, QDoubleSpinBox)
        self.assertIsInstance(response_window_widget, QDoubleSpinBox)
        self.assertIsInstance(threshold_widget, QDoubleSpinBox)
        self.assertIsInstance(ordinary_widget, QDoubleSpinBox)
        assert isinstance(epoch_window_widget, QDoubleSpinBox)
        assert isinstance(response_window_widget, QDoubleSpinBox)
        assert isinstance(threshold_widget, QDoubleSpinBox)
        assert isinstance(ordinary_widget, QDoubleSpinBox)
        self.assertEqual(epoch_window_widget.decimals(), 3)
        self.assertEqual(response_window_widget.decimals(), 3)
        self.assertEqual(threshold_widget.decimals(), 3)
        self.assertEqual(ordinary_widget.decimals(), 6)

    def test_text_parameters_use_shared_placeholder_style(self) -> None:
        """Confirm custom workflow text fields use napari placeholders.

        Inputs: string and path workflow parameter specs.
        Outputs: line edits with shared ellipsis placeholder text.
        """
        _ = QApplication.instance() or QApplication([])
        note_widget = _parameter_widget(
            CustomParameterSpec(
                name="note",
                label="Note",
                kind="str",
                default="",
                description="Free text saved with this run",
            )
        )
        path_widget = _parameter_widget(
            CustomParameterSpec(
                name="output_name",
                label="Output file",
                kind="path",
                default=Path("result.csv"),
                description="Relative output path",
                role="output_name",
            )
        )

        self.assertIsInstance(note_widget, QLineEdit)
        self.assertIsInstance(path_widget, QLineEdit)
        assert isinstance(note_widget, QLineEdit)
        assert isinstance(path_widget, QLineEdit)
        self.assertEqual(
            note_widget.placeholderText(),
            "Free text saved with this run...",
        )
        self.assertTrue(note_widget.font().italic())
        self.assertEqual(path_widget.placeholderText(), "Output path...")
        self.assertFalse(path_widget.font().italic())

    def test_custom_workflow_sync_result_shows_publish_table_path(self) -> None:
        """Confirm cached custom tables display the publish path."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            local_path = (
                root
                / "cache"
                / "custom_outputs"
                / "direction-selectivity"
                / "1.0"
                / "dsi.csv"
            )
            file_path = (
                root
                / "cache"
                / "custom_outputs"
                / "direction-selectivity"
                / "1.0"
                / "dsi_notes.txt"
            )
            sync_plan = AnalysisSyncPlan(
                local_root=root / "cache",
                publish_root=root / "publish",
                local_paths=(local_path, file_path),
            )
            result = CustomResult(
                message="ok",
                files=(file_path,),
                tables=(CustomTable("Direction selectivity", local_path),),
            )

            display_result = _custom_workflow_display_result(result, sync_plan)

            self.assertEqual(
                display_result.files,
                (
                    root
                    / "publish"
                    / "custom_outputs"
                    / "direction-selectivity"
                    / "1.0"
                    / "dsi_notes.txt",
                ),
            )
            self.assertEqual(display_result.tables[0].path, local_path)
            self.assertEqual(
                display_result.tables[0].display_path,
                root
                / "publish"
                / "custom_outputs"
                / "direction-selectivity"
                / "1.0"
                / "dsi.csv",
            )

    def test_custom_workflow_panel_previews_returned_tables(self) -> None:
        """Confirm custom workflow tables render in the Custom tab."""
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            table_path = root / "direction_selectivity.csv"
            table_path.write_text(
                "roi_label,direction_selectivity_index\n"
                "roi_0001,0.241\n"
                "roi_0002,-0.341\n",
                encoding="utf-8",
            )
            panel = CustomWorkflowPanel(workflow_paths=(), on_run=_unused_custom_run)

            panel._render_result(
                CustomResult(
                    message="ok",
                    tables=(
                        CustomTable(
                            "Direction selectivity",
                            table_path,
                            highlighted_rows=(1,),
                        ),
                    ),
                )
            )
            table = panel.findChild(QTableWidget)

            self.assertIsNotNone(table)
            assert table is not None
            header_roi = table.horizontalHeaderItem(0)
            header_dsi = table.horizontalHeaderItem(1)
            roi_item = table.item(0, 0)
            dsi_item = table.item(0, 1)
            highlighted_item = table.item(1, 1)
            self.assertIsNotNone(header_roi)
            self.assertIsNotNone(header_dsi)
            self.assertIsNotNone(roi_item)
            self.assertIsNotNone(dsi_item)
            self.assertIsNotNone(highlighted_item)
            assert header_roi is not None
            assert header_dsi is not None
            assert roi_item is not None
            assert dsi_item is not None
            assert highlighted_item is not None
            self.assertEqual(header_roi.text(), "roi_label")
            self.assertEqual(
                header_dsi.text(),
                "direction_selectivity_index",
            )
            self.assertEqual(roi_item.text(), "roi_0001")
            self.assertEqual(dsi_item.text(), "0.241")
            background = highlighted_item.background().color()
            foreground = highlighted_item.foreground().color()
            self.assertNotEqual(background.getRgb(), QColor("#fff2a8").getRgb())
            self.assertNotEqual(background.getRgb(), foreground.getRgb())

    def test_custom_workflow_panel_shows_one_selectable_output_path(self) -> None:
        """Confirm custom outputs show one compact selectable folder path."""
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            local_output_root = root / "custom_outputs" / "response_kernels"
            local_version_root = local_output_root / "1.0"
            publish_output_root = (
                root / "published" / "custom_outputs" / "response_kernels"
            )
            publish_version_root = publish_output_root / "1.0"
            ipsi_path = (
                local_version_root / "response_kernels_group_01_epochs_2_Both_ipsi.csv"
            )
            contra_path = (
                local_version_root
                / "response_kernels_group_01_epochs_2_Both_contra.csv"
            )
            summary_path = local_version_root / "response_kernels_summary.csv"
            summary_display_path = publish_version_root / "response_kernels_summary.csv"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text("name,count\nBoth,2\n", encoding="utf-8")
            panel = CustomWorkflowPanel(
                workflow_paths=(),
                on_run=_unused_custom_run,
            )

            panel._render_result(
                CustomResult(
                    message="ok",
                    files=(ipsi_path, contra_path),
                    tables=(
                        CustomTable(
                            "Kernel summary",
                            summary_path,
                            display_path=summary_display_path,
                        ),
                    ),
                )
            )

            labels = [label.text() for label in panel.findChildren(QLabel)]
            output_labels = [
                label
                for label in panel.findChildren(QLabel)
                if label.text().startswith("Output path:")
            ]
            self.assertEqual(
                [label.text() for label in output_labels],
                [f"Output path: {publish_output_root}/"],
            )
            self.assertFalse(any(text.startswith("File:") for text in labels))
            self.assertNotIn(str(ipsi_path), labels)
            self.assertNotIn(str(contra_path), labels)
            self.assertEqual(
                panel.horizontalScrollBarPolicy(),
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
            )
            self.assertTrue(
                output_labels[0].textInteractionFlags()
                & Qt.TextInteractionFlag.TextSelectableByMouse
            )

    def test_custom_workflow_panel_clears_result_after_failure(self) -> None:
        """Confirm failed reruns remove stale custom workflow outputs."""
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            table_path = root / "direction_selectivity.csv"
            table_path.write_text("roi_label,dsi\nroi_0001,0.5\n", encoding="utf-8")
            workflow = CustomWorkflow(
                id="direction-selectivity",
                name="Direction selectivity",
                version="1.0",
                description="Computes a direction-selectivity index.",
                params_type=None,
                author="twopy",
                requires_rois=True,
                output_prefix="direction-selectivity",
                function=_unused_custom_run,
                source_path=root / "workflow.py",
                source_hash="abc123",
            )
            outcomes: list[CustomResult | Exception] = [
                CustomResult(
                    message="ok",
                    tables=(CustomTable("Direction selectivity", table_path),),
                ),
                ValueError("missing stimulus output"),
            ]

            def run_once(
                workflow: CustomWorkflow,
                params: object | None,
            ) -> CustomResult:
                """Return the next result or failure for the panel callback."""
                del workflow, params
                outcome = outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

            with patch(
                "twopy.napari.custom_tab.discover_custom_workflows",
                return_value=WorkflowDiscoveryResult((workflow,), ()),
            ):
                panel = CustomWorkflowPanel(
                    workflow_paths=(root / "workflow.py",),
                    on_run=run_once,
                )

            panel._run_selected_workflow()
            self.assertIsNotNone(panel.findChild(QTableWidget))

            panel._run_selected_workflow()

            self.assertIsNone(panel.findChild(QTableWidget))
            labels = {label.text() for label in panel.findChildren(QLabel)}
            self.assertIn(
                "Custom workflow failed: missing stimulus output",
                labels,
            )
            self.assertIn("No custom workflow outputs.", labels)

    def test_custom_workflow_panel_empty_result_replaces_stale_table(self) -> None:
        """Confirm empty successful results remove stale table previews."""
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            table_path = root / "direction_selectivity.csv"
            table_path.write_text("roi_label,dsi\nroi_0001,0.5\n", encoding="utf-8")
            panel = CustomWorkflowPanel(
                workflow_paths=(),
                on_run=_unused_custom_run,
            )

            panel._render_result(
                CustomResult(
                    message="ok",
                    tables=(CustomTable("Direction selectivity", table_path),),
                )
            )
            self.assertIsNotNone(panel.findChild(QTableWidget))

            panel._render_result(CustomResult(message="no display outputs"))

            self.assertIsNone(panel.findChild(QTableWidget))
            labels = {label.text() for label in panel.findChildren(QLabel)}
            self.assertIn("No custom workflow outputs.", labels)

    def test_create_viewer_uses_twopy_version_window_title(self) -> None:
        """Confirm the launcher brands the top-level napari window.

        Inputs: patched napari Viewer constructor.
        Outputs: viewer construction receives the twopy name, version, and icon.
        """
        _ = QApplication.instance() or QApplication([])
        viewer = _FakeViewer()

        with patch("napari.Viewer", return_value=viewer) as viewer_constructor:
            created = create_viewer()

        self.assertIs(created, viewer)
        self.assertEqual(APPLICATION_TITLE, f"twopy {__version__}")
        viewer_constructor.assert_called_once_with(
            title=APPLICATION_TITLE,
            show_welcome_screen=False,
        )
        self.assertFalse(viewer.welcome_screen.visible)
        self.assertEqual(viewer.text_overlay.text, EMPTY_VIEWER_MESSAGE)
        self.assertTrue(viewer.text_overlay.text.startswith(f"{APPLICATION_TITLE}\n"))
        self.assertTrue(viewer.text_overlay.visible)
        self.assertEqual(viewer.text_overlay.position, "top_center")
        self.assertEqual(viewer.text_overlay.font_size, 18)
        self.assertEqual(viewer.text_overlay.color, "#c8cdd3")
        self.assertFalse(viewer.text_overlay.box)
        icon = viewer.window._qt_window.window_icon
        self.assertIsNotNone(icon)
        assert icon is not None
        self.assertFalse(icon.isNull())

    def test_empty_viewer_message_only_clears_twopy_text(self) -> None:
        """Confirm hiding the empty message leaves other overlay owners alone.

        Inputs: fake viewer whose text overlay belongs to another feature.
        Outputs: unchanged text and visibility after hiding twopy's message.
        """
        viewer = _FakeViewer()
        viewer.text_overlay.text = "existing HUD"
        viewer.text_overlay.visible = True

        hide_empty_viewer_message(viewer)

        self.assertEqual(viewer.text_overlay.text, "existing HUD")
        self.assertTrue(viewer.text_overlay.visible)

    def test_empty_viewer_message_can_show_update_command(self) -> None:
        """Confirm update notices extend only twopy's empty canvas text."""
        viewer = _FakeViewer()
        viewer.text_overlay.text = EMPTY_VIEWER_MESSAGE
        viewer.text_overlay.visible = True

        refresh_empty_viewer_message(
            viewer,
            update_command="python -m pip install -U twopy",
        )

        self.assertEqual(
            viewer.text_overlay.text,
            f"{EMPTY_VIEWER_MESSAGE}\n\npython -m pip install -U twopy",
        )
        self.assertEqual(
            viewer.text_overlay.text,
            empty_viewer_message(update_command="python -m pip install -U twopy"),
        )
        hide_empty_viewer_message(viewer)
        self.assertEqual(viewer.text_overlay.text, "")
        self.assertFalse(viewer.text_overlay.visible)

    def test_empty_viewer_update_does_not_replace_other_overlay_text(self) -> None:
        """Confirm update notices do not replace another overlay owner."""
        viewer = _FakeViewer()
        viewer.text_overlay.text = "existing HUD"
        viewer.text_overlay.visible = True

        refresh_empty_viewer_message(
            viewer,
            update_command="python -m pip install -U twopy",
        )

        self.assertEqual(viewer.text_overlay.text, "existing HUD")
        self.assertTrue(viewer.text_overlay.visible)

    def test_cleared_recording_selection_restores_empty_viewer_message(self) -> None:
        """Confirm no-recording selection brings back the twopy empty canvas.

        Inputs: empty loaded-recording state after the Load tab clears a view.
        Outputs: napari welcome hidden and twopy empty text shown.
        """
        viewer = _FakeViewer()
        state = NapariControlState(
            viewer=viewer,
            roi_labels_layer=None,
            roi_save_file=Path("rois.h5"),
            recording=None,
            response_plot_widget=None,
        )

        select_loaded_recording(state, None)

        self.assertFalse(viewer.welcome_screen.visible)
        self.assertEqual(viewer.text_overlay.text, EMPTY_VIEWER_MESSAGE)
        self.assertTrue(viewer.text_overlay.visible)

    def test_opens_empty_roi_labels_layer_by_default(self) -> None:
        """Confirm new recordings open with an editable empty ROI layer.

        Inputs: tiny converted recording and fake viewer.
        Outputs: one zero-valued labels layer matching the movie frame shape.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            viewer = _FakeViewer()
            viewer.text_overlay.text = EMPTY_VIEWER_MESSAGE
            viewer.text_overlay.visible = True

            opened = open_recording_in_napari(recording_path, viewer=viewer)

            self.assertIs(opened.viewer, viewer)
            self.assertEqual(len(viewer.labels), 1)
            self.assertIs(opened.roi_labels_layer, viewer.labels[0])
            self.assertEqual(len(viewer.images), 1)
            self.assertIs(opened.mean_image_layer, viewer.images[0])
            self.assertIsNone(opened.movie_layer)
            self.assertEqual(len(viewer.window.dock_widgets), 2)
            self.assertIs(
                opened.response_plot_dock_widget,
                viewer.window.dock_widgets[0],
            )
            self.assertIs(
                opened.twopy_sidebar_dock_widget,
                viewer.window.dock_widgets[1],
            )
            self.assertIs(
                opened.response_plot_widget, viewer.window.dock_widgets[0].widget
            )
            self.assertIs(
                opened.twopy_sidebar_widget, viewer.window.dock_widgets[1].widget
            )
            self.assertEqual(viewer.labels[0].options["opacity"], 0.5)
            self.assertEqual(viewer.labels[0].options["blending"], "additive")
            self.assertEqual(viewer.labels[0].brush_size, 6)
            self.assertEqual(viewer.images[0].options["opacity"], 0.5)
            self.assertEqual(viewer.images[0].options["gamma"], 1.3)
            self.assertEqual(viewer.images[0].options["contrast_limits"], (4.3, 7.0))
            self.assertEqual(viewer.images[0].contrast_limits_range, (4.0, 7.0))
            self.assertEqual(viewer.text_overlay.text, "")
            self.assertFalse(viewer.text_overlay.visible)
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
                    "Custom",
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
            self.assertIn("Reconvert selected", options_buttons)
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
            self.assertTrue(action_buttons["Reconvert selected"].isEnabled())
            self.assertTrue(action_buttons["Save loaded list"].isEnabled())
            group_titles = {
                group.title() for group in options_widget.findChildren(QGroupBox)
            }
            self.assertIn("Recording", group_titles)
            self.assertIn("Microscope", group_titles)
            self.assertIn("Outputs", group_titles)
            self.assertIn("Status", group_titles)
            self.assertIn("Plot", group_titles)
            self.assertIn("Smoothing", group_titles)
            np.testing.assert_array_equal(
                roi_label_image_from_layer(viewer.labels[0]),
                np.zeros((2, 2), dtype=np.int64),
            )

    def test_can_open_recording_without_controls(self) -> None:
        """Confirm scripts can load viewer layers without twopy docks.

        Inputs: tiny converted recording and fake viewer.
        Outputs: mean image and editable ROI layer without sidebar widgets.
        """
        with temporary_directory() as temp_dir:
            recording_path = _write_converted_recording(Path(temp_dir))
            viewer = _FakeViewer()

            opened = open_recording_in_napari(
                recording_path,
                viewer=viewer,
                add_controls=False,
            )

            self.assertIs(opened.viewer, viewer)
            self.assertEqual(len(viewer.images), 1)
            self.assertEqual(len(viewer.labels), 1)
            self.assertEqual(viewer.window.dock_widgets, [])
            self.assertIsNone(opened.load_widget)
            self.assertIsNone(opened.loaded_recordings_widget)
            self.assertIsNone(opened.twopy_sidebar_widget)
            self.assertIsNone(opened.twopy_sidebar_dock_widget)
            self.assertIsNone(opened.response_plot_widget)
            self.assertIsNone(opened.response_plot_dock_widget)
            self.assertIsNone(opened.response_options_widget)
            self.assertIsNone(opened.trial_timeline_controller)

    def test_can_open_recording_without_roi_labels_layer(self) -> None:
        """Confirm read-only display can skip the editable ROI layer.

        Inputs: tiny converted recording and fake viewer.
        Outputs: mean image only and a coherent returned recording view.
        """
        with temporary_directory() as temp_dir:
            recording_path = _write_converted_recording(Path(temp_dir))
            viewer = _FakeViewer()

            opened = open_recording_in_napari(
                recording_path,
                viewer=viewer,
                add_roi_labels_layer=False,
                add_controls=False,
            )

            self.assertIs(opened.viewer, viewer)
            self.assertEqual(len(viewer.images), 1)
            self.assertEqual(len(viewer.labels), 0)
            self.assertIs(opened.mean_image_layer, viewer.images[0])
            self.assertIsNone(opened.movie_layer)
            self.assertIsNone(opened.roi_labels_layer)
            self.assertEqual(viewer.window.dock_widgets, [])

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
            process_qt_events_until(lambda: len(viewer.reset_view_calls) == 1)

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
            self.assertEqual(
                viewer.reset_view_calls,
                [{"margin": 0.05, "reset_camera_angle": False}],
            )
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


def _unused_custom_run(*args: object) -> CustomResult:
    """Raise if a custom workflow panel test unexpectedly runs a workflow."""
    raise AssertionError(args)


if __name__ == "__main__":
    unittest.main()
