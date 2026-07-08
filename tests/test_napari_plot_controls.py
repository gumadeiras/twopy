"""Napari plot layout and display-control tests.

Inputs: shared fake napari state and tiny response data.
Outputs: assertions for plot-control behavior.
"""

from dataclasses import replace

from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QSizePolicy, QStyleOptionViewItem

from tests.napari_support import (
    PLOT_CONTROL_WIDTH,
    PLOT_DROPDOWN_WIDTH,
    Any,
    DeltaFOverFOptions,
    DeltaFOverFOptionsWidget,
    NapariAdapterTestCase,
    NormalizationOptions,
    NormalizationOptionsWidget,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QScrollArea,
    QSpinBox,
    Qt,
    QWidget,
    ResponseMapOptions,
    ResponseMapOptionsWidget,
    ResponseProcessingOptions,
    ResponseProcessingOptionsWidget,
    ResponseWindowOptions,
    ResponseWindowOptionsWidget,
    SpatialCrop,
    cast,
    create_response_plot_widget,
    np,
    plot_display_options_group,
    unittest,
)
from tests.recording_data import minimal_recording_data
from twopy.napari.plotting.motion_summary import MotionSummaryWidget
from twopy.napari.plotting.panels import (
    SidebarTextLabel,
    response_metadata_tab,
)


class NapariPlotControlsTest(NapariAdapterTestCase):
    """Napari plot layout and display-control tests."""

    def test_metadata_tab_uses_sidebar_width_and_selectable_text(self) -> None:
        """Confirm Metadata text wraps in the sidebar and can be copied.

        Inputs: Metadata-tab labels with long path-like text.
        Outputs: the scroll area is vertical-only and labels are selectable,
        wrapped, and shrinkable to the sidebar width.
        """
        _ = QApplication.instance() or QApplication([])
        analysis_text = "Analysis output: /very/long/path/to/analysis_outputs.h5"
        labels = (
            SidebarTextLabel("Recording: /very/long/path/to/a/recording"),
            SidebarTextLabel("Microscope: scan settings"),
            SidebarTextLabel(analysis_text),
            SidebarTextLabel("ROI output: /very/long/path/to/rois.h5"),
            SidebarTextLabel("Saved analysis."),
            SidebarTextLabel(
                "Update available!\n"
                "Latest version is 0.3.6.\n"
                "To update, run:\n"
                "python -m pip install -U twopy",
            ),
        )

        tab = response_metadata_tab(
            recording_summary_label=labels[0],
            microscope_summary_label=labels[1],
            analysis_output_label=labels[2],
            roi_output_label=labels[3],
            status_label=labels[4],
            update_notice_label=labels[5],
        )

        self.assertIsInstance(tab, QScrollArea)
        scroll_tab = cast(QScrollArea, tab)
        self.assertEqual(
            scroll_tab.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        content = scroll_tab.widget()
        if content is None:
            self.fail("Metadata tab is missing scroll content")
        self.assertEqual(
            content.sizePolicy().horizontalPolicy(),
            QSizePolicy.Policy.Ignored,
        )
        for label in labels:
            self.assertTrue(
                label.textInteractionFlags()
                & Qt.TextInteractionFlag.TextSelectableByMouse,
            )
            self.assertTrue(label.wordWrap())
            self.assertTrue(label.hasHeightForWidth())
            self.assertEqual(
                label.sizePolicy().horizontalPolicy(),
                QSizePolicy.Policy.Ignored,
            )
            self.assertEqual(label.minimumWidth(), 1)
        self.assertEqual(labels[2].text(), analysis_text)
        self.assertGreater(
            labels[2].heightForWidth(90),
            labels[2].heightForWidth(400),
        )
        self.assertFalse(hasattr(labels[2], "verticalScrollBar"))

    def test_motion_summary_uses_metadata_tab_width(self) -> None:
        """Confirm Motion metadata stays inside the fixed sidebar width.

        Inputs: a recording with a crop, signed x/y movement, and one bad frame.
        Outputs: summary text reports crop/motion metrics, and the plot can shrink
        with the Metadata tab instead of widening the sidebar.
        """
        _ = QApplication.instance() or QApplication([])
        widget = MotionSummaryWidget()
        recording = minimal_recording_data(
            movie_shape=(4, 5, 6),
            alignment_valid_crop=SpatialCrop(
                axis0_start=1,
                axis0_stop=4,
                axis1_start=2,
                axis1_stop=5,
                original_shape=(5, 6),
                source="alignment_valid_crop",
            ),
            alignment_offset_pixels=np.array(
                [[0.0, 0.0], [1.0, -1.0], [3.0, 0.0], [2.0, 1.0]],
                dtype=np.float64,
            ),
            alignment_shift_pixels=np.array([0.0, 1.0, 3.0, 2.0]),
            motion_artifact_mask=np.array([False, False, True, False]),
        )

        widget.set_recording(recording)

        self.assertEqual(widget.title(), "Motion")
        labels = widget.findChildren(SidebarTextLabel)
        self.assertEqual(len(labels), 1)
        self.assertIn("Visible crop: 3 x 3 px from a 6 x 5 px frame", labels[0].text())
        self.assertIn(
            "Removed border: left 2 px, right 1 px, top 1 px, bottom 1 px",
            labels[0].text(),
        )
        self.assertIn("high-motion 1/4 frames", labels[0].text())
        self.assertIn(
            "Plot: left-right movement in blue, up-down movement in yellow",
            labels[0].text(),
        )
        layout = widget.layout()
        if layout is None:
            self.fail("Motion summary is missing a layout")
        plot_item = layout.itemAt(1)
        if plot_item is None:
            self.fail("Motion summary is missing a plot item")
        plot = plot_item.widget()
        if plot is None:
            self.fail("Motion summary is missing a plot")
        self.assertEqual(plot.minimumWidth(), 1)
        self.assertEqual(
            plot.sizePolicy().horizontalPolicy(),
            QSizePolicy.Policy.Ignored,
        )
        offset_pixels = cast(Any, plot)._values
        np.testing.assert_array_equal(
            offset_pixels,
            np.array([[0.0, 0.0], [1.0, -1.0], [3.0, 0.0], [2.0, 1.0]]),
        )

    def test_motion_summary_falls_back_to_total_movement(self) -> None:
        """Confirm old recordings show magnitude-only movement information.

        Inputs: a recording without signed x/y alignment offsets.
        Outputs: summary text asks for reconversion and plot uses total movement.
        """
        _ = QApplication.instance() or QApplication([])
        widget = MotionSummaryWidget()
        recording = replace(
            minimal_recording_data(
                movie_shape=(3, 5, 6),
                alignment_shift_pixels=np.array([0.0, 2.0, 1.0], dtype=np.float64),
                motion_artifact_mask=np.array([False, True, False]),
            ),
            alignment_offset_pixels=None,
        )

        widget.set_recording(recording)

        labels = widget.findChildren(SidebarTextLabel)
        self.assertEqual(len(labels), 1)
        self.assertIn(
            "Reconvert this recording to see independent x/y pixel movement",
            labels[0].text(),
        )
        layout = widget.layout()
        if layout is None:
            self.fail("Motion summary is missing a layout")
        plot_item = layout.itemAt(1)
        if plot_item is None:
            self.fail("Motion summary is missing a plot item")
        plot = plot_item.widget()
        if plot is None:
            self.fail("Motion summary is missing a plot")
        np.testing.assert_array_equal(
            cast(Any, plot)._values,
            np.array([[0.0], [2.0], [1.0]], dtype=np.float64),
        )

    def test_motion_summary_plot_renders_motion_data(self) -> None:
        """Confirm the inline Motion plot paints real movement data.

        Inputs: a Motion widget with finite and non-finite x/y movement values.
        Outputs: the plot renders without crashing after values are sanitized.
        """
        _ = QApplication.instance() or QApplication([])
        widget = MotionSummaryWidget()
        recording = minimal_recording_data(
            movie_shape=(4, 5, 6),
            alignment_offset_pixels=np.array(
                [[0.0, 0.0], [np.nan, 1.0], [np.inf, -1.0], [2.0, 0.0]],
                dtype=np.float64,
            ),
            alignment_shift_pixels=np.array([0.0, np.nan, np.inf, 2.0]),
            motion_artifact_mask=np.array([False, True, False, False]),
        )
        widget.set_recording(recording)
        labels = widget.findChildren(SidebarTextLabel)
        self.assertEqual(len(labels), 1)
        self.assertIn("95% of frames moved <= 1.90 px", labels[0].text())
        self.assertIn("max 2.00 px", labels[0].text())
        layout = widget.layout()
        if layout is None:
            self.fail("Motion summary is missing a layout")
        plot_item = layout.itemAt(1)
        if plot_item is None:
            self.fail("Motion summary is missing a plot item")
        plot = plot_item.widget()
        if plot is None:
            self.fail("Motion summary is missing a plot")

        plot.resize(300, 120)
        pixmap = QPixmap(plot.size())
        pixmap.fill()
        plot.render(pixmap)

        self.assertFalse(pixmap.isNull())

    def test_loaded_recording_updates_metadata_motion_summary(self) -> None:
        """Confirm response-widget loading routes motion metadata to the tab.

        Inputs: a response widget and one recording with nonzero x/y movement.
        Outputs: the Metadata-tab Motion section reflects the loaded recording.
        """
        _ = QApplication.instance() or QApplication([])
        response_widget = cast(Any, create_response_plot_widget(None))
        recording = minimal_recording_data(
            movie_shape=(3, 4, 4),
            alignment_offset_pixels=np.array(
                [[0.0, 0.0], [2.0, -1.0], [1.0, 1.0]],
                dtype=np.float64,
            ),
            alignment_shift_pixels=np.array([0.0, 2.24, 1.41], dtype=np.float64),
            motion_artifact_mask=np.array([False, True, False]),
        )

        response_widget.load_recording(recording)

        motion_widget = response_widget._motion_summary_widget
        labels = motion_widget.findChildren(SidebarTextLabel)
        self.assertEqual(len(labels), 1)
        self.assertIn("high-motion 1/3 frames", labels[0].text())

    def test_plot_tab_option_sections_have_expected_order(self) -> None:
        """Confirm the Plot tab presents response options in workflow order.

        Inputs: a newly created response widget.
        Outputs: the Plot tab orders Plot, Response window, Heatmap, dF/F,
        Normalization, then processing controls.
        """
        _ = QApplication.instance() or QApplication([])
        response_widget = cast(Any, create_response_plot_widget(None))

        self.assertIsInstance(
            response_widget._response_window_options_widget,
            ResponseWindowOptionsWidget,
        )
        self.assertIsInstance(
            response_widget._response_map_options_widget,
            ResponseMapOptionsWidget,
        )
        self.assertIsInstance(
            response_widget._delta_f_over_f_options_widget,
            DeltaFOverFOptionsWidget,
        )
        self.assertEqual(response_widget._plot_tabs.tabText(0), "Responses")
        self.assertEqual(response_widget._plot_tabs.tabText(1), "Heatmaps")
        self.assertIsInstance(
            response_widget._normalization_options_widget,
            NormalizationOptionsWidget,
        )
        self.assertLess(
            response_widget._plot_options_layout.indexOf(
                response_widget._response_window_options_widget,
            ),
            response_widget._plot_options_layout.indexOf(
                response_widget._response_map_options_widget,
            ),
        )
        self.assertLess(
            response_widget._plot_options_layout.indexOf(
                response_widget._response_map_options_widget,
            ),
            response_widget._plot_options_layout.indexOf(
                response_widget._delta_f_over_f_options_widget,
            ),
        )
        self.assertLess(
            response_widget._plot_options_layout.indexOf(
                response_widget._delta_f_over_f_options_widget,
            ),
            response_widget._plot_options_layout.indexOf(
                response_widget._normalization_options_widget,
            ),
        )
        self.assertLess(
            response_widget._plot_options_layout.indexOf(
                response_widget._normalization_options_widget,
            ),
            response_widget._plot_options_layout.indexOf(
                response_widget._processing_options_widget,
            ),
        )

    def test_plot_tab_show_sem_starts_unchecked(self) -> None:
        """Confirm the default Plot tab hides SEM bands.

        Inputs: a newly created response widget.
        Outputs: the Show SEM checkbox starts unchecked.
        """
        _ = QApplication.instance() or QApplication([])
        response_widget = cast(Any, create_response_plot_widget(None))
        display_item = response_widget._plot_display_options_layout.itemAt(0)
        if display_item is None:
            self.fail("Plot display options group is missing")
        display_group = display_item.widget()
        if display_group is None:
            self.fail("Plot display options group is not a widget")
        layout = cast(QFormLayout, display_group.layout())
        sem_item = layout.itemAt(0, QFormLayout.ItemRole.FieldRole)
        if sem_item is None:
            self.fail("Plot display show SEM checkbox is missing")
        sem_widget = sem_item.widget()
        if not isinstance(sem_widget, QCheckBox):
            self.fail("Plot display show SEM row is not a QCheckBox")

        self.assertFalse(response_widget._show_sem)
        self.assertFalse(sem_widget.isChecked())

    def test_response_map_options_show_only_current_mode_rows(self) -> None:
        """Confirm Heatmap controls hide parameters for the inactive mode.

        Inputs: heatmap options widget switched from pixel mode to window mode.
        Outputs: only sigma is shown for pixel mode, and only preset, size, and
        stride are shown for window mode.
        """
        _ = QApplication.instance() or QApplication([])
        widget = cast(Any, ResponseMapOptionsWidget(ResponseMapOptions(mode="pixel")))
        layout = cast(QFormLayout, widget._form_layout)

        def row_label(field: QWidget) -> QLabel:
            label = layout.labelForField(field)
            if not isinstance(label, QLabel):
                self.fail("Heatmap option row is missing a label")
            return label

        def row_label_at(row: int) -> QLabel:
            item = layout.itemAt(row, QFormLayout.ItemRole.LabelRole)
            widget_at_row = item.widget() if item is not None else None
            if not isinstance(widget_at_row, QLabel):
                self.fail("Heatmap option row is missing its ordered label")
            return widget_at_row

        def row_is_hidden(field: QWidget) -> bool:
            label = row_label(field)
            return label.isHidden() and field.isHidden()

        self.assertTrue(widget._shared_limits.isChecked())
        self.assertEqual(row_label_at(0).text(), "Shared limits")
        self.assertEqual(row_label_at(1).text(), "Mode")
        self.assertEqual(row_label_at(2).text(), "Sigma")
        self.assertEqual(row_label(widget._shared_limits).text(), "Shared limits")
        self.assertEqual(row_label(widget._pixel_smoothing_sigma).text(), "Sigma")
        self.assertEqual(row_label(widget._window_preset).text(), "Preset")
        self.assertEqual(row_label(widget._window_size).text(), "Size")
        self.assertFalse(row_is_hidden(widget._shared_limits))
        self.assertFalse(row_is_hidden(widget._pixel_smoothing_sigma))
        self.assertTrue(row_is_hidden(widget._window_preset))
        self.assertTrue(row_is_hidden(widget._window_size))
        self.assertTrue(row_is_hidden(widget._window_stride))

        mode_combo = cast(QComboBox, widget._mode)
        mode_combo.setCurrentIndex(mode_combo.findData("window"))

        self.assertTrue(row_is_hidden(widget._pixel_smoothing_sigma))
        self.assertFalse(row_is_hidden(widget._window_preset))
        self.assertFalse(row_is_hidden(widget._window_size))
        self.assertFalse(row_is_hidden(widget._window_stride))

        self.assertFalse(row_is_hidden(widget._shared_limits))

    def test_response_window_options_auto_default_and_manual_bounds(self) -> None:
        """Confirm response-window controls default to auto and cap manual values.

        Inputs: response-window widget with a two-second interleave limit.
        Outputs: auto starts checked, manual fields are disabled, and manual
        spin boxes cap at the supplied limit when auto is unchecked.
        """
        _ = QApplication.instance() or QApplication([])
        widget = cast(
            Any,
            ResponseWindowOptionsWidget(
                ResponseWindowOptions(),
                max_window_seconds=2.0,
            ),
        )

        self.assertTrue(widget._auto.isChecked())
        self.assertFalse(widget._pre_seconds.isEnabled())
        self.assertFalse(widget._post_seconds.isEnabled())
        self.assertEqual(widget._pre_seconds.maximum(), 2.0)
        self.assertEqual(widget._post_seconds.maximum(), 2.0)

        widget._auto.setChecked(False)

        self.assertTrue(widget._pre_seconds.isEnabled())
        self.assertTrue(widget._post_seconds.isEnabled())

    def test_delta_f_over_f_controls_use_readable_menu_labels(self) -> None:
        """Confirm Plot-tab menus show readable labels, not stored values.

        Inputs: dF/F and processing option widgets with default settings.
        Outputs: dropdowns hide internal option names.
        """
        _ = QApplication.instance() or QApplication([])
        dff_widget = cast(Any, DeltaFOverFOptionsWidget(DeltaFOverFOptions()))
        processing_widget = cast(
            Any,
            ResponseProcessingOptionsWidget(ResponseProcessingOptions()),
        )

        background_labels = tuple(
            dff_widget._background_method.itemText(index)
            for index in range(dff_widget._background_method.count())
        )
        baseline_mode_labels = tuple(
            dff_widget._baseline_mode.itemText(index)
            for index in range(dff_widget._baseline_mode.count())
        )
        fit_labels = tuple(
            dff_widget._fit_mode.itemText(index)
            for index in range(dff_widget._fit_mode.count())
        )
        smoothing_labels = tuple(
            processing_widget._smoothing_method.itemText(index)
            for index in range(processing_widget._smoothing_method.count())
        )
        correlation_labels = tuple(
            processing_widget._correlation_reference.itemText(index)
            for index in range(processing_widget._correlation_reference.count())
        )

        self.assertIn("global percentile", background_labels)
        self.assertIn("shared y-stripe P%", background_labels)
        self.assertIn("ROI y-stripe P%", background_labels)
        self.assertNotIn("roi_y_stripe_percentile", background_labels)
        self.assertEqual(
            baseline_mode_labels,
            ("baseline epoch", "no baseline epoch"),
        )
        self.assertNotIn("no_true_interleave", baseline_mode_labels)
        rich_label_role = int(Qt.ItemDataRole.UserRole) + 1
        self.assertEqual(
            dff_widget._background_method.itemData(2, rich_label_role),
            "shared y-stripe P<sub>%</sub>",
        )
        self.assertEqual(
            dff_widget._background_method.itemData(3, rich_label_role),
            "ROI y-stripe P<sub>%</sub>",
        )
        self.assertIn("bounded tau", fit_labels)
        self.assertIn("bounded tau/amplitude", fit_labels)
        self.assertIn("log-linear", fit_labels)
        self.assertNotIn("direct_bounded_tau", fit_labels)
        self.assertIn("moving average", smoothing_labels)
        self.assertNotIn("moving_average", smoothing_labels)
        self.assertIn("epoch mean", correlation_labels)
        self.assertNotIn("epoch_mean", correlation_labels)

    def test_delta_f_over_f_rich_menu_label_fits_combo_height(self) -> None:
        """Confirm y-stripe background labels have enough vertical room.

        Inputs: dF/F controls with rich ``P%`` background menu labels.
        Outputs: the rich-text delegate fits inside the closed combo box.
        """
        _ = QApplication.instance() or QApplication([])
        widget = cast(Any, DeltaFOverFOptionsWidget(DeltaFOverFOptions()))
        combo = widget._background_method
        option = QStyleOptionViewItem()
        option.font = combo.font()
        option.widget = combo

        item_size = combo.itemDelegate().sizeHint(
            option,
            combo.model().index(2, 0),
        )

        self.assertLessEqual(item_size.height(), combo.sizeHint().height())

    def test_plot_display_size_label_preserves_ui_text(self) -> None:
        """Confirm the Plot subsection puts show SEM before size.

        Inputs: plot display options group with default numeric bounds.
        Outputs: first field row is the SEM toggle, then the configured size label.
        """
        _ = QApplication.instance() or QApplication([])
        group = plot_display_options_group(
            show_sem_checkbox=QCheckBox("show SEM"),
            plot_size_spin=QSpinBox(),
            x_min=0.0,
            x_max=1.0,
            y_min=-1.0,
            y_max=1.0,
            on_change=lambda **_kwargs: None,
        )
        layout = cast(QFormLayout, group.layout())
        sem_item = layout.itemAt(0, QFormLayout.ItemRole.FieldRole)
        if sem_item is None:
            self.fail("Plot display show SEM checkbox is missing")
        sem_widget = sem_item.widget()
        if not isinstance(sem_widget, QCheckBox):
            self.fail("Plot display show SEM row is not a QCheckBox")
        self.assertEqual(sem_widget.text(), "show SEM")
        self.assertEqual(sem_widget.minimumWidth(), PLOT_DROPDOWN_WIDTH)
        self.assertEqual(sem_widget.maximumWidth(), PLOT_DROPDOWN_WIDTH)

        label_item = layout.itemAt(1, QFormLayout.ItemRole.LabelRole)
        if label_item is None:
            self.fail("Plot display size label is missing")
        label_widget = label_item.widget()
        if not isinstance(label_widget, QLabel):
            self.fail("Plot display size label is not a QLabel")

        self.assertEqual(label_widget.text(), "Size")
        self.assertTrue(layout.formAlignment() & Qt.AlignmentFlag.AlignHCenter)

    def test_plot_display_numeric_controls_share_width_and_rounding(self) -> None:
        """Confirm Plot numeric controls use one width and two-decimal bounds.

        Inputs: plot display options group with fractional axis bounds.
        Outputs: size and axis limit controls share width, and limits round to
        two decimal places.
        """
        _ = QApplication.instance() or QApplication([])
        group = plot_display_options_group(
            show_sem_checkbox=QCheckBox("show SEM"),
            plot_size_spin=QSpinBox(),
            x_min=1.123,
            x_max=1.456,
            y_min=-1.123,
            y_max=-1.456,
            on_change=lambda **_kwargs: None,
        )
        layout = cast(QFormLayout, group.layout())
        size_item = layout.itemAt(1, QFormLayout.ItemRole.FieldRole)
        if size_item is None:
            self.fail("Plot display size control is missing")
        size_spin = size_item.widget()
        if not isinstance(size_spin, QSpinBox):
            self.fail("Plot display size control is not a QSpinBox")

        axis_spins: list[QDoubleSpinBox] = []
        for row in range(2, 6):
            axis_item = layout.itemAt(row, QFormLayout.ItemRole.FieldRole)
            if axis_item is None:
                self.fail("Plot display axis control is missing")
            axis_spin = axis_item.widget()
            if not isinstance(axis_spin, QDoubleSpinBox):
                self.fail("Plot display axis control is not a QDoubleSpinBox")
            axis_spins.append(axis_spin)

        widths = {
            size_spin.minimumWidth(),
            *(spin.minimumWidth() for spin in axis_spins),
        }
        max_widths = {
            size_spin.maximumWidth(),
            *(spin.maximumWidth() for spin in axis_spins),
        }
        self.assertEqual(len(widths), 1)
        self.assertEqual(len(max_widths), 1)
        self.assertEqual([spin.decimals() for spin in axis_spins], [2, 2, 2, 2])
        self.assertEqual(
            [spin.value() for spin in axis_spins],
            [1.12, 1.46, -1.12, -1.46],
        )

    def test_plot_tab_option_controls_use_shared_widths(self) -> None:
        """Confirm Plot-tab fields use shared widths by control type.

        Inputs: response-window, dF/F, normalization, and processing widgets.
        Outputs: dropdowns and checkboxes use the wider width, while spin boxes
        use the compact control width.
        """
        _ = QApplication.instance() or QApplication([])
        response_window_widget = cast(
            Any,
            ResponseWindowOptionsWidget(ResponseWindowOptions()),
        )
        dff_widget = cast(Any, DeltaFOverFOptionsWidget(DeltaFOverFOptions()))
        normalization_widget = cast(
            Any,
            NormalizationOptionsWidget(
                NormalizationOptions(method="epoch_abs_peak", epoch_number=2),
            ),
        )
        processing_widget = cast(
            Any,
            ResponseProcessingOptionsWidget(ResponseProcessingOptions()),
        )
        for widget in (
            response_window_widget,
            dff_widget,
            normalization_widget,
            processing_widget,
        ):
            widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            widget.ensurePolished()
            widget.adjustSize()
        QApplication.processEvents()

        wide_controls: list[QWidget] = [
            response_window_widget._auto,
            dff_widget._baseline_mode,
            dff_widget._background_method,
            dff_widget._baseline_epoch,
            dff_widget._use_full_baseline,
            dff_widget._fit_mode,
            dff_widget._apply_motion_mask,
            normalization_widget._epoch,
            processing_widget._smoothing_method,
            processing_widget._low_pass_method,
            processing_widget._correlation_reference,
            processing_widget._correlation_window_has_stop,
        ]
        dropdowns: list[QComboBox] = [
            dff_widget._baseline_mode,
            dff_widget._background_method,
            dff_widget._baseline_epoch,
            dff_widget._fit_mode,
            normalization_widget._epoch,
            processing_widget._smoothing_method,
            processing_widget._low_pass_method,
            processing_widget._correlation_reference,
        ]
        compact_controls: list[QWidget] = [
            response_window_widget._pre_seconds,
            response_window_widget._post_seconds,
            dff_widget._baseline_seconds,
            processing_widget._smoothing_window_frames,
            processing_widget._smoothing_polynomial_order,
            processing_widget._low_pass_cutoff_hz,
            processing_widget._low_pass_order,
            processing_widget._minimum_correlation,
            processing_widget._correlation_window_start,
            processing_widget._correlation_window_stop,
        ]

        self.assertTrue(
            all(
                isinstance(
                    control,
                    QSpinBox | QDoubleSpinBox,
                )
                for control in compact_controls
            ),
        )
        self.assertEqual(
            {control.width() for control in compact_controls},
            {PLOT_CONTROL_WIDTH},
        )
        self.assertEqual(
            {control.minimumWidth() for control in wide_controls},
            {PLOT_DROPDOWN_WIDTH},
        )
        self.assertEqual(
            {control.maximumWidth() for control in wide_controls},
            {PLOT_DROPDOWN_WIDTH},
        )
        self.assertEqual(
            {control.width() for control in wide_controls},
            {PLOT_DROPDOWN_WIDTH},
        )
        self.assertEqual(
            normalization_widget._normalize_to_epoch_abs_peak.width(),
            230,
        )
        popup_view_widths: set[int] = set()
        popup_window_widths: set[int] = set()
        for control in dropdowns:
            view = control.view()
            self.assertIsNotNone(view)
            if view is None:
                continue
            popup_view_widths.add(view.width())
            popup = view.window()
            self.assertIsNotNone(popup)
            if popup is not None:
                popup_window_widths.add(popup.width())
        self.assertEqual(popup_view_widths, {PLOT_DROPDOWN_WIDTH})
        self.assertEqual(popup_window_widths, {PLOT_DROPDOWN_WIDTH})

    def test_plot_tab_decimal_controls_show_two_places(self) -> None:
        """Confirm Plot-tab dF/F and processing decimals show two places.

        Inputs: dF/F and processing option widgets.
        Outputs: every decimal spin box in those subsections has two decimals.
        """
        _ = QApplication.instance() or QApplication([])
        dff_widget = DeltaFOverFOptionsWidget(DeltaFOverFOptions())
        processing_widget = ResponseProcessingOptionsWidget(
            ResponseProcessingOptions(),
        )

        self.assertEqual(dff_widget._baseline_seconds.decimals(), 2)
        self.assertEqual(processing_widget._low_pass_cutoff_hz.decimals(), 2)
        self.assertEqual(processing_widget._minimum_correlation.decimals(), 2)
        self.assertEqual(processing_widget._correlation_window_start.decimals(), 2)
        self.assertEqual(processing_widget._correlation_window_stop.decimals(), 2)


if __name__ == "__main__":
    unittest.main()
