"""Napari plot layout and display-control tests.

Inputs: shared fake napari state and tiny response data.
Outputs: assertions for plot-control behavior.
"""

from tests.napari_support import (
    PLOT_CONTROL_WIDTH,
    PLOT_DROPDOWN_WIDTH,
    RESPONSE_HEATMAP_COLORMAP,
    Any,
    DeltaFOverFOptions,
    DeltaFOverFOptionsWidget,
    EpochResponseMap,
    NapariAdapterTestCase,
    NormalizationOptions,
    NormalizationOptionsWidget,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    Qt,
    QWidget,
    ResponseMapData,
    ResponseMapOptions,
    ResponseMapOptionsWidget,
    ResponseProcessingOptions,
    ResponseProcessingOptionsWidget,
    ResponseWindowOptions,
    ResponseWindowOptionsWidget,
    SpatialCrop,
    cast,
    create_response_plot_widget,
    display_response_limit,
    display_response_values,
    np,
    plot_display_options_group,
    unittest,
)


class NapariPlotControlsTest(NapariAdapterTestCase):
    """Napari plot layout and display-control tests."""

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

    def test_response_heatmap_colormap_has_black_zero_without_white_band(self) -> None:
        """Confirm the signed heatmap colorbar has a black neutral point.

        Inputs: the shared response heatmap colormap sampled across the full
        signed response range.
        Outputs: zero is black and no sampled color is close to white.
        """
        colors = np.asarray(RESPONSE_HEATMAP_COLORMAP(np.linspace(0.0, 1.0, 257)))
        zero_color = colors[128, :3]
        low_positive_color = colors[154, :3]
        mid_positive_color = colors[192, :3]
        high_positive_color = colors[-1, :3]

        self.assertLess(float(np.max(zero_color)), 0.02)
        self.assertFalse(np.any(np.all(colors[:, :3] > 0.9, axis=1)))
        self.assertLess(float(low_positive_color[0]), float(mid_positive_color[0]))
        self.assertLess(float(mid_positive_color[0]), float(high_positive_color[0]))

    def test_response_heatmap_display_uses_robust_limits(self) -> None:
        """Confirm outliers do not set the heatmap color limit.

        Inputs: one response map with a single extreme pixel.
        Outputs: the visual limit follows the 95th percentile and clips the
        outlier.
        """
        response = np.concatenate(
            (np.linspace(0.01, 0.20, 100, dtype=np.float64), np.array([1.0])),
        ).reshape(101, 1)
        map_data = ResponseMapData(
            mean_image=np.ones((101, 1), dtype=np.float64),
            epochs=(
                EpochResponseMap(
                    epoch_name="Odor",
                    epoch_number=1,
                    response_values=response,
                    trial_count=1,
                ),
            ),
            options=ResponseMapOptions(),
            spatial_crop=SpatialCrop(0, 101, 0, 1, (101, 1), "test"),
            response_scale=1.0,
        )
        epoch = map_data.epochs[0]

        limit = display_response_limit(
            map_data,
            epoch,
            shared_limits=False,
        )
        values = display_response_values(
            map_data,
            epoch,
            shared_limits=False,
        )

        self.assertLess(limit, 0.20)
        self.assertGreater(limit, 0.18)
        self.assertEqual(float(values[-1, 0]), limit)

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
                NormalizationOptions(method="epoch_peak", epoch_number=2),
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
            widget.show()
        QApplication.processEvents()

        wide_controls: list[QWidget] = [
            response_window_widget._auto,
            dff_widget._baseline_mode,
            dff_widget._background_method,
            dff_widget._baseline_epoch,
            dff_widget._use_full_baseline,
            dff_widget._fit_mode,
            dff_widget._apply_motion_mask,
            normalization_widget._normalize_to_epoch_peak,
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
