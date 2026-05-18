"""Napari trial timeline tests.

Inputs: shared fake napari state and tiny converted recordings.
Outputs: assertions for one napari workflow area.
"""

from tests.napari_support import (
    TRIAL_TIMELINE_DOCK_NAME,
    NapariAdapterTestCase,
    Path,
    QApplication,
    TrialTimelineController,
    TrialTimelineData,
    TrialTimelineWindow,
    _FakeLayer,
    _FakeViewer,
    _timeline_photodiode,
    _two_window_timeline,
    _write_converted_recording,
    cast,
    current_trial_text,
    current_trial_window,
    load_converted_recording,
    np,
    open_recording_in_napari,
    patch,
    resolve_trial_timeline_data,
    temporary_directory,
    trial_timeline,
    unittest,
)


class NapariTimelineTest(NapariAdapterTestCase):
    """Napari trial timeline tests."""

    def test_trial_timeline_follows_current_movie_frame(self) -> None:
        """Confirm the timeline rail and HUD follow napari's frame slider.

        Inputs: a loaded movie and patched photodiode-aligned trial windows.
        Outputs: a bottom timeline dock, visible HUD text, and click-to-seek
        updates on the movie dimension.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(
                root,
                movie_values=np.arange(16, dtype=np.float64).reshape(4, 2, 2),
            )
            viewer = _FakeViewer()
            timeline = _two_window_timeline(frame_count=4)

            with patch(
                "twopy.napari.trial_timeline.resolve_trial_timeline_data",
                return_value=timeline,
            ):
                opened = open_recording_in_napari(
                    recording_path,
                    viewer=viewer,
                    movie_frame_range=(0, None),
                )

            self.assertEqual(
                viewer.window.dock_widgets[-1].name,
                TRIAL_TIMELINE_DOCK_NAME,
            )
            self.assertEqual(viewer.window.dock_widgets[-1].area, "bottom")
            self.assertTrue(viewer.text_overlay.visible)
            self.assertIn("Trial 1/2", viewer.text_overlay.text)
            self.assertIn("Epoch 1: Gray", viewer.text_overlay.text)

            viewer.dims.set_current_step(0, 2)

            self.assertIn("Trial 2/2", viewer.text_overlay.text)
            self.assertIn("Epoch 2: Odor", viewer.text_overlay.text)

            controller = cast(TrialTimelineController, opened.trial_timeline_controller)
            controller.widget.frame_selected.emit(3)

            self.assertEqual(viewer.dims.current_step[0], 3)
            self.assertNotIn("frame", viewer.text_overlay.text)

    def test_trial_timeline_uses_real_photodiode_epoch_mapping(self) -> None:
        """Confirm timeline data is built from converted timing evidence.

        Inputs: synthetic high-rate photodiode events and stimulus epoch rows.
        Outputs: opening the recording creates a timeline whose HUD reports the
        real epoch names from converted stimulus parameters.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(
                root,
                movie_values=np.arange(48, dtype=np.float64).reshape(12, 2, 2),
                stimulus_data=np.array(
                    [
                        [0.0, 1.0],
                        [0.3, 1.0],
                        [0.4, 2.0],
                        [0.8, 2.0],
                    ],
                    dtype=np.float64,
                ),
                high_res_pd=_timeline_photodiode(),
                stimulus_parameters_json=(
                    '[{"epochName": "Gray"}, {"epochName": "Odor"}]'
                ),
            )
            viewer = _FakeViewer()

            opened = open_recording_in_napari(
                recording_path,
                viewer=viewer,
                movie_frame_range=(0, None),
            )
            timeline = resolve_trial_timeline_data(opened.recording)

            self.assertIsNotNone(timeline)
            assert timeline is not None
            self.assertEqual(
                tuple(window.epoch_name for window in timeline.windows),
                ("Gray", "Odor"),
            )
            self.assertIn("Epoch 1: Gray", viewer.text_overlay.text)

    def test_trial_timeline_does_not_seek_outside_loaded_preview(self) -> None:
        """Confirm full-recording rail clicks cannot jump partial previews.

        Inputs: a timeline for six frames with only frames two through three
        loaded in the napari movie layer.
        Outputs: seeking an unloaded frame leaves the displayed stack index
        unchanged, while seeking a loaded frame updates it.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(
                root,
                movie_values=np.arange(24, dtype=np.float64).reshape(6, 2, 2),
            )
            viewer = _FakeViewer()
            timeline = _two_window_timeline(frame_count=6)

            with patch(
                "twopy.napari.trial_timeline.resolve_trial_timeline_data",
                return_value=timeline,
            ):
                opened = open_recording_in_napari(
                    recording_path,
                    viewer=viewer,
                    movie_frame_range=(2, 3),
                )
            controller = cast(TrialTimelineController, opened.trial_timeline_controller)

            controller.widget.frame_selected.emit(0)
            self.assertEqual(viewer.dims.current_step[0], 0)

            controller.widget.frame_selected.emit(3)
            self.assertEqual(viewer.dims.current_step[0], 1)

    def test_trial_timeline_restores_existing_text_overlay(self) -> None:
        """Confirm timeline HUD does not permanently own napari text overlay.

        Inputs: viewer with an existing text overlay before timeline activation.
        Outputs: clearing timeline context restores the prior overlay fields.
        """
        _ = QApplication.instance() or QApplication([])
        viewer = _FakeViewer()
        viewer.text_overlay.text = "existing"
        viewer.text_overlay.visible = True
        viewer.text_overlay.position = "top_left"
        viewer.text_overlay.font_size = 16
        viewer.text_overlay.color = "yellow"
        viewer.text_overlay.box = False
        viewer.text_overlay.box_color = (1.0, 0.0, 0.0, 1.0)
        controller = TrialTimelineController(viewer)
        timeline = _two_window_timeline(frame_count=4)

        with temporary_directory() as temp_dir:
            recording = load_converted_recording(
                _write_converted_recording(
                    Path(temp_dir),
                    movie_values=np.arange(16, dtype=np.float64).reshape(4, 2, 2),
                ),
            )
            movie_layer = _FakeLayer(
                name="aligned movie",
                data=np.zeros((4, 2, 2), dtype=np.float64),
                options={"metadata": {}},
            )
            movie_layer.metadata = {}
            with patch(
                "twopy.napari.trial_timeline.resolve_trial_timeline_data",
                return_value=timeline,
            ):
                controller.set_context(recording, movie_layer)

        controller.set_context(None, None)

        self.assertEqual(viewer.text_overlay.text, "existing")
        self.assertTrue(viewer.text_overlay.visible)
        self.assertEqual(viewer.text_overlay.position, "top_left")
        self.assertEqual(viewer.text_overlay.font_size, 16)
        self.assertEqual(viewer.text_overlay.color, "yellow")
        self.assertFalse(viewer.text_overlay.box)
        self.assertEqual(viewer.text_overlay.box_color, (1.0, 0.0, 0.0, 1.0))

    def test_current_trial_lookup_handles_frames_between_windows(self) -> None:
        """Confirm trial lookup returns concise text inside and outside windows.

        Inputs: two timeline windows and representative frame positions.
        Outputs: matching windows and HUD text for in-trial and no-trial frames.
        """
        timeline = TrialTimelineData(
            frame_count=8,
            windows=(
                TrialTimelineWindow(0, 1, 3, 1, "Gray"),
                TrialTimelineWindow(1, 5, 7, 2, "Odor"),
            ),
            start_frames=(1, 5),
            stop_frames=(3, 7),
        )

        self.assertIsNone(current_trial_window(timeline, 0))
        self.assertEqual(current_trial_window(timeline, 1), timeline.windows[0])
        self.assertIsNone(current_trial_window(timeline, 4))
        self.assertEqual(current_trial_window(timeline, 6), timeline.windows[1])
        self.assertEqual(current_trial_text(timeline, 4), "No trial")

    def test_trial_timeline_colors_interleave_epochs_gray(self) -> None:
        """Confirm baseline-like epochs use neutral gray in the rail.

        Inputs: one interleave timeline window and one odor timeline window.
        Outputs: interleave uses the neutral baseline color while odor keeps a
        distinct condition color.
        """
        interleave_color = trial_timeline._timeline_window_color(
            TrialTimelineWindow(0, 0, 2, 2, "Baseline Interleave"),
        )
        odor_color = trial_timeline._timeline_window_color(
            TrialTimelineWindow(1, 2, 4, 2, "Odor"),
        )

        self.assertEqual(interleave_color.getRgb()[:3], (145, 150, 160))
        self.assertNotEqual(odor_color.getRgb()[:3], (145, 150, 160))


if __name__ == "__main__":
    unittest.main()
