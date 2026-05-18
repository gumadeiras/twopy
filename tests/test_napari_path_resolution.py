"""Napari recording path resolution tests.

Inputs: shared fake napari state and tiny converted recordings.
Outputs: assertions for one napari workflow area.
"""

from tests.converted_files import write_converted_recording_files
from tests.napari_support import (
    NapariAdapterTestCase,
    Path,
    _fake_convert_recording,
    _FakeViewer,
    _load_recording_widget,
    _write_converted_recording,
    _write_source_recording_shape,
    add_twopy_magicgui_controls,
    chdir,
    patch,
    resolve_launch_recording_path,
    resolve_or_convert_recording,
    resolve_recording_paths,
    temporary_directory,
    unittest,
)

from twopy.config import load_config, resolve_analysis_cache_dir


class NapariPathResolutionTest(NapariAdapterTestCase):
    """Napari recording path resolution tests."""

    def test_failed_single_load_keeps_visible_recording_picker_text(self) -> None:
        """Confirm failed single loads do not reset the recording picker.

        Inputs: one loaded recording and a bad manually entered recording path.
        Outputs: the error is returned and the user-entered recording field
        stays visible for correction.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir) / "twopy"
            root.mkdir()
            _write_converted_recording(root)
            viewer = _FakeViewer()
            control_docks = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            load_widget = _load_recording_widget(control_docks.load_widget)

            load_widget(recording_folder=root)
            user_text = root / "manual-entry"
            load_widget.recording_folder.line_edit.value = str(user_text)
            result = load_widget(
                recording_folder=user_text,
                roi_file_to_load=Path("default"),
            )

            self.assertIn("Could not find recording_data.h5", str(result))
            self.assertEqual(
                load_widget.recording_folder.line_edit.value,
                str(user_text),
            )

    def test_recording_path_resolution_reports_available_files(self) -> None:
        """Confirm folder resolution finds optional movie and ROI paths.

        Inputs: converted output folder with ``aligned_movie.h5`` and
            ``rois.h5``.
        Outputs: concrete paths for the viewer and controls.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            roi_path = root / "rois.h5"
            roi_path.touch()

            paths = resolve_recording_paths(root)

            self.assertEqual(paths.recording_data_path, recording_path.resolve())
            self.assertEqual(paths.movie_path, (root / "aligned_movie.h5").resolve())
            self.assertEqual(paths.roi_file_to_load, roi_path.resolve())
            self.assertEqual(paths.roi_save_file, roi_path.resolve())

    def test_recording_path_resolution_accepts_widget_strings(self) -> None:
        """Confirm magicgui string paths resolve the same way as ``Path`` values.

        Inputs: converted output folder passed as text, matching magicgui
            ``FileEdit`` callback values.
        Outputs: concrete recording, movie, and ROI paths.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            roi_path = root / "rois.h5"
            roi_path.touch()

            paths = resolve_recording_paths(str(root))

            self.assertEqual(paths.recording_data_path, recording_path.resolve())
            self.assertEqual(paths.movie_path, (root / "aligned_movie.h5").resolve())
            self.assertEqual(paths.roi_file_to_load, roi_path.resolve())
            self.assertEqual(paths.roi_save_file, roi_path.resolve())

    def test_recording_path_resolution_converts_external_source_to_cache(
        self,
    ) -> None:
        """Confirm configured caching routes external source conversion to cache.

        Inputs: source-shaped recording folder outside configured ``data_path``.
        Outputs: converted paths in the stable external cache plus a flag saying
            conversion ran.
        """
        with temporary_directory() as temp_dir:
            source_dir = Path(temp_dir)
            _write_source_recording_shape(source_dir)
            expected_dir = resolve_analysis_cache_dir(
                load_config(),
                source_dir.resolve(),
            )

            with patch(
                "twopy.napari.loading.convert_recording_to_twopy",
                side_effect=_fake_convert_recording,
            ) as convert:
                resolved = resolve_or_convert_recording(source_dir)

            self.assertTrue(resolved.was_converted)
            convert.assert_called_once_with(
                source_dir.resolve(),
                output_dir=expected_dir,
            )
            self.assertEqual(
                resolved.paths.recording_data_path,
                (expected_dir / "recording_data.h5").resolve(),
            )
            self.assertEqual(
                resolved.paths.movie_path,
                (expected_dir / "aligned_movie.h5").resolve(),
            )

    def test_recording_path_resolution_uses_local_analysis_cache(self) -> None:
        """Confirm source loading converts into local cache when enabled.

        Inputs: a source-shaped recording under configured ``data_path``.
        Outputs: converted paths under ``analysis_cache_dir``.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            source_dir = data_root / "fly" / "stim" / "2023" / "10_17"
            cache_root = root / "cache"
            _write_source_recording_shape(source_dir)
            (root / "config.yml").write_text(
                f"database_path: {root / 'db'}\n"
                f"data_path: {data_root.resolve()}\n"
                "database_access: copy\n"
                "analysis_caching: true\n"
                f"analysis_cache_dir: {cache_root}\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )
            original_cwd = Path.cwd()
            try:
                chdir(root)
                with patch(
                    "twopy.napari.loading.convert_recording_to_twopy",
                    side_effect=_fake_convert_recording,
                ) as convert:
                    resolved = resolve_or_convert_recording(source_dir)
            finally:
                chdir(original_cwd)

            expected_dir = cache_root / "fly" / "stim" / "2023" / "10_17"
            self.assertTrue(resolved.was_converted)
            convert.assert_called_once_with(
                source_dir.resolve(), output_dir=expected_dir
            )
            self.assertEqual(
                resolved.paths.recording_data_path,
                (expected_dir / "recording_data.h5").resolve(),
            )
            self.assertEqual(
                resolved.paths.movie_path,
                (expected_dir / "aligned_movie.h5").resolve(),
            )

    def test_recording_path_resolution_uses_existing_cache_for_unavailable_source(
        self,
    ) -> None:
        """Confirm unavailable source paths can reopen existing cached data.

        Inputs: missing source recording folder under ``data_path`` and a
            matching converted cache folder.
        Outputs: resolved paths point at the existing cached HDF5 files.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            source_dir = data_root / "fly" / "stim" / "2023" / "10_17"
            cache_root = root / "cache"
            cache_dir = cache_root / "fly" / "stim" / "2023" / "10_17"
            cache_dir.mkdir(parents=True)
            _write_converted_recording(cache_dir, source_session_dir=source_dir)
            (root / "config.yml").write_text(
                f"database_path: {root / 'db'}\n"
                f"data_path: {data_root.resolve()}\n"
                "database_access: copy\n"
                "analysis_caching: true\n"
                f"analysis_cache_dir: {cache_root}\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )
            original_cwd = Path.cwd()
            try:
                chdir(root)
                with patch(
                    "twopy.napari.loading.convert_recording_to_twopy",
                    side_effect=AssertionError("conversion should not run"),
                ):
                    resolved = resolve_or_convert_recording(source_dir)
            finally:
                chdir(original_cwd)

            self.assertFalse(resolved.was_converted)
            self.assertTrue(resolved.source_unavailable)
            self.assertEqual(
                resolved.paths.recording_data_path,
                (cache_dir / "recording_data.h5").resolve(),
            )
            self.assertEqual(
                resolved.paths.movie_path,
                (cache_dir / "aligned_movie.h5").resolve(),
            )

    def test_recording_path_resolution_pulls_published_rois_into_cache(self) -> None:
        """Confirm cached source loads reuse published ROI files locally.

        Inputs: source recording, cache config, and a published ``rois.h5``.
        Outputs: resolved ROI file path points at the local cache copy.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            source_dir = data_root / "fly" / "stim" / "2023" / "10_17"
            cache_root = root / "cache"
            publish_dir = source_dir / "twopy"
            _write_source_recording_shape(source_dir)
            publish_dir.mkdir()
            (publish_dir / "rois.h5").write_text("published", encoding="utf-8")
            (root / "config.yml").write_text(
                f"database_path: {root / 'db'}\n"
                f"data_path: {data_root.resolve()}\n"
                "database_access: copy\n"
                "analysis_caching: true\n"
                f"analysis_cache_dir: {cache_root}\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )
            original_cwd = Path.cwd()
            try:
                chdir(root)
                with patch(
                    "twopy.napari.loading.convert_recording_to_twopy",
                    side_effect=_fake_convert_recording,
                ):
                    resolved = resolve_or_convert_recording(source_dir)
            finally:
                chdir(original_cwd)

            expected_roi = cache_root / "fly" / "stim" / "2023" / "10_17" / "rois.h5"
            self.assertEqual(resolved.paths.roi_file_to_load, expected_roi.resolve())
            self.assertEqual(expected_roi.read_text(encoding="utf-8"), "published")

    def test_external_source_cache_pulls_published_analysis_outputs(self) -> None:
        """Confirm external cached source loads refresh saved analysis files.

        Inputs: source recording outside configured ``data_path``, an existing
            external cache entry, and published source-local analysis outputs.
        Outputs: the existing cache is reused and saved analysis files are
            copied beside the cached converted files.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "configured_data"
            source_dir = root / "external" / "fly" / "stim" / "2023" / "10_17"
            cache_root = root / "cache"
            publish_dir = source_dir / "twopy"
            _write_source_recording_shape(source_dir)
            publish_dir.mkdir()
            (publish_dir / "rois.h5").write_text("published-rois", encoding="utf-8")
            (publish_dir / "analysis_outputs.h5").write_text(
                "published-analysis",
                encoding="utf-8",
            )
            config_path = root / "config.yml"
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                f"data_path: {data_root}\n"
                "database_access: copy\n"
                "analysis_caching: true\n"
                f"analysis_cache_dir: {cache_root}\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )
            expected_dir = resolve_analysis_cache_dir(
                load_config(config_path),
                source_dir.resolve(),
            )
            expected_dir.mkdir(parents=True)
            _write_converted_recording(expected_dir, source_session_dir=source_dir)

            original_cwd = Path.cwd()
            try:
                chdir(root)
                with patch(
                    "twopy.napari.loading.convert_recording_to_twopy",
                    side_effect=AssertionError("conversion should not run"),
                ):
                    resolved = resolve_or_convert_recording(source_dir)
            finally:
                chdir(original_cwd)

            self.assertFalse(resolved.was_converted)
            self.assertEqual(
                resolved.paths.recording_data_path,
                (expected_dir / "recording_data.h5").resolve(),
            )
            self.assertEqual(
                (expected_dir / "rois.h5").read_text(encoding="utf-8"),
                "published-rois",
            )
            self.assertEqual(
                (expected_dir / "analysis_outputs.h5").read_text(encoding="utf-8"),
                "published-analysis",
            )

    def test_recording_path_resolution_localizes_selected_converted_output(
        self,
    ) -> None:
        """Confirm direct converted selections are copied to cache before use.

        Inputs: a converted folder on a publish path with source metadata.
        Outputs: resolved paths point at the local analysis cache.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            source_dir = data_root / "fly" / "stim" / "2023" / "10_17"
            cache_root = root / "cache"
            publish_dir = root / "publish"
            _write_source_recording_shape(source_dir)
            publish_dir.mkdir()
            _write_converted_recording(publish_dir, source_session_dir=source_dir)
            (root / "config.yml").write_text(
                f"database_path: {root / 'db'}\n"
                f"data_path: {data_root.resolve()}\n"
                "database_access: copy\n"
                "analysis_caching: true\n"
                f"analysis_cache_dir: {cache_root}\n"
                f"analysis_output: {publish_dir}\n",
                encoding="utf-8",
            )
            original_cwd = Path.cwd()
            try:
                chdir(root)
                resolved = resolve_or_convert_recording(publish_dir)
            finally:
                chdir(original_cwd)

            expected_dir = cache_root / "fly" / "stim" / "2023" / "10_17"
            self.assertFalse(resolved.was_converted)
            self.assertEqual(
                resolved.paths.recording_data_path,
                (expected_dir / "recording_data.h5").resolve(),
            )
            self.assertEqual(
                resolved.paths.movie_path,
                (expected_dir / "aligned_movie.h5").resolve(),
            )

    def test_recording_path_resolution_refreshes_stale_localized_output(
        self,
    ) -> None:
        """Confirm direct converted selections replace older cache copies.

        Inputs: a newer selected converted file and an older local cache copy.
        Outputs: local cached recording data is refreshed from the selected file.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            source_dir = data_root / "fly" / "stim" / "2023" / "10_17"
            cache_root = root / "cache"
            publish_dir = root / "publish"
            local_dir = cache_root / "fly" / "stim" / "2023" / "10_17"
            _write_source_recording_shape(source_dir)
            publish_dir.mkdir()
            local_dir.mkdir(parents=True)
            _write_converted_recording(publish_dir, source_session_dir=source_dir)
            _write_converted_recording(local_dir, source_session_dir=source_dir)
            source_recording = publish_dir / "recording_data.h5"
            local_recording = local_dir / "recording_data.h5"
            source_recording.touch()
            (root / "config.yml").write_text(
                f"database_path: {root / 'db'}\n"
                f"data_path: {data_root.resolve()}\n"
                "database_access: copy\n"
                "analysis_caching: true\n"
                f"analysis_cache_dir: {cache_root}\n"
                f"analysis_output: {publish_dir}\n",
                encoding="utf-8",
            )
            original_cwd = Path.cwd()
            try:
                chdir(root)
                resolved = resolve_or_convert_recording(publish_dir)
            finally:
                chdir(original_cwd)

            self.assertEqual(
                resolved.paths.recording_data_path,
                local_recording.resolve(),
            )
            self.assertEqual(
                local_recording.stat().st_mtime_ns,
                source_recording.stat().st_mtime_ns,
            )

    def test_source_recording_validation_error_is_reported(self) -> None:
        """Confirm malformed source folders do not look like missing HDF5.

        Inputs: source-shaped recording folder with two real TIFF movies.
        Outputs: the source discovery error reaches the caller.
        """
        with temporary_directory() as temp_dir:
            source_dir = Path(temp_dir)
            _write_source_recording_shape(source_dir)
            (source_dir / "second_movie.tif").touch()

            with self.assertRaisesRegex(ValueError, "raw TIFF movie"):
                resolve_or_convert_recording(source_dir)

    def test_recording_path_resolution_repairs_missing_cached_movie(self) -> None:
        """Confirm incomplete cached twopy output is regenerated in place.

        Inputs: source-shaped folder with external cache ``recording_data.h5``
            but no ``aligned_movie.h5``.
        Outputs: conversion called with the existing cache directory.
        """
        with temporary_directory() as temp_dir:
            source_dir = Path(temp_dir)
            _write_source_recording_shape(source_dir)
            output_dir = resolve_analysis_cache_dir(
                load_config(),
                source_dir.resolve(),
            )
            output_dir.mkdir()
            (output_dir / "recording_data.h5").touch()

            with patch(
                "twopy.napari.loading.convert_recording_to_twopy",
                side_effect=_fake_convert_recording,
            ) as convert:
                resolved = resolve_or_convert_recording(source_dir)

            self.assertTrue(resolved.was_converted)
            convert.assert_called_once_with(
                source_dir.resolve(),
                output_dir=output_dir.resolve(),
            )
            self.assertEqual(
                resolved.paths.movie_path,
                (output_dir / "aligned_movie.h5").resolve(),
            )

    def test_recording_path_resolution_repairs_mirrored_output_from_source_attr(
        self,
    ) -> None:
        """Confirm missing movie repair works outside the source folder.

        Inputs: converted ``recording_data.h5`` with a ``source_session_dir``
            attribute and no ``aligned_movie.h5``.
        Outputs: conversion called with the selected converted output folder.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            output_dir = root / "converted"
            _write_source_recording_shape(source_dir)
            write_converted_recording_files(
                output_dir,
                source_session_dir=source_dir,
            )
            (output_dir / "aligned_movie.h5").unlink()

            with patch(
                "twopy.napari.loading.convert_recording_to_twopy",
                side_effect=_fake_convert_recording,
            ) as convert:
                resolved = resolve_or_convert_recording(output_dir)

            self.assertTrue(resolved.was_converted)
            convert.assert_called_once_with(
                source_dir.resolve(),
                output_dir=output_dir.resolve(),
            )
            self.assertEqual(
                resolved.paths.movie_path,
                (output_dir / "aligned_movie.h5").resolve(),
            )

    def test_launch_recording_path_returns_none_when_no_default_exists(self) -> None:
        """Confirm no-path app launch can start empty instead of failing.

        Inputs: temporary directory without converted recording files.
        Outputs: ``None`` so the launcher can open an empty viewer.
        """
        with temporary_directory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                chdir(temp_dir)

                resolved = resolve_launch_recording_path(None)
            finally:
                chdir(original_cwd)

            self.assertIsNone(resolved)

    def test_launch_recording_path_resolves_from_current_directory(self) -> None:
        """Confirm launcher can run with no path from a converted folder.

        Inputs: temporary directory containing ``recording_data.h5``.
        Outputs: resolved path to that file.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = root / "recording_data.h5"
            recording_path.touch()
            original_cwd = Path.cwd()
            try:
                chdir(root)

                resolved = resolve_launch_recording_path(None)
            finally:
                chdir(original_cwd)

            self.assertEqual(resolved, recording_path.resolve())

    def test_launch_recording_path_resolves_from_source_recording_directory(
        self,
    ) -> None:
        """Confirm launcher can run from a source recording with twopy output.

        Inputs: temporary directory containing ``twopy/recording_data.h5``.
        Outputs: resolved path to the converted recording file.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = root / "twopy" / "recording_data.h5"
            recording_path.parent.mkdir()
            recording_path.touch()
            original_cwd = Path.cwd()
            try:
                chdir(root)

                resolved = resolve_launch_recording_path(None)
            finally:
                chdir(original_cwd)

            self.assertEqual(resolved, recording_path.resolve())


if __name__ == "__main__":
    unittest.main()
