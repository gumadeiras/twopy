"""Shared converted-file and response-data helpers for napari tests.

Inputs: tiny arrays and temporary folders.
Outputs: converted HDF5 files and response objects used by napari tests.
"""

# ruff: noqa: I001

from tests.converted_files import write_converted_recording_files
from tests.napari_support.base import (
    ConvertedRecording,
    EpochResponseMap,
    Path,
    ResponseMapData,
    ResponseMapOptions,
    SpatialCrop,
    TrialTimelineData,
    TrialTimelineWindow,
    np,
    npt,
    sqlite3,
    unittest,
)


def _write_database(path: Path) -> None:
    """Create the smallest SQLite DB needed by napari DB-search tests.

    Args:
        path: SQLite database path to create.

    Returns:
        None. The function writes one fly and one stimulus row.
    """
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE fly (
                flyId INTEGER PRIMARY KEY,
                genotype TEXT,
                cellType TEXT,
                fluorescentProtein TEXT,
                eye TEXT,
                surgeon TEXT
            )
            """,
        )
        connection.execute(
            """
            CREATE TABLE stimulusPresentation (
                stimulusPresentationId INTEGER PRIMARY KEY,
                fly INT,
                relativeDataPath TEXT,
                stimulusFunction TEXT,
                date TEXT,
                dataQuality INT
            )
            """,
        )
        connection.execute(
            """
            INSERT INTO fly
                (flyId, genotype, cellType, fluorescentProtein, eye, surgeon)
            VALUES
                (10923, 'gh146gal4', 'ALPN', 'g6f', 'left', 'Gustavo')
            """,
        )
        connection.execute(
            """
            INSERT INTO stimulusPresentation
                (
                    stimulusPresentationId,
                    fly,
                    relativeDataPath,
                    stimulusFunction,
                    date,
                    dataQuality
                )
            VALUES
                (
                    20005,
                    10923,
                    'genotype\\stimulus\\2023\\10_17\\10_02_49',
                    'combo_stim_singles',
                    '2023-10-17 10:03:14',
                    1
                )
            """,
        )
        connection.commit()
    finally:
        connection.close()


def _write_converted_recording(
    root: Path,
    *,
    movie_values: npt.NDArray[np.float64] | None = None,
    alignment_valid_crop: SpatialCrop | None = None,
    source_session_dir: Path | None = None,
    stimulus_data: npt.NDArray[np.float64] | None = None,
    stimulus_data_column_names: tuple[str, ...] = ("time_seconds", "epoch_number"),
    high_res_pd: npt.NDArray[np.float64] | None = None,
    stimulus_parameters_json: str = "[]",
) -> Path:
    """Write a tiny converted recording for adapter tests.

    Args:
        root: Temporary directory receiving HDF5 files.
        movie_values: Optional movie array shaped ``(frames, axis0, axis1)``.
        alignment_valid_crop: Optional spatial crop metadata.
        source_session_dir: Optional source recording folder stored in the
            converted recording metadata.
        stimulus_data: Optional stimulus rows shaped ``(rows, 2)`` containing
            ``time_seconds`` and ``epoch_number``.
        stimulus_data_column_names: Column names stored with ``stimulus_data``.
        high_res_pd: Optional high-rate photodiode vector.
        stimulus_parameters_json: JSON list of stimulus epoch parameter
            dictionaries.

    Returns:
        Path to ``recording_data.h5``.
    """
    return write_converted_recording_files(
        root,
        movie_values=movie_values,
        alignment_valid_crop=alignment_valid_crop,
        source_session_dir=source_session_dir,
        acquisition_metadata={"acq.frameRate": 10.0, "acq.zoomFactor": 2.0},
        run_metadata={"rig_name": "TestRig"},
        stimulus_data=stimulus_data,
        stimulus_data_column_names=stimulus_data_column_names,
        stimulus_parameters_json=stimulus_parameters_json,
        high_res_pd=high_res_pd,
    )


def _two_window_timeline(*, frame_count: int) -> TrialTimelineData:
    """Return two simple trial windows for timeline tests.

    Args:
        frame_count: Movie frame count stored on the timeline.

    Returns:
        Timeline data with one gray and one odor window.
    """
    windows = (
        TrialTimelineWindow(0, 0, 2, 1, "Gray"),
        TrialTimelineWindow(1, 2, frame_count, 2, "Odor"),
    )
    return TrialTimelineData(
        frame_count=frame_count,
        windows=windows,
        start_frames=tuple(window.start_frame for window in windows),
        stop_frames=tuple(window.stop_frame for window in windows),
    )


def _timeline_photodiode() -> npt.NDArray[np.float64]:
    """Return a synthetic photodiode trace with start and long end flashes."""
    values = np.zeros(120, dtype=np.float64)
    values[0:2] = 1.0
    values[80:116] = 1.0
    return values


def _tiny_response_map_data(*, epoch_name: str = "Odor") -> ResponseMapData:
    """Build one tiny response heatmap data object for napari plot tests.

    Args:
        epoch_name: Name to store on the only heatmap epoch.

    Returns:
        One-epoch heatmap data with normalized signed response values.
    """
    return ResponseMapData(
        mean_image=np.ones((2, 2), dtype=np.float64),
        epochs=(
            EpochResponseMap(
                epoch_name=epoch_name,
                epoch_number=1,
                response_values=np.full((2, 2), 0.25, dtype=np.float64),
                trial_count=1,
            ),
        ),
        options=ResponseMapOptions(),
        spatial_crop=SpatialCrop(0, 2, 0, 2, (2, 2), "test"),
        response_scale=4.0,
    )


def _write_source_recording_shape(root: Path) -> None:
    """Write the required source file names without real imaging contents.

    Args:
        root: Temporary source recording folder.

    Returns:
        None. The folder shape is enough for napari conversion routing tests
        because the actual converter is patched there.
    """
    root.mkdir(parents=True, exist_ok=True)
    stimulus_dir = root / "stimulusData"
    stimulus_dir.mkdir()
    for path in (
        root / "alignedMovie.mat",
        root / "stimulus_name_changes_001.tif",
        root / "stimulus_name_changes_001_ch1_disinterleaved_alignment.txt",
        root / "defaultAlignChannel.txt",
        root / "highResPd.mat",
        root / "imageDescription.mat",
        root / "imagingResPd.mat",
        stimulus_dir / "filebackup.zip",
    ):
        path.touch()


def _fake_convert_recording(
    source_dir: Path,
    output_dir: Path | None = None,
    **_kwargs: object,
) -> ConvertedRecording:
    """Create tiny placeholder converted files for routing tests.

    Args:
        source_dir: Source recording folder passed by napari loading.
        output_dir: Optional output folder requested by napari loading.
        _kwargs: Ignored converter keyword arguments.

    Returns:
        Converted recording summary pointing at placeholder files.
    """
    destination = output_dir or source_dir / "twopy"
    destination.mkdir(parents=True, exist_ok=True)
    recording_path = destination / "recording_data.h5"
    movie_path = destination / "aligned_movie.h5"
    recording_path.touch()
    movie_path.touch()
    return ConvertedRecording(
        path=recording_path,
        movie_path=movie_path,
        source_session_dir=source_dir,
        movie_shape=(1, 1, 1),
        mean_image_start_frame=0,
        mean_image_stop_frame=1,
    )


if __name__ == "__main__":
    unittest.main()
