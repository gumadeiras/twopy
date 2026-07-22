"""Tests for local napari convenience state.

Inputs: isolated state files and concurrent folder updates.
Outputs: assertions that independent UI preferences remain available.
"""

import unittest
from pathlib import Path
from threading import Event, Lock, Thread
from unittest.mock import patch

from tests.tempdir import temporary_directory
from twopy.napari import state as napari_state


class NapariStateTest(unittest.TestCase):
    """Local napari convenience-state tests."""

    def test_concurrent_folder_writes_preserve_both_preferences(self) -> None:
        """Confirm concurrent UI-state writes do not replace another key.

        Inputs: manual and CSV folder writes that overlap at disk output.
        Outputs: both independent folder preferences remain in the state file.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            manual_folder = root / "manual"
            csv_folder = root / "csv"
            state_file = root / "state.json"
            manual_folder.mkdir()
            csv_folder.mkdir()
            first_write_entered = Event()
            second_write_entered = Event()
            release_first_write = Event()
            call_lock = Lock()
            write_count = 0
            original_write = napari_state._write_state_data

            def delayed_write(
                data: dict[str, object],
                selected_state_file: Path | None,
            ) -> None:
                """Pause the first disk write so a second thread can overlap."""
                nonlocal write_count
                with call_lock:
                    write_count += 1
                    call_number = write_count
                if call_number == 1:
                    first_write_entered.set()
                    release_first_write.wait(timeout=2)
                else:
                    second_write_entered.set()
                original_write(data, selected_state_file)

            with patch.object(
                napari_state,
                "_write_state_data",
                side_effect=delayed_write,
            ):
                manual_thread = Thread(
                    target=napari_state.write_last_recording_folder,
                    kwargs={"folder": manual_folder, "state_file": state_file},
                )
                csv_thread = Thread(
                    target=napari_state.write_last_recording_csv_folder,
                    kwargs={"folder": csv_folder, "state_file": state_file},
                )
                manual_thread.start()
                self.assertTrue(first_write_entered.wait(timeout=2))
                csv_thread.start()
                second_write_entered.wait(timeout=0.2)
                release_first_write.set()
                manual_thread.join(timeout=2)
                csv_thread.join(timeout=2)

            self.assertFalse(manual_thread.is_alive())
            self.assertFalse(csv_thread.is_alive())
            self.assertEqual(
                napari_state.read_last_recording_folder(state_file=state_file),
                manual_folder.resolve(),
            )
            self.assertEqual(
                napari_state.read_last_recording_csv_folder(state_file=state_file),
                csv_folder.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
