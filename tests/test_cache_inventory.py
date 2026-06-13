"""Tests for bounded analysis-cache inventory and cleanup.

Inputs: temporary cache folders with small converted-recording stand-ins.
Outputs: assertions that cleanup removes only safe whole-recording entries.
"""

import shutil
import time
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch

import h5py

from tests.tempdir import temporary_directory
from twopy.analysis_cache import (
    copy_converted_files_to_publish,
    refresh_cached_analysis_outputs,
)
from twopy.cache_inventory import (
    CACHE_INVENTORY_FILENAME,
    enforce_analysis_cache_limit,
    record_analysis_cache_refresh,
    record_analysis_cache_sync,
    record_analysis_cache_use,
    record_analysis_cache_write,
)
from twopy.config import TwopyConfig, load_config
from twopy.filenames import (
    ALIGNED_MOVIE_FILENAME,
    ANALYSIS_OUTPUT_FILENAME,
    RECORDING_DATA_FILENAME,
)


class CacheInventoryTest(unittest.TestCase):
    """Tests cache inventory bookkeeping and eviction safety."""

    def test_cleanup_removes_oldest_published_entry_until_under_limit(self) -> None:
        """Confirm cleanup evicts the oldest safe whole recording.

        Inputs: two published cache entries and a cap that only permits one.
        Outputs: the older entry is removed and the recently used entry remains.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root, max_bytes=1)
            old_entry = _create_published_entry(config, "old", payload_size=2048)
            new_entry = _create_published_entry(config, "new", payload_size=2048)
            config = replace(
                config,
                analysis_cache_max_bytes=_entry_size(new_entry.cache_root) + 512,
            )
            record_analysis_cache_sync(
                config,
                local_root=old_entry.cache_root,
                publish_root=old_entry.publish_root,
            )
            time.sleep(0.001)
            record_analysis_cache_sync(
                config,
                local_root=new_entry.cache_root,
                publish_root=new_entry.publish_root,
            )
            record_analysis_cache_use(config, new_entry.cache_root)

            result = enforce_analysis_cache_limit(config)

            self.assertFalse(old_entry.cache_root.exists())
            self.assertTrue(new_entry.cache_root.is_dir())
            self.assertEqual(result.removed_roots, (old_entry.cache_root,))
            self.assertLessEqual(result.bytes_after, config.analysis_cache_max_bytes)

    def test_cleanup_keeps_local_only_entry_even_when_over_limit(self) -> None:
        """Confirm unsynced local writes are never evicted.

        Inputs: one cache entry marked local-only and a cap below its size.
        Outputs: cleanup reports the entry as skipped and leaves files in place.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root, max_bytes=1)
            entry = _create_cache_entry(config, "dirty", payload_size=2048)
            record_analysis_cache_write(config, entry.cache_root)

            result = enforce_analysis_cache_limit(config)

            self.assertTrue(entry.cache_root.is_dir())
            self.assertEqual(result.removed_roots, ())
            self.assertEqual(result.skipped_roots, (entry.cache_root,))
            self.assertGreater(result.bytes_after, config.analysis_cache_max_bytes)

    def test_cleanup_keeps_entry_when_source_is_unavailable(self) -> None:
        """Confirm cache-only recordings are protected from eviction.

        Inputs: one published cache entry whose source folder disappears.
        Outputs: cleanup skips the entry instead of deleting the only usable copy.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root, max_bytes=1)
            entry = _create_published_entry(config, "missing-source", 2048)
            shutil.rmtree(entry.source_root)

            result = enforce_analysis_cache_limit(config)

            self.assertTrue(entry.cache_root.is_dir())
            self.assertEqual(result.removed_roots, ())
            self.assertEqual(result.skipped_roots, (entry.cache_root,))

    def test_cleanup_keeps_entry_when_publish_copy_is_stale(self) -> None:
        """Confirm stale publish output blocks eviction.

        Inputs: one published cache entry whose local file changes afterward
            without a matching publish copy.
        Outputs: cleanup skips the entry so local changes are not lost.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root, max_bytes=1)
            entry = _create_published_entry(config, "stale-publish", 2048)
            local_movie = entry.cache_root / ALIGNED_MOVIE_FILENAME
            local_movie.write_bytes(b"new-local-movie")

            result = enforce_analysis_cache_limit(config)

            self.assertTrue(entry.cache_root.is_dir())
            self.assertEqual(result.removed_roots, ())
            self.assertEqual(result.skipped_roots, (entry.cache_root,))

    def test_cleanup_respects_protected_roots(self) -> None:
        """Confirm active recordings are not evicted.

        Inputs: two published entries, one protected, and a tight cache limit.
        Outputs: cleanup skips the protected entry and removes the other one.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root, max_bytes=1)
            protected = _create_published_entry(config, "protected", 2048)
            removable = _create_published_entry(config, "removable", 2048)
            config = replace(
                config,
                analysis_cache_max_bytes=_entry_size(protected.cache_root) + 512,
            )
            record_analysis_cache_sync(
                config,
                local_root=protected.cache_root,
                publish_root=protected.publish_root,
            )
            record_analysis_cache_sync(
                config,
                local_root=removable.cache_root,
                publish_root=removable.publish_root,
            )

            result = enforce_analysis_cache_limit(
                config,
                protected_roots=(protected.cache_root,),
            )

            self.assertTrue(protected.cache_root.is_dir())
            self.assertFalse(removable.cache_root.exists())
            self.assertEqual(result.removed_roots, (removable.cache_root,))
            self.assertEqual(result.skipped_roots, (protected.cache_root,))

    def test_corrupt_inventory_rebuild_keeps_unknown_entry(self) -> None:
        """Confirm rebuilt entries must not be deleted as published.

        Inputs: a corrupt inventory file plus one previously published cache entry.
        Outputs: cleanup rebuilds inventory from disk and skips the unknown entry.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root, max_bytes=1)
            entry = _create_published_entry(config, "rebuild", 2048)
            inventory_path = config.analysis_cache_dir / CACHE_INVENTORY_FILENAME
            inventory_path.parent.mkdir(parents=True, exist_ok=True)
            inventory_path.write_text("{not json", encoding="utf-8")

            result = enforce_analysis_cache_limit(config)

            self.assertTrue(entry.cache_root.is_dir())
            self.assertEqual(result.removed_roots, ())
            self.assertEqual(result.skipped_roots, (entry.cache_root,))

    def test_refresh_updates_size_before_later_cleanup(self) -> None:
        """Confirm publish refreshes cannot leave stale inventory size.

        Inputs: a small published cache entry refreshed with a large output.
        Outputs: later cleanup sees the larger size and can remove the entry.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            config_path = _write_config(root, max_gb=0.001)
            config = load_config(config_path)
            entry = _create_published_entry(config, "refresh", payload_size=128)
            (entry.cache_root / ANALYSIS_OUTPUT_FILENAME).unlink()
            (entry.publish_root / ANALYSIS_OUTPUT_FILENAME).write_bytes(
                b"x" * 2 * 1024 * 1024,
            )

            refreshed = refresh_cached_analysis_outputs(
                source_session_dir=entry.source_root,
                local_output_dir=entry.cache_root,
                config_path=config_path,
            )
            cleanup = enforce_analysis_cache_limit(config)

            self.assertEqual(
                refreshed,
                (entry.cache_root / ANALYSIS_OUTPUT_FILENAME,),
            )
            self.assertFalse(entry.cache_root.exists())
            self.assertEqual(cleanup.removed_roots, (entry.cache_root,))

    def test_refresh_preserves_local_only_state(self) -> None:
        """Confirm publish refreshes do not mark dirty local files safe.

        Inputs: a local-only cache entry plus a refreshed published file.
        Outputs: cleanup still skips the entry even when it is over the limit.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root, max_bytes=1)
            entry = _create_published_entry(config, "dirty-refresh", payload_size=128)
            record_analysis_cache_write(config, entry.cache_root)

            record_analysis_cache_refresh(
                config,
                local_root=entry.cache_root,
                publish_root=entry.publish_root,
            )
            result = enforce_analysis_cache_limit(config)

            self.assertTrue(entry.cache_root.exists())
            self.assertEqual(result.removed_roots, ())
            self.assertEqual(result.skipped_roots, (entry.cache_root,))

    def test_cleanup_skips_entry_when_directory_removal_fails(self) -> None:
        """Confirm cleanup failure on one entry does not abort cleanup.

        Inputs: two published entries and a removal failure for the oldest.
        Outputs: cleanup skips the failed entry and removes the next candidate.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root, max_bytes=1)
            failed = _create_published_entry(config, "failed", payload_size=2048)
            removable = _create_published_entry(config, "removable-after-fail", 2048)
            original_rmtree = shutil.rmtree

            def remove_or_fail(path: Path) -> None:
                if path == failed.cache_root:
                    raise OSError("busy")
                original_rmtree(path)

            with patch(
                "twopy.cache_inventory.shutil.rmtree",
                side_effect=remove_or_fail,
            ):
                result = enforce_analysis_cache_limit(config)

            self.assertTrue(failed.cache_root.exists())
            self.assertFalse(removable.cache_root.exists())
            self.assertEqual(result.skipped_roots, (failed.cache_root,))
            self.assertEqual(result.removed_roots, (removable.cache_root,))

    def test_cache_use_does_not_rescan_existing_entry_size(self) -> None:
        """Confirm reopening a cache entry only updates last-use policy state.

        Inputs: an inventory entry whose size has already been recorded.
        Outputs: use recording succeeds without walking the entry files again.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            config = _config(root, max_bytes=4096)
            entry = _create_published_entry(config, "used", 2048)

            with patch(
                "twopy.cache_inventory._directory_size",
                side_effect=AssertionError("cache use should not rescan size"),
            ):
                record_analysis_cache_use(config, entry.cache_root)

    def test_direct_publish_marks_cache_entry_removable(self) -> None:
        """Confirm Python copy-back records a successful publish sync.

        Inputs: one cache entry copied with ``copy_converted_files_to_publish``.
        Outputs: later cleanup can remove the entry when it is over the limit.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            config_path = _write_config(root, max_gb=0.001)
            config = load_config(config_path)
            entry = _create_cache_entry(
                config,
                "direct-publish",
                payload_size=2 * 1024 * 1024,
            )
            (entry.cache_root / "analysis_outputs.h5").unlink()

            result = copy_converted_files_to_publish(
                recording_data_path=entry.cache_root / RECORDING_DATA_FILENAME,
                movie_path=entry.cache_root / ALIGNED_MOVIE_FILENAME,
                source_session_dir=entry.source_root,
                config_path=config_path,
            )
            cleanup = enforce_analysis_cache_limit(config)

            self.assertIsNotNone(result)
            self.assertFalse(entry.cache_root.exists())
            self.assertEqual(cleanup.removed_roots, (entry.cache_root,))


@dataclass(frozen=True)
class _CacheEntryPaths:
    """Paths for one test cache entry."""

    cache_root: Path
    publish_root: Path
    source_root: Path


def _config(root: Path, *, max_bytes: int) -> TwopyConfig:
    """Return a test config with mirrored cache and publish roots."""
    data_root = root / "data"
    return TwopyConfig(
        database_path=root / "db",
        data_paths=(data_root,),
        database_access="copy",
        analysis_output=root / "publish",
        analysis_caching=True,
        analysis_cache_dir=root / "cache",
        analysis_cache_max_bytes=max_bytes,
    )


def _write_config(root: Path, *, max_gb: float) -> Path:
    """Write a config file for helpers that load config from disk."""
    config_path = root / "config.yml"
    config_path.write_text(
        f"database_path: {root / 'db'}\n"
        "data_paths:\n"
        f"  - {root / 'data'}\n"
        "database_access: copy\n"
        f"analysis_output: {root / 'publish'}\n"
        "analysis_caching: true\n"
        f"analysis_cache_dir: {root / 'cache'}\n"
        f"analysis_cache_max_gb: {max_gb}\n",
        encoding="utf-8",
    )
    return config_path


def _create_published_entry(
    config: TwopyConfig,
    name: str,
    payload_size: int,
) -> _CacheEntryPaths:
    """Create a cache entry with current publish copies."""
    entry = _create_cache_entry(config, name, payload_size=payload_size)
    shutil.copytree(entry.cache_root, entry.publish_root)
    record_analysis_cache_sync(
        config,
        local_root=entry.cache_root,
        publish_root=entry.publish_root,
    )
    return entry


def _create_cache_entry(
    config: TwopyConfig,
    name: str,
    *,
    payload_size: int,
) -> _CacheEntryPaths:
    """Create one converted-recording-like cache entry."""
    source_root = config.data_paths[0] / name
    cache_root = config.analysis_cache_dir / name
    publish_root = Path(config.analysis_output) / name
    source_root.mkdir(parents=True)
    cache_root.mkdir(parents=True)
    with h5py.File(cache_root / RECORDING_DATA_FILENAME, "w") as h5_file:
        h5_file.attrs["source_session_dir"] = str(source_root)
        h5_file.create_dataset("payload", data=b"metadata")
    (cache_root / ALIGNED_MOVIE_FILENAME).write_bytes(b"x" * payload_size)
    (cache_root / "analysis_outputs.h5").write_bytes(b"analysis")
    return _CacheEntryPaths(
        cache_root=cache_root.resolve(strict=False),
        publish_root=publish_root.resolve(strict=False),
        source_root=source_root.resolve(strict=False),
    )


def _entry_size(path: Path) -> int:
    """Return regular file bytes below one test entry."""
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
