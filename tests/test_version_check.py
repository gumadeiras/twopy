"""Tests for the napari update notice.

Inputs: fake package-index metadata and isolated local state files.
Outputs: assertions that twopy shows only a safe pip command when newer.
"""

from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tests.tempdir import temporary_directory
from twopy.napari.state import read_version_check_cache
from twopy.napari.version_check import (
    PIP_UPDATE_COMMAND,
    PublishedVersion,
    check_for_update,
    fetch_pypi_published_version,
    update_notice_for_versions,
)


class VersionCheckTest(unittest.TestCase):
    """Tests package-version update notices for napari."""

    def test_newer_final_release_returns_only_pip_command(self) -> None:
        """Confirm a newer release produces the copyable update command."""
        notice = update_notice_for_versions("0.3.2", "0.3.3")

        self.assertIsNotNone(notice)
        assert notice is not None
        self.assertEqual(notice.command, PIP_UPDATE_COMMAND)

    def test_same_or_older_release_returns_no_notice(self) -> None:
        """Confirm current installs do not show an update command."""
        self.assertIsNone(update_notice_for_versions("0.3.2", "0.3.2"))
        self.assertIsNone(update_notice_for_versions("0.3.2", "0.3.1"))

    def test_prerelease_is_hidden_for_final_current_install(self) -> None:
        """Confirm prereleases do not nag users on stable installs."""
        self.assertIsNone(update_notice_for_versions("0.3.2", "0.3.3rc1"))

    def test_check_writes_successful_lookup_to_cache(self) -> None:
        """Confirm a network lookup stores the latest version."""
        with temporary_directory() as temp_dir:
            state_file = Path(temp_dir) / "state.json"
            now = datetime(2026, 6, 18, 12, tzinfo=UTC)

            notice = check_for_update(
                current_version="0.3.2",
                now=now,
                state_file=state_file,
                fetch_latest=lambda: PublishedVersion(
                    version="0.3.3",
                ),
            )

            self.assertIsNotNone(notice)
            assert notice is not None
            self.assertEqual(notice.command, PIP_UPDATE_COMMAND)
            cache = read_version_check_cache(state_file=state_file)
            self.assertIsNotNone(cache)
            assert cache is not None
            self.assertEqual(cache.checked_at, now.isoformat())
            self.assertEqual(cache.latest_version, "0.3.3")

    def test_fresh_cache_skips_network_lookup(self) -> None:
        """Confirm recent cache entries prevent a launch-time PyPI request."""
        with temporary_directory() as temp_dir:
            state_file = Path(temp_dir) / "state.json"
            now = datetime(2026, 6, 18, 12, tzinfo=UTC)
            _write_cache(state_file, checked_at=now, latest_version="0.3.3")

            notice = check_for_update(
                current_version="0.3.2",
                now=now + timedelta(hours=12),
                state_file=state_file,
                fetch_latest=_failing_fetch,
            )

            self.assertIsNotNone(notice)
            assert notice is not None
            self.assertEqual(notice.command, PIP_UPDATE_COMMAND)

    def test_stale_cache_is_used_when_network_fails(self) -> None:
        """Confirm old cache still helps when PyPI is unreachable."""
        with temporary_directory() as temp_dir:
            state_file = Path(temp_dir) / "state.json"
            now = datetime(2026, 6, 18, 12, tzinfo=UTC)
            _write_cache(
                state_file,
                checked_at=now - timedelta(days=7),
                latest_version="0.3.3",
            )

            notice = check_for_update(
                current_version="0.3.2",
                now=now,
                state_file=state_file,
                fetch_latest=_failing_fetch,
            )

            self.assertIsNotNone(notice)
            assert notice is not None
            self.assertEqual(notice.command, PIP_UPDATE_COMMAND)

    def test_cache_does_not_warn_after_user_updates(self) -> None:
        """Confirm cached metadata stops showing once the install catches up."""
        with temporary_directory() as temp_dir:
            state_file = Path(temp_dir) / "state.json"
            now = datetime(2026, 6, 18, 12, tzinfo=UTC)
            _write_cache(state_file, checked_at=now, latest_version="0.3.3")

            notice = check_for_update(
                current_version="0.3.3",
                now=now + timedelta(hours=1),
                state_file=state_file,
                fetch_latest=_failing_fetch,
            )

            self.assertIsNone(notice)

    def test_fetch_pypi_published_version_validates_response_shape(self) -> None:
        """Confirm PyPI JSON parsing accepts only expected string fields."""
        response = _FakeResponse(
            {
                "info": {
                    "version": "0.3.3",
                },
            },
        )

        published = fetch_pypi_published_version(open_url=response.open)

        self.assertEqual(published.version, "0.3.3")
        self.assertEqual(response.opened_url, "https://pypi.org/pypi/twopy/json")

    def test_fetch_pypi_published_version_rejects_missing_version(self) -> None:
        """Confirm malformed package metadata fails before display."""
        response = _FakeResponse({"info": {}})

        with self.assertRaisesRegex(ValueError, "version"):
            fetch_pypi_published_version(open_url=response.open)


def _write_cache(
    state_file: Path,
    *,
    checked_at: datetime,
    latest_version: str,
) -> None:
    """Write a version-check cache record for one test."""
    state_file.write_text(
        json.dumps(
            {
                "version_check_checked_at": checked_at.isoformat(),
                "version_check_latest_version": latest_version,
            },
        ),
        encoding="utf-8",
    )


def _failing_fetch() -> PublishedVersion:
    """Raise like an unavailable package index."""
    msg = "network unavailable"
    raise OSError(msg)


class _FakeResponse:
    """Small fake ``urlopen`` response for PyPI JSON tests."""

    def __init__(self, payload: dict[str, object]) -> None:
        """Store payload returned by the fake response."""
        self._payload = payload
        self.opened_url = ""

    def open(self, url: str, *, timeout: float) -> _FakeResponse:
        """Return this response and remember the requested URL."""
        _ = timeout
        self.opened_url = url
        return self

    def __enter__(self) -> _FakeResponse:
        """Return the readable response."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        """Close the fake response."""
        _ = exc_type, exc, traceback
        return None

    def read(self, size: int = -1, /) -> bytes:
        """Return encoded JSON bytes."""
        _ = size
        return json.dumps(self._payload).encode("utf-8")


if __name__ == "__main__":
    unittest.main()
