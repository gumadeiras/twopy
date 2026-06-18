"""Check whether PyPI has a newer twopy release for the napari app.

Inputs: installed twopy version, PyPI JSON metadata, and local UI cache.
Outputs: an optional copyable update notice for the Metadata tab.

The app only tells users how to update. It does not choose or run an
environment manager, because users may launch twopy from pip, conda, or another
managed Python environment.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen

from packaging.version import InvalidVersion, Version

from twopy._version import __version__
from twopy.napari.state import (
    VersionCheckCache,
    read_version_check_cache,
    write_version_check_cache,
)

__all__ = [
    "DEFAULT_CHECK_INTERVAL",
    "PIP_UPDATE_COMMAND",
    "PublishedVersion",
    "UpdateNotice",
    "check_for_update",
    "fetch_pypi_published_version",
    "update_notice_for_versions",
]

DEFAULT_CHECK_INTERVAL = timedelta(days=1)
DEFAULT_TIMEOUT_SECONDS = 3.0
PYPI_PROJECT_NAME = "twopy"
PIP_UPDATE_COMMAND = "python -m pip install -U twopy"


class _ReadableResponse(Protocol):
    """HTTP response shape needed to read PyPI JSON bytes."""

    def read(self, size: int = -1, /) -> bytes:
        """Read response bytes."""


class _UrlOpenResponse(Protocol):
    """Context-manager response returned by ``urllib.request.urlopen``."""

    def __enter__(self) -> _ReadableResponse:
        """Return the readable HTTP response."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        """Close the response context."""


class UrlOpen(Protocol):
    """Callable shape for opening a URL with a timeout."""

    def __call__(self, url: str, *, timeout: float) -> _UrlOpenResponse:
        """Open one URL."""


@dataclass(frozen=True)
class PublishedVersion:
    """Package version reported by a package index.

    Args:
        version: Latest version text from package metadata.

    The value is checked before display so malformed metadata cannot produce an
    update notice.
    """

    version: str


@dataclass(frozen=True)
class UpdateNotice:
    """Copyable update command for a newer twopy release.

    Args:
        current_version: Installed twopy version.
        latest_version: Newer version available from the package index.
        command: Update command shown to the user.

    The Metadata tab shows text the user can copy. The command stays plain so
    twopy does not assume how the user's Python environment is managed.
    """

    current_version: str
    latest_version: str
    command: str = PIP_UPDATE_COMMAND

    @property
    def text(self) -> str:
        """Return the copyable user-facing update notice."""
        return (
            "Update available!\n"
            f"Latest version is {self.latest_version}.\n"
            "To update, run:\n"
            f"{self.command}"
        )


def check_for_update(
    *,
    current_version: str = __version__,
    project_name: str = PYPI_PROJECT_NAME,
    now: datetime | None = None,
    max_age: timedelta = DEFAULT_CHECK_INTERVAL,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    state_file: Path | None = None,
    fetch_latest: Callable[[], PublishedVersion] | None = None,
) -> UpdateNotice | None:
    """Return an update notice when a newer twopy release is known.

    Args:
        current_version: Installed twopy version.
        project_name: Package name to query from PyPI.
        now: Current UTC time override for tests.
        max_age: Minimum time between network checks.
        timeout_seconds: Network timeout for PyPI.
        state_file: Optional state file override for tests.
        fetch_latest: Optional no-argument fetcher used by tests.

    Returns:
        Update notice, or ``None`` when twopy is current or the check cannot
        prove that a newer final release exists.

    A fresh cache avoids one network request per launch only after it has
    already seen a newer version. If PyPI is unreachable and an older cache
    proves a newer version exists, twopy can still show the update command.
    """
    checked_now = _utc(now)
    cache = read_version_check_cache(state_file=state_file)
    if _cache_is_fresh(
        cache,
        now=checked_now,
        max_age=max_age,
        current_version=current_version,
    ):
        return _notice_from_cache(current_version, cache)

    try:
        published = (
            fetch_latest()
            if fetch_latest is not None
            else fetch_pypi_published_version(
                project_name=project_name,
                timeout_seconds=timeout_seconds,
            )
        )
    except (OSError, TimeoutError, URLError, ValueError, json.JSONDecodeError):
        return _notice_from_cache(current_version, cache)

    if not _is_valid_version_text(published.version):
        return _notice_from_cache(current_version, cache)

    write_version_check_cache(
        VersionCheckCache(
            checked_at=checked_now.isoformat(),
            latest_version=published.version,
        ),
        state_file=state_file,
    )
    return update_notice_for_versions(current_version, published.version)


def fetch_pypi_published_version(
    *,
    project_name: str = PYPI_PROJECT_NAME,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    open_url: UrlOpen = urlopen,
) -> PublishedVersion:
    """Fetch the latest package version from PyPI JSON metadata.

    Args:
        project_name: PyPI project name.
        timeout_seconds: Network timeout in seconds.
        open_url: URL opener override for tests.

    Returns:
        Published version.

    Raises:
        ValueError: If PyPI returns JSON without the expected string fields.
        OSError: If the network request fails.
    """
    url = f"https://pypi.org/pypi/{quote(project_name)}/json"
    with open_url(url, timeout=timeout_seconds) as response:
        payload = response.read()
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        msg = "PyPI response must be a JSON object."
        raise ValueError(msg)
    info = decoded.get("info")
    if not isinstance(info, dict):
        msg = "PyPI response is missing package info."
        raise ValueError(msg)
    version = info.get("version")
    if not isinstance(version, str):
        msg = "PyPI package info is missing a version."
        raise ValueError(msg)
    return PublishedVersion(version=version)


def update_notice_for_versions(
    current_version: str,
    latest_version: str,
) -> UpdateNotice | None:
    """Compare installed and published versions.

    Args:
        current_version: Installed twopy version.
        latest_version: Version reported by PyPI.

    Returns:
        Update notice only when ``latest_version`` is a newer final release.
    """
    try:
        current = Version(current_version)
        latest = Version(latest_version)
    except InvalidVersion:
        return None
    if latest.is_prerelease and not current.is_prerelease:
        return None
    if latest <= current:
        return None
    return UpdateNotice(current_version=current_version, latest_version=latest_version)


def _notice_from_cache(
    current_version: str,
    cache: VersionCheckCache | None,
) -> UpdateNotice | None:
    """Return an update notice from cached metadata when it is usable."""
    if cache is None:
        return None
    return update_notice_for_versions(current_version, cache.latest_version)


def _cache_is_fresh(
    cache: VersionCheckCache | None,
    *,
    now: datetime,
    max_age: timedelta,
    current_version: str,
) -> bool:
    """Return whether the cache is young enough to skip a network check."""
    if cache is None:
        return False
    checked_at = _parse_cached_datetime(cache.checked_at)
    if checked_at is None:
        return False
    try:
        current = Version(current_version)
        latest = Version(cache.latest_version)
    except InvalidVersion:
        return False
    if latest <= current:
        return False
    return now - checked_at <= max_age


def _parse_cached_datetime(value: str) -> datetime | None:
    """Parse one cached UTC timestamp."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return _utc(parsed)


def _utc(value: datetime | None) -> datetime:
    """Return a timezone-aware UTC datetime."""
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _is_valid_version_text(value: str) -> bool:
    """Return whether ``value`` is a PEP 440 version."""
    try:
        Version(value)
    except InvalidVersion:
        return False
    return True
