"""Canonical temporary-directory helpers for tests.

Inputs: requests for isolated temporary folders.
Outputs: temporary directories exposed as canonical ``Path`` objects.

macOS can expose the same temporary directory through both ``/var`` and
``/private/var``. Tests compare paths often, so this helper resolves the folder
once at creation time and keeps assertions independent of that OS detail.
"""

import tempfile
from pathlib import Path
from types import TracebackType

__all__ = ["TemporaryDirectory", "temporary_directory"]


class TemporaryDirectory:
    """Temporary directory whose public path is canonicalized.

    Inputs: none.
    Outputs: a cleanup-owning temporary directory with ``Path`` and string
        accessors.

    The object supports both ``with temporary_directory() as path`` and explicit
    cleanup through ``cleanup()`` for unittest ``setUp``/``tearDown`` lifecycles.
    """

    def __init__(self) -> None:
        """Create one temporary directory and canonicalize its public path."""
        self._directory = tempfile.TemporaryDirectory()
        self.path = Path(self._directory.name).resolve(strict=False)
        self.name = str(self.path)

    def __enter__(self) -> Path:
        """Return the canonical temporary directory path."""
        return self.path

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Clean up the temporary directory when leaving a context manager."""
        self.cleanup()

    def cleanup(self) -> None:
        """Remove the temporary directory and its contents."""
        self._directory.cleanup()


def temporary_directory() -> TemporaryDirectory:
    """Return one cleanup-owning canonical temporary directory.

    Args:
        None.

    Returns:
        Temporary directory wrapper that yields a canonical ``Path``.
    """
    return TemporaryDirectory()
