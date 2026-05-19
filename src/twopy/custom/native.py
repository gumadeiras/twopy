"""Expose workflow files that ship with twopy.

Native workflows use the same discovery and validation path as local workflow
files. The Custom tab can treat both the same way.
"""

from pathlib import Path

import twopy.custom.native_workflows as native_workflows

__all__ = ["native_custom_workflow_paths"]


def native_custom_workflow_paths() -> tuple[Path, ...]:
    """Return the package folder that contains twopy workflows.

    Args:
        None.

    Returns:
        Workflow directory scanned before machine-local workflow paths.
    """
    package_file = native_workflows.__file__
    if package_file is None:
        msg = "twopy native workflow package has no filesystem path"
        raise RuntimeError(msg)
    return (Path(package_file).parent,)
