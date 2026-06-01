"""Workflow declaration objects used by discovery and execution.

Inputs: metadata supplied by the ``@workflow`` decorator and validated source
files discovered at run time.
Outputs: immutable workflow definitions that the Custom tab and script runner
can execute.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from twopy.custom.results import CustomResult

__all__ = [
    "CustomWorkflow",
    "CustomWorkflowFunction",
    "WorkflowDeclaration",
]


class CustomWorkflowFunction(Protocol):
    """Function shape accepted by custom workflow discovery."""

    def __call__(self, *args: object) -> CustomResult:
        """Run the workflow.

        Args:
            *args: Validated workflow call arguments.

        Returns:
            Workflow result.
        """
        ...


@dataclass(frozen=True)
class WorkflowDeclaration:
    """Raw metadata attached by the ``@workflow`` decorator.

    Args:
        id: Stable workflow id.
        name: GUI label.
        version: Workflow version.
        description: One-sentence description.
        params_type: Optional frozen dataclass for GUI controls.
        author: Optional workflow author string.
        requires_rois: Whether ROIs must exist before running.
        output_prefix: Optional output subfolder name.
        function: Workflow function.

    Returns:
        Declaration attached to a workflow function.
    """

    id: str
    name: str
    version: str
    description: str
    params_type: type[object] | None
    author: str | None
    requires_rois: bool
    output_prefix: str | None
    function: CustomWorkflowFunction


@dataclass(frozen=True)
class CustomWorkflow:
    """One workflow that passed discovery checks.

    Args:
        id: Stable workflow id.
        name: GUI label.
        version: Workflow version.
        description: One-sentence description.
        params_type: Optional frozen dataclass for GUI controls.
        author: Optional workflow author string.
        requires_rois: Whether ROIs must exist before running.
        output_prefix: Safe output subfolder name.
        function: Workflow function.
        source_path: Python file that defines the workflow.
        source_hash: SHA-256 hash of the source file at discovery time.

    Returns:
        Workflow definition shown in the Custom tab.
    """

    id: str
    name: str
    version: str
    description: str
    params_type: type[object] | None
    author: str | None
    requires_rois: bool
    output_prefix: str
    function: CustomWorkflowFunction
    source_path: Path
    source_hash: str
