"""Decorator used by workflow files.

``@workflow`` stores metadata on the function. Discovery imports workflow files
and reads that metadata, so reloads do not depend on global registration state.
"""

from collections.abc import Callable

from twopy.custom.types import CustomResult, WorkflowDeclaration

__all__ = [
    "WORKFLOW_DECLARATION_ATTRIBUTE",
    "declared_workflow",
    "workflow",
]

WORKFLOW_DECLARATION_ATTRIBUTE = "__twopy_custom_workflow__"


def workflow(
    *,
    id: str,
    name: str,
    version: str,
    description: str,
    params: type[object] | None = None,
    author: str | None = None,
    requires_rois: bool = True,
    output_prefix: str | None = None,
) -> Callable[[Callable[..., CustomResult]], Callable[..., CustomResult]]:
    """Mark a function as a twopy custom workflow.

    Args:
        id: Stable workflow id.
        name: GUI label.
        version: Workflow version.
        description: One-sentence description.
        params: Optional frozen dataclass for GUI controls.
        author: Optional workflow author string.
        requires_rois: Whether ROIs must exist before running.
        output_prefix: Optional safe output subfolder name.

    Returns:
        Decorator that attaches workflow metadata to the function.
    """

    def decorate(function: Callable[..., CustomResult]) -> Callable[..., CustomResult]:
        """Attach a workflow declaration to one function."""
        declaration = WorkflowDeclaration(
            id=id,
            name=name,
            version=version,
            description=description,
            params_type=params,
            author=author,
            requires_rois=requires_rois,
            output_prefix=output_prefix,
            function=function,
        )
        setattr(function, WORKFLOW_DECLARATION_ATTRIBUTE, declaration)
        return function

    return decorate


def declared_workflow(value: object) -> WorkflowDeclaration | None:
    """Return a workflow declaration attached to an object.

    Args:
        value: Imported module object to inspect.

    Returns:
        Workflow declaration, or ``None`` when the object is not a workflow.
    """
    declaration = getattr(value, WORKFLOW_DECLARATION_ATTRIBUTE, None)
    if isinstance(declaration, WorkflowDeclaration):
        return declaration
    return None
