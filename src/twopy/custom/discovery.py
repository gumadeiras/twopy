"""Find custom workflow files and keep invalid ones out of the Run button.

Discovery is strict on purpose. Bad workflow files are reported in the Custom
tab, but they are not listed as runnable workflows.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import re
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import get_type_hints

from twopy.custom.parameters import CustomParameterError, validate_parameter_type
from twopy.custom.registry import declared_workflow
from twopy.custom.types import (
    CustomResult,
    CustomRunContext,
    CustomWorkflow,
    WorkflowDeclaration,
)

__all__ = [
    "CustomWorkflowError",
    "WorkflowDiscoveryResult",
    "discover_custom_workflows",
]

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_VERSION_PATTERN = re.compile(r"^[0-9]+[.][0-9]+$")


@dataclass(frozen=True)
class CustomWorkflowError:
    """One workflow file that could not be loaded.

    Args:
        path: File or configured path that failed.
        message: Rejection reason shown in the Custom tab.

    Returns:
        Error record for GUI display.
    """

    path: Path
    message: str


@dataclass(frozen=True)
class WorkflowDiscoveryResult:
    """Loaded workflows plus files that were rejected.

    Args:
        workflows: Workflows available to run.
        errors: Import, validation, duplicate, or path errors.

    Returns:
        Discovery result for the Custom tab.
    """

    workflows: tuple[CustomWorkflow, ...]
    errors: tuple[CustomWorkflowError, ...]


def discover_custom_workflows(paths: tuple[Path, ...]) -> WorkflowDiscoveryResult:
    """Load valid workflows from configured files and directories.

    Args:
        paths: Python files or directories to scan.

    Returns:
        Runnable workflows plus rejection errors.
    """
    workflows: list[CustomWorkflow] = []
    errors: list[CustomWorkflowError] = []
    for workflow_path in paths:
        for file_path in _workflow_files(workflow_path, errors):
            imported = _import_workflow_file(file_path)
            if isinstance(imported, CustomWorkflowError):
                errors.append(imported)
                continue
            source_hash = _source_hash(file_path)
            declarations = _workflow_declarations(imported)
            if len(declarations) == 0:
                errors.append(
                    CustomWorkflowError(file_path, "no @workflow declarations found")
                )
                continue
            for declaration in declarations:
                try:
                    workflows.append(
                        _validate_workflow_declaration(
                            declaration,
                            source_path=file_path,
                            source_hash=source_hash,
                        )
                    )
                except ValueError as error:
                    errors.append(CustomWorkflowError(file_path, str(error)))
    workflows, duplicate_errors = _remove_duplicate_workflows(tuple(workflows))
    errors.extend(duplicate_errors)
    return WorkflowDiscoveryResult(
        workflows=tuple(sorted(workflows, key=lambda workflow: workflow.name.lower())),
        errors=tuple(errors),
    )


def _workflow_files(
    configured_path: Path,
    errors: list[CustomWorkflowError],
) -> tuple[Path, ...]:
    """Return Python workflow files from one configured path."""
    path = configured_path.expanduser()
    if not path.exists():
        errors.append(CustomWorkflowError(path, "path does not exist"))
        return ()
    if path.is_file():
        if path.suffix != ".py":
            errors.append(CustomWorkflowError(path, "workflow file must end in .py"))
            return ()
        return (path,)
    if not path.is_dir():
        errors.append(CustomWorkflowError(path, "path is not a file or directory"))
        return ()
    return tuple(
        sorted(
            child
            for child in path.iterdir()
            if (
                child.is_file()
                and child.suffix == ".py"
                and child.name != "__init__.py"
                and not child.name.startswith("_")
            )
        )
    )


def _import_workflow_file(path: Path) -> ModuleType | CustomWorkflowError:
    """Import a workflow file without reusing another workflow module name."""
    module_name = f"twopy_custom_{hashlib.sha256(str(path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return CustomWorkflowError(path, "could not create import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    workflow_parent = str(path.parent)
    added_parent = workflow_parent not in sys.path
    if added_parent:
        sys.path.insert(0, workflow_parent)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(module_name, None)
        return CustomWorkflowError(path, f"import failed: {error}")
    finally:
        if added_parent:
            with suppress(ValueError):
                sys.path.remove(workflow_parent)
    return module


def _workflow_declarations(module: ModuleType) -> tuple[WorkflowDeclaration, ...]:
    """Return ``@workflow`` declarations found in one module."""
    declarations: list[WorkflowDeclaration] = []
    for value in vars(module).values():
        declaration = declared_workflow(value)
        if declaration is not None and declaration not in declarations:
            declarations.append(declaration)
    return tuple(declarations)


def _validate_workflow_declaration(
    declaration: WorkflowDeclaration,
    *,
    source_path: Path,
    source_hash: str,
) -> CustomWorkflow:
    """Turn one declaration into a runnable workflow or raise a clear error."""
    for field_name, value in (
        ("id", declaration.id),
        ("name", declaration.name),
        ("version", declaration.version),
        ("description", declaration.description),
    ):
        _validate_required_text(field_name, value)
    if _ID_PATTERN.fullmatch(declaration.id) is None:
        msg = f"workflow id {declaration.id!r} must be slug-like"
        raise ValueError(msg)
    if _VERSION_PATTERN.fullmatch(declaration.version) is None:
        msg = f"workflow version {declaration.version!r} must look like MAJOR.MINOR"
        raise ValueError(msg)
    if not isinstance(declaration.requires_rois, bool):
        msg = "requires_rois must be true or false"
        raise ValueError(msg)
    output_prefix = declaration.output_prefix or declaration.id
    if _ID_PATTERN.fullmatch(output_prefix) is None:
        msg = f"workflow output_prefix {output_prefix!r} must be slug-like"
        raise ValueError(msg)
    try:
        validate_parameter_type(declaration.params_type)
    except CustomParameterError as error:
        msg = f"invalid workflow parameters: {error}"
        raise ValueError(msg) from error
    _validate_workflow_signature(declaration)
    return CustomWorkflow(
        id=declaration.id,
        name=declaration.name,
        version=declaration.version,
        description=declaration.description,
        params_type=declaration.params_type,
        author=declaration.author,
        requires_rois=declaration.requires_rois,
        output_prefix=output_prefix,
        function=declaration.function,
        source_path=source_path,
        source_hash=source_hash,
    )


def _validate_workflow_signature(declaration: WorkflowDeclaration) -> None:
    """Confirm a workflow function has the supported signature."""
    signature = inspect.signature(declaration.function)
    parameters = tuple(signature.parameters.values())
    expected_count = 2 if declaration.params_type is not None else 1
    if len(parameters) != expected_count:
        signature_text = "run(ctx: CustomRunContext) -> CustomResult"
        if declaration.params_type is not None:
            signature_text = (
                "run(ctx: CustomRunContext, params: Params) -> CustomResult"
            )
        msg = f"workflow function signature must be {signature_text}"
        raise ValueError(msg)
    if any(
        parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in parameters
    ):
        msg = "workflow function parameters must be positional-or-keyword"
        raise ValueError(msg)
    type_hints = get_type_hints(declaration.function)
    if type_hints.get(parameters[0].name) is not CustomRunContext:
        msg = "workflow first parameter must be annotated as CustomRunContext"
        raise ValueError(msg)
    if declaration.params_type is not None:
        params_annotation = type_hints.get(parameters[1].name)
        if params_annotation is not declaration.params_type:
            msg = (
                "workflow params parameter must be annotated with its params dataclass"
            )
            raise ValueError(msg)
    if type_hints.get("return") is not CustomResult:
        msg = "workflow return annotation must be CustomResult"
        raise ValueError(msg)


def _validate_required_text(field_name: str, value: object) -> None:
    """Confirm one required text metadata field is non-empty."""
    if not isinstance(value, str) or value.strip() == "":
        msg = f"workflow {field_name} must be a non-empty string"
        raise ValueError(msg)


def _source_hash(path: Path) -> str:
    """Return the SHA-256 hash of a workflow source file."""
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_duplicate_workflows(
    workflows: tuple[CustomWorkflow, ...],
) -> tuple[list[CustomWorkflow], tuple[CustomWorkflowError, ...]]:
    """Reject duplicate workflow id/version pairs."""
    by_key: dict[tuple[str, str], list[CustomWorkflow]] = {}
    for workflow in workflows:
        by_key.setdefault((workflow.id, workflow.version), []).append(workflow)
    duplicate_keys = {key for key, values in by_key.items() if len(values) > 1}
    if len(duplicate_keys) == 0:
        return list(workflows), ()
    errors: list[CustomWorkflowError] = []
    kept = [
        workflow
        for workflow in workflows
        if (workflow.id, workflow.version) not in duplicate_keys
    ]
    for key in sorted(duplicate_keys):
        duplicated = by_key[key]
        paths = ", ".join(str(workflow.source_path) for workflow in duplicated)
        errors.append(
            CustomWorkflowError(
                duplicated[0].source_path,
                f"duplicate workflow id/version {key[0]} {key[1]} in {paths}",
            )
        )
    return kept, tuple(errors)
