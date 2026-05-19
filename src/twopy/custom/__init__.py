"""Public API for twopy custom workflows.

Workflow files should import from this module instead of twopy internals. twopy
validates workflow metadata before showing a workflow in the napari Custom tab.
"""

from twopy.custom.discovery import (
    CustomWorkflowError,
    WorkflowDiscoveryResult,
    discover_custom_workflows,
)
from twopy.custom.native import native_custom_workflow_paths
from twopy.custom.parameters import (
    CustomParameterSpec,
    ParameterRole,
    build_parameter_object,
    parameter_specs,
    parameter_values,
    validate_parameter_type,
)
from twopy.custom.provenance import (
    custom_result_artifact_paths,
    provenance_sidecar_path,
    validate_custom_result,
    write_result_provenance,
    write_workflow_provenance_for_path,
)
from twopy.custom.registry import workflow
from twopy.custom.types import (
    CustomEpoch,
    CustomLinePlot,
    CustomResult,
    CustomRunContext,
    CustomTable,
    CustomWorkflow,
    CustomWorkflowProvenance,
    workflow_provenance,
)

__all__ = [
    "CustomEpoch",
    "CustomLinePlot",
    "CustomParameterSpec",
    "CustomResult",
    "CustomRunContext",
    "CustomTable",
    "CustomWorkflow",
    "CustomWorkflowError",
    "CustomWorkflowProvenance",
    "ParameterRole",
    "WorkflowDiscoveryResult",
    "build_parameter_object",
    "custom_result_artifact_paths",
    "discover_custom_workflows",
    "native_custom_workflow_paths",
    "parameter_specs",
    "parameter_values",
    "provenance_sidecar_path",
    "validate_custom_result",
    "validate_parameter_type",
    "workflow",
    "workflow_provenance",
    "write_result_provenance",
    "write_workflow_provenance_for_path",
]
