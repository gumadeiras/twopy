"""Public API for twopy custom workflows.

Workflow files should import from this module instead of twopy internals. twopy
validates workflow metadata before showing a workflow in the napari Custom tab.
"""

from twopy.analysis.responses import finite_mean_and_sem
from twopy.custom.context import CustomEpoch, CustomRunContext
from twopy.custom.discovery import (
    CustomWorkflowError,
    WorkflowDiscoveryResult,
    discover_custom_workflows,
)
from twopy.custom.metadata import CustomRecordingMetadata
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
    CustomWorkflowProvenance,
    custom_result_artifact_paths,
    provenance_sidecar_path,
    validate_custom_result,
    workflow_provenance,
    write_result_provenance,
    write_workflow_provenance_for_path,
)
from twopy.custom.registry import workflow
from twopy.custom.results import (
    CustomLineBand,
    CustomLinePlot,
    CustomResult,
    CustomTable,
)
from twopy.custom.runner import (
    CustomWorkflowRun,
    run_custom_workflow,
)
from twopy.custom.workflows import CustomWorkflow

__all__ = [
    "CustomEpoch",
    "CustomLineBand",
    "CustomLinePlot",
    "CustomRecordingMetadata",
    "CustomParameterSpec",
    "CustomResult",
    "CustomRunContext",
    "CustomTable",
    "CustomWorkflow",
    "CustomWorkflowError",
    "CustomWorkflowProvenance",
    "CustomWorkflowRun",
    "ParameterRole",
    "WorkflowDiscoveryResult",
    "build_parameter_object",
    "custom_result_artifact_paths",
    "discover_custom_workflows",
    "finite_mean_and_sem",
    "native_custom_workflow_paths",
    "parameter_specs",
    "parameter_values",
    "provenance_sidecar_path",
    "run_custom_workflow",
    "validate_custom_result",
    "validate_parameter_type",
    "workflow",
    "workflow_provenance",
    "write_result_provenance",
    "write_workflow_provenance_for_path",
]
