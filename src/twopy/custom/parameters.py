"""Turn workflow parameter dataclasses into simple GUI controls.

Only scalar fields are supported. Workflows can still read their own config
files, but the Custom tab stays small and easy to inspect.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints

from twopy.custom.types import CustomParameterValue

__all__ = [
    "CustomParameterSpec",
    "ParameterRole",
    "build_parameter_object",
    "parameter_values",
    "parameter_specs",
    "validate_parameter_type",
]

ParameterKind = Literal["bool", "int", "float", "str", "path", "choice"]
ParameterRole = Literal[
    "baseline_epoch",
    "comparison_epoch",
    "epoch",
    "epoch_window_start",
    "epoch_window_stop",
    "output_name",
    "response_metric",
    "response_window_start",
    "response_window_stop",
    "roi_limit",
    "roi_selector",
    "table_highlight_threshold",
]

_ROLE_KINDS: dict[ParameterRole, tuple[ParameterKind, ...]] = {
    "baseline_epoch": ("str",),
    "comparison_epoch": ("str",),
    "epoch": ("str",),
    "epoch_window_start": ("float",),
    "epoch_window_stop": ("float",),
    "output_name": ("path", "str"),
    "response_metric": ("str",),
    "response_window_start": ("float",),
    "response_window_stop": ("float",),
    "roi_limit": ("int",),
    "roi_selector": ("str",),
    "table_highlight_threshold": ("float",),
}
_ROLE_VALUES: dict[object, ParameterRole] = {role: role for role in _ROLE_KINDS}
_MISSING = object()


class CustomParameterError(ValueError):
    """Raised when workflow parameters cannot be shown or saved."""


@dataclass(frozen=True)
class CustomParameterSpec:
    """One workflow parameter shown in the Custom tab.

    Args:
        name: Dataclass field name.
        label: GUI label.
        kind: Control kind inferred from the type annotation.
        default: Default field value.
        description: Optional help text.
        role: Optional twopy-managed control role.
        choices: Choice labels for ``Literal`` or ``Enum`` fields.
        minimum: Optional numeric minimum.
        maximum: Optional numeric maximum.
        step: Optional numeric step.
        decimals: Optional decimal precision for float controls.

    Returns:
        Parameter specification for the GUI.
    """

    name: str
    label: str
    kind: ParameterKind
    default: object
    description: str
    role: ParameterRole | None = None
    choices: tuple[object, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    decimals: int | None = None


def validate_parameter_type(params_type: type[object] | None) -> None:
    """Check that a workflow parameter dataclass can be shown in the GUI.

    Args:
        params_type: Optional parameter dataclass.

    Returns:
        None.
    """
    if params_type is None:
        return
    if not is_dataclass(params_type):
        msg = "workflow params must be None or a dataclass type"
        raise CustomParameterError(msg)
    dataclass_params = getattr(params_type, "__dataclass_params__", None)
    if dataclass_params is None or not bool(dataclass_params.frozen):
        msg = "workflow params dataclass must be frozen"
        raise CustomParameterError(msg)
    parameter_specs(params_type)


def parameter_specs(params_type: type[object]) -> tuple[CustomParameterSpec, ...]:
    """Return Custom tab parameter specs for a frozen dataclass.

    Args:
        params_type: Frozen dataclass type.

    Returns:
        Parameter specs in dataclass field order.
    """
    type_hints = get_type_hints(params_type)
    specs: list[CustomParameterSpec] = []
    for field in fields(params_type):
        if field.default is MISSING:
            msg = f"parameter {field.name!r} must provide a default value"
            raise CustomParameterError(msg)
        annotation = type_hints.get(field.name)
        if annotation is None:
            msg = f"parameter {field.name!r} must have a type annotation"
            raise CustomParameterError(msg)
        kind, choices = _parameter_kind(annotation)
        role = _field_metadata_role(field.metadata)
        _validate_role_kind(field.name, role, kind)
        specs.append(
            CustomParameterSpec(
                name=field.name,
                label=_field_metadata_string(field.metadata, "label", field.name),
                kind=kind,
                default=field.default,
                description=_field_metadata_string(field.metadata, "description", ""),
                role=role,
                choices=choices,
                minimum=_field_metadata_float(field.metadata, "min"),
                maximum=_field_metadata_float(field.metadata, "max"),
                step=_field_metadata_float(field.metadata, "step"),
                decimals=_field_metadata_int(field.metadata, "decimals"),
            )
        )
    return tuple(specs)


def build_parameter_object(
    params_type: type[object] | None,
    values: Mapping[str, object],
) -> object | None:
    """Build a workflow parameter object from GUI values.

    Args:
        params_type: Optional frozen dataclass type.
        values: Field values keyed by dataclass field name.

    Returns:
        Dataclass instance or ``None``.
    """
    if params_type is None:
        return None
    specs = parameter_specs(params_type)
    coerced: dict[str, object] = {}
    for spec in specs:
        value = values[spec.name]
        coerced[spec.name] = _coerce_parameter_value(spec, value)
    return params_type(**coerced)


def parameter_values(params: object | None) -> dict[str, CustomParameterValue]:
    """Return parameter values that can be written to workflow metadata.

    Args:
        params: Parameter dataclass instance or ``None``.

    Returns:
        Dictionary safe for YAML and HDF5 metadata.
    """
    if params is None:
        return {}
    if not is_dataclass(params):
        msg = "custom workflow parameters must be a dataclass instance"
        raise CustomParameterError(msg)
    return {
        field.name: _metadata_parameter_value(getattr(params, field.name))
        for field in fields(params)
    }


def _parameter_kind(annotation: object) -> tuple[ParameterKind, tuple[object, ...]]:
    """Return the control kind and choices for one annotation."""
    origin = get_origin(annotation)
    if origin is Literal:
        choices = get_args(annotation)
        if len(choices) == 0:
            msg = "Literal parameters must include at least one choice"
            raise CustomParameterError(msg)
        for choice in choices:
            if not isinstance(choice, str | int | float | bool):
                msg = f"Unsupported Literal choice {choice!r}"
                raise CustomParameterError(msg)
        return "choice", tuple(choices)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return "choice", tuple(annotation)
    if annotation is bool:
        return "bool", ()
    if annotation is int:
        return "int", ()
    if annotation is float:
        return "float", ()
    if annotation is str:
        return "str", ()
    if annotation is Path:
        return "path", ()
    if origin in {UnionType, Union}:
        msg = "optional and union parameters are not supported in the GUI"
        raise CustomParameterError(msg)
    msg = f"Unsupported parameter type {annotation!r}"
    raise CustomParameterError(msg)


def _coerce_parameter_value(spec: CustomParameterSpec, value: object) -> object:
    """Convert one GUI value to the dataclass field value."""
    if spec.kind == "path":
        return Path(str(value)).expanduser()
    if spec.kind == "choice":
        return _coerce_choice(spec, value)
    if spec.kind == "bool":
        return bool(value)
    if spec.kind == "int":
        return _int_parameter_value(value)
    if spec.kind == "float":
        return _float_parameter_value(value)
    if spec.kind == "str":
        return str(value)
    msg = f"Unsupported parameter kind {spec.kind!r}"
    raise CustomParameterError(msg)


def _coerce_choice(spec: CustomParameterSpec, value: object) -> object:
    """Return the selected choice value for a choice parameter."""
    for choice in spec.choices:
        if value == choice:
            return choice
        if _choice_label(choice) == str(value):
            return choice
    msg = f"Unknown choice {value!r} for parameter {spec.name!r}"
    raise CustomParameterError(msg)


def _int_parameter_value(value: object) -> int:
    """Return one GUI value as an integer."""
    if isinstance(value, str | int | float):
        return int(value)
    msg = f"Expected an integer-compatible parameter value, got {value!r}"
    raise CustomParameterError(msg)


def _float_parameter_value(value: object) -> float:
    """Return one GUI value as a float."""
    if isinstance(value, str | int | float):
        return float(value)
    msg = f"Expected a float-compatible parameter value, got {value!r}"
    raise CustomParameterError(msg)


def _choice_label(value: object) -> str:
    """Return a display label for one parameter choice."""
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _field_metadata_string(
    metadata: Mapping[object, object],
    key: str,
    default: str,
) -> str:
    """Return one optional string field metadata value."""
    value = metadata.get(key, default)
    if isinstance(value, str):
        return value
    msg = f"parameter metadata {key!r} must be a string"
    raise CustomParameterError(msg)


def _field_metadata_float(
    metadata: Mapping[object, object],
    key: str,
) -> float | None:
    """Return one optional numeric field metadata value."""
    value = metadata.get(key, _MISSING)
    if value is _MISSING:
        return None
    if isinstance(value, int | float):
        return float(value)
    msg = f"parameter metadata {key!r} must be numeric"
    raise CustomParameterError(msg)


def _field_metadata_int(
    metadata: Mapping[object, object],
    key: str,
) -> int | None:
    """Return one optional integer field metadata value."""
    value = metadata.get(key, _MISSING)
    if value is _MISSING:
        return None
    if type(value) is int and value >= 0:
        return value
    msg = f"parameter metadata {key!r} must be a non-negative integer"
    raise CustomParameterError(msg)


def _field_metadata_role(
    metadata: Mapping[object, object],
) -> ParameterRole | None:
    """Return one optional standard parameter role."""
    value = metadata.get("twopy_role", None)
    if value is None:
        return None
    role = _ROLE_VALUES.get(value)
    if role is not None:
        return role
    allowed = ", ".join(repr(role) for role in _ROLE_KINDS)
    msg = f"parameter metadata 'twopy_role' must be one of {allowed}"
    raise CustomParameterError(msg)


def _validate_role_kind(
    name: str,
    role: ParameterRole | None,
    kind: ParameterKind,
) -> None:
    """Check that a standard role uses the right field type."""
    if role is not None:
        _require_role_kind(name, role, kind, _ROLE_KINDS[role])


def _require_role_kind(
    name: str,
    role: ParameterRole,
    kind: ParameterKind,
    allowed: tuple[ParameterKind, ...],
) -> None:
    """Raise a clear error when a role is attached to the wrong type."""
    if kind in allowed:
        return
    allowed_text = " or ".join(allowed)
    msg = f"parameter {name!r} role {role!r} requires {allowed_text}; got {kind}"
    raise CustomParameterError(msg)


def _metadata_parameter_value(value: object) -> CustomParameterValue:
    """Return one parameter value in a metadata-safe form."""
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    msg = f"Unsupported custom workflow parameter value {value!r}."
    raise ValueError(msg)
