"""Recording metadata exposed to custom workflow code.

Inputs: loaded converted recording objects.
Outputs: immutable metadata snapshots with typed lookup helpers.

The custom workflow API exposes this snapshot instead of the internal
``RecordingData`` object so workflow code can inspect metadata without mutating
loaded recording state.
"""

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from twopy.converted import RecordingData

__all__ = ["CustomRecordingMetadata"]


@dataclass(frozen=True)
class CustomRecordingMetadata:
    """Converted recording metadata exposed to custom workflows.

    Args:
        recording_path: Converted ``recording_data.h5`` file used by the run.
        source_session_dir: Source microscope session directory.
        run: Run-level metadata such as rig name, genotype, and hemisphere.
        acquisition: Microscope acquisition metadata such as frame rate and zoom.
        synchronization: Photodiode and clock-alignment metadata.
        stimulus_parameters: Per-epoch stimulus parameter dictionaries.
        stimulus_function_lookup: Mapping from stimulus ids to source function names.
        stimulus_specific_columns: Metadata for converted stimulus-specific columns.

    Returns:
        Immutable metadata snapshot with typed read helpers.

    Workflows use this object to inspect converted metadata without depending on
    the internal ``RecordingData`` object or mutating the loaded recording.
    """

    recording_path: Path
    source_session_dir: Path
    run: Mapping[str, object]
    acquisition: Mapping[str, object]
    synchronization: Mapping[str, str]
    stimulus_parameters: tuple[Mapping[str, object], ...]
    stimulus_function_lookup: Mapping[str, str]
    stimulus_specific_columns: Mapping[str, Mapping[str, object]]

    def value(self, section: str, key: str, default: object = None) -> object:
        """Return one raw metadata value from a named section.

        Args:
            section: Metadata section name, such as ``"run"`` or
                ``"acquisition"``.
            key: Metadata key inside that section.
            default: Value returned when the key is absent.

        Returns:
            Raw metadata value or ``default``.
        """
        return self._section(section).get(key, default)

    def text(
        self,
        section: str,
        key: str,
        default: str | None = None,
    ) -> str | None:
        """Return one metadata value as text.

        Args:
            section: Metadata section name.
            key: Metadata key inside that section.
            default: Value returned when the key is absent.

        Returns:
            Text value, or ``default`` when absent.
        """
        value = self.value(section, key, default)
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def float(
        self,
        section: str,
        key: str,
        default: builtins.float | None = None,
    ) -> builtins.float | None:
        """Return one metadata value as a float.

        Args:
            section: Metadata section name.
            key: Metadata key inside that section.
            default: Value returned when the key is absent.

        Returns:
            Floating-point value, or ``default`` when absent.
        """
        value = self.value(section, key, default)
        if value is None:
            return None
        if isinstance(value, bool):
            msg = f"Metadata field {section}.{key} is boolean, not numeric."
            raise ValueError(msg)
        if not isinstance(value, str | bytes | int | builtins.float):
            msg = f"Metadata field {section}.{key} cannot be read as a float."
            raise ValueError(msg)
        try:
            return builtins.float(value)
        except (TypeError, ValueError) as error:
            msg = f"Metadata field {section}.{key} cannot be read as a float."
            raise ValueError(msg) from error

    def int(
        self,
        section: str,
        key: str,
        default: builtins.int | None = None,
    ) -> builtins.int | None:
        """Return one metadata value as an integer.

        Args:
            section: Metadata section name.
            key: Metadata key inside that section.
            default: Value returned when the key is absent.

        Returns:
            Integer value, or ``default`` when absent.
        """
        value = self.value(section, key, default)
        if value is None:
            return None
        if isinstance(value, bool):
            msg = f"Metadata field {section}.{key} is boolean, not integer."
            raise ValueError(msg)
        if not isinstance(value, str | bytes | int | builtins.float):
            msg = f"Metadata field {section}.{key} cannot be read as an integer."
            raise ValueError(msg)
        try:
            as_float = builtins.float(value)
        except (TypeError, ValueError) as error:
            msg = f"Metadata field {section}.{key} cannot be read as an integer."
            raise ValueError(msg) from error
        if not as_float.is_integer():
            msg = f"Metadata field {section}.{key} is not an integer."
            raise ValueError(msg)
        return builtins.int(as_float)

    def _section(self, section: str) -> Mapping[str, object]:
        """Return a key-value metadata section by name."""
        if section == "run":
            return self.run
        if section == "acquisition":
            return self.acquisition
        if section == "synchronization":
            return self.synchronization
        if section == "stimulus_function_lookup":
            return self.stimulus_function_lookup
        if section == "stimulus_specific_columns":
            return self.stimulus_specific_columns
        msg = f"Unknown recording metadata section {section!r}."
        raise ValueError(msg)

    @classmethod
    def _from_recording(cls, recording: RecordingData) -> CustomRecordingMetadata:
        """Return a public metadata snapshot for an internal recording object."""
        return cls(
            recording_path=recording.path,
            source_session_dir=recording.source_session_dir,
            run=_readonly_mapping(recording.run_metadata),
            acquisition=_readonly_mapping(recording.acquisition_metadata),
            synchronization=_readonly_string_mapping(
                recording.synchronization_metadata
            ),
            stimulus_parameters=tuple(
                _readonly_mapping(parameters)
                for parameters in recording.stimulus_parameters
            ),
            stimulus_function_lookup=_readonly_string_mapping(
                recording.stimulus_function_lookup,
            ),
            stimulus_specific_columns=_readonly_nested_mapping(
                recording.stimulus_specific_columns,
            ),
        )


def _readonly_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    """Return a read-only recursive copy of one metadata mapping."""
    return MappingProxyType(
        {key: _readonly_metadata_value(value) for key, value in values.items()},
    )


def _readonly_string_mapping(values: Mapping[str, str]) -> Mapping[str, str]:
    """Return a read-only shallow copy of one string metadata mapping."""
    return MappingProxyType(dict(values))


def _readonly_nested_mapping(
    values: Mapping[str, Mapping[str, object]],
) -> Mapping[str, Mapping[str, object]]:
    """Return a read-only copy of nested stimulus metadata mappings."""
    return MappingProxyType(
        {key: _readonly_mapping(nested) for key, nested in values.items()},
    )


def _readonly_metadata_value(value: object) -> object:
    """Return one metadata value with mutable containers frozen."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _readonly_metadata_value(nested)
                for key, nested in value.items()
            },
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(_readonly_metadata_value(item) for item in value)
    return value
