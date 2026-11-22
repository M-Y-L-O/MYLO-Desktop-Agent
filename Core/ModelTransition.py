from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple


TRANSITION_SCHEMA_VERSION = "1.0"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def state_dict_fingerprint(state: Mapping[str, Any]) -> str:
    manifest = {}
    for key in sorted(state):
        tensor = state[key]
        value = tensor.detach().cpu().contiguous()
        try:
            payload = value.numpy().tobytes(order="C")
        except (TypeError, RuntimeError):
            payload = bytes(value.untyped_storage())
        manifest[str(key)] = {
            "shape": [int(dimension) for dimension in value.shape],
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return canonical_fingerprint(manifest)


@dataclass(frozen=True)
class ParameterMapping:
    source_key: Optional[str]
    target_key: Optional[str]
    mapping_kind: str
    key_resolution: str
    source_shape: Tuple[int, ...] = ()
    target_shape: Tuple[int, ...] = ()
    source_dtype: Optional[str] = None
    target_dtype: Optional[str] = None
    copied_elements: int = 0
    source_elements: int = 0
    target_elements: int = 0
    source_elements_used: Optional[int] = None
    target_elements_initialized: Optional[int] = None
    note: Optional[str] = None

    @property
    def mapped_source_elements(self) -> int:
        return int(
            self.source_elements_used
            if self.source_elements_used is not None
            else self.copied_elements
        )

    @property
    def mapped_target_elements(self) -> int:
        return int(
            self.target_elements_initialized
            if self.target_elements_initialized is not None
            else self.copied_elements
        )

    @property
    def source_coverage(self) -> float:
        if self.source_elements <= 0:
            return 0.0
        return min(1.0, self.mapped_source_elements / self.source_elements)

    @property
    def target_coverage(self) -> float:
        if self.target_elements <= 0:
            return 0.0
        return min(1.0, self.mapped_target_elements / self.target_elements)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_key": self.source_key,
            "target_key": self.target_key,
            "mapping_kind": self.mapping_kind,
            "key_resolution": self.key_resolution,
            "source_shape": list(self.source_shape),
            "target_shape": list(self.target_shape),
            "source_dtype": self.source_dtype,
            "target_dtype": self.target_dtype,
            "copied_elements": self.copied_elements,
            "source_elements": self.source_elements,
            "target_elements": self.target_elements,
            "source_elements_used": self.source_elements_used,
            "target_elements_initialized": self.target_elements_initialized,
            "source_coverage": round(self.source_coverage, 8),
            "target_coverage": round(self.target_coverage, 8),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParameterMapping":
        fields = {
            "source_key": data.get("source_key"),
            "target_key": data.get("target_key"),
            "mapping_kind": str(data.get("mapping_kind", "unknown")),
            "key_resolution": str(data.get("key_resolution", "unknown")),
            "source_shape": tuple(data.get("source_shape", ())),
            "target_shape": tuple(data.get("target_shape", ())),
            "source_dtype": data.get("source_dtype"),
            "target_dtype": data.get("target_dtype"),
            "copied_elements": int(data.get("copied_elements", 0)),
            "source_elements": int(data.get("source_elements", 0)),
            "target_elements": int(data.get("target_elements", 0)),
            "source_elements_used": data.get("source_elements_used"),
            "target_elements_initialized": data.get("target_elements_initialized"),
            "note": data.get("note"),
        }
        return cls(**fields)


@dataclass
class StateMappingReport:
    parameter_mappings: List[ParameterMapping] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    policy: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        source_records = [
            mapping for mapping in self.parameter_mappings if mapping.source_key is not None
        ]
        target_records = [
            mapping for mapping in self.parameter_mappings if mapping.target_key is not None
        ]
        copied_records = [
            mapping
            for mapping in self.parameter_mappings
            if mapping.mapping_kind
            in {"exact_copy", "dtype_cast_copy", "overlap_copy"}
        ]

        source_elements = sum(mapping.source_elements for mapping in source_records)
        target_elements = sum(mapping.target_elements for mapping in target_records)
        copied_elements = sum(mapping.copied_elements for mapping in copied_records)
        source_elements_used = sum(
            mapping.mapped_source_elements for mapping in source_records
        )
        target_elements_initialized = sum(
            mapping.mapped_target_elements for mapping in target_records
        )

        return {
            "source_tensor_count": len(source_records),
            "target_tensor_count": len(target_records),
            "exact_tensor_count": sum(
                mapping.mapping_kind == "exact_copy" for mapping in self.parameter_mappings
            ),
            "overlap_tensor_count": sum(
                mapping.mapping_kind == "overlap_copy" for mapping in self.parameter_mappings
            ),
            "dtype_cast_tensor_count": sum(
                mapping.mapping_kind == "dtype_cast_copy"
                for mapping in self.parameter_mappings
            ),
            "initialized_target_tensor_count": sum(
                mapping.mapping_kind == "initialized_target"
                for mapping in self.parameter_mappings
            ),
            "unmapped_source_tensor_count": sum(
                mapping.mapping_kind == "unmapped_source"
                for mapping in self.parameter_mappings
            ),
            "copied_elements": copied_elements,
            "source_elements_used": source_elements_used,
            "target_elements_initialized": target_elements_initialized,
            "source_elements": source_elements,
            "target_elements": target_elements,
            "source_coverage": round(
                source_elements_used / source_elements if source_elements else 0.0, 8
            ),
            "target_coverage": round(
                target_elements_initialized / target_elements
                if target_elements
                else 0.0,
                8,
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": TRANSITION_SCHEMA_VERSION,
            "policy": dict(self.policy),
            "summary": self.summary(),
            "parameter_mappings": [
                mapping.to_dict() for mapping in self.parameter_mappings
            ],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StateMappingReport":
        return cls(
            parameter_mappings=[
                ParameterMapping.from_dict(mapping)
                for mapping in data.get("parameter_mappings", [])
            ],
            warnings=list(data.get("warnings", [])),
            policy=dict(data.get("policy", {})),
        )


@dataclass
class TransitionEvidence:
    static: Dict[str, Any] = field(default_factory=dict)
    execution: Dict[str, Any] = field(default_factory=dict)
    recovery: Dict[str, Any] = field(default_factory=dict)
    collection_cost: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": TRANSITION_SCHEMA_VERSION,
            "static": _jsonable(self.static),
            "execution": _jsonable(self.execution),
            "recovery": _jsonable(self.recovery),
            "collection_cost": _jsonable(self.collection_cost),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TransitionEvidence":
        return cls(
            static=dict(data.get("static", {})),
            execution=dict(data.get("execution", {})),
            recovery=dict(data.get("recovery", {})),
            collection_cost=dict(data.get("collection_cost", {})),
            warnings=list(data.get("warnings", [])),
        )


@dataclass
class ModelTransition:
    parent_revision_id: str
    child_revision_id: str
    transformation: Dict[str, Any]
    parent_architecture_fingerprint: str
    child_architecture_fingerprint: str
    state_mapping: Optional[StateMappingReport] = None
    evidence: Optional[TransitionEvidence] = None
    decision: Dict[str, Any] = field(default_factory=dict)
    outcome: Dict[str, Any] = field(default_factory=dict)
    status: str = "created"
    transition_id: str = ""

    def __post_init__(self) -> None:
        if not self.transition_id:
            identity = {
                "parent_revision_id": self.parent_revision_id,
                "child_revision_id": self.child_revision_id,
                "transformation": self.transformation,
                "parent_architecture_fingerprint": self.parent_architecture_fingerprint,
                "child_architecture_fingerprint": self.child_architecture_fingerprint,
            }
            self.transition_id = canonical_fingerprint(identity)

    @classmethod
    def create(
        cls,
        parent_revision_id: str,
        child_revision_id: str,
        transformation: Mapping[str, Any],
        parent_architecture: Any,
        child_architecture: Any,
    ) -> "ModelTransition":
        return cls(
            parent_revision_id=parent_revision_id,
            child_revision_id=child_revision_id,
            transformation=dict(transformation),
            parent_architecture_fingerprint=canonical_fingerprint(parent_architecture),
            child_architecture_fingerprint=canonical_fingerprint(child_architecture),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": TRANSITION_SCHEMA_VERSION,
            "transition_id": self.transition_id,
            "parent_revision_id": self.parent_revision_id,
            "child_revision_id": self.child_revision_id,
            "transformation": _jsonable(self.transformation),
            "parent_architecture_fingerprint": self.parent_architecture_fingerprint,
            "child_architecture_fingerprint": self.child_architecture_fingerprint,
            "state_mapping": self.state_mapping.to_dict() if self.state_mapping else None,
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "decision": _jsonable(self.decision),
            "outcome": _jsonable(self.outcome),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelTransition":
        state_mapping_data = data.get("state_mapping")
        evidence_data = data.get("evidence")
        return cls(
            parent_revision_id=str(data["parent_revision_id"]),
            child_revision_id=str(data["child_revision_id"]),
            transformation=dict(data.get("transformation", {})),
            parent_architecture_fingerprint=str(
                data["parent_architecture_fingerprint"]
            ),
            child_architecture_fingerprint=str(
                data["child_architecture_fingerprint"]
            ),
            state_mapping=(
                StateMappingReport.from_dict(state_mapping_data)
                if isinstance(state_mapping_data, Mapping)
                else None
            ),
            evidence=(
                TransitionEvidence.from_dict(evidence_data)
                if isinstance(evidence_data, Mapping)
                else None
            ),
            decision=dict(data.get("decision", {})),
            outcome=dict(data.get("outcome", {})),
            status=str(data.get("status", "created")),
            transition_id=str(data.get("transition_id", "")),
        )
