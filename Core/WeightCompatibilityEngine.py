from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .ModelTransition import ParameterMapping, StateMappingReport


class WeightCompatibilityEngine:
    @staticmethod
    def transfer_weights(
        source_state_dict: Dict[str, torch.Tensor],
        target_module: nn.Module,
        *,
        explicit_key_mapping: Optional[Mapping[str, str]] = None,
        allow_partial: bool = True,
        allow_ambiguous_key_mapping: bool = True,
    ) -> Dict[str, Any]:
        """Transfer compatible state and return an auditable mapping report.

        The legacy list fields remain in the response for existing callers. New
        research code should consume ``state_mapping`` and distinguish exact
        copies from overlap copies.
        """
        target_state_dict = target_module.state_dict()
        new_state_dict: Dict[str, torch.Tensor] = {}
        mappings = []
        warnings = []
        claimed_targets = set()

        for source_key, source_tensor in source_state_dict.items():
            target_key, resolution, resolution_warning = (
                WeightCompatibilityEngine._resolve_key_mapping_detailed(
                    source_key,
                    target_state_dict,
                    explicit_key_mapping=explicit_key_mapping,
                    allow_ambiguous=allow_ambiguous_key_mapping,
                )
            )
            if resolution_warning:
                warnings.append(resolution_warning)

            source_shape = WeightCompatibilityEngine._shape(source_tensor)
            source_elements = WeightCompatibilityEngine._numel(source_tensor)
            source_dtype = WeightCompatibilityEngine._dtype(source_tensor)

            if target_key not in target_state_dict:
                mappings.append(
                    ParameterMapping(
                        source_key=source_key,
                        target_key=None,
                        mapping_kind="unmapped_source",
                        key_resolution=resolution,
                        source_shape=source_shape,
                        source_dtype=source_dtype,
                        source_elements=source_elements,
                        note="No target parameter could be resolved.",
                    )
                )
                continue

            target_tensor = target_state_dict[target_key]
            target_shape = WeightCompatibilityEngine._shape(target_tensor)
            target_elements = WeightCompatibilityEngine._numel(target_tensor)
            target_dtype = WeightCompatibilityEngine._dtype(target_tensor)

            if target_key in claimed_targets:
                warning = (
                    f"Source parameter '{source_key}' also resolved to already-mapped "
                    f"target '{target_key}' and was not transferred."
                )
                warnings.append(warning)
                mappings.append(
                    ParameterMapping(
                        source_key=source_key,
                        target_key=None,
                        mapping_kind="unmapped_source",
                        key_resolution="target_already_mapped",
                        source_shape=source_shape,
                        source_dtype=source_dtype,
                        source_elements=source_elements,
                        note=warning,
                    )
                )
                continue

            if source_shape == target_shape:
                new_state_dict[target_key] = source_tensor
                claimed_targets.add(target_key)
                mapping_kind = (
                    "exact_copy"
                    if source_dtype == target_dtype
                    else "dtype_cast_copy"
                )
                mappings.append(
                    ParameterMapping(
                        source_key=source_key,
                        target_key=target_key,
                        mapping_kind=mapping_kind,
                        key_resolution=resolution,
                        source_shape=source_shape,
                        target_shape=target_shape,
                        source_dtype=source_dtype,
                        target_dtype=target_dtype,
                        copied_elements=target_elements,
                        source_elements=source_elements,
                        target_elements=target_elements,
                        note=(
                            None
                            if mapping_kind == "exact_copy"
                            else "Tensor shape matched, but loading casts the source dtype."
                        ),
                    )
                )
                continue

            transferred = None
            if allow_partial:
                transferred = WeightCompatibilityEngine._partial_shape_transfer(
                    source_tensor, target_tensor
                )

            if transferred is None:
                mappings.append(
                    ParameterMapping(
                        source_key=source_key,
                        target_key=None,
                        mapping_kind="unmapped_source",
                        key_resolution=resolution,
                        source_shape=source_shape,
                        target_shape=target_shape,
                        source_dtype=source_dtype,
                        target_dtype=target_dtype,
                        source_elements=source_elements,
                        target_elements=target_elements,
                        note=(
                            "Partial transfer is disabled or tensor ranks are incompatible."
                        ),
                    )
                )
                continue

            copied_elements = WeightCompatibilityEngine._overlap_elements(
                source_shape, target_shape
            )
            new_state_dict[target_key] = transferred
            claimed_targets.add(target_key)
            mappings.append(
                ParameterMapping(
                    source_key=source_key,
                    target_key=target_key,
                    mapping_kind="overlap_copy",
                    key_resolution=resolution,
                    source_shape=source_shape,
                    target_shape=target_shape,
                    source_dtype=source_dtype,
                    target_dtype=target_dtype,
                    copied_elements=copied_elements,
                    source_elements=source_elements,
                    target_elements=target_elements,
                    note=(
                        "Copied the shared tensor prefix; this is not a "
                        "function-preservation guarantee."
                    ),
                )
            )

        for target_key, target_tensor in target_state_dict.items():
            if target_key in claimed_targets:
                continue
            new_state_dict[target_key] = target_tensor
            mappings.append(
                ParameterMapping(
                    source_key=None,
                    target_key=target_key,
                    mapping_kind="initialized_target",
                    key_resolution="target_initialization",
                    target_shape=WeightCompatibilityEngine._shape(target_tensor),
                    target_dtype=WeightCompatibilityEngine._dtype(target_tensor),
                    target_elements=WeightCompatibilityEngine._numel(target_tensor),
                )
            )

        target_module.load_state_dict(new_state_dict, strict=False)

        state_mapping = StateMappingReport(
            parameter_mappings=mappings,
            warnings=warnings,
            policy={
                "allow_partial": allow_partial,
                "allow_ambiguous_key_mapping": allow_ambiguous_key_mapping,
                "explicit_key_mapping": bool(explicit_key_mapping),
            },
        )
        structured_report = state_mapping.to_dict()

        matched_keys = [
            mapping.target_key
            for mapping in mappings
            if mapping.mapping_kind
            in {"exact_copy", "dtype_cast_copy", "overlap_copy"}
            and mapping.target_key is not None
        ]
        unmatched_source = [
            mapping.source_key
            for mapping in mappings
            if mapping.mapping_kind == "unmapped_source"
            and mapping.source_key is not None
        ]
        unmatched_target = [
            mapping.target_key
            for mapping in mappings
            if mapping.mapping_kind == "initialized_target"
            and mapping.target_key is not None
        ]

        return {
            "matched_keys": matched_keys,
            "unmatched_source": unmatched_source,
            "unmatched_target": unmatched_target,
            "mapping_summary": structured_report["summary"],
            "parameter_mappings": structured_report["parameter_mappings"],
            "mapping_warnings": structured_report["warnings"],
            "state_mapping": structured_report,
        }

    @staticmethod
    def _resolve_key_mapping_detailed(
        source_key: str,
        target_state_dict: Mapping[str, torch.Tensor],
        *,
        explicit_key_mapping: Optional[Mapping[str, str]],
        allow_ambiguous: bool,
    ) -> Tuple[str, str, Optional[str]]:
        if explicit_key_mapping and source_key in explicit_key_mapping:
            return explicit_key_mapping[source_key], "explicit", None

        if source_key in target_state_dict:
            return source_key, "direct", None

        prefixed = f"node_modules.{source_key}"
        if prefixed in target_state_dict:
            return prefixed, "node_modules_prefix", None

        suffix = source_key.split(".")[-1]
        suffix_candidates = [
            target_key
            for target_key in target_state_dict
            if target_key.endswith(f".{suffix}") or target_key == suffix
        ]
        node_candidates = []
        for target_key in suffix_candidates:
            target_parts = target_key.split(".")
            if len(target_parts) >= 2 and target_parts[-2] in source_key:
                node_candidates.append(target_key)

        if len(node_candidates) == 1:
            return node_candidates[0], "node_and_suffix", None
        if len(suffix_candidates) == 1:
            return suffix_candidates[0], "unique_suffix", None

        candidates = node_candidates or suffix_candidates
        if candidates and allow_ambiguous:
            selected = candidates[0]
            warning = (
                f"Ambiguous mapping for source parameter '{source_key}': "
                f"selected '{selected}' from {candidates}."
            )
            return selected, "ambiguous_suffix_first", warning
        if candidates:
            warning = (
                f"Ambiguous mapping for source parameter '{source_key}' was refused: "
                f"{candidates}."
            )
            return source_key, "ambiguous_refused", warning
        return source_key, "unresolved", None

    @staticmethod
    def _resolve_key_mapping(
        source_key: str, target_state_dict: Dict[str, torch.Tensor]
    ) -> str:
        target_key, _, _ = WeightCompatibilityEngine._resolve_key_mapping_detailed(
            source_key,
            target_state_dict,
            explicit_key_mapping=None,
            allow_ambiguous=True,
        )
        return target_key

    @staticmethod
    def _partial_shape_transfer(
        source_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not isinstance(source_tensor, torch.Tensor) or not isinstance(
            target_tensor, torch.Tensor
        ):
            return None
        if len(source_tensor.shape) != len(target_tensor.shape):
            return None

        target = target_tensor.clone()
        common_slices = tuple(
            slice(0, min(source_size, target_size))
            for source_size, target_size in zip(
                source_tensor.shape, target_tensor.shape
            )
        )
        target[common_slices] = source_tensor[common_slices]
        return target

    @staticmethod
    def _shape(tensor: Any) -> Tuple[int, ...]:
        shape = getattr(tensor, "shape", ())
        return tuple(int(dimension) for dimension in shape)

    @staticmethod
    def _numel(tensor: Any) -> int:
        numel = getattr(tensor, "numel", None)
        return int(numel()) if callable(numel) else 0

    @staticmethod
    def _dtype(tensor: Any) -> Optional[str]:
        dtype = getattr(tensor, "dtype", None)
        return str(dtype) if dtype is not None else None

    @staticmethod
    def _overlap_elements(
        source_shape: Tuple[int, ...], target_shape: Tuple[int, ...]
    ) -> int:
        if len(source_shape) != len(target_shape):
            return 0
        overlap = 1
        for source_size, target_size in zip(source_shape, target_shape):
            overlap *= min(source_size, target_size)
        return overlap
