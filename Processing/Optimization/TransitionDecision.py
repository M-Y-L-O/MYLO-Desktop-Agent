from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


RISK_WEIGHTS = {
    "output_drift": 0.50,
    "calibration_loss": 0.35,
    "prediction_disagreement": 0.10,
    "unmapped_child_state": 0.05,
}


@dataclass(frozen=True)
class TransitionDecision:
    action: str
    accepted: bool
    reasons: tuple[str, ...]
    observed: Dict[str, Any]
    policy: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "observed": self.observed,
            "policy": self.policy,
        }


@dataclass(frozen=True)
class EvidenceDecisionPolicy:
    mode: str = "evidence"
    max_relative_l2_delta: float = 0.5
    function_preserving_tolerance: float = 1e-5
    max_calibration_loss_ratio: float = 2.0
    min_prediction_agreement: float = 0.75
    risk_threshold: float = 0.75
    hard_risk_threshold: float = 1.0
    min_static_width_retention: float = 0.5

    def __post_init__(self) -> None:
        if self.max_relative_l2_delta < 0:
            raise ValueError("max_relative_l2_delta cannot be negative.")
        if self.max_calibration_loss_ratio < 1.0:
            raise ValueError("max_calibration_loss_ratio must be at least 1.")
        if not 0.0 <= self.min_prediction_agreement <= 1.0:
            raise ValueError("min_prediction_agreement must be in [0, 1].")
        if self.risk_threshold <= 0:
            raise ValueError("risk_threshold must be positive.")
        if self.hard_risk_threshold < self.risk_threshold:
            raise ValueError("hard_risk_threshold cannot be below risk_threshold.")
        if not 0.0 <= self.min_static_width_retention <= 1.0:
            raise ValueError("min_static_width_retention must be in [0, 1].")

    def to_dict(self) -> Dict[str, Any]:
        return self._policy_metadata(self.mode.strip().lower())

    def decide(
        self,
        execution: Mapping[str, Any],
        *,
        semantic_class: str,
        static: Optional[Mapping[str, Any]] = None,
    ) -> TransitionDecision:
        normalized_mode = self.mode.strip().lower()
        if normalized_mode == "always_train":
            observed = self._observed(execution, static)
            observed["decision_tier"] = "unranked"
            return TransitionDecision(
                action="train",
                accepted=True,
                reasons=("Evidence gate disabled for baseline comparison.",),
                observed=observed,
                policy=self._policy_metadata(normalized_mode),
            )
        if normalized_mode == "static_only":
            return self._decide_static(static or {}, semantic_class)
        if normalized_mode not in {"evidence", "execution_evidence"}:
            raise ValueError(
                "Decision mode must be 'evidence', 'execution_evidence', "
                "'static_only', or 'always_train'."
            )
        return self._decide_execution(execution, static or {}, semantic_class)

    def _decide_static(
        self,
        static: Mapping[str, Any],
        semantic_class: str,
    ) -> TransitionDecision:
        reasons = []
        mapping = static.get("state_mapping") or {}
        target_coverage = _optional_float(mapping.get("target_coverage"))
        if target_coverage is not None and target_coverage < 1.0:
            reasons.append(
                f"State mapping covers only {target_coverage:.1%} of the child state."
            )

        transformation = static.get("transformation") or {}
        old_width = _optional_float(transformation.get("old_width"))
        new_width = _optional_float(transformation.get("new_width"))
        width_retention = None
        if old_width and new_width is not None:
            width_retention = min(1.0, new_width / old_width)
        if (
            semantic_class != "function_preserving"
            and width_retention is not None
            and width_retention < self.min_static_width_retention
        ):
            reasons.append(
                f"Approximate edit retains only {width_retention:.1%} of the original width."
            )

        observed = self._observed({}, static)
        observed.update(
            {
                "target_coverage": target_coverage,
                "width_retention": width_retention,
                "risk_score": (
                    max(0.0, 1.0 - width_retention)
                    if width_retention is not None
                    else 0.0
                ),
                "decision_tier": "rejected" if reasons else "preferred",
            }
        )
        return TransitionDecision(
            action="reject" if reasons else "train",
            accepted=not reasons,
            reasons=tuple(reasons) if reasons else ("Transition passed the static evidence gate.",),
            observed=observed,
            policy=self._policy_metadata("static_only"),
        )

    def _decide_execution(
        self,
        execution: Mapping[str, Any],
        static: Mapping[str, Any],
        semantic_class: str,
    ) -> TransitionDecision:
        rejection_reasons = []
        if not execution.get("comparable_outputs", False):
            rejection_reasons.append("Parent and child outputs are not comparable.")
        if not execution.get("parent_finite", False) or not execution.get("child_finite", False):
            rejection_reasons.append("Parent or child produced non-finite outputs.")

        relative_l2 = _optional_float(execution.get("relative_l2_delta"))
        if relative_l2 is None:
            rejection_reasons.append("Output drift evidence is missing or non-finite.")
        drift_component = _normalized_component(relative_l2, self.max_relative_l2_delta)
        if relative_l2 is not None and relative_l2 > self.max_relative_l2_delta:
            rejection_reasons.append(
                f"Relative output drift {relative_l2:.6g} exceeds {self.max_relative_l2_delta:.6g}."
            )
        if (
            semantic_class == "function_preserving"
            and relative_l2 is not None
            and relative_l2 > self.function_preserving_tolerance
        ):
            rejection_reasons.append(
                "A function-preserving edit violated its output-invariance tolerance."
            )

        parent_loss = _optional_float(execution.get("parent_loss"))
        child_loss = _optional_float(execution.get("child_loss"))
        loss_ratio = None
        loss_component = 0.0
        if parent_loss is not None and child_loss is not None:
            loss_ratio = child_loss / max(abs(parent_loss), 1e-12)
            loss_component = _normalized_excess_ratio(
                loss_ratio, self.max_calibration_loss_ratio
            )
            if loss_ratio > self.max_calibration_loss_ratio:
                rejection_reasons.append(
                    f"Calibration loss ratio {loss_ratio:.6g} exceeds {self.max_calibration_loss_ratio:.6g}."
                )
        else:
            rejection_reasons.append("Calibration loss evidence is missing or non-finite.")

        agreement = _optional_float(execution.get("prediction_agreement"))
        agreement_component = 0.0
        if agreement is not None and self.min_prediction_agreement > 0:
            agreement_component = max(
                0.0,
                (self.min_prediction_agreement - agreement)
                / self.min_prediction_agreement,
            )
            if agreement < self.min_prediction_agreement:
                rejection_reasons.append(
                    f"Prediction agreement {agreement:.6g} is below {self.min_prediction_agreement:.6g}."
                )

        mapping = static.get("state_mapping") or {}
        target_coverage = _optional_float(mapping.get("target_coverage"))
        mapping_component = max(0.0, 1.0 - target_coverage) if target_coverage is not None else 0.0
        risk_components = {
            "output_drift": drift_component,
            "calibration_loss": loss_component,
            "prediction_disagreement": agreement_component,
            "unmapped_child_state": mapping_component,
        }
        risk_score = sum(
            RISK_WEIGHTS[name] * value
            for name, value in risk_components.items()
        )
        if risk_score >= self.hard_risk_threshold:
            rejection_reasons.append(
                f"Combined transition risk {risk_score:.6g} reaches the "
                f"{self.hard_risk_threshold:.6g} hard-rejection threshold."
            )

        deferred = not rejection_reasons and risk_score >= self.risk_threshold
        if deferred:
            reasons = (
                f"Combined transition risk {risk_score:.6g} reaches the "
                f"{self.risk_threshold:.6g} low-priority threshold.",
            )
        elif rejection_reasons:
            reasons = tuple(rejection_reasons)
        else:
            reasons = ("Transition passed the execution evidence gate.",)

        observed = self._observed(execution, static)
        observed.update(
            {
                "calibration_loss_ratio": loss_ratio,
                "target_coverage": target_coverage,
                "risk_components": risk_components,
                "risk_score": risk_score,
                "risk_threshold": self.risk_threshold,
                "hard_risk_threshold": self.hard_risk_threshold,
                "decision_tier": (
                    "rejected"
                    if rejection_reasons
                    else "deferred"
                    if deferred
                    else "preferred"
                ),
            }
        )
        return TransitionDecision(
            action="reject" if rejection_reasons else "defer" if deferred else "train",
            accepted=not rejection_reasons,
            reasons=reasons,
            observed=observed,
            policy=self._policy_metadata(self.mode),
        )

    def _policy_metadata(self, mode: str) -> Dict[str, Any]:
        return {
            "version": "evidence_policy_v3",
            "mode": mode,
            "max_relative_l2_delta": self.max_relative_l2_delta,
            "function_preserving_tolerance": self.function_preserving_tolerance,
            "max_calibration_loss_ratio": self.max_calibration_loss_ratio,
            "min_prediction_agreement": self.min_prediction_agreement,
            "risk_threshold": self.risk_threshold,
            "hard_risk_threshold": self.hard_risk_threshold,
            "min_static_width_retention": self.min_static_width_retention,
            "risk_weights": dict(RISK_WEIGHTS),
        }

    @staticmethod
    def _observed(
        execution: Mapping[str, Any],
        static: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        keys = (
            "comparable_outputs",
            "parent_finite",
            "child_finite",
            "relative_l2_delta",
            "cosine_similarity",
            "prediction_agreement",
            "parent_loss",
            "child_loss",
            "loss_delta",
            "calibration_batches",
            "calibration_examples",
        )
        observed = {key: execution.get(key) for key in keys if key in execution}
        transformation = (static or {}).get("transformation") or {}
        if transformation:
            observed["transformation_mode"] = transformation.get("mode")
            observed["semantic_class"] = transformation.get("semantic_class")
        return observed


def _normalized_component(value: Optional[float], threshold: float) -> float:
    if value is None or threshold <= 0:
        return 0.0
    return min(2.0, max(0.0, value / threshold))


def _normalized_excess_ratio(value: float, threshold: float) -> float:
    if threshold <= 1.0:
        return 2.0 if value > threshold else 0.0
    return min(2.0, max(0.0, (value - 1.0) / (threshold - 1.0)))


def _optional_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed is not None and math.isfinite(parsed) else None
