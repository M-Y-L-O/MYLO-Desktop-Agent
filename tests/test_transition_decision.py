import unittest

from Processing.Optimization.TransitionDecision import EvidenceDecisionPolicy


class EvidenceDecisionPolicyTests(unittest.TestCase):
    def test_always_train_mode_is_an_explicit_baseline(self):
        decision = EvidenceDecisionPolicy(mode="always_train").decide(
            {"child_finite": False}, semantic_class="approximate"
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.action, "train")

    def test_rejects_function_preserving_invariant_violation(self):
        decision = EvidenceDecisionPolicy().decide(
            {
                "comparable_outputs": True,
                "parent_finite": True,
                "child_finite": True,
                "relative_l2_delta": 0.01,
            },
            semantic_class="function_preserving",
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.action, "reject")

    def test_allows_finite_recoverable_transition(self):
        decision = EvidenceDecisionPolicy().decide(
            {
                "comparable_outputs": True,
                "parent_finite": True,
                "child_finite": True,
                "relative_l2_delta": 0.2,
                "parent_loss": 1.0,
                "child_loss": 1.4,
            },
            semantic_class="approximate",
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.action, "train")

    def test_rejects_destructive_transition_with_combined_risk(self):
        decision = EvidenceDecisionPolicy().decide(
            {
                "comparable_outputs": True,
                "parent_finite": True,
                "child_finite": True,
                "relative_l2_delta": 0.8082,
                "parent_loss": 0.2029,
                "child_loss": 0.832,
            },
            semantic_class="approximate",
        )

        self.assertFalse(decision.accepted)
        self.assertGreaterEqual(decision.observed["risk_score"], 1.0)
        self.assertIn("risk_components", decision.observed)

    def test_static_only_rejects_severe_narrowing(self):
        decision = EvidenceDecisionPolicy(mode="static_only").decide(
            {},
            semantic_class="approximate",
            static={
                "transformation": {"old_width": 32, "new_width": 8},
                "state_mapping": {"target_coverage": 1.0},
            },
        )

        self.assertFalse(decision.accepted)
        self.assertAlmostEqual(decision.observed["width_retention"], 0.25)

    def test_execution_evidence_is_the_named_evidence_mode(self):
        decision = EvidenceDecisionPolicy(mode="execution_evidence").decide(
            {
                "comparable_outputs": True,
                "parent_finite": True,
                "child_finite": True,
                "relative_l2_delta": 0.1,
                "parent_loss": 1.0,
                "child_loss": 1.1,
            },
            semantic_class="approximate",
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.policy["mode"], "execution_evidence")

    def test_policy_serialization_captures_replay_configuration(self):
        policy = EvidenceDecisionPolicy(mode="execution_evidence").to_dict()

        self.assertEqual(policy["version"], "evidence_policy_v3")
        self.assertEqual(policy["mode"], "execution_evidence")
        self.assertIn("risk_weights", policy)

    def test_execution_gate_fails_closed_on_nonfinite_metrics(self):
        decision = EvidenceDecisionPolicy().decide(
            {
                "comparable_outputs": True,
                "parent_finite": True,
                "child_finite": True,
                "relative_l2_delta": float("nan"),
                "parent_loss": 1.0,
                "child_loss": float("inf"),
            },
            semantic_class="approximate",
        )

        self.assertFalse(decision.accepted)
        self.assertTrue(any("non-finite" in reason for reason in decision.reasons))

    def test_combined_risk_can_defer_without_discarding_candidate(self):
        decision = EvidenceDecisionPolicy().decide(
            {
                "comparable_outputs": True,
                "parent_finite": True,
                "child_finite": True,
                "relative_l2_delta": 0.4571409,
                "parent_loss": 0.2029254,
                "child_loss": 0.386124,
            },
            semantic_class="approximate",
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.action, "defer")
        self.assertEqual(decision.observed["decision_tier"], "deferred")


if __name__ == "__main__":
    unittest.main()
