import json
import unittest


try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from Core.ArchitectureDescriptor import ArchitectureDescriptor
    from Core.DescriptorModelBuilder import DescriptorModelBuilder
    from Processing.Optimization.RecoveryHarness import RecoveryExperimentConfig
    from Processing.Optimization.TransitionExperiment import (
        LinearWidthRecoveryExperiment,
    )
    from Processing.Optimization.TransitionDecision import EvidenceDecisionPolicy
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    ArchitectureDescriptor = None
    DescriptorModelBuilder = None
    RecoveryExperimentConfig = None
    LinearWidthRecoveryExperiment = None
    EvidenceDecisionPolicy = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the active interpreter")
class LinearWidthRecoveryExperimentTests(unittest.TestCase):
    def test_runs_complete_transition_loop(self):
        torch.manual_seed(13)
        descriptor = ArchitectureDescriptor.default_feedforward(3, 2, hidden=4)
        parent = DescriptorModelBuilder.build(descriptor)
        inputs = torch.randn(12, 3)
        targets = torch.randn(12, 2)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)

        result = LinearWidthRecoveryExperiment.run(
            descriptor,
            parent,
            "dense1",
            6,
            loader,
            loader,
            nn.MSELoss(),
            RecoveryExperimentConfig(
                max_steps=2,
                evaluation_interval=1,
                learning_rate=1e-3,
                recovery_tolerance=1e-5,
                seed=5,
            ),
            parent_revision_id="test-parent",
            test_loader=loader,
            decision_policy=EvidenceDecisionPolicy(mode="execution_evidence"),
        )

        self.assertEqual(result.transition.status, "trained")
        self.assertEqual(result.plan.semantic_class, "function_preserving")
        self.assertAlmostEqual(
            result.transition.evidence.execution["relative_l2_delta"],
            0.0,
            places=5,
        )
        self.assertEqual(result.recovery.recovery_step, 0)
        self.assertEqual(result.recovery.optimizer_steps, 2)
        self.assertTrue(result.decision.accepted)
        self.assertEqual(result.transition.decision["action"], "train")
        self.assertFalse(result.held_out_test["used_for_selection"])
        json.dumps(result.to_dict())


if __name__ == "__main__":
    unittest.main()
