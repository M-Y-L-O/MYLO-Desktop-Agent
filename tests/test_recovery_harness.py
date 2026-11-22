import unittest


try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from Core.ModelTransition import ModelTransition
    from Processing.Optimization.RecoveryHarness import (
        RecoveryExperimentConfig,
        TransitionRecoveryHarness,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    ModelTransition = None
    RecoveryExperimentConfig = None
    TransitionRecoveryHarness = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the active interpreter")
class TransitionRecoveryHarnessTests(unittest.TestCase):
    def test_records_budget_and_updates_transition(self):
        torch.manual_seed(4)
        parent = nn.Linear(2, 1)
        child = nn.Linear(2, 1)
        child.load_state_dict(parent.state_dict())

        inputs = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        )
        targets = inputs.sum(dim=1, keepdim=True)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
        transition = ModelTransition.create(
            "parent",
            "child",
            {"type": "identity_test"},
            {"nodes": []},
            {"nodes": []},
        )
        config = RecoveryExperimentConfig(
            max_steps=3,
            evaluation_interval=1,
            learning_rate=1e-2,
            seed=11,
        )

        result = TransitionRecoveryHarness.run(
            parent,
            child,
            loader,
            loader,
            nn.MSELoss(),
            config,
            transition=transition,
        )

        self.assertEqual(result.recovery_step, 0)
        self.assertEqual(result.optimizer_steps, 3)
        self.assertEqual(result.examples_seen, 6)
        self.assertEqual(len(result.checkpoints), 4)
        self.assertIn(result.selected_step, {0, 1, 2, 3})
        self.assertIsNotNone(result.best_validation_loss)
        self.assertEqual(transition.status, "trained")
        self.assertTrue(transition.outcome["recovered"])
        self.assertEqual(transition.evidence.recovery["optimizer_steps"], 3)

    def test_stops_when_validation_does_not_meaningfully_improve(self):
        torch.manual_seed(8)
        parent = nn.Linear(2, 1)
        child = nn.Linear(2, 1)
        child.load_state_dict(parent.state_dict())
        inputs = torch.randn(8, 2)
        targets = torch.randn(8, 1)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

        result = TransitionRecoveryHarness.run(
            parent,
            child,
            loader,
            loader,
            nn.MSELoss(),
            RecoveryExperimentConfig(
                max_steps=5,
                evaluation_interval=1,
                learning_rate=1e-12,
                early_stopping_patience_evaluations=1,
                minimum_relative_improvement=0.5,
            ),
        )

        self.assertTrue(result.stopped_early)
        self.assertEqual(result.stop_reason, "validation_plateau")
        self.assertEqual(result.optimizer_steps, 1)


if __name__ == "__main__":
    unittest.main()
