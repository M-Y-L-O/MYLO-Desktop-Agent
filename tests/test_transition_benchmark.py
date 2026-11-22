import json
import unittest


try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from Core.ArchitectureDescriptor import ArchitectureDescriptor
    from Core.DescriptorModelBuilder import DescriptorModelBuilder
    from Processing.Data.DataPipeline import DataPipelineResult
    from Processing.Optimization.RecoveryHarness import RecoveryExperimentConfig
    from Processing.Optimization.TransitionBenchmark import (
        LinearWidthBenchmark,
        LinearWidthCandidate,
    )
    from Processing.Optimization.TransitionDecision import EvidenceDecisionPolicy
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the active interpreter")
class LinearWidthBenchmarkTests(unittest.TestCase):
    def test_compares_modes_without_using_test_for_selection(self):
        torch.manual_seed(21)
        descriptor = ArchitectureDescriptor.default_feedforward(3, 2, hidden=4)
        parent = DescriptorModelBuilder.build(descriptor)
        inputs = torch.randn(30, 3)
        targets = torch.randn(30, 2)

        train = TensorDataset(inputs[:18], targets[:18])
        validation = TensorDataset(inputs[18:24], targets[18:24])
        test = TensorDataset(inputs[24:], targets[24:])
        pipeline = DataPipelineResult(
            train_loader=DataLoader(train, batch_size=3, shuffle=True),
            val_loader=DataLoader(validation, batch_size=3),
            test_loader=DataLoader(test, batch_size=3),
            scalers={},
            input_shape=[-1, 3],
            output_shape=[-1, 2],
            feature_names=["a", "b", "c"],
            split_metadata={"test_used_for_selection": False},
        )

        result = LinearWidthBenchmark.run(
            descriptor,
            parent.state_dict(),
            [
                LinearWidthCandidate("dense1", 6),
                LinearWidthCandidate("dense1", 2),
            ],
            pipeline,
            nn.MSELoss(),
            RecoveryExperimentConfig(
                max_steps=1,
                evaluation_interval=1,
                learning_rate=1e-3,
                seed=9,
            ),
            parent_revision_id="benchmark-parent",
            decision_policies={
                "evidence": EvidenceDecisionPolicy(mode="evidence"),
                "static_only": EvidenceDecisionPolicy(mode="static_only"),
                "always_train": EvidenceDecisionPolicy(mode="always_train"),
            },
            optimizer_step_budget=2,
            steps_per_candidate=1,
            calibration_batches=2,
            test_evaluation_scope="feasible_frontier",
        )

        self.assertEqual(len(result["modes"]["evidence"]["candidates"]), 2)
        self.assertEqual(len(result["modes"]["always_train"]["candidates"]), 2)
        self.assertEqual(len(result["modes"]["static_only"]["candidates"]), 2)
        for mode in result["modes"].values():
            self.assertEqual(mode["parent_baseline"]["candidate_id"], "parent")
            self.assertGreater(len(mode["pareto_frontier"]), 0)
            self.assertLessEqual(mode["budget"]["optimizer_steps_used"], 2)
            if mode["held_out_test"] is not None:
                self.assertFalse(mode["held_out_test"]["used_for_selection"])
            self.assertIsNotNone(mode["held_out_frontier"])
            self.assertEqual(
                {record["candidate_id"] for record in mode["held_out_frontier"]},
                {
                    record["candidate_id"]
                    for record in mode["feasible_pareto_frontier"]
                },
            )
            self.assertTrue(
                all(
                    not record["used_for_selection"]
                    for record in mode["held_out_frontier"]
                )
            )
        self.assertIn("static_only_optimizer_steps", result["comparison"])
        self.assertEqual(
            result["modes"]["always_train"]["scheduling"]["strategy"],
            "provided_order",
        )
        self.assertEqual(
            result["modes"]["static_only"]["scheduling"]["scheduled_order"][0],
            "dense1:2",
        )
        self.assertIn("validation_pareto", result["comparison"])
        self.assertIn(
            "feasible_pareto_frontier", result["modes"]["evidence"]
        )
        self.assertTrue(
            all(
                record["feasible"]
                for record in result["modes"]["evidence"][
                    "feasible_pareto_frontier"
                ]
            )
        )
        self.assertTrue(
            result["comparison"]["validation_pareto"][
                "higher_hypervolume_is_better"
            ]
        )
        self.assertEqual(
            result["protocol"]["schema_version"],
            "linear_width_benchmark_protocol_v1",
        )
        self.assertEqual(len(result["protocol"]["protocol_fingerprint"]), 64)
        self.assertIn("held_out_selected_frontier", result["comparison"])
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
