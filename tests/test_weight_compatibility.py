import unittest


try:
    import torch
    import torch.nn as nn

    from Core.WeightCompatibilityEngine import WeightCompatibilityEngine
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None
    nn = None
    WeightCompatibilityEngine = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the active interpreter")
class WeightCompatibilityEngineTests(unittest.TestCase):
    def test_report_distinguishes_exact_and_overlap_copy(self):
        target = nn.Linear(3, 3)
        source = {
            "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "bias": torch.tensor([1.0, 2.0]),
        }

        report = WeightCompatibilityEngine.transfer_weights(source, target)
        summary = report["mapping_summary"]

        self.assertEqual(summary["overlap_tensor_count"], 2)
        self.assertEqual(summary["initialized_target_tensor_count"], 0)
        self.assertAlmostEqual(summary["target_coverage"], 8 / 12)
        self.assertTrue(
            all(
                mapping["mapping_kind"] == "overlap_copy"
                for mapping in report["parameter_mappings"]
            )
        )

    def test_explicit_mapping_is_recorded(self):
        target = nn.Linear(2, 1, bias=False)
        source = {"legacy.weight": torch.tensor([[2.0, 3.0]])}

        report = WeightCompatibilityEngine.transfer_weights(
            source,
            target,
            explicit_key_mapping={"legacy.weight": "weight"},
        )

        mapping = report["parameter_mappings"][0]
        self.assertEqual(mapping["mapping_kind"], "exact_copy")
        self.assertEqual(mapping["key_resolution"], "explicit")
        torch.testing.assert_close(target.weight, source["legacy.weight"])

    def test_ambiguous_suffix_can_be_refused(self):
        class TwoWeights(nn.Module):
            def __init__(self):
                super().__init__()
                self.left = nn.Linear(2, 1, bias=False)
                self.right = nn.Linear(2, 1, bias=False)

        target = TwoWeights()
        report = WeightCompatibilityEngine.transfer_weights(
            {"legacy.weight": torch.ones(1, 2)},
            target,
            allow_ambiguous_key_mapping=False,
        )

        self.assertEqual(report["unmatched_source"], ["legacy.weight"])
        self.assertEqual(report["mapping_summary"]["initialized_target_tensor_count"], 2)
        self.assertTrue(report["mapping_warnings"])


if __name__ == "__main__":
    unittest.main()
