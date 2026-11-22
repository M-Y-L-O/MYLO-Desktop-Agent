import unittest

try:
    import torch
    import torch.nn as nn

    from Processing.Optimization.TransitionEvidence import TransitionEvidenceCollector
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None
    nn = None
    TransitionEvidenceCollector = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the active interpreter")
class TransitionEvidenceCollectorTests(unittest.TestCase):
    def test_static_evidence_identifies_changed_nodes(self):
        parent = {
            "input_shape": [-1, 2],
            "output_shape": [-1, 1],
            "nodes": [
                {"id": "dense", "type": "Linear", "params": {"out_features": 2}}
            ],
            "edges": [{"source": "input", "target": "dense"}],
        }
        child = {
            "input_shape": [-1, 2],
            "output_shape": [-1, 1],
            "nodes": [
                {"id": "dense", "type": "Linear", "params": {"out_features": 4}},
                {"id": "relu", "type": "ReLU", "params": {}},
            ],
            "edges": [
                {"source": "input", "target": "dense"},
                {"source": "dense", "target": "relu"},
            ],
        }

        evidence = TransitionEvidenceCollector.collect_static(parent, child)

        self.assertEqual(evidence["node_count_delta"], 1)
        self.assertEqual(evidence["added_node_ids"], ["relu"])
        self.assertEqual(evidence["changed_node_ids"], ["dense"])

    def test_execution_evidence_measures_functional_drift(self):
        parent = nn.Linear(2, 2, bias=False)
        child = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            parent.weight.copy_(torch.eye(2))
            child.weight.copy_(torch.eye(2))
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        evidence = TransitionEvidenceCollector.collect_execution(parent, child, inputs)

        self.assertTrue(evidence["comparable_outputs"])
        self.assertEqual(evidence["mean_squared_delta"], 0.0)
        self.assertEqual(evidence["relative_l2_delta"], 0.0)
        self.assertAlmostEqual(evidence["cosine_similarity"], 1.0)


if __name__ == "__main__":
    unittest.main()
