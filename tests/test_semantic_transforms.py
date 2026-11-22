import json
import unittest


try:
    import torch

    from Core.ArchitectureDescriptor import ArchitectureDescriptor, Edge
    from Core.DescriptorModelBuilder import DescriptorModelBuilder
    from Core.SemanticTransforms import (
        LinearWidthTransformation,
        TransformationError,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None
    ArchitectureDescriptor = None
    Edge = None
    DescriptorModelBuilder = None
    LinearWidthTransformation = None
    TransformationError = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the active interpreter")
class LinearWidthTransformationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.descriptor = ArchitectureDescriptor.default_feedforward(
            input_dim=3,
            output_dim=2,
            hidden=4,
        )
        self.parent_model = DescriptorModelBuilder.build(self.descriptor)
        self.parent_model.eval()

    def test_widening_preserves_eval_function(self):
        applied = LinearWidthTransformation.apply(
            self.descriptor,
            self.parent_model.state_dict(),
            "dense1",
            7,
            parent_revision_id="parent-1",
        )
        inputs = torch.randn(11, 3)
        applied.child_model.eval()

        with torch.inference_mode():
            parent_output = self.parent_model(inputs)
            child_output = applied.child_model(inputs)

        torch.testing.assert_close(parent_output, child_output, rtol=1e-5, atol=1e-6)
        self.assertEqual(applied.plan.mode, "widen")
        self.assertEqual(applied.plan.semantic_class, "function_preserving")
        self.assertEqual(applied.transition.status, "initialized")
        self.assertEqual(
            applied.transition.state_mapping.summary()["target_coverage"], 1.0
        )
        json.dumps(applied.transition.to_dict())

    def test_narrowing_is_recorded_as_approximate(self):
        applied = LinearWidthTransformation.apply(
            self.descriptor,
            self.parent_model.state_dict(),
            "dense1",
            2,
            parent_revision_id="parent-1",
        )

        self.assertEqual(applied.plan.mode, "narrow")
        self.assertEqual(applied.plan.semantic_class, "approximate")
        kinds = {
            mapping.mapping_kind
            for mapping in applied.transition.state_mapping.parameter_mappings
        }
        self.assertIn("structured_narrow_transform", kinds)

    def test_branching_path_is_refused(self):
        branched = ArchitectureDescriptor.from_dict(self.descriptor.to_dict())
        branched.edges.append(Edge(source="dense1", target="out_linear"))

        with self.assertRaises(TransformationError):
            LinearWidthTransformation.plan(branched, "dense1", 6)


if __name__ == "__main__":
    unittest.main()
