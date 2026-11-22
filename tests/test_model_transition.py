import importlib.util
import sys
import unittest
from pathlib import Path


def _load_transition_module():
    try:
        from Core import ModelTransition as model_transition_class
        from Core.ModelTransition import (
            ParameterMapping,
            StateMappingReport,
            canonical_fingerprint,
            state_dict_fingerprint,
        )

        return (
            model_transition_class,
            ParameterMapping,
            StateMappingReport,
            canonical_fingerprint,
            state_dict_fingerprint,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise

    module_path = Path(__file__).parents[1] / "Core" / "ModelTransition.py"
    spec = importlib.util.spec_from_file_location("mylo_model_transition", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return (
        module.ModelTransition,
        module.ParameterMapping,
        module.StateMappingReport,
        module.canonical_fingerprint,
        module.state_dict_fingerprint,
    )


(
    ModelTransition,
    ParameterMapping,
    StateMappingReport,
    canonical_fingerprint,
    state_dict_fingerprint,
) = _load_transition_module()


class ModelTransitionTests(unittest.TestCase):
    class _FakeArray:
        def __init__(self, payload):
            self.payload = payload

        def tobytes(self, order="C"):
            return self.payload

    class _FakeTensor:
        def __init__(self, payload, shape=(1,), dtype="float32"):
            self.payload = payload
            self.shape = shape
            self.dtype = dtype

        def detach(self):
            return self

        def cpu(self):
            return self

        def contiguous(self):
            return self

        def numpy(self):
            return ModelTransitionTests._FakeArray(self.payload)

    def test_fingerprint_is_canonical(self):
        left = {"nodes": [{"id": "a", "params": {"width": 8}}], "version": 1}
        right = {"version": 1, "nodes": [{"params": {"width": 8}, "id": "a"}]}
        changed = {"version": 1, "nodes": [{"params": {"width": 9}, "id": "a"}]}

        self.assertEqual(canonical_fingerprint(left), canonical_fingerprint(right))
        self.assertNotEqual(canonical_fingerprint(left), canonical_fingerprint(changed))

    def test_state_fingerprint_is_order_independent_and_weight_sensitive(self):
        left = {
            "b": self._FakeTensor(b"second"),
            "a": self._FakeTensor(b"first", shape=(2,)),
        }
        reordered = {"a": left["a"], "b": left["b"]}
        changed = {**left, "a": self._FakeTensor(b"changed", shape=(2,))}

        self.assertEqual(
            state_dict_fingerprint(left), state_dict_fingerprint(reordered)
        )
        self.assertNotEqual(
            state_dict_fingerprint(left), state_dict_fingerprint(changed)
        )

    def test_mapping_report_separates_exact_partial_and_initialized_state(self):
        report = StateMappingReport(
            parameter_mappings=[
                ParameterMapping(
                    source_key="a",
                    target_key="a",
                    mapping_kind="exact_copy",
                    key_resolution="direct",
                    source_shape=(2, 3),
                    target_shape=(2, 3),
                    copied_elements=6,
                    source_elements=6,
                    target_elements=6,
                ),
                ParameterMapping(
                    source_key="b",
                    target_key="b",
                    mapping_kind="overlap_copy",
                    key_resolution="direct",
                    source_shape=(2, 4),
                    target_shape=(2, 5),
                    copied_elements=8,
                    source_elements=8,
                    target_elements=10,
                ),
                ParameterMapping(
                    source_key=None,
                    target_key="c",
                    mapping_kind="initialized_target",
                    key_resolution="target_initialization",
                    target_shape=(4,),
                    target_elements=4,
                ),
            ]
        )

        summary = report.summary()
        self.assertEqual(summary["exact_tensor_count"], 1)
        self.assertEqual(summary["overlap_tensor_count"], 1)
        self.assertEqual(summary["initialized_target_tensor_count"], 1)
        self.assertEqual(summary["copied_elements"], 14)
        self.assertEqual(summary["source_coverage"], 1.0)
        self.assertEqual(summary["target_coverage"], 0.7)

    def test_transition_identity_is_stable_and_edit_sensitive(self):
        parent = {"nodes": [{"id": "dense", "params": {"width": 8}}]}
        child = {"nodes": [{"id": "dense", "params": {"width": 12}}]}
        first = ModelTransition.create(
            "parent-1",
            "child-1",
            {"type": "scale_width", "factor": 1.5},
            parent,
            child,
        )
        second = ModelTransition.create(
            "parent-1",
            "child-1",
            {"factor": 1.5, "type": "scale_width"},
            parent,
            child,
        )
        different = ModelTransition.create(
            "parent-1",
            "child-1",
            {"type": "scale_width", "factor": 2.0},
            parent,
            child,
        )

        self.assertEqual(first.transition_id, second.transition_id)
        self.assertNotEqual(first.transition_id, different.transition_id)

        restored = ModelTransition.from_dict(first.to_dict())
        self.assertEqual(restored.to_dict(), first.to_dict())


if __name__ == "__main__":
    unittest.main()
