import unittest


try:
    import pandas as pd
    import torch

    from Processing.Data.DataPipeline import DataPipeline
except ModuleNotFoundError as exc:
    if exc.name not in {"torch", "pandas", "sklearn"}:
        raise
    pd = None
    torch = None
    DataPipeline = None


@unittest.skipIf(torch is None, "Data pipeline dependencies are not installed")
class DataPipelineTests(unittest.TestCase):
    def test_creates_reproducible_three_way_split(self):
        frame = pd.DataFrame(
            {
                "x": list(range(100)),
                "y": [float(value * 2) for value in range(100)],
            }
        )

        first = DataPipeline.prepare_data(
            frame, ["x"], ["y"], val_split=0.2, test_split=0.1, random_state=17
        )
        second = DataPipeline.prepare_data(
            frame, ["x"], ["y"], val_split=0.2, test_split=0.1, random_state=17
        )

        self.assertEqual(len(first.train_loader.dataset), 70)
        self.assertEqual(len(first.val_loader.dataset), 20)
        self.assertEqual(len(first.test_loader.dataset), 10)
        self.assertEqual(first.split_metadata, second.split_metadata)
        self.assertFalse(first.split_metadata["test_used_for_selection"])

    def test_classification_is_stratified_and_train_encoded(self):
        frame = pd.DataFrame(
            {
                "value": list(range(60)),
                "kind": ["a", "b", "c"] * 20,
                "target": [0, 1] * 30,
            }
        )

        result = DataPipeline.prepare_data(
            frame,
            ["value", "kind"],
            ["target"],
            problem_type="classification",
            encoding="one-hot",
            val_split=0.2,
            test_split=0.1,
            random_state=3,
        )

        self.assertEqual(result.split_metadata["strategy"], "stratified_random")
        self.assertEqual(result.output_shape, [-1, 2])
        self.assertEqual(len(result.feature_names), 4)
        self.assertEqual(
            result.scalers["feature_encoder"]["columns"]["kind"]["kind"],
            "one-hot",
        )

    def test_sequence_data_uses_chronological_partitions(self):
        frame = pd.DataFrame(
            {"x": list(range(40)), "y": [float(value) for value in range(40)]}
        )

        result = DataPipeline.prepare_data(
            frame,
            ["x"],
            ["y"],
            sequence_length=3,
            val_split=0.2,
            test_split=0.1,
        )

        self.assertEqual(result.split_metadata["strategy"], "chronological")
        self.assertEqual(result.input_shape, [-1, 3, 1])
        self.assertEqual(len(result.train_loader.dataset), 26)
        self.assertEqual(len(result.val_loader.dataset), 6)
        self.assertEqual(len(result.test_loader.dataset), 2)


if __name__ == "__main__":
    unittest.main()
