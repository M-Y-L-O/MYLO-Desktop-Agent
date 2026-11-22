import tempfile
import unittest

from Processing.Optimization.ResearchRunStore import ResearchRunStore


class ResearchRunStoreTests(unittest.TestCase):
    def test_saves_lists_and_loads_run_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ResearchRunStore(directory)
            metadata = store.save(
                run_type="benchmark",
                project_id="project-a",
                request={"seed": 7},
                result={"score": 0.5},
            )

            loaded = store.load(metadata["run_id"])
            self.assertEqual(loaded["schema_version"], store.SCHEMA_VERSION)
            self.assertEqual(loaded["request"]["seed"], 7)
            self.assertEqual(loaded["result"]["score"], 0.5)
            self.assertEqual(len(store.list(project_id="project-a")), 1)
            self.assertEqual(store.list(project_id="project-b"), [])

    def test_refuses_invalid_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ResearchRunStore(directory)
            with self.assertRaises(ValueError):
                store.load("../outside")


if __name__ == "__main__":
    unittest.main()
