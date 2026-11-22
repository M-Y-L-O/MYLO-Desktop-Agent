import os
import unittest
from unittest.mock import patch


try:
    import torch

    from Processing.Optimization.TransitionService import (
        ResearchExperimentError,
        discover_linear_width_candidates,
        research_api_enabled,
        research_status,
        run_project_linear_width_benchmark,
        run_project_linear_width_experiment,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None
    ResearchExperimentError = None
    discover_linear_width_candidates = None
    research_api_enabled = None
    research_status = None
    run_project_linear_width_benchmark = None
    run_project_linear_width_experiment = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the active interpreter")
class TransitionServiceTests(unittest.TestCase):
    def test_switch_defaults_to_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(research_api_enabled())
            self.assertFalse(research_status()["enabled"])

    def test_switch_accepts_explicit_true(self):
        with patch.dict(os.environ, {"MYLO_RESEARCH_API_ENABLED": "true"}):
            self.assertTrue(research_api_enabled())
            self.assertFalse(
                research_status()["production_optimizer_uses_transitions"]
            )

    def test_disabled_service_refuses_before_project_access(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ResearchExperimentError):
                run_project_linear_width_experiment(None, {})
            with self.assertRaises(ResearchExperimentError):
                run_project_linear_width_benchmark(None, {})

    def test_candidate_discovery_identifies_supported_hidden_linear(self):
        from Core.ArchitectureDescriptor import ArchitectureDescriptor

        descriptor = ArchitectureDescriptor.default_feedforward(3, 2, hidden=4)
        project = type("Project", (), {"modelFilepath": "model.pt2"})()
        with patch.dict(os.environ, {"MYLO_RESEARCH_API_ENABLED": "true"}):
            with patch(
                "Processing.Optimization.TransitionService._load_descriptor",
                return_value=descriptor,
            ):
                result = discover_linear_width_candidates(project)

        self.assertGreater(result["candidateCount"], 0)
        self.assertEqual(result["eligibleNodes"][0]["nodeId"], "dense1")


if __name__ == "__main__":
    unittest.main()
