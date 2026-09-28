import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.agents.writer_agent import WriterAgent
from backend.agents.orchestrator import AgentOrchestrator
from backend.config import get_groq_fallback_chain


class TestGroqWriterAgent(unittest.TestCase):

    def test_pure_groq_initialization(self):
        """Verify WriterAgent initializes as Groq-only without any Gemini dependencies."""
        writer = WriterAgent(groq_api_key="gsk_test_key", model_name="llama-3.1-8b-instant")
        self.assertEqual(writer.model_name, "llama-3.1-8b-instant")
        self.assertEqual(writer.fallback_models, get_groq_fallback_chain("llama-3.1-8b-instant"))
        self.assertIsNotNone(writer.client)
        self.assertFalse(hasattr(writer, "gemini_api_key"))

    def test_model_name_normalization(self):
        """Verify normalization of various aliases for Llama 3.1 and Llama 3.8."""
        # Llama 3.1
        self.assertEqual(WriterAgent.normalize_model_name("llama-3.1"), "llama-3.1-8b-instant")
        self.assertEqual(WriterAgent.normalize_model_name("llama 3.1"), "llama-3.1-8b-instant")
        self.assertEqual(WriterAgent.normalize_model_name("llama3.1"), "llama-3.1-8b-instant")
        self.assertEqual(WriterAgent.normalize_model_name("llama-3.1-8b-instant"), "llama-3.1-8b-instant")

        # Llama 3.8 / 3 8B
        self.assertEqual(WriterAgent.normalize_model_name("llama-3.8"), "llama3-8b-8192")
        self.assertEqual(WriterAgent.normalize_model_name("llama 3.8"), "llama3-8b-8192")
        self.assertEqual(WriterAgent.normalize_model_name("llama3.8"), "llama3-8b-8192")
        self.assertEqual(WriterAgent.normalize_model_name("llama 3 8b"), "llama3-8b-8192")
        self.assertEqual(WriterAgent.normalize_model_name("llama-3-8b"), "llama3-8b-8192")
        self.assertEqual(WriterAgent.normalize_model_name("llama3-8b-8192"), "llama3-8b-8192")

        # Non-Groq string defaults to Llama 3.1
        self.assertEqual(WriterAgent.normalize_model_name("gemini-3.6-flash"), "llama-3.1-8b-instant")

    def test_fallback_chain_starting_from_38(self):
        """Verify fallback chain when Llama 3.8 is the primary model."""
        writer = WriterAgent(model_name="llama 3.8")
        self.assertEqual(writer.model_name, "llama3-8b-8192")
        self.assertEqual(writer.fallback_models, get_groq_fallback_chain("llama3-8b-8192"))

    def test_groq_direct_summary_generation(self):
        """Verify direct summary generation calls Groq chat completion API."""
        writer = WriterAgent(groq_api_key="gsk_test")
        
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "# Summary: Quantum Computing\n\nExecutive synthesis generated via Groq Llama 3.1."
        mock_res = MagicMock()
        mock_res.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_res
        writer.client = mock_client

        summary = writer.generate_direct_summary("Quantum Computing", tone="Objective", target_words=500)
        self.assertIn("Quantum Computing", summary)
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "llama-3.1-8b-instant")

    def test_groq_report_generation(self):
        """Verify research report generation calls Groq chat completion API."""
        writer = WriterAgent(groq_api_key="gsk_test")
        
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "# Research Report\n\n## Key Findings\n- Breakthrough verified via Groq Llama."
        mock_res = MagicMock()
        mock_res.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_res
        writer.client = mock_client

        report = writer.generate_report(
            objective="Battery Tech",
            evidence=[{"title": "Doc1", "content": "Evidence text", "url": "https://example.com"}]
        )
        self.assertIn("Breakthrough verified via Groq Llama", report)
        mock_client.chat.completions.create.assert_called_once()

    def test_groq_revision_loop(self):
        """Verify report revision calls Groq chat completion API."""
        writer = WriterAgent(groq_api_key="gsk_test")
        
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "# Revised Report\n\nCorrected and grounded via Groq."
        mock_res = MagicMock()
        mock_res.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_res
        writer.client = mock_client

        revised = writer.revise_report(
            objective="Battery Tech",
            draft_report="# Draft",
            feedback={"issues": ["Needs citations"], "unsupported_claims": [], "recommendations": []},
            evidence=[{"title": "Doc1", "content": "Evidence text", "url": "https://example.com"}]
        )
        self.assertIn("Revised Report", revised)

    def test_deterministic_fallback_when_no_groq_key(self):
        """Verify deterministic grounded fallback report is generated when Groq key is not present."""
        writer = WriterAgent(groq_api_key="")
        writer._call_groq_chain = lambda *args, **kwargs: None

        report = writer.generate_report(
            objective="Autonomous Drones",
            evidence=[{"title": "Drone Study", "content": "Flight time increased by 30%.", "url": "https://drones.org"}]
        )
        self.assertIn("Drone Study", report)
        self.assertIn("Autonomous Drones", report)

    def test_orchestrator_architecture_separation(self):
        """Verify AgentOrchestrator uses Groq WriterAgent while other agents use Gemini."""
        orchestrator = AgentOrchestrator(model_name="gemini-3.7-flash")
        
        # WriterAgent must be Groq-powered
        self.assertEqual(orchestrator.writer_agent.model_name, "llama-3.1-8b-instant")
        self.assertIn("llama3-8b-8192", orchestrator.writer_agent.fallback_models)
        self.assertFalse(hasattr(orchestrator.writer_agent, "gemini_api_key"))

        # Other agents remain Gemini-powered
        self.assertEqual(orchestrator.manager.model, "gemini-3.7-flash")
        self.assertEqual(orchestrator.curator_agent.model_name, "gemini-3.7-flash")
        self.assertEqual(orchestrator.reviewer_agent.model_name, "gemini-3.7-flash")

        # Changing Gemini model in orchestrator does NOT alter Groq WriterAgent
        orchestrator.set_model("gemini-3.8-flash")
        self.assertEqual(orchestrator.reviewer_agent.model_name, "gemini-3.8-flash")
        self.assertEqual(orchestrator.writer_agent.model_name, "llama-3.1-8b-instant")

        # Explicitly setting WriterAgent model to Llama 3.8
        orchestrator.set_writer_model("llama 3.8")
        self.assertEqual(orchestrator.writer_agent.model_name, "llama3-8b-8192")


if __name__ == "__main__":
    unittest.main()
