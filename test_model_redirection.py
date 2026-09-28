"""
Test script to verify dynamic model redirection across Gemini 3.6, 3.7, 3.8, and 3.5 Flash.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.config import get_fallback_chain, config, SUPPORTED_FLASH_MODELS
from backend.agents.writer_agent import WriterAgent
from backend.agents.reviewer_agent import ReviewerAgent


class TestModelRedirection(unittest.TestCase):

    def test_fallback_chains_order(self):
        """Verify priority order for each 3.x Flash model."""
        # 3.6 Flash fallback order
        chain_36 = get_fallback_chain("gemini-3.6-flash")
        self.assertEqual(chain_36[0], "gemini-3.6-flash")
        self.assertIn("gemini-3.7-flash", chain_36)
        self.assertIn("gemini-3.8-flash", chain_36)
        self.assertIn("gemini-3.5-flash", chain_36)
        self.assertEqual(chain_36, ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.5-flash"])

        # 3.7 Flash fallback order
        chain_37 = get_fallback_chain("gemini-3.7-flash")
        self.assertEqual(chain_37[0], "gemini-3.7-flash")
        self.assertEqual(chain_37, ["gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash"])

        # 3.8 Flash fallback order
        chain_38 = get_fallback_chain("gemini-3.8-flash")
        self.assertEqual(chain_38[0], "gemini-3.8-flash")
        self.assertEqual(chain_38, ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"])

        # Deprecated alias normalization
        chain_dep = get_fallback_chain("gemini-2.5-flash")
        self.assertEqual(chain_dep[0], "gemini-3.6-flash")

    def test_writer_agent_groq_llama_redirection(self):
        """Simulate Groq Llama 3.1 rate limit and verify redirect to Llama 3.8 (llama3-8b-8192)."""
        writer = WriterAgent(groq_api_key="gsk_mock", model_name="llama-3.1-8b-instant")
        
        call_models = []

        def mock_create(model, messages, **kwargs):
            call_models.append(model)
            if model == "llama-3.1-8b-instant":
                raise Exception("429 RateLimitError: rate limit reached on llama-3.1-8b-instant")
            mock_res = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = f"# Generated with {model}"
            mock_res.choices = [mock_choice]
            return mock_res

        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create
        writer.client = mock_client
        report = writer.draft_web_report("AI Ethics", [{"title": "Doc1", "content": "Evidence"}])

        # Verify that llama-3.1-8b-instant was attempted, failed, and redirected to llama3-8b-8192!
        self.assertEqual(call_models, ["llama-3.1-8b-instant", "llama3-8b-8192"])
        self.assertIn("Generated with llama3-8b-8192", report)

    def test_reviewer_agent_redirection_from_38(self):
        """Simulate starting with 3.8, failing, and redirecting to 3.7 then 3.6."""
        reviewer = ReviewerAgent(gemini_api_key="mock_key", model_name="gemini-3.8-flash")
        
        call_models = []

        def mock_generate(model, contents, **kwargs):
            call_models.append(model)
            if model == "gemini-3.8-flash":
                raise Exception("503 UNAVAILABLE: high demand spike")
            if model == "gemini-3.7-flash":
                raise Exception("503 UNAVAILABLE: high demand spike")
            mock_res = MagicMock()
            mock_res.text = '{"decision": "APPROVE", "overall_score": 0.95, "factual_grounding": 0.95, "completeness": 0.95, "citation_quality": 0.95, "issues": [], "unsupported_claims": [], "recommendations": [], "summary": "Looks good"}'
            return mock_res

        reviewer.client.models.generate_content = mock_generate
        result = reviewer.review_report("AI Ethics", "# Draft Report", [{"title": "Doc1", "content": "Evidence"}])

        self.assertEqual(call_models, ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"])
        self.assertEqual(result.decision, "APPROVE")


if __name__ == "__main__":
    unittest.main()
