import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from backend.main import app
from backend.agents.chat_agent import SummaryChatAgent


class TestChatAssistant(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.sample_summary = """
# Quantum Computing Advances in 2026

Recent quantum processors have surpassed 1,000 physical qubits with error-mitigation techniques.
Neutral atom architectures and topological qubits have emerged as promising competitors to superconducting circuits.

## Key Findings
1. Quantum error correction achieved below fault-tolerance thresholds in dual-rail qubit experiments.
2. Major limitation: Cryogenic cooling overhead and coherence times remain a practical barrier.
3. Industry applications include molecular simulation for green ammonia synthesis.
"""

    def test_validation_empty_summary(self):
        """Endpoint should return 400 when research summary is missing or empty."""
        response = self.client.post("/api/chat", json={
            "question": "What are the key findings?",
            "summary": "   ",
            "topic": "Quantum Computing"
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("only available after a research summary is generated", response.json()["detail"])

    def test_validation_empty_question(self):
        """Endpoint should return 400 when question is empty."""
        response = self.client.post("/api/chat", json={
            "question": "   ",
            "summary": self.sample_summary,
            "topic": "Quantum Computing"
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("Question cannot be empty", response.json()["detail"])

    def test_chat_agent_system_instruction(self):
        """Verify prompt builder includes strict boundary guardrails and research summary."""
        agent = SummaryChatAgent()
        prompt = agent._build_system_instruction(
            topic="Quantum Computing",
            summary=self.sample_summary,
            sources=[{"title": "Nature Physics", "url": "https://nature.com/example", "snippet": "Dual rail qubits"}]
        )
        self.assertIn("ONLY ANSWER ABOUT THE RESEARCH SUMMARY", prompt)
        self.assertIn("STRICT GUARDRAIL FOR OUT-OF-SCOPE QUESTIONS", prompt)
        self.assertIn("Quantum Computing Advances in 2026", prompt)
        self.assertIn("Nature Physics", prompt)

    def test_chat_agent_answer_mocked(self):
        """Verify chat agent returns structured answer with model info."""
        agent = SummaryChatAgent(gemini_api_key="mock_key")
        
        mock_response = MagicMock()
        mock_response.text = "Based on the summary, the main limitation is cryogenic cooling overhead."
        
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        agent.gemini_client = mock_client

        result = agent.answer_question(
            question="What is the main limitation?",
            summary=self.sample_summary,
            topic="Quantum Computing"
        )
        self.assertEqual(result["status"], "success")
        self.assertIn("cryogenic cooling overhead", result["answer"])
        self.assertEqual(result["provider"], "Gemini")

    def test_chat_agent_groq_fallback(self):
        """Verify failover to Groq when Gemini fails."""
        agent = SummaryChatAgent(gemini_api_key="mock_key", groq_api_key="mock_groq")
        
        mock_gemini = MagicMock()
        mock_gemini.models.generate_content.side_effect = Exception("503 Service Unavailable")
        agent.gemini_client = mock_gemini

        mock_groq = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "According to the research summary, dual-rail qubits achieved fault tolerance."
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_groq.chat.completions.create.return_value = mock_completion
        agent.groq_client = mock_groq

        result = agent.answer_question(
            question="What did dual-rail qubits achieve?",
            summary=self.sample_summary,
            topic="Quantum Computing"
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["provider"], "Groq")
        self.assertIn("dual-rail qubits", result["answer"])


if __name__ == "__main__":
    unittest.main()
