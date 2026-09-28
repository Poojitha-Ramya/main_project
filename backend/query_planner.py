import os
import json
import re
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from google import genai
from backend.config import config, get_fallback_chain

# Ensure .env is loaded
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class QueryAnalyzer:
    """Analyzes search queries to identify characteristics such as recency requirements."""

    RECENT_KEYWORDS = {
        "latest",
        "recent",
        "today",
        "current",
        "currently",
        "new",
        "newest",
        "this week",
        "this month",
        "this year",
        "last 7 days",
        "2026"
    }

    def requires_recent_sources(self, query: str) -> bool:
        """
        Check if the research question asks for recent or current information.
        """
        query = query.lower()
        return any(
            keyword in query
            for keyword in self.RECENT_KEYWORDS
        )

    def get_time_range(self, query: str) -> Optional[str]:
        """
        Returns 'week' if user prefers very recent info (last 7 days / latest / today),
        'month' if generally recent, or None.
        """
        q = query.lower()
        if any(k in q for k in ("today", "this week", "latest", "newest", "last 7 days")):
            return "week"
        elif self.requires_recent_sources(query):
            return "month"
        return None


class QueryPlanner:
    """Generates focused research subqueries from a user's question using Gemini."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.analyzer = QueryAnalyzer()

    def generate_subqueries(
        self,
        question: str,
        number_of_queries: int = 5
    ) -> List[str]:
        """
        Deconstructs a user's research topic into focused search subqueries using Gemini.
        """
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY", "")
            if self.api_key:
                self.client = genai.Client(api_key=self.api_key)

        if not self.client:
            raise ValueError("GEMINI_API_KEY is missing. Please set it in your .env file.")

        # Check if query requires recent or current information
        is_recent = self.analyzer.requires_recent_sources(question)
        recency_instruction = (
            "- The user requires recent/current sources. Ensure queries focus on the latest developments, recent news, and 2026 data.\n"
            if is_recent else ""
        )

        prompt = f"""
You are a research planning assistant.

The user wants to research this question:

"{question}"

Break this broad research question into
{number_of_queries} focused search queries.

Each query should investigate a different
important aspect of the original question.

Rules:
- Queries must be useful for web search.
- Avoid duplicate queries.
- Keep each query concise and informative.
- Every query must strictly focus on the specific core subject matter of the user question.
- Do NOT generate generic queries about admissions, exam cutoffs, courses, fees, or recruitment unless explicitly requested.
- Do not answer the question.
{recency_instruction}- Return ONLY a JSON array of strings.

Example:

User question:
"How is AI used in fraud detection?"

Output:
[
    "AI techniques used for financial fraud detection",
    "machine learning algorithms for fraud detection",
    "deep learning for transaction fraud detection",
    "real-time AI fraud detection in banking",
    "challenges and limitations of AI fraud detection"
]
"""

        response = None
        models_to_try = get_fallback_chain(getattr(config, "DEFAULT_MODEL", "gemini-3.6-flash"))
        last_error = None

        for idx, model_name in enumerate(models_to_try):
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if response and response.text:
                    break
            except Exception as e:
                last_error = e
                next_model = models_to_try[idx + 1] if idx + 1 < len(models_to_try) else "fallback heuristic"
                print(f"[QueryPlanner] Model {model_name} unavailable: {e}. Redirecting to {next_model}...")
                continue

        if not response or not response.text:
            raise ValueError(f"Failed to generate subqueries from Gemini: {last_error}")

        text = (response.text or "").strip()

        # Remove markdown code fences if Gemini adds them
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        try:
            queries = json.loads(text)

            if not isinstance(queries, list):
                raise ValueError("Gemini did not return a list.")

            return [str(q).strip() for q in queries if str(q).strip()]

        except json.JSONDecodeError:
            # Fallback: attempt to find JSON array inside text
            match = re.search(r"\[\s*\"[^\"]+\"(?:\s*,\s*\"[^\"]+\")*\s*\]", text)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            raise ValueError(
                f"Could not parse Gemini response:\n{text}"
            )


if __name__ == "__main__":
    planner = QueryPlanner()
    try:
        sample_queries = planner.generate_subqueries("Latest advances in solid-state batteries", number_of_queries=3)
        print("Generated subqueries:")
        for q in sample_queries:
            print(f"- {q}")
    except Exception as err:
        print(f"Test run note: {err}")
