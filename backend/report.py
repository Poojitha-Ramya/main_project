import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from backend.config import config, get_fallback_chain

# Ensure .env is loaded regardless of execution context
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class ReportGenerator:
    """Uses Gemini to synthesize relevant research findings into a comprehensive Markdown report."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    def generate(
        self,
        topic: str,
        findings: List[Dict[str, Any]],
        report_type: str = "Summary - Short and fast (~2 min)",
        tone: str = "Objective - Impartial and unbiased presentation of facts and findings"
    ) -> str:
        """
        Compile vetted research findings into a structured Markdown report using Gemini.
        """
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY", "")
            if self.api_key:
                self.client = genai.Client(api_key=self.api_key)

        if not self.client:
            raise ValueError("GEMINI_API_KEY is not set in your .env file.")

        if not findings:
            return (
                f"# Research Report: {topic}\n\n"
                "## Summary\n\nNo relevant sources could be retrieved for this topic. "
                "Please verify your research topic keywords or API credentials.\n"
            )

        # Prepare the filtered research context for Gemini
        research_context = ""
        for i, item in enumerate(findings, start=1):
            title = item.get("title", "Untitled Source")
            url = item.get("url", "")
            content = item.get("content", "")

            research_context += f"""
--- SOURCE {i} ---
Title: {title}
URL: {url}
Content:
{content}
"""

        prompt = f"""
You are an expert autonomous AI research assistant.

The user requested research on the following topic:

"{topic}"

Below are verified research findings collected and filtered from live web sources:

{research_context}

Instructions:
Using ONLY the information provided in the research findings above:
1. Write an executive summary that provides a high-level overview.
2. Identify the most important Key Findings in bullet points with concise explanations.
3. Provide a thorough, well-structured Detailed Analysis organized logically by subtopic.
4. If sources present conflicting viewpoints or dates, highlight the nuances.
5. Strictly adhere to facts in the sources; do not hallucinate details not present.
6. Format your entire response in GitHub-flavored Markdown.

Use this structure:

## Executive Summary

[Concise overview of the topic based on the sources]

## Key Findings

- **[Key Theme 1]**: [Explanation and evidence]
- **[Key Theme 2]**: [Explanation and evidence]
- **[Key Theme 3]**: [Explanation and evidence]

## Detailed Analysis

### [Subtopic 1]
[In-depth discussion with facts and figures]

### [Subtopic 2]
[In-depth discussion with facts and figures]

Do NOT create a "Sources" or "References" section, as the system appends the verified source list automatically.
"""

        # Send research context to Gemini with multi-model fallback and redirection
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
                next_model = models_to_try[idx + 1] if idx + 1 < len(models_to_try) else "error summary"
                print(f"[ReportGenerator] Model {model_name} unavailable: {e}. Redirecting to {next_model}...")
                continue

        if not response or not response.text:
            synthesis = f"Unable to generate synthesis from the provided sources: {last_error}"
        else:
            synthesis = response.text.strip()

        # Build final Markdown report with verified sources
        report = f"# Research Report: {topic}\n\n"
        report += synthesis
        report += "\n\n## Sources & References\n\n"

        for item in findings:
            title = item.get("title", "Source")
            url = item.get("url", "#")
            report += f"- [{title}]({url})\n"

        return report


if __name__ == "__main__":
    reporter = ReportGenerator()
    test_findings = [{
        "title": "Quantum Leap 2026",
        "url": "https://example.com/quantum",
        "content": "Logical qubits have achieved below 0.1% physical error rates with surface codes in 2026."
    }]
    try:
        report = reporter.generate("Quantum Computing Error Correction", test_findings)
        print("Generated Report Preview:\n")
        print(report[:400])
    except Exception as err:
        print(f"Test run note: {err}")
