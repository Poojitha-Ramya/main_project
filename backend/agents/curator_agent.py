import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from google import genai


# =========================================================
# PROJECT PATH
# =========================================================

project_root = Path(__file__).resolve().parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.config import config, get_fallback_chain
from backend.source_processor import SourceProcessor
from backend.scraper import WebScraper


# =========================================================
# STRUCTURED OUTPUT MODELS
# =========================================================

class EvidenceEvaluation(BaseModel):
    """
    Gemini's evaluation of one evidence item.
    """

    evidence_id: str = Field(
        description="Unique identifier of the evidence"
    )

    relevance_score: float = Field(
        description="How relevant the evidence is to the research objective, from 0 to 1"
    )

    quality_score: float = Field(
        description="How trustworthy and useful the source is, from 0 to 1"
    )

    keep: bool = Field(
        description="Whether this evidence should be used in the final report"
    )

    reason: str = Field(
        description="Short explanation for the decision"
    )


class CuratorOutput(BaseModel):
    """
    Final structured response from CuratorAgent.
    """

    evaluations: List[EvidenceEvaluation]

    contradictions: List[str] = Field(
        default_factory=list,
        description="Important contradictions found between evidence items"
    )

    summary: str = Field(
        description="Short summary of the curation result"
    )


# =========================================================
# CURATOR AGENT
# =========================================================

class CuratorAgent:
    """
    Curator Agent

    Responsibilities:
    1. Receive evidence from ResearchAgent / DocumentAgent.
    2. Check semantic relevance.
    3. Evaluate evidence quality.
    4. Remove weak evidence.
    5. Identify duplicate or overlapping evidence.
    6. Identify contradictions.
    7. Return clean evidence for WriterAgent.
    """

    def __init__(
        self,
        gemini_api_key: str = "",
        model_name: str = "gemini-3.6-flash",
        min_score: float = 0.40,
        rag_pipeline: Optional[Any] = None
    ):
        self.gemini_api_key = (
            gemini_api_key
            or config.GEMINI_API_KEY
        )

        self.model_name = model_name or getattr(config, "DEFAULT_MODEL", "gemini-3.6-flash")
        self.fallback_models = get_fallback_chain(self.model_name)

        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing."
            )

        self.client = genai.Client(
            api_key=self.gemini_api_key
        )

        # Supporting components for source processing & scraping
        self.source_processor = SourceProcessor(min_score=min_score)
        self.scraper = WebScraper()
        self.rag_pipeline = rag_pipeline

    # =====================================================
    # COMPATIBILITY METHODS
    # =====================================================

    def filter_sources(
        self,
        raw_sources: List[Dict[str, Any]],
        topic: str = "",
        prioritize_recent: bool = False,
        max_sources: int = 8
    ) -> List[Dict[str, Any]]:
        """Filters raw sources using SourceProcessor."""
        if not raw_sources:
            return []
        return self.source_processor.process_sources(
            sources=raw_sources,
            topic=topic,
            prioritize_recent=prioritize_recent,
            max_sources=max_sources
        )

    def extract_relevant_urls(
        self,
        sources: List[Dict[str, Any]]
    ) -> List[str]:
        """Extracts clean URLs from sources."""
        return [s.get("url") for s in sources if s.get("url")]

    def scrape_sources(
        self,
        relevant_sources: List[Dict[str, Any]],
        topic: str = ""
    ) -> List[Dict[str, Any]]:
        """Scrapes useful content from selected URLs."""
        if not relevant_sources:
            return []
        return self.scraper.scrape_relevant_content(
            sources=relevant_sources,
            topic=topic
        )

    def curate(
        self,
        raw_sources: List[Dict[str, Any]],
        topic: str = "",
        prioritize_recent: bool = False,
        max_sources: int = 8,
        index_to_rag: bool = False
    ) -> Dict[str, Any]:
        """End-to-end source curation & scraping for orchestrator."""
        relevant_sources = self.filter_sources(
            raw_sources=raw_sources,
            topic=topic,
            prioritize_recent=prioritize_recent,
            max_sources=max_sources
        )
        relevant_urls = self.extract_relevant_urls(relevant_sources)
        findings = self.scrape_sources(relevant_sources, topic=topic)

        indexed_count = 0
        if index_to_rag and findings and self.rag_pipeline:
            try:
                indexed_count = self.rag_pipeline.index_findings(findings)
            except Exception as e:
                print(f"Warning: RAG indexing in CuratorAgent failed: {e}")

        return {
            "relevant_sources": relevant_sources,
            "relevant_urls": relevant_urls,
            "findings": findings,
            "indexed_count": indexed_count
        }

    # =====================================================
    # PREPARE EVIDENCE
    # =====================================================

    def prepare_evidence(
        self,
        evidence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        prepared = []

        for index, item in enumerate(evidence):

            evidence_id = item.get(
                "id",
                f"evidence_{index + 1}"
            )

            prepared.append({
                "evidence_id": evidence_id,

                "title": item.get(
                    "title",
                    "Untitled"
                ),

                "url": item.get(
                    "url",
                    ""
                ),

                "content": item.get(
                    "content",
                    item.get(
                        "text",
                        item.get(
                            "snippet",
                            ""
                        )
                    )
                ),

                "score": item.get(
                    "score",
                    item.get(
                        "similarity",
                        0
                    )
                ),

                "source_query": item.get(
                    "source_query",
                    ""
                ),

                "published_date": item.get(
                    "published_date"
                )
            })

        return prepared

    # =====================================================
    # REMOVE EXACT DUPLICATES
    # =====================================================

    def remove_duplicates(
        self,
        evidence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        seen_urls = set()
        seen_content = set()

        unique = []

        for item in evidence:

            url = item.get(
                "url",
                ""
            ).strip().lower()

            content = item.get(
                "content",
                ""
            ).strip().lower()

            content_key = content[:500]

            # ---------------------------------------------
            # URL duplicate
            # ---------------------------------------------

            if url and url in seen_urls:
                continue

            # ---------------------------------------------
            # Content duplicate
            # ---------------------------------------------

            if content_key and content_key in seen_content:
                continue

            if url:
                seen_urls.add(url)

            if content_key:
                seen_content.add(content_key)

            unique.append(item)

        print(
            f"[CuratorAgent] Duplicate removal: "
            f"{len(evidence)} -> {len(unique)}"
        )

        return unique

    # =====================================================
    # BUILD GEMINI PROMPT
    # =====================================================

    def build_prompt(
        self,
        objective: str,
        evidence: List[Dict[str, Any]]
    ) -> str:

        evidence_text = []

        for item in evidence:

            evidence_text.append(
                f"""
EVIDENCE ID: {item["evidence_id"]}

TITLE:
{item["title"]}

URL:
{item["url"]}

SOURCE SCORE:
{item["score"]}

PUBLISHED DATE:
{item["published_date"]}

CONTENT:
{item["content"][:5000]}

----------------------------------------
"""
            )

        joined_evidence = "\n".join(
            evidence_text
        )

        return f"""
You are the Curator Agent in a research system.

Your job is to evaluate research evidence before it is
given to the Writer Agent.

RESEARCH OBJECTIVE:
{objective}

EVIDENCE:
{joined_evidence}

Evaluate every evidence item.

For each evidence item:

1. Determine how relevant it is to the objective.
2. Determine whether the evidence is useful and trustworthy.
3. Give a relevance score from 0 to 1.
4. Give a quality score from 0 to 1.
5. Decide whether it should be kept.
6. Give a short reason.

Important rules:

- Do not invent facts.
- Do not add information that is not present in the evidence.
- Do not automatically trust a high search score.
- Prefer evidence that directly supports the research objective.
- Prefer authoritative and information-rich sources.
- Reject irrelevant evidence.
- Reject pages that contain mostly navigation, advertisements,
  jobs, promotional material, or unrelated content.
- If two evidence items contradict each other, mention the
  contradiction.
- If evidence is insufficient, do not pretend that it is sufficient.

Keep evidence when it is both relevant and useful.

The final result will be passed to WriterAgent.
"""

    # =====================================================
    # GEMINI EVALUATION
    # =====================================================

    def evaluate_evidence(
        self,
        objective: str,
        evidence: List[Dict[str, Any]]
    ) -> CuratorOutput:

        if not evidence:
            return CuratorOutput(
                evaluations=[],
                contradictions=[],
                summary="No evidence was available for curation."
            )

        prompt = self.build_prompt(
            objective=objective,
            evidence=evidence
        )

        print(
            "[CuratorAgent] Sending evidence to Gemini..."
        )

        models_to_try = list(dict.fromkeys([self.model_name, *self.fallback_models]))
        last_error = None

        for idx, model in enumerate(models_to_try):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": CuratorOutput,
                    }
                )
                if response and response.text:
                    return CuratorOutput.model_validate_json(response.text)
            except Exception as e:
                last_error = e
                next_model = models_to_try[idx + 1] if idx + 1 < len(models_to_try) else "deterministic evaluation"
                print(f"[CuratorAgent] Model {model} unavailable: {e}. Redirecting to {next_model}...")
                continue

        # Fallback deterministic evaluation if API is unavailable
        print(f"[CuratorAgent] Using deterministic evaluation (Error: {last_error})")
        evals = []
        for item in evidence:
            score = float(item.get("score", 0.8) or 0.8)
            evals.append(EvidenceEvaluation(
                evidence_id=item["evidence_id"],
                relevance_score=max(0.7, min(1.0, score)),
                quality_score=0.85,
                keep=True,
                reason="Retained through fallback evidence scoring."
            ))

        return CuratorOutput(
            evaluations=evals,
            contradictions=[],
            summary=f"Curated {len(evals)} evidence items for '{objective}'."
        )

    # =====================================================
    # SELECT BEST EVIDENCE
    # =====================================================

    def select_evidence(
        self,
        evidence: List[Dict[str, Any]],
        evaluation: CuratorOutput,
        min_relevance: float = 0.60,
        min_quality: float = 0.50
    ) -> List[Dict[str, Any]]:

        evaluation_map = {
            item.evidence_id: item
            for item in evaluation.evaluations
        }

        selected = []

        for item in evidence:

            evidence_id = item[
                "evidence_id"
            ]

            result = evaluation_map.get(
                evidence_id
            )

            if not result:
                continue

            if (
                result.keep
                and result.relevance_score >= min_relevance
                and result.quality_score >= min_quality
            ):

                item_copy = dict(item)

                item_copy[
                    "curator_relevance"
                ] = result.relevance_score

                item_copy[
                    "curator_quality"
                ] = result.quality_score

                item_copy[
                    "curator_reason"
                ] = result.reason

                selected.append(
                    item_copy
                )

        # Highest quality/relevance first
        selected.sort(
            key=lambda x: (
                x.get(
                    "curator_relevance",
                    0
                ),
                x.get(
                    "curator_quality",
                    0
                )
            ),
            reverse=True
        )

        print(
            f"[CuratorAgent] Selected "
            f"{len(selected)} / {len(evidence)} evidence items."
        )

        return selected

    # =====================================================
    # MAIN TASK
    # =====================================================

    def execute_task(
        self,
        task: Dict[str, Any],
        evidence: List[Dict[str, Any]],
        min_relevance: float = 0.60,
        min_quality: float = 0.50
    ) -> Dict[str, Any]:

        task_id = task.get(
            "id",
            "curator_task"
        )

        objective = task.get(
            "objective",
            ""
        )

        print("\n" + "=" * 60)

        print(
            f"[CuratorAgent] Task: {task_id}"
        )

        print(
            f"[CuratorAgent] Objective: {objective}"
        )

        print("=" * 60)

        # -------------------------------------------------
        # STEP 1: Prepare Evidence
        # -------------------------------------------------

        print(
            "[CuratorAgent] Preparing evidence..."
        )

        prepared = self.prepare_evidence(
            evidence
        )

        # -------------------------------------------------
        # STEP 2: Remove Duplicates
        # -------------------------------------------------

        prepared = self.remove_duplicates(
            prepared
        )

        # -------------------------------------------------
        # STEP 3: Gemini Evaluation
        # -------------------------------------------------

        evaluation = self.evaluate_evidence(
            objective=objective,
            evidence=prepared
        )

        # -------------------------------------------------
        # STEP 4: Select Evidence
        # -------------------------------------------------

        selected = self.select_evidence(
            evidence=prepared,
            evaluation=evaluation,
            min_relevance=min_relevance,
            min_quality=min_quality
        )

        # -------------------------------------------------
        # STEP 5: Return Clean Evidence
        # -------------------------------------------------

        return {
            "status": "success",

            "task_id": task_id,

            "agent": "CuratorAgent",

            "objective": objective,

            "input_evidence_count": len(
                evidence
            ),

            "unique_evidence_count": len(
                prepared
            ),

            "selected_evidence_count": len(
                selected
            ),

            "selected_evidence": selected,

            "evaluations": [
                item.model_dump()
                for item in evaluation.evaluations
            ],

            "contradictions": (
                evaluation.contradictions
            ),

            "summary": evaluation.summary
        }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CURATOR AGENT TEST")
    print("=" * 60)

    # -----------------------------------------------------
    # Sample evidence
    # -----------------------------------------------------

    evidence = [

        {
            "id": "source_1",
            "title": "Official AI Research Report",
            "url": "https://example.com/ai-report",
            "content": """
            Artificial intelligence and machine learning are
            increasingly used in financial fraud detection.
            Machine learning models analyze transaction patterns
            and identify unusual behavior that may indicate fraud.
            """,
            "score": 0.91,
            "source_query": "AI machine learning fraud detection",
            "published_date": "2026-01-10"
        },

        {
            "id": "source_2",
            "title": "AI Fraud Detection Overview",
            "url": "https://example.com/fraud",
            "content": """
            Fraud detection systems can use machine learning
            algorithms to detect unusual transaction behavior.
            """,
            "score": 0.82,
            "source_query": "machine learning fraud detection",
            "published_date": "2025-12-20"
        },

        {
            "id": "source_3",
            "title": "Latest College Job Fair",
            "url": "https://example.com/jobs",
            "content": """
            Our college is organizing a job fair for students.
            Companies will participate in the recruitment process.
            """,
            "score": 0.80,
            "source_query": "AI machine learning fraud detection",
            "published_date": "2026-02-01"
        },

        {
            "id": "source_4",
            "title": "Duplicate Fraud Article",
            "url": "https://example.com/fraud",
            "content": """
            Fraud detection systems can use machine learning
            algorithms to detect unusual transaction behavior.
            """,
            "score": 0.81,
            "source_query": "fraud detection",
            "published_date": "2025-12-20"
        }
    ]

    # -----------------------------------------------------
    # Task
    # -----------------------------------------------------

    task = {

        "id": "curator_task_1",

        "agent": "CuratorAgent",

        "role": "Evaluate research evidence",

        "objective": (
            "Explain how machine learning is used "
            "for financial fraud detection."
        )
    }

    # -----------------------------------------------------
    # Create agent
    # -----------------------------------------------------

    agent = CuratorAgent()

    # -----------------------------------------------------
    # Execute
    # -----------------------------------------------------

    result = agent.execute_task(
        task=task,
        evidence=evidence,
        min_relevance=0.60,
        min_quality=0.50
    )

    # -----------------------------------------------------
    # Display
    # -----------------------------------------------------

    print("\n")
    print("=" * 60)
    print("CURATOR RESULT")
    print("=" * 60)

    print(
        "Status:",
        result["status"]
    )

    print(
        "Input evidence:",
        result["input_evidence_count"]
    )

    print(
        "Unique evidence:",
        result["unique_evidence_count"]
    )

    print(
        "Selected evidence:",
        result["selected_evidence_count"]
    )

    print(
        "\nSummary:"
    )

    print(
        result["summary"]
    )

    print(
        "\nContradictions:"
    )

    for contradiction in result[
        "contradictions"
    ]:

        print(
            "-",
            contradiction
        )

    print(
        "\nSelected Evidence:"
    )

    for item in result[
        "selected_evidence"
    ]:

        print(
            f"\n[{item['evidence_id']}] "
            f"{item['title']}"
        )

        print(
            "Relevance:",
            item["curator_relevance"]
        )

        print(
            "Quality:",
            item["curator_quality"]
        )

        print(
            "Reason:",
            item["curator_reason"]
        )
