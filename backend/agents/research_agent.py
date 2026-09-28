import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# =========================================================
# PROJECT ROOT
# =========================================================

project_root = Path(__file__).resolve().parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# =========================================================
# PROJECT IMPORTS
# =========================================================

from backend.query_planner import QueryPlanner
from backend.search import SearchEngine
from backend.source_processor import SourceProcessor
from backend.scraper import WebScraper
from backend.rag import RAGPipeline
from backend.config import config


class ResearchAgent:
    """
    Research Agent

    Specializes in web-based research.

    Responsibilities:
    1. Receive a research task from ManagerAgent.
    2. Analyze recency requirements.
    3. Generate focused search subqueries.
    4. Search Tavily.
    5. Filter candidate sources.
    6. Scrape relevant webpages.
    7. Index scraped content into Web RAG.
    8. Retrieve relevant evidence.
    9. Return research findings to the orchestrator.

    The ResearchAgent does NOT:
    - decide the overall research strategy
    - write the final report
    - review the final report
    """

    def __init__(
        self,
        gemini_api_key: str = "",
        tavily_api_key: str = ""
    ):

        # -------------------------------------------------
        # API KEYS
        # -------------------------------------------------

        self.gemini_api_key = (
            gemini_api_key
            or config.GEMINI_API_KEY
        )

        self.tavily_api_key = (
            tavily_api_key
            or config.TAVILY_API_KEY
        )

        # -------------------------------------------------
        # EXISTING COMPONENTS
        # -------------------------------------------------

        self.query_planner = QueryPlanner(
            api_key=self.gemini_api_key
        )

        self.search_engine = SearchEngine(
            api_key=self.tavily_api_key
        )

        self.source_processor = SourceProcessor()

        self.scraper = WebScraper()

        # -------------------------------------------------
        # WEB RAG
        # -------------------------------------------------

        self.rag = RAGPipeline(
            api_key=self.gemini_api_key,
            chunk_size=600,
            chunk_overlap=100,
            collection_name="research_chunks",
            persist_directory="./chroma_db"
        )

    # =====================================================
    # 1. PLAN SUBQUERIES
    # =====================================================

    def plan_subqueries(
        self,
        topic: str,
        number_of_queries: int = 5
    ) -> List[str]:
        """
        Converts a research topic/task into
        focused search queries.
        """

        if not topic or not topic.strip():
            return []

        try:

            queries = (
                self.query_planner.generate_subqueries(
                    topic,
                    number_of_queries=number_of_queries
                )
            )

        except Exception as e:

            print(
                f"[ResearchAgent] "
                f"Subquery generation failed: {e}"
            )

            # Safe fallback
            queries = [topic]

        # -------------------------------------------------
        # Remove duplicates
        # -------------------------------------------------

        cleaned_queries = []

        seen = set()

        for query in queries:

            query = str(query).strip()

            if not query:
                continue

            normalized = query.lower()

            if normalized not in seen:

                seen.add(normalized)
                cleaned_queries.append(query)

        return cleaned_queries[:number_of_queries]

    # =====================================================
    # 2. GET TIME RANGE
    # =====================================================

    def get_time_range(
        self,
        topic: str
    ) -> Optional[str]:
        """
        Determines the Tavily time range required
        for the research topic.
        """

        if hasattr(
            self.query_planner,
            "analyzer"
        ):

            return (
                self.query_planner
                .analyzer
                .get_time_range(topic)
            )

        return None

    # =====================================================
    # 3. CHECK RECENCY
    # =====================================================

    def requires_recent(
        self,
        topic: str
    ) -> bool:
        """
        Determines whether the research requires
        recent information.
        """

        if hasattr(
            self.query_planner,
            "analyzer"
        ):

            return (
                self.query_planner
                .analyzer
                .requires_recent_sources(topic)
            )

        return False

    # =====================================================
    # 4. SEARCH WEB
    # =====================================================

    def search_web(
        self,
        subqueries: List[str],
        max_results_per_query: int = 5,
        time_range: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes Tavily search across all subqueries.
        """

        if not subqueries:
            return []

        return self.search_engine.search_multiple(
            queries=subqueries,
            max_results_per_query=max_results_per_query,
            time_range=time_range
        )

    # =====================================================
    # 5. FILTER SOURCES
    # =====================================================

    def filter_sources(
        self,
        raw_sources: List[Dict[str, Any]],
        topic: str,
        is_recent: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Removes irrelevant, blocked, duplicate and
        low-quality sources.
        """

        if not raw_sources:
            return []

        relevant_sources = (
            self.source_processor.process_sources(
                raw_sources,
                topic=topic,
                prioritize_recent=is_recent,
                max_sources=8
            )
        )

        print(
            f"[ResearchAgent] "
            f"Source filtering: "
            f"{len(raw_sources)} -> "
            f"{len(relevant_sources)}"
        )

        return relevant_sources

    # =====================================================
    # 6. SCRAPE SOURCES
    # =====================================================

    def scrape_sources(
        self,
        sources: List[Dict[str, Any]],
        topic: str
    ) -> List[Dict[str, Any]]:
        """
        Scrapes useful content from selected URLs.
        """

        if not sources:
            return []

        findings = (
            self.scraper.scrape_relevant_content(
                sources,
                topic=topic
            )
        )

        print(
            f"[ResearchAgent] "
            f"Successfully scraped "
            f"{len(findings)} sources."
        )

        return findings

    # =====================================================
    # 7. INDEX FINDINGS INTO WEB RAG
    # =====================================================

    def index_into_rag(
        self,
        findings: List[Dict[str, Any]]
    ) -> int:
        """
        Converts scraped content into chunks,
        generates embeddings and stores them in ChromaDB.
        """

        if not findings:
            return 0

        try:

            self.rag.clear()

        except Exception as e:

            print(
                f"[ResearchAgent] "
                f"RAG clear warning: {e}"
            )

        try:
            chunk_count = self.rag.index_findings(findings)
        except Exception as e:
            print(f"[ResearchAgent] RAG indexing warning: {e}. Continuing with direct findings.")
            chunk_count = len(findings)

        print(
            f"[ResearchAgent] "
            f"Indexed {chunk_count} "
            f"RAG chunks."
        )

        return chunk_count

    # =====================================================
    # 8. RETRIEVE EVIDENCE
    # =====================================================

    def retrieve_evidence(
        self,
        subqueries: List[str],
        top_k_per_query: int = 3,
        min_score: float = 0.50
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant chunks from
        Web RAG.
        """

        if not subqueries:
            return []

        evidence = (
            self.rag.enhance_report_context(
                subqueries=subqueries,
                top_k_per_query=top_k_per_query,
                min_score=min_score
            )
        )

        print(
            f"[ResearchAgent] "
            f"Retrieved {len(evidence)} "
            f"evidence chunks."
        )

        return evidence

    # =====================================================
    # 9. COMPLETE RESEARCH TASK
    # =====================================================

    def execute_task(
        self,
        task: Dict[str, Any],
        subquery_count: int = 5,
        max_results_per_query: int = 5
    ) -> Dict[str, Any]:
        """
        Executes ONE task received from ManagerAgent.

        Workflow:

        Manager Task
             ↓
        Query Planner
             ↓
        Tavily
             ↓
        Source Processor
             ↓
        Scraper
             ↓
        Web RAG
             ↓
        Evidence
        """

        task_id = task.get(
            "id",
            "unknown_task"
        )

        objective = task.get(
            "objective",
            ""
        ).strip()

        search_query = task.get(
            "search_query",
            ""
        ).strip()

        # -------------------------------------------------
        # Use Manager's search query when available.
        # Otherwise use objective.
        # -------------------------------------------------

        research_topic = (
            search_query
            or objective
        )

        if not research_topic:

            raise ValueError(
                "Research task must contain "
                "an objective or search_query."
            )

        print("\n" + "=" * 60)

        print(
            f"RESEARCH AGENT"
        )

        print(
            f"Task ID: {task_id}"
        )

        print(
            f"Role: {task.get('role', 'researcher')}"
        )

        print(
            f"Objective: {objective}"
        )

        print("=" * 60)

        # -------------------------------------------------
        # STEP 1: RECENCY
        # -------------------------------------------------

        print(
            "\n[1/7] Analyzing recency..."
        )

        time_range = self.get_time_range(
            research_topic
        )

        is_recent = self.requires_recent(
            research_topic
        )

        print(
            f"Time range: "
            f"{time_range or 'all'}"
        )

        print(
            f"Requires recent: "
            f"{is_recent}"
        )

        # -------------------------------------------------
        # STEP 2: QUERY PLANNING
        # -------------------------------------------------

        print(
            "\n[2/7] Generating search subqueries..."
        )

        subqueries = self.plan_subqueries(
            research_topic,
            number_of_queries=subquery_count
        )

        print(
            f"Generated {len(subqueries)} queries:"
        )

        for query in subqueries:

            print(
                f"  -> {query}"
            )

        # -------------------------------------------------
        # STEP 3: TAVILY SEARCH
        # -------------------------------------------------

        print(
            "\n[3/7] Searching Tavily..."
        )

        raw_sources = self.search_web(
            subqueries=subqueries,
            max_results_per_query=max_results_per_query,
            time_range=time_range
        )

        print(
            f"Retrieved "
            f"{len(raw_sources)} raw sources."
        )

        # -------------------------------------------------
        # STEP 4: SOURCE PROCESSING
        # -------------------------------------------------

        print(
            "\n[4/7] Filtering sources..."
        )

        relevant_sources = self.filter_sources(
            raw_sources=raw_sources,
            topic=objective or research_topic,
            is_recent=is_recent
        )

        # -------------------------------------------------
        # STEP 5: SCRAPING
        # -------------------------------------------------

        print(
            "\n[5/7] Scraping relevant sources..."
        )

        findings = self.scrape_sources(
            sources=relevant_sources,
            topic=objective or research_topic
        )

        # -------------------------------------------------
        # STEP 6: RAG INDEXING
        # -------------------------------------------------

        print(
            "\n[6/7] Indexing findings into Web RAG..."
        )

        rag_chunk_count = (
            self.index_into_rag(
                findings
            )
        )

        # -------------------------------------------------
        # STEP 7: RAG RETRIEVAL
        # -------------------------------------------------

        print(
            "\n[7/7] Retrieving relevant evidence..."
        )

        evidence = self.retrieve_evidence(
            subqueries=subqueries,
            top_k_per_query=3,
            min_score=0.50
        )

        # =================================================
        # RETURN RESULT
        # =================================================

        result = {

            "task_id": task_id,

            "agent": "ResearchAgent",

            "role": task.get(
                "role",
                "researcher"
            ),

            "objective": objective,

            "status": "completed",

            "subqueries": subqueries,

            "time_range": time_range,

            "is_recent": is_recent,

            "raw_sources": raw_sources,

            "relevant_sources": relevant_sources,

            "findings": findings,

            "rag_chunk_count": rag_chunk_count,

            "evidence": evidence,

            "source_count": len(
                relevant_sources
            ),

            "findings_count": len(
                findings
            ),

            "evidence_count": len(
                evidence
            )
        }

        print(
            "\n[ResearchAgent] "
            "Research task completed."
        )

        return result

    def execute_research(
        self,
        topic: str,
        subquery_count: int = 5,
        max_results_per_query: int = 5
    ) -> Dict[str, Any]:
        """
        Coordinates full research retrieval for orchestrator / backward compatibility.
        Delegates to execute_task.
        """
        task = {
            "id": "research_task_main",
            "agent": "ResearchAgent",
            "role": "lead_researcher",
            "objective": topic,
            "search_query": topic,
            "priority": "HIGH"
        }
        return self.execute_task(
            task=task,
            subquery_count=subquery_count,
            max_results_per_query=max_results_per_query
        )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    agent = ResearchAgent()

    test_task = {

        "id": "task_1",

        "agent": "ResearchAgent",

        "role": "technology_researcher",

        "objective": (
            "Research recent breakthroughs "
            "in solid-state battery technology"
        ),

        "search_query": (
            "latest solid-state battery breakthroughs"
        ),

        "priority": "HIGH"
    }

    result = agent.execute_task(
        task=test_task,
        subquery_count=2,
        max_results_per_query=2
    )

    print("\n" + "=" * 60)
    print("RESEARCH AGENT RESULT")
    print("=" * 60)

    print(
        "Status:",
        result["status"]
    )

    print(
        "Task:",
        result["task_id"]
    )

    print(
        "Subqueries:",
        len(result["subqueries"])
    )

    print(
        "Raw sources:",
        len(result["raw_sources"])
    )

    print(
        "Relevant sources:",
        result["source_count"]
    )

    print(
        "Scraped findings:",
        result["findings_count"]
    )

    print(
        "RAG chunks:",
        result["rag_chunk_count"]
    )

    print(
        "Evidence chunks:",
        result["evidence_count"]
    )