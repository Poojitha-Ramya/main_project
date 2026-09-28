import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.agents import (
    ManagerAgent,
    ResearchAgent,
    DocumentAgent,
    DocumentLoader,
    CuratorAgent,
    WriterAgent,
    ReviewerAgent,
    AgentOrchestrator
)


def test_agent_package_exports():
    print("\n--- 1. Testing Agent Package Exports ---")
    assert ManagerAgent is not None
    assert ResearchAgent is not None
    assert DocumentAgent is not None
    assert DocumentLoader is not None
    assert CuratorAgent is not None
    assert WriterAgent is not None
    assert ReviewerAgent is not None
    assert AgentOrchestrator is not None
    print("[PASS] All 8 agent entities successfully imported from backend.agents.")


def test_manager_agent():
    print("\n--- 2. Testing ManagerAgent Strategy & Routing ---")
    manager = ManagerAgent()
    
    # Auto with document -> DOCUMENT or HYBRID
    mode_doc = manager.determine_mode("Summarize uploaded document", user_documents=[{"title": "doc.txt", "content": "Sample content"}], requested_mode="auto")
    assert mode_doc.upper() in ("DOCUMENT", "HYBRID"), f"Expected DOCUMENT or HYBRID mode, got {mode_doc}"
    
    # Auto without document -> WEB
    mode_web = manager.determine_mode("Solid state battery", user_documents=None, requested_mode="auto")
    assert mode_web.upper() in ("WEB", "RESEARCH"), f"Expected WEB mode, got {mode_web}"
    
    # Explicit document
    mode_exp = manager.determine_mode("Topic", user_documents=None, requested_mode="document")
    assert mode_exp.upper() == "DOCUMENT"

    plan = manager.formulate_plan("AI Ethics", mode="research")
    assert len(plan) >= 1, f"Expected at least 1 step in research plan, got {len(plan)}"
    print("[PASS] ManagerAgent correctly routes execution modes and formulates plans.")


def test_research_agent_interfaces():
    print("\n--- 3. Testing ResearchAgent Interfaces ---")
    agent = ResearchAgent(gemini_api_key="dummy", tavily_api_key="dummy")
    assert hasattr(agent, "plan_subqueries")
    assert hasattr(agent, "search_web")
    assert hasattr(agent, "execute_research")
    
    is_recent = agent.requires_recent("Latest AI models 2026")
    assert is_recent is True
    print("[PASS] ResearchAgent query planning and recency interfaces verified.")


def test_document_agent():
    print("\n--- 4. Testing DocumentAgent & DocumentLoader ---")
    doc_agent = DocumentAgent(gemini_api_key="dummy")
    assert doc_agent.is_supported("paper.pdf") is True
    assert doc_agent.is_supported("notes.md") is True
    assert doc_agent.is_supported("data.txt") is True
    assert doc_agent.is_supported("image.png") is False

    raw_bytes = b"Section 1: Quantum Supremacy.\nSection 2: Error mitigation with Surface Codes."
    parsed = doc_agent.load_from_bytes(raw_bytes, "quantum_report.txt")
    assert parsed["title"] == "quantum_report.txt"
    assert "Section 1" in parsed["content"]

    chunks = doc_agent.chunk_documents([parsed])
    assert len(chunks) >= 1
    print(f"[PASS] DocumentAgent parsed and chunked test document into {len(chunks)} chunk(s).")


def test_curator_agent():
    print("\n--- 5. Testing CuratorAgent Quality Filtration ---")
    curator = CuratorAgent(gemini_api_key="dummy")
    mock_sources = [
        {"title": "Quantum Advances 2026", "url": "https://nature.com/article1", "snippet": "Breakthrough in topological qubits.", "score": 0.95},
        {"title": "Duplicate URL", "url": "https://nature.com/article1", "snippet": "Duplicate.", "score": 0.95},
        {"title": "Blocked Social", "url": "https://twitter.com/post1", "snippet": "Check out twitter", "score": 0.9},
        {"title": "Irrelevant Recipe", "url": "https://recipes.com/soup", "snippet": "Tomato soup recipe", "score": 0.1}
    ]
    filtered = curator.filter_sources(mock_sources, topic="Quantum Advances")
    urls = curator.extract_relevant_urls(filtered)
    assert len(urls) == 1, f"Expected 1 filtered URL, got {len(urls)}"
    assert "nature.com" in urls[0]
    print(f"[PASS] CuratorAgent filtered sources down to {urls}.")


def test_writer_and_reviewer_agents():
    print("\n--- 6. Testing WriterAgent & ReviewerAgent ---")
    writer = WriterAgent(gemini_api_key="dummy")
    assert hasattr(writer, "draft_research_report")
    assert hasattr(writer, "draft_document_report")

    reviewer = ReviewerAgent(gemini_api_key="dummy")
    draft = "## Executive Summary\n\nQuantum processors made substantial progress."
    sources = [{"title": "Nature Physics", "url": "https://nature.com/physics"}]
    polished = reviewer.review_and_polish("Quantum Processors", draft, sources)
    assert "https://nature.com/physics" in polished
    print("[PASS] WriterAgent and ReviewerAgent contract and safety fallbacks verified.")


async def test_orchestrator_integration():
    print("\n--- 7. Testing AgentOrchestrator End-to-End ---")
    orchestrator = AgentOrchestrator()
    
    # Backward compatibility checks
    assert hasattr(orchestrator, "query_planner")
    assert hasattr(orchestrator, "search_engine")
    assert hasattr(orchestrator, "source_processor")
    assert hasattr(orchestrator, "scraper")
    assert hasattr(orchestrator, "report_generator")
    assert hasattr(orchestrator, "chunker")

    sample_doc = [{
        "title": "Battery_Milestone_2026.txt",
        "content": "The solid-state battery achieved 520 Wh/kg in continuous lab cycling."
    }]

    res = await orchestrator.run_research("What energy density was achieved?", user_documents=sample_doc, mode="document")
    assert res.get("mode") == "document"
    assert res.get("subqueries") == []
    assert res.get("findings_count") > 0
    assert len(res.get("report", "")) > 50
    print("[PASS] AgentOrchestrator coordinated Document Mode with 0 subqueries.")


def test_review_revision_loop():
    print("\n--- 8. Testing Conditional Review-Revision Loop ---")
    orchestrator = AgentOrchestrator()

    # Test Case 1: Reviewer APPROVES -> revision should NOT be called (0 revisions)
    calls = {"revise_count": 0}

    def mock_revise(objective, draft_report, feedback, evidence):
        calls["revise_count"] += 1
        return draft_report + "\n[Revised]"

    orchestrator.writer_agent.revise_report = mock_revise

    class MockApproveResult:
        decision = "APPROVE"
        overall_score = 0.95
        issues = []
        unsupported_claims = []
        recommendations = []
        summary = "All good"

    orchestrator.reviewer_agent.review_report = lambda objective, report, evidence: MockApproveResult()

    result_approved = orchestrator._run_review_and_revision_loop(
        topic="Test Topic",
        draft_report="Initial Draft",
        evidence=[],
        sources=[]
    )
    assert calls["revise_count"] == 0, f"Expected 0 revisions when approved, got {calls['revise_count']}"
    print("  -> Passed: Revision loop DID NOT run when review approved.")

    # Test Case 2: Reviewer REJECTS first, then APPROVES -> revision runs exactly once
    review_state = {"attempt": 0}

    class MockRejectResult:
        decision = "REJECT"
        overall_score = 0.40
        issues = ["Missing details"]
        unsupported_claims = ["Unproven claim"]
        recommendations = ["Add more data"]
        summary = "Needs work"

    def mock_review_dynamic(objective, report, evidence):
        review_state["attempt"] += 1
        if review_state["attempt"] == 1:
            return MockRejectResult()
        return MockApproveResult()

    orchestrator.reviewer_agent.review_report = mock_review_dynamic

    result_revised = orchestrator._run_review_and_revision_loop(
        topic="Test Topic",
        draft_report="Initial Draft",
        evidence=[],
        sources=[]
    )
    assert calls["revise_count"] == 1, f"Expected exactly 1 revision, got {calls['revise_count']}"
    assert "[Revised]" in result_revised
    print("  -> Passed: Revision loop executed ONLY when review rejected!")
    print("[PASS] Conditional review-revision loop logic fully verified.")


def main():
    test_agent_package_exports()
    test_manager_agent()
    test_research_agent_interfaces()
    test_document_agent()
    test_curator_agent()
    test_writer_and_reviewer_agents()
    asyncio.run(test_orchestrator_integration())
    test_review_revision_loop()
    print("\n=======================================================")
    print("ALL MULTI-AGENT ARCHITECTURE UNIT & INTEGRATION TESTS PASSED!")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
