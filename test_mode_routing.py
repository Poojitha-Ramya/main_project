import asyncio
import sys
from pathlib import Path

# Ensure root is in path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.researcher import ResearchCoordinator

async def main():
    print("\n=======================================================")
    print("Testing Dual-Mode Routing Architecture")
    print("=======================================================")

    coordinator = ResearchCoordinator()

    # ---------------------------------------------------------
    # TEST 1: DOCUMENT MODE
    # Flow: Read doc -> Chunk doc -> Gemini / RAG -> Summary
    # Constraint: subqueries == [] (Strictly NO Tavily, NO subqueries, NO scraper)
    # ---------------------------------------------------------
    print("\n[TEST 1] Testing DOCUMENT MODE execution...")
    sample_docs = [
        {
            "title": "Quantum_Hardware_2026.txt",
            "content": (
                "In early 2026, researchers demonstrated topological qubit coherence "
                "surpassing 1.2 milliseconds at 15 millikelvin. The system utilized "
                "Majorana zero modes on semiconductor-superconductor nanowires. "
                "Error rates dropped below the fault-tolerance threshold of 0.1%."
            )
        }
    ]

    doc_result = await coordinator.run_research(
        topic="What is the coherence time and error rate achieved?",
        user_documents=sample_docs,
        mode="document"
    )

    print("Document Mode Verification:")
    print(f"  Mode: {doc_result.get('mode')}")
    print(f"  Subqueries count: {len(doc_result.get('subqueries', []))}")
    print(f"  Findings/Chunks count: {doc_result.get('findings_count')}")
    print(f"  Report character length: {len(doc_result.get('report', ''))}")
    print("  Report preview:\n", doc_result.get('report')[:350], "...\n")

    assert doc_result.get("mode") == "document", f"Expected mode 'document', got {doc_result.get('mode')}"
    assert len(doc_result.get("subqueries", [])) == 0, f"Expected 0 subqueries in Document Mode, got {len(doc_result.get('subqueries', []))}"
    assert doc_result.get("findings_count") > 0, "Expected at least 1 chunk in findings_count"
    assert len(doc_result.get("report", "")) > 50, "Expected non-empty report"
    print(">>> [TEST 1 PASSED]: Document Mode executes cleanly with 0 subqueries and grounded synthesis!")

    # ---------------------------------------------------------
    # TEST 2: AUTO MODE WITH UPLOADED DOC
    # Flow: Automatically routes to DOCUMENT MODE
    # ---------------------------------------------------------
    print("\n[TEST 2] Testing AUTO MODE with uploaded document...")
    auto_result = await coordinator.run_research(
        topic="Summarize topological qubit achievements",
        user_documents=sample_docs,
        mode="auto"
    )
    assert auto_result.get("mode") == "document"
    assert len(auto_result.get("subqueries", [])) == 0
    print(">>> [TEST 2 PASSED]: Auto Mode cleanly routes to Document Mode when documents are present!")

    print("\n=======================================================")
    print("ALL ROUTING & CONSTRAINT TESTS PASSED!")
    print("=======================================================")

if __name__ == "__main__":
    asyncio.run(main())
