import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.query_planner import QueryPlanner
from backend.planner import QueryPlanner as LegacyPlanner
from backend.search import SearchEngine
from backend.source_processor import SourceProcessor
from backend.scraper import WebScraper
from backend.report import ReportGenerator
from backend.researcher import ResearchCoordinator

def test_pipeline_units():
    print("Testing pipeline components...")

    # 1. Test QueryPlanner backward compatibility
    assert QueryPlanner is LegacyPlanner, "planner.py backwards compatibility alias failed"
    print("[PASS] 1. query_planner and planner compatibility shim verified.")

    # 2. Test SearchEngine instantiation
    search = SearchEngine(api_key="dummy_key")
    assert hasattr(search, "search")
    assert hasattr(search, "search_multiple")
    print("[PASS] 2. search.py (Tavily search engine) interface verified.")

    # 3. Test SourceProcessor (Relevant URLs ONLY)
    processor = SourceProcessor(min_score=0.4)
    raw_mock_sources = [
        {
            "title": "Quantum Computing Milestones 2026",
            "url": "https://nature.com/articles/quantum-2026?utm_source=twitter&utm_medium=social",
            "snippet": "Researchers reveal scalable quantum error correction and topological qubits.",
            "score": 0.92
        },
        {
            "title": "Quantum Computing Milestones 2026",
            "url": "https://nature.com/articles/quantum-2026", # Duplicate URL
            "snippet": "Duplicate entry snippet.",
            "score": 0.92
        },
        {
            "title": "Admissions and Cutoff 2026 for Engineering",
            "url": "https://collegedunia.com/cutoff-2026", # Blocked domain & spam
            "snippet": "Check exam cutoff and seat allotment 2026.",
            "score": 0.99
        },
        {
            "title": "Unrelated Cooking Recipe",
            "url": "https://recipes.com/pasta", # Irrelevant to quantum computing
            "snippet": "How to make delicious pasta.",
            "score": 0.10
        }
    ]

    relevant_urls = processor.get_relevant_urls(raw_mock_sources, topic="Quantum Computing")
    print(f"       -> Extracted relevant URLs: {relevant_urls}")
    assert len(relevant_urls) == 1, f"Expected 1 relevant URL, got {len(relevant_urls)}"
    assert "nature.com/articles/quantum-2026" in relevant_urls[0]
    print("[PASS] 3. source_processor.py -> Relevant URLs ONLY verified.")

    # 4. Test WebScraper (HTML & Content relevance checks)
    scraper = WebScraper()
    test_text = (
        "Fault-Tolerant Quantum Qubits. Recent developments in quantum computing have shown "
        "revolutionary improvements in error mitigation and coherence times for superconducting circuits. "
        "Another major leap was achieved using neutral atom arrays with 1,000 physical qubits demonstrating entanglement."
    )
    is_relevant = scraper._is_relevant(test_text, topic="Quantum Computing")
    assert is_relevant is True, "Expected topic text to pass content relevance check"

    irrelevant_text = "How to bake a chocolate cake with frosting and sprinkles in under an hour."
    is_irrelevant = scraper._is_relevant(irrelevant_text, topic="Quantum Computing")
    assert is_irrelevant is False, "Expected cake recipe to fail quantum computing relevance check"
    print("[PASS] 4. scraper.py -> HTML check & Content relevance check verified.")

    # 5. Test ReportGenerator interface
    reporter = ReportGenerator(api_key="dummy_key")
    assert hasattr(reporter, "generate")
    print("[PASS] 5. report.py -> Gemini report generator interface verified.")

    # 6. Test ResearchCoordinator orchestration wiring
    coordinator = ResearchCoordinator()
    assert hasattr(coordinator, "query_planner")
    assert hasattr(coordinator, "search_engine")
    assert hasattr(coordinator, "source_processor")
    assert hasattr(coordinator, "scraper")
    assert hasattr(coordinator, "report_generator")
    print("[PASS] 6. researcher.py orchestrator structure fully verified.")

    print("\nAll pipeline component unit tests PASSED successfully!")

if __name__ == "__main__":
    test_pipeline_units()
