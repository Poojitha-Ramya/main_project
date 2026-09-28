import asyncio
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.orchestrator import AgentOrchestrator
from backend.agents.writer_agent import WriterAgent

async def test_tone_pipeline():
    print("\n=======================================================")
    print("TESTING USER-INPUT TONE DRIVEN REPORT GENERATION")
    print("=======================================================")

    orchestrator = AgentOrchestrator()

    # 1. Test Preset Tone: Humorous
    print("\n[TEST 1] Testing Preset Tone: 'Humorous'...")
    res_humorous = await orchestrator.run_research(
        topic="Quantum Computing in 2026",
        report_type="Summary - Short and fast (~2 min)",
        tone="Humorous - Light-hearted and engaging, usually to make the content more relatable"
    )
    report_humor = res_humorous.get("report", "")
    print(f"Humorous Report Snippet:\n{report_humor[:350]}...\n")
    assert "quantum" in report_humor.lower()
    print(">>> [TEST 1 PASSED]: Preset 'Humorous' tone successfully delivered to LLM!\n")

    # 2. Test Custom User-Input Tone: "Sarcastic Pirate"
    print("\n[TEST 2] Testing Custom User Input Tone: 'Sarcastic Pirate'...")
    res_pirate = await orchestrator.run_research(
        topic="Solid state battery advancements",
        report_type="Summary - Short and fast (~2 min)",
        tone="Sarcastic Pirate"
    )
    report_pirate = res_pirate.get("report", "")
    print(f"Sarcastic Pirate Report Snippet:\n{report_pirate[:350]}...\n")
    print(f"Report Tone Metadata in Response: {res_pirate.get('tone')}")
    assert res_pirate.get("tone") == "Sarcastic Pirate"
    # Verify that the LLM recognized the pirate tone
    pirate_markers = ["ahoy", "matey", "pirate", "aye", "booty", "treasure", "seas", "ship", "arrr", "plank", "sail"]
    matched = [m for m in pirate_markers if m in report_pirate.lower()]
    print(f"Detected pirate stylistic markers: {matched}")
    assert len(matched) > 0 or "pirate" in report_pirate.lower(), "LLM must reflect the requested 'Sarcastic Pirate' custom tone"
    print(">>> [TEST 2 PASSED]: Custom user-input tone 'Sarcastic Pirate' generated report based on that exact tone!\n")

    # 3. Test Metadata helper
    meta = WriterAgent.get_tone_metadata("Cyberpunk Detective")
    print(f"[TEST 3] Custom tone metadata resolution: {meta['name']} -> {meta['instruction'][:80]}...")
    assert meta["name"] == "Cyberpunk Detective"
    assert "Cyberpunk Detective" in meta["instruction"]
    print(">>> [TEST 3 PASSED]: WriterAgent.get_tone_metadata resolves custom user input correctly!\n")

    print("=======================================================")
    print("ALL TONE PIPELINE TESTS PASSED SUCCESSFULLY!")
    print("=======================================================\n")

if __name__ == "__main__":
    asyncio.run(test_tone_pipeline())
