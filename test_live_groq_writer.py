import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.config import config
from backend.agents.writer_agent import WriterAgent

def run_live_writer_tests():
    print("=" * 70)
    print("TESTING WRITER AGENT WITH LIVE GROQ API")
    print("=" * 70)

    print(f"[Config] GROQ_API_KEY detected: {bool(config.GROQ_API_KEY)} (Length: {len(config.GROQ_API_KEY)})")
    print(f"[Config] Default GROQ Model: {getattr(config, 'GROQ_WRITER_MODEL', 'llama-3.1-8b-instant')}")

    if not config.GROQ_API_KEY:
        print("[ERROR] GROQ_API_KEY is not configured in .env file.")
        return

    # Sample curated evidence for grounded testing
    sample_evidence = [
        {
            "title": "Quantum Supremacy and Error Correction 2026",
            "url": "https://nature.com/articles/quantum-2026",
            "content": (
                "In 2026, researchers achieved fault-tolerant quantum error correction with logical qubits "
                "demonstrating sub-0.01% error rates using surface codes. Modular quantum architectures "
                "now scale past 1,000 physical qubits while maintaining coherence across multi-chip interconnects."
            )
        },
        {
            "title": "Commercial Quantum Deployments in Finance & Cryptography",
            "url": "https://ieee.org/quantum-deployments",
            "content": (
                "Financial institutions have begun transitioning to post-quantum cryptography (PQC) standards "
                "such as ML-KEM and ML-DSA while utilizing quantum-assisted Monte Carlo simulations for portfolio "
                "risk analysis with a 40x speedup compared to classical supercomputing clusters."
            )
        }
    ]

    # -------------------------------------------------------------
    # TEST 1: Writer Agent with Groq Llama 3.1 (llama-3.1-8b-instant)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[TEST 1] Testing WriterAgent on Groq Llama 3.1 ('llama-3.1-8b-instant')...")
    print("-" * 70)

    writer_31 = WriterAgent(model_name="llama-3.1-8b-instant")
    print(f"WriterAgent active model: {writer_31.model_name}")
    print(f"WriterAgent fallback chain: {writer_31.fallback_models}")

    t0 = time.time()
    report_31 = writer_31.generate_report(
        objective="Analyze recent breakthroughs in quantum computing and commercial adoption in 2026",
        evidence=sample_evidence,
        report_type="Summary - Short and fast (~2 min)",
        tone="Objective"
    )
    t_31 = time.time() - t0

    print(f"\n[TEST 1 SUCCESS] Generated in {t_31:.2f} seconds!")
    print("\n--- REPORT OUTPUT (Llama 3.1) ---")
    print(report_31[:800] + ("\n... [truncated]" if len(report_31) > 800 else ""))

    # -------------------------------------------------------------
    # TEST 2: Writer Agent with Groq Llama 3.8 / 3 8B ('llama 3.8')
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[TEST 2] Testing WriterAgent on Groq Llama 3.8 ('llama 3.8' alias)...")
    print("-" * 70)

    writer_38 = WriterAgent(model_name="llama 3.8")
    print(f"WriterAgent active model: {writer_38.model_name}")
    print(f"WriterAgent fallback chain: {writer_38.fallback_models}")

    t0 = time.time()
    report_38 = writer_38.generate_report(
        objective="Explain post-quantum cryptography migration and risk analysis",
        evidence=sample_evidence,
        report_type="Summary - Short and fast (~2 min)",
        tone="Analytical"
    )
    t_38 = time.time() - t0

    print(f"\n[TEST 2 SUCCESS] Generated in {t_38:.2f} seconds!")
    print("\n--- REPORT OUTPUT (Llama 3.8 / Llama 3 8B) ---")
    print(report_38[:800] + ("\n... [truncated]" if len(report_38) > 800 else ""))

    # -------------------------------------------------------------
    # TEST 3: Direct LLM Summary (~500 words) via Groq
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[TEST 3] Testing Direct LLM Executive Summary via Groq...")
    print("-" * 70)

    t0 = time.time()
    direct_summary = writer_31.generate_direct_summary(
        objective="Solid State Batteries in Electric Vehicles",
        tone="Informative",
        target_words=500
    )
    t_summary = time.time() - t0

    words = len(direct_summary.split())
    print(f"\n[TEST 3 SUCCESS] Generated in {t_summary:.2f} seconds ({words} words)!")
    print("\n--- DIRECT SUMMARY OUTPUT ---")
    print(direct_summary[:800] + ("\n... [truncated]" if len(direct_summary) > 800 else ""))

    print("\n" + "=" * 70)
    print(">>> ALL LIVE GROQ WRITER AGENT TESTS COMPLETED SUCCESSFULLY! <<<")
    print("=" * 70)


if __name__ == "__main__":
    run_live_writer_tests()
