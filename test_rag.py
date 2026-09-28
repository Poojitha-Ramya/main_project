import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.chunker import TextChunker
from backend.embeddings import EmbeddingEngine
from backend.vector_store import VectorStore
from backend.rag import RAGPipeline


def test_chunker_unit():
    print("\n--- Testing 1. Chunker ---")
    chunker = TextChunker(chunk_size=150, chunk_overlap=30)
    text = (
        "First sentence discussing technology. Second sentence going deeper into solid state physics. "
        "Third sentence analyzing material properties. Fourth sentence reviewing energy density and lifetime."
    )
    chunks = chunker.chunk_text(text, metadata={"title": "Test Doc", "url": "https://test.com"})
    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
    assert chunks[0]["metadata"]["title"] == "Test Doc"
    assert chunks[0]["chunk_id"].startswith("https___test_com")
    print(f"[PASS] Chunker produced {len(chunks)} chunks with metadata and overlap.")


def test_vector_store_unit():
    print("\n--- Testing 2. VectorStore (ChromaDB) ---")
    store = VectorStore(collection_name="test_unit_store")
    store.clear()
    mock_chunks = [
        {"chunk_id": "c1", "text": "Solid-state batteries achieve 450 Wh/kg.", "metadata": {"topic": "battery"}},
        {"chunk_id": "c2", "text": "Quantum computers operate with superconducting qubits.", "metadata": {"topic": "quantum"}},
        {"chunk_id": "c3", "text": "Solar cells convert sunlight using photovoltaic silicon wafers.", "metadata": {"topic": "solar"}}
    ]
    # Orthogonal toy embeddings
    mock_embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]
    store.add_documents(mock_chunks, mock_embeddings)
    assert store.size() == 3, f"Expected store size 3, got {store.size()}"

    # Search query aligned with quantum
    matches = store.search([0.1, 0.95, 0.0], top_k=2)
    assert len(matches) == 2
    assert matches[0]["chunk_id"] == "c2", f"Expected c2 to match best, got {matches[0]['chunk_id']}"
    assert matches[0]["similarity"] > 0.9
    print(f"[PASS] VectorStore cosine similarity matched {matches[0]['chunk_id']} (similarity: {matches[0]['similarity']}).")


def test_embeddings_unit():
    print("\n--- Testing 3. Embeddings (Gemini API) ---")
    engine = EmbeddingEngine()
    query_vec = engine.embed_query("Solid-state battery electrolyte")
    assert len(query_vec) == 3072, f"Expected 3072 dimensions, got {len(query_vec)}"
    
    batch_vecs = engine.embed_texts(["Electrolyte chemistry", "Cathode development"])
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 3072
    print(f"[PASS] EmbeddingEngine generated 3072-dimensional vector embeddings via Gemini.")


def test_rag_pipeline_end_to_end():
    print("\n--- Testing 4. End-to-End RAG Pipeline (ChromaDB) ---")
    rag = RAGPipeline(chunk_size=300, chunk_overlap=50, collection_name="test_rag_pipeline")
    rag.clear()

    sample_findings = [
        {
            "title": "Battery Milestone 2026",
            "url": "https://nature.com/articles/battery-2026",
            "content": (
                "Researchers developed an ultra-thin sulfide solid electrolyte that operates stably at room temperature. "
                "The solid-state cell achieved an exceptional energy density of 480 Wh/kg and retained 90% capacity after 1,000 cycles."
            )
        },
        {
            "title": "Quantum Error Correction 2026",
            "url": "https://science.org/articles/quantum-2026",
            "content": (
                "Scientists have demonstrated fault-tolerant logical qubits on a 1,000-qubit processor. "
                "The two-qubit gate error was maintained below 0.08%, exceeding the fault-tolerant threshold."
            )
        }
    ]

    indexed_count = rag.index_findings(sample_findings)
    assert indexed_count > 0, "Expected chunks to be indexed"
    print(f"       -> Indexed {indexed_count} chunks into vector database.")

    # Test retrieval
    results = rag.retrieve("What energy density was achieved in the solid state battery?", top_k=1)
    assert len(results) == 1
    assert "480 Wh/kg" in results[0]["text"]
    print(f"       -> Retrieved relevant chunk with similarity: {results[0]['similarity']}")

    # Test Question Answering
    qa = rag.answer_question("What energy density did the solid-state cell achieve?")
    assert "480" in qa["answer"], f"Expected '480' in answer, got: {qa['answer']}"
    print(f"       -> Grounded QA Answer: {qa['answer']}")
    print("[PASS] RAGPipeline end-to-end indexing, retrieval, and QA verified successfully.")


if __name__ == "__main__":
    test_chunker_unit()
    test_vector_store_unit()
    test_embeddings_unit()
    test_rag_pipeline_end_to_end()
    print("\nAll RAG unit & integration tests PASSED!")
