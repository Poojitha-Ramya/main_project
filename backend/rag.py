import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path so 'backend.xxx' imports work from anywhere
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from google import genai

from backend.chunker import TextChunker
from backend.embeddings import EmbeddingEngine
from backend.vector_store import VectorStore
from backend.config import config, get_fallback_chain


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Pipeline:
        Findings
            ↓
        TextChunker
            ↓
        Gemini EmbeddingEngine
            ↓
        ChromaDB VectorStore
            ↓
        Semantic Retrieval
            ↓
        Gemini LLM
            ↓
        Grounded Answer
    """

    # Use designated generation model with resilient redirection across 3.6, 3.7, 3.8, and 3.5.
    DEFAULT_GENERATION_MODEL = getattr(config, "DEFAULT_MODEL", "gemini-3.6-flash")

    def __init__(
        self,
        api_key: str = "",
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        collection_name: str = "research_chunks",
        persist_directory: Optional[str] = "./chroma_db",
        generation_model: str = DEFAULT_GENERATION_MODEL
    ):
        """
        Initialize the RAG pipeline.

        Args:
            api_key:
                Gemini API key.

            chunk_size:
                Maximum target size of each chunk.

            chunk_overlap:
                Number of overlapping characters/sentences.

            collection_name:
                ChromaDB collection name.

            persist_directory:
                Directory where ChromaDB data is stored.
                Use None for temporary in-memory storage.

            generation_model:
                Gemini model used to generate final answers.
        """

        # ----------------------------------------------------
        # Gemini API
        # ----------------------------------------------------

        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. "
                "Please set it in your .env file."
            )

        self.client = genai.Client(api_key=self.api_key)

        self.generation_model = generation_model

        # ----------------------------------------------------
        # RAG Components
        # ----------------------------------------------------

        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.embedding_engine = EmbeddingEngine(
            api_key=self.api_key
        )

        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )

        print("[RAG] Pipeline initialized.")
        print(
            f"[RAG] ChromaDB collection: "
            f"{collection_name}"
        )

        if persist_directory:
            print(
                f"[RAG] Persistent storage: "
                f"{persist_directory}"
            )
        else:
            print("[RAG] Storage mode: in-memory")

        print(
            f"[RAG] Generation model: "
            f"{self.generation_model}"
        )

    # ========================================================
    # INDEXING
    # ========================================================

    def index_findings(
        self,
        findings: List[Dict[str, Any]]
    ) -> int:
        """
        Convert scraped findings into chunks, generate embeddings,
        and store everything in ChromaDB.

        Args:
            findings:
                Scraped research findings.

        Returns:
            Total number of chunks currently stored.
        """

        if not findings:
            print("[RAG] No findings to index.")
            return 0

        print("\n[RAG] ===== INDEXING =====")

        # ----------------------------------------------------
        # STEP 1: CHUNKING
        # ----------------------------------------------------

        print(
            f"[RAG] Received {len(findings)} findings."
        )

        chunks = self.chunker.chunk_findings(findings)

        if not chunks:
            print("[RAG] No chunks were generated.")
            return 0

        print(
            f"[RAG] Created {len(chunks)} text chunks."
        )

        # ----------------------------------------------------
        # STEP 2: EMBEDDINGS
        # ----------------------------------------------------

        texts = [
            chunk["text"]
            for chunk in chunks
            if chunk.get("text")
        ]

        if not texts:
            print("[RAG] No valid text available for embeddings.")
            return 0

        print(
            f"[RAG] Generating embeddings for "
            f"{len(texts)} chunks..."
        )

        embeddings = self.embedding_engine.embed_texts(
            texts,
            batch_size=16
        )

        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: "
                f"{len(texts)} texts but "
                f"{len(embeddings)} embeddings."
            )

        print(
            f"[RAG] Generated {len(embeddings)} embeddings."
        )

        # ----------------------------------------------------
        # STEP 3: CHROMADB
        # ----------------------------------------------------

        print("[RAG] Storing chunks in ChromaDB...")

        total_count = self.vector_store.add_documents(
            chunks,
            embeddings
        )

        print(
            f"[RAG] ChromaDB now contains "
            f"{total_count} chunks."
        )

        print("[RAG] ===== INDEXING COMPLETE =====\n")

        return total_count

    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve semantically relevant chunks from ChromaDB.

        Process:

            Query
              ↓
            Gemini embedding
              ↓
            ChromaDB similarity search
              ↓
            Relevant chunks
        """

        query = (query or "").strip()

        if not query:
            return []

        if self.vector_store.size() == 0:
            print("[RAG] Vector store is empty.")
            return []

        print(
            f"[RAG] Retrieving top {top_k} chunks "
            f"for: {query}"
        )

        # ----------------------------------------------------
        # Convert question into vector
        # ----------------------------------------------------

        query_vector = self.embedding_engine.embed_query(
            query
        )

        # ----------------------------------------------------
        # Search ChromaDB
        # ----------------------------------------------------

        results = self.vector_store.search(
            query_embedding=query_vector,
            top_k=top_k,
            min_score=min_score
        )

        print(
            f"[RAG] Retrieved {len(results)} relevant chunks."
        )

        return results

    # ========================================================
    # MULTI-QUERY RETRIEVAL
    # ========================================================

    def enhance_report_context(
        self,
        subqueries: List[str],
        top_k_per_query: int = 3,
        min_score: float = 0.50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for multiple research subqueries.

        Useful when generating a comprehensive research report.

        Example:

            Query 1 → 3 chunks
            Query 2 → 3 chunks
            Query 3 → 3 chunks

        Duplicate chunks are removed.
        """

        if not subqueries:
            return []

        print(
            f"\n[RAG] Retrieving context for "
            f"{len(subqueries)} subqueries..."
        )

        collected_chunks = []
        seen_chunk_ids = set()

        for index, query in enumerate(
            subqueries,
            start=1
        ):

            print(
                f"[RAG] Subquery {index}/{len(subqueries)}: "
                f"{query}"
            )

            results = self.retrieve(
                query=query,
                top_k=top_k_per_query,
                min_score=min_score
            )

            for item in results:

                chunk_id = item.get("chunk_id")

                if not chunk_id:
                    continue

                if chunk_id in seen_chunk_ids:
                    continue

                seen_chunk_ids.add(chunk_id)
                collected_chunks.append(item)

        print(
            f"[RAG] Unique context chunks collected: "
            f"{len(collected_chunks)}"
        )

        return collected_chunks

    # ========================================================
    # CONTEXT BUILDING
    # ========================================================

    def _build_context(
        self,
        chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Convert retrieved chunks into a context block
        that can be supplied to Gemini.
        """

        context_parts = []

        for index, chunk in enumerate(
            chunks,
            start=1
        ):

            text = chunk.get(
                "text",
                ""
            )

            metadata = chunk.get(
                "metadata",
                {}
            )

            title = metadata.get(
                "title",
                "Untitled Source"
            )

            url = metadata.get(
                "url",
                ""
            )

            similarity = chunk.get(
                "similarity",
                0.0
            )

            published_date = metadata.get(
                "published_date"
            )

            source_query = metadata.get(
                "source_query",
                ""
            )

            context_parts.append(
                f"""
[SOURCE {index}]
Title: {title}
URL: {url}
Similarity: {similarity}
Published Date: {published_date or "Unknown"}
Original Search Query: {source_query}

Content:
{text}
"""
            )

        return "\n".join(context_parts)

    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate an answer using ONLY retrieved context.
        """

        if not retrieved_chunks:
            return (
                "I could not find enough relevant information "
                "in the indexed research sources to answer "
                "this question."
            )

        context = self._build_context(
            retrieved_chunks
        )

        prompt = f"""
You are an evidence-based research assistant.

USER QUESTION:
{question}

RESEARCH CONTEXT:
{context}

INSTRUCTIONS:

1. Answer the user's question using ONLY the research
   context provided above.

2. Do not invent facts.

3. Do not use information from your general knowledge
   when it is not present in the provided context.

4. If the context does not contain enough information,
   clearly say that the available research sources
   do not provide enough information.

5. Prefer information supported by multiple sources
   when available.

6. When making an important claim, mention the source
   title and URL from the provided context.

7. Do not treat the similarity score as evidence.
   It only indicates semantic relevance.

8. Give a clear, concise and well-structured answer.

FINAL ANSWER:
"""

        # Try designated generation model with resilient fallbacks and redirection
        models_to_try = get_fallback_chain(self.generation_model)

        last_error = None
        for idx, model_name in enumerate(models_to_try):
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                if response and response.text:
                    return response.text.strip()

            except Exception as error:
                last_error = error
                next_model = models_to_try[idx + 1] if idx + 1 < len(models_to_try) else "retrieved context fallback"
                print(f"[RAG] Model {model_name} unavailable: {error}. Redirecting to {next_model}...")
                continue

        print(
            f"[RAG] Generation failed: {last_error}. Using retrieved context fallback."
        )

        # Resilient fallback: Ground directly on the most relevant retrieved chunk
        if retrieved_chunks:
            top_chunk = retrieved_chunks[0]
            top_text = top_chunk.get("text", "").strip()
            top_title = top_chunk.get("metadata", {}).get("title", "Retrieved Source")
            return f"Based on the retrieved research context from {top_title}: {top_text}"

        return (
            "I was unable to generate an answer "
            "from the retrieved research context."
        )

    # ========================================================
    # QUESTION ANSWERING
    # ========================================================

    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.50
    ) -> Dict[str, Any]:
        """
        Complete RAG question-answering pipeline.

        Question
           ↓
        Embedding
           ↓
        ChromaDB
           ↓
        Relevant chunks
           ↓
        Gemini
           ↓
        Answer
        """

        question = (question or "").strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        print("\n[RAG] ===== QUESTION ANSWERING =====")

        # ----------------------------------------------------
        # STEP 1: RETRIEVE
        # ----------------------------------------------------

        chunks = self.retrieve(
            query=question,
            top_k=top_k,
            min_score=min_score
        )

        if not chunks:

            return {
                "question": question,
                "answer": (
                    "No sufficiently relevant information "
                    "was found in the indexed research sources."
                ),
                "retrieved_chunks": [],
                "sources": []
            }

        # ----------------------------------------------------
        # STEP 2: GENERATE
        # ----------------------------------------------------

        answer = self.generate_answer(
            question=question,
            retrieved_chunks=chunks
        )

        # ----------------------------------------------------
        # STEP 3: EXTRACT SOURCES
        # ----------------------------------------------------

        sources = []

        seen_urls = set()

        for chunk in chunks:

            metadata = chunk.get(
                "metadata",
                {}
            )

            title = metadata.get(
                "title",
                "Untitled Source"
            )

            url = metadata.get(
                "url",
                ""
            )

            if url and url not in seen_urls:

                seen_urls.add(url)

                sources.append({
                    "title": title,
                    "url": url
                })

        print(
            f"[RAG] Answer generated using "
            f"{len(chunks)} chunks from "
            f"{len(sources)} sources."
        )

        print("[RAG] ===== QUESTION COMPLETE =====\n")

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": chunks,
            "sources": sources
        }

    # ========================================================
    # CLEAR VECTOR STORE
    # ========================================================

    def clear(self) -> None:
        """
        Completely clear the ChromaDB research collection.
        """

        print("[RAG] Clearing ChromaDB collection...")

        self.vector_store.clear()

        print("[RAG] ChromaDB collection cleared.")

    # ========================================================
    # STATISTICS
    # ========================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Return RAG/vector database statistics.
        """

        return self.vector_store.get_stats()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\nStarting RAG pipeline test...\n")

    try:

        rag = RAGPipeline(
            collection_name="research_chunks",
            persist_directory="./chroma_db"
        )

        # ----------------------------------------------------
        # Sample research findings
        # ----------------------------------------------------

        sample_findings = [

            {
                "title": "Solid State Battery Guide 2026",

                "url":
                    "https://example.com/solid-state",

                "content":
                    (
                        "Solid-state batteries replace the "
                        "liquid electrolyte with a solid electrolyte. "
                        "This design can improve safety and energy density. "
                        "Researchers are testing lithium metal anodes "
                        "with ceramic and sulfide electrolytes. "
                        "Commercial automotive deployment is being "
                        "targeted by several manufacturers."
                    ),

                "score": 0.92,

                "published_date":
                    "2026-01-15",

                "priority":
                    "HIGH",

                "source_query":
                    "solid state battery breakthroughs"
            },

            {
                "title": "Quantum Computing Milestones",

                "url":
                    "https://example.com/quantum",

                "content":
                    (
                        "Quantum computers use quantum bits called "
                        "qubits. Researchers are improving quantum "
                        "error correction and increasing the number "
                        "of controllable physical qubits."
                    ),

                "score": 0.88,

                "published_date":
                    "2026-02-10",

                "priority":
                    "HIGH",

                "source_query":
                    "quantum computing breakthroughs"
            }

        ]

        # ----------------------------------------------------
        # Index
        # ----------------------------------------------------

        total = rag.index_findings(
            sample_findings
        )

        print(
            f"Indexed chunks: {total}"
        )

        # ----------------------------------------------------
        # Ask
        # ----------------------------------------------------

        result = rag.answer_question(
            question=(
                "What technology is used to replace "
                "the liquid electrolyte in solid-state batteries?"
            ),
            top_k=5,
            min_score=0.50
        )

        print("\nANSWER:")
        print(result["answer"])

        print("\nSOURCES:")

        for source in result["sources"]:
            print(
                f"- {source['title']}: "
                f"{source['url']}"
            )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        print("\nRAG STATISTICS:")
        print(rag.get_stats())

    except Exception as error:

        print(
            f"\nRAG test failed: {error}"
        )
