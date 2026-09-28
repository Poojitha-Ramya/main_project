import io
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pypdf

# =========================================================
# PROJECT ROOT
# =========================================================

project_root = Path(__file__).resolve().parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# =========================================================
# PROJECT IMPORTS
# =========================================================

from backend.config import config
from backend.chunker import TextChunker
from backend.rag import RAGPipeline


# =========================================================
# DOCUMENT LOADER
# =========================================================

class DocumentLoader:
    """
    Parses and extracts text from user-uploaded documents (PDF, TXT, Markdown).
    """

    SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")

    @staticmethod
    def is_supported(filename: str) -> bool:
        return filename.lower().endswith(DocumentLoader.SUPPORTED_EXTENSIONS)

    def load_from_bytes(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        filename_lower = filename.lower()
        extracted_text = ""
        pages_count = 1

        if filename_lower.endswith(".pdf"):
            try:
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                pages_count = len(reader.pages)
                page_texts = []
                for idx, page in enumerate(reader.pages):
                    page_content = page.extract_text() or ""
                    if page_content.strip():
                        page_texts.append(page_content.strip())
                extracted_text = "\n\n".join(page_texts)
            except Exception as e:
                raise ValueError(f"Failed to read PDF '{filename}': {str(e)}")

        elif filename_lower.endswith((".txt", ".md")):
            try:
                extracted_text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    extracted_text = file_bytes.decode("latin-1")
                except Exception as e:
                    raise ValueError(f"Failed to decode text file '{filename}': {str(e)}")
        else:
            raise ValueError(
                f"Unsupported file format for '{filename}'. Supported formats: PDF, TXT, MD."
            )

        extracted_text = re.sub(r"[ \t]+", " ", extracted_text)
        extracted_text = re.sub(r"\n\s*\n", "\n\n", extracted_text).strip()

        if not extracted_text:
            raise ValueError(f"File '{filename}' contains no readable text.")

        return {
            "title": filename,
            "url": f"uploaded://{filename}",
            "content": extracted_text,
            "char_count": len(extracted_text),
            "pages": pages_count,
            "source_type": "uploaded_document"
        }


# =========================================================
# DOCUMENT RAG
# =========================================================

class DocumentRAG(RAGPipeline):
    """
    Document RAG Pipeline powered by ChromaDB & Gemini embeddings.
    """

    def __init__(
        self,
        api_key: str = "",
        collection_name: str = "document_chunks",
        persist_directory: Optional[str] = "./chroma_db"
    ):
        super().__init__(
            api_key=api_key,
            chunk_size=600,
            chunk_overlap=100,
            collection_name=collection_name,
            persist_directory=persist_directory
        )

    def index_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> int:
        """
        Normalizes uploaded documents and stores them in ChromaDB.
        Supports documents with either 'content' or 'text' keys.
        """
        normalized_docs = []
        for d in documents:
            title = d.get("title", "Document")
            content = d.get("content") or d.get("text") or ""
            url = d.get("url", f"uploaded://{title}")
            normalized_docs.append({
                "title": title,
                "url": url,
                "content": content,
                "score": 1.0,
                "priority": "HIGH"
            })
        return self.index_findings(normalized_docs)


# =========================================================
# DOCUMENT AGENT
# =========================================================

class DocumentAgent:
    """
    Document Agent

    Responsible for research over uploaded documents.

    Workflow:

        Uploaded Documents
              ↓
        DocumentAgent
              ↓
        Document RAG
              ↓
        Semantic Retrieval
              ↓
        Relevant Evidence
              ↓
        Return to Orchestrator

    The DocumentAgent does NOT:
        - perform web search
        - use Tavily
        - write the final report
        - review the final report
    """

    def __init__(
        self,
        gemini_api_key: str = "",
        document_rag: Optional[Any] = None,
        rag_pipeline: Optional[Any] = None
    ):
        # -------------------------------------------------
        # Gemini API key
        # -------------------------------------------------
        self.gemini_api_key = (
            gemini_api_key
            or config.GEMINI_API_KEY
        )

        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing."
            )

        # -------------------------------------------------
        # Components
        # -------------------------------------------------
        self.loader = DocumentLoader()
        self.chunker = TextChunker()

        # Document RAG
        effective_rag = document_rag or rag_pipeline
        self.document_rag = effective_rag or DocumentRAG(
            api_key=self.gemini_api_key,
            collection_name="document_chunks",
            persist_directory="./chroma_db"
        )

    def is_supported(self, filename: str) -> bool:
        return self.loader.is_supported(filename)

    def load_from_bytes(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        return self.loader.load_from_bytes(file_bytes, filename)

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for d in documents:
            normalized.append({
                "title": d.get("title", "Document"),
                "url": d.get("url", f"uploaded://{d.get('title', 'doc')}"),
                "content": d.get("content") or d.get("text") or "",
                "score": 1.0,
                "priority": "HIGH"
            })
        return self.chunker.chunk_findings(normalized)

    # =====================================================
    # 1. INDEX DOCUMENTS
    # =====================================================

    def index_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Sends uploaded documents to Document RAG
        for processing and indexing.
        """
        if not documents:
            return {
                "status": "no_documents",
                "document_count": 0,
                "chunk_count": 0
            }

        print(
            f"\n[DocumentAgent] "
            f"Indexing {len(documents)} documents..."
        )

        try:
            chunk_count = self.document_rag.index_documents(
                documents
            )
        except Exception as e:
            print(
                f"[DocumentAgent] "
                f"Document indexing failed: {e}"
            )
            return {
                "status": "failed",
                "document_count": len(documents),
                "chunk_count": 0,
                "error": str(e)
            }

        print(
            f"[DocumentAgent] "
            f"Indexed {chunk_count} chunks."
        )

        return {
            "status": "completed",
            "document_count": len(documents),
            "chunk_count": chunk_count
        }

    # =====================================================
    # 2. RETRIEVE DOCUMENT EVIDENCE
    # =====================================================

    def retrieve_evidence(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.50
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant document chunks
        for the user's question.
        """
        if not question or not question.strip():
            return []

        print(
            "\n[DocumentAgent] "
            "Retrieving relevant document evidence..."
        )

        try:
            results = self.document_rag.retrieve(
                query=question,
                top_k=top_k,
                min_score=min_score
            )
        except Exception as e:
            print(
                f"[DocumentAgent] "
                f"Document retrieval failed: {e}"
            )
            return []

        print(
            f"[DocumentAgent] "
            f"Retrieved {len(results)} "
            f"document evidence chunks."
        )

        return results

    # =====================================================
    # 3. EXECUTE DOCUMENT TASK
    # =====================================================

    def execute_task(
        self,
        task: Dict[str, Any],
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        min_score: float = 0.50
    ) -> Dict[str, Any]:
        """
        Executes one document research task.

        Workflow:

            Manager Task
                 ↓
            DocumentAgent
                 ↓
            Document RAG indexing
                 ↓
            Semantic retrieval
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

        print("\n" + "=" * 60)
        print("DOCUMENT AGENT")
        print("=" * 60)

        print(
            f"Task ID: {task_id}"
        )

        print(
            f"Objective: {objective}"
        )

        # -------------------------------------------------
        # STEP 1: Indexing
        # -------------------------------------------------
        print(
            "\n[1/2] Indexing uploaded documents..."
        )

        indexing_result = self.index_documents(
            documents
        )

        # -------------------------------------------------
        # STEP 2: Retrieval
        # -------------------------------------------------
        print(
            "\n[2/2] Retrieving relevant evidence..."
        )

        evidence = self.retrieve_evidence(
            question=objective,
            top_k=top_k,
            min_score=min_score
        )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------
        result = {
            "task_id": task_id,
            "agent": "DocumentAgent",
            "role": task.get(
                "role",
                "document_researcher"
            ),
            "objective": objective,
            "status": (
                "completed"
                if indexing_result.get("status") == "completed"
                else indexing_result.get("status", "completed")
            ),
            "document_count": indexing_result.get(
                "document_count", len(documents)
            ),
            "rag_chunk_count": indexing_result.get(
                "chunk_count", 0
            ),
            "evidence": evidence,
            "evidence_count": len(evidence)
        }

        if "error" in indexing_result:
            result["error"] = indexing_result["error"]

        print(
            "\n[DocumentAgent] "
            "Document task completed."
        )

        return result


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    agent = DocumentAgent()

    # -----------------------------------------------------
    # Sample document for verification
    # -----------------------------------------------------
    test_documents = [
        {
            "title": "sample_document.txt",
            "text": """
            Quantum computing uses quantum mechanical
            phenomena such as superposition and
            entanglement to process information.

            Quantum algorithms such as Shor's algorithm
            and Grover's algorithm demonstrate potential
            advantages for particular computational tasks.
            """
        }
    ]

    test_task = {
        "id": "document_task_1",
        "agent": "DocumentAgent",
        "role": "document_researcher",
        "objective": (
            "Explain the quantum computing concepts "
            "discussed in the uploaded document."
        ),
        "search_query": None,
        "priority": "HIGH"
    }

    result = agent.execute_task(
        task=test_task,
        documents=test_documents,
        top_k=3,
        min_score=0.50
    )

    print("\n" + "=" * 60)
    print("DOCUMENT AGENT RESULT")
    print("=" * 60)

    print(
        "Status:",
        result["status"]
    )

    print(
        "Documents:",
        result["document_count"]
    )

    print(
        "RAG chunks:",
        result["rag_chunk_count"]
    )

    print(
        "Evidence:",
        result["evidence_count"]
    )