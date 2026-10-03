import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path so 'backend.xxx' imports work from any directory
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import re
from fastapi import FastAPI, HTTPException, UploadFile, File, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from backend.agents import AgentOrchestrator, DocumentLoader, SummaryChatAgent
from backend.rag import RAGPipeline
from backend.config import config
from backend.pdf_export import build_pdf_report

app = FastAPI(
    title="Mini Researcher API",
    description="Autonomous web research & document RAG API powered by Tavily, ChromaDB & Gemini",
    version="0.2.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure absolute path for ChromaDB storage so it is consistent regardless of CWD
CHROMA_DIR = str(Path(__file__).resolve().parent / "chroma_db")
rag_pipeline = RAGPipeline(collection_name="research_chunks", persist_directory=CHROMA_DIR)
doc_loader = DocumentLoader()
coordinator = AgentOrchestrator(rag_pipeline=rag_pipeline)
chat_assistant = SummaryChatAgent()

# In-memory session store for active user-uploaded documents
uploaded_documents_store: List[Dict[str, Any]] = []


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Research topic or question")
    mode: Optional[str] = Field("auto", description="Execution mode: 'auto', 'document', or 'web'")
    include_uploaded: bool = Field(True, description="Whether to incorporate uploaded documents into research")
    model: Optional[str] = Field(None, description="Target Gemini model (e.g. 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash')")
    report_type: Optional[str] = Field("Summary - Short and fast (~2 min)", description="Report type (Summary, Deep Research, Multi Agents, Detailed)")
    tone: Optional[str] = Field("Objective - Impartial and unbiased presentation of facts and findings", description="Tone of voice")


@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "message": "Mini Researcher API is running.",
        "gemini_configured": bool(config.GEMINI_API_KEY),
        "groq_configured": bool(config.GROQ_API_KEY),
        "writer_agent_provider": "Groq",
        "writer_model": getattr(coordinator.writer_agent, "model_name", "llama-3.1-8b-instant"),
        "writer_fallback_models": getattr(coordinator.writer_agent, "fallback_models", ["llama-3.1-8b-instant", "llama3-8b-8192"]),
        "default_model": getattr(config, "DEFAULT_MODEL", "gemini-3.6-flash"),
        "fallback_models": ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.5-flash"],
        "tavily_configured": bool(config.TAVILY_API_KEY),
        "uploaded_docs_count": len(uploaded_documents_store)
    }


@app.get("/api/documents")
def get_uploaded_documents():
    """
    Returns list of all currently active uploaded documents.
    """
    return {
        "documents": [
            {
                "id": doc.get("title"),
                "filename": doc.get("title"),
                "char_count": doc.get("char_count", 0),
                "pages": doc.get("pages", 1),
                "url": doc.get("url")
            }
            for doc in uploaded_documents_store
        ]
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a user document (PDF, TXT, or Markdown), parses its content,
    and indexes it into the ChromaDB vector store.
    """
    filename = file.filename or "uploaded_document.txt"
    if not DocumentLoader.is_supported(filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{filename}'. Please upload PDF (.pdf), Text (.txt), or Markdown (.md)."
        )

    try:
        content_bytes = await file.read()
        if not content_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Parse document text and metadata
        parsed_doc = doc_loader.load_from_bytes(content_bytes, filename)

        # Index in ChromaDB RAG vector store
        chunks_indexed = rag_pipeline.index_findings([parsed_doc])

        # Remove existing doc with same name to avoid duplicates
        global uploaded_documents_store
        uploaded_documents_store = [d for d in uploaded_documents_store if d.get("title") != filename]
        uploaded_documents_store.append(parsed_doc)

        return {
            "status": "success",
            "message": f"Successfully indexed '{filename}' into ChromaDB.",
            "document": {
                "id": filename,
                "filename": filename,
                "char_count": parsed_doc["char_count"],
                "pages": parsed_doc["pages"],
                "chunks_indexed": chunks_indexed
            },
            "total_documents": len(uploaded_documents_store)
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {str(e)}")


@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    """
    Removes a document from active session memory.
    """
    global uploaded_documents_store
    initial_len = len(uploaded_documents_store)
    uploaded_documents_store = [d for d in uploaded_documents_store if d.get("title") != filename]

    if len(uploaded_documents_store) == initial_len:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")

    return {
        "status": "success",
        "message": f"Document '{filename}' removed.",
        "remaining_documents": len(uploaded_documents_store)
    }


@app.post("/api/research")
async def start_research(payload: ResearchRequest):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if payload.mode == "document" and not (payload.include_uploaded and uploaded_documents_store):
        raise HTTPException(
            status_code=400,
            detail="Document Mode is selected, but no uploaded document was found. Please upload a PDF, TXT, or MD document first or switch to Web Research Mode."
        )

    try:
        user_docs = uploaded_documents_store if payload.include_uploaded and uploaded_documents_store else None
        result = await coordinator.run_research(
            query,
            user_documents=user_docs,
            mode=payload.mode or "auto",
            model=payload.model,
            report_type=payload.report_type or "Summary - Short and fast (~2 min)",
            tone=payload.tone or "Objective - Impartial and unbiased presentation of facts and findings"
        )
        return result
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question or doubt regarding the research summary")
    summary: str = Field(..., min_length=1, description="The research summary report content")
    topic: Optional[str] = Field("Research Summary", description="Topic of the research report")
    history: Optional[List[Dict[str, str]]] = Field(default=[], description="Previous conversation turns")
    sources: Optional[List[Dict[str, Any]]] = Field(default=[], description="Sources associated with the research")
    model: Optional[str] = Field(None, description="Optional target model")


@app.post("/api/chat")
async def chat_with_summary_assistant(payload: ChatRequest):
    """
    Clarifies doubts and answers questions strictly based on the generated research summary.
    Enforces strict guardrails: only answers about the research summary, and politely redirects out-of-scope questions.
    Only available after research summary has been generated.
    """
    q = payload.question.strip()
    summary = payload.summary.strip()

    if not summary:
        raise HTTPException(
            status_code=400,
            detail="Chat Assistant is only available after a research summary is generated. Please run research first."
        )
    if not q:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        response = chat_assistant.answer_question(
            question=q,
            summary=summary,
            topic=payload.topic or "Research Summary",
            history=payload.history or [],
            sources=payload.sources or [],
            model=payload.model
        )
        return response
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat assistant failed: {str(e)}")


class ExportPdfRequest(BaseModel):
    topic: str
    report: str
    report_type: Optional[str] = "Standard"
    tone: Optional[str] = "Objective"
    model: Optional[str] = "Gemini 3.5 Flash Lite"
    sources: Optional[List[Dict[str, Any]]] = []


@app.post("/api/export-pdf")
def export_pdf_endpoint(payload: ExportPdfRequest):
    """
    Generates an executive, publication-grade vector PDF report using ReportLab.
    Returns the file stream with an explicit Content-Disposition attachment header,
    guaranteeing that Chromium/Edge names the file properly with the .pdf extension.
    """
    try:
        clean_topic = re.sub(r"[^a-zA-Z0-9]+", "-", payload.topic.strip().lower()).strip("-")[:40] or "research-report"
        filename = f"{clean_topic}.pdf"

        pdf_bytes = build_pdf_report(
            topic=payload.topic,
            report_markdown=payload.report,
            report_type=payload.report_type,
            tone=payload.tone,
            model=payload.model,
            sources=payload.sources or []
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Type": "application/pdf",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF export failed: {str(e)}")


@app.post("/api/export-pdf-download")
def export_pdf_download_form(
    topic: str = Form("Research Report"),
    report: str = Form(""),
    report_type: str = Form("Standard"),
    tone: str = Form("Objective"),
    model: str = Form("Gemini 3.5 Flash Lite")
):
    """
    Direct HTML Form submit endpoint for browser downloads.
    Avoids client-side blob URLs completely, guaranteeing that Edge and Chrome save the file
    with the exact filename and .pdf extension via HTTP Content-Disposition attachment.
    """
    try:
        clean_topic = re.sub(r"[^a-zA-Z0-9]+", "-", topic.strip().lower()).strip("-")[:40] or "research-report"
        filename = f"{clean_topic}.pdf"

        pdf_bytes = build_pdf_report(
            topic=topic,
            report_markdown=report,
            report_type=report_type,
            tone=tone,
            model=model,
            sources=[]
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Type": "application/pdf",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF export failed: {str(e)}")


# Mount frontend (built React app from dist or fallback to raw static files)
frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"

if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
elif frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=config.HOST, port=config.PORT, reload=True)
