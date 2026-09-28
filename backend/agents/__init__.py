"""
Backend Agents Package

Exports specialized agents for the autonomous research pipeline:
- ManagerAgent: High-level request analysis and mode routing
- ResearchAgent: Subquery planning and Tavily web retrieval
- DocumentAgent: Document parsing, chunking, and document RAG
- CuratorAgent: Source filtering, URL extraction, and content scraping
- WriterAgent: Synthesis and Markdown report drafting
- ReviewerAgent: Quality control, citation verification, and polishing
- AgentOrchestrator: Multi-agent pipeline orchestrator
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.agents.manager_agent import ManagerAgent, ResearchPlan, ResearchTask
    from backend.agents.research_agent import ResearchAgent
    from backend.agents.document_agent import DocumentAgent, DocumentLoader
    from backend.agents.curator_agent import CuratorAgent
    from backend.agents.writer_agent import WriterAgent
    from backend.agents.reviewer_agent import ReviewerAgent
    from backend.agents.orchestrator import AgentOrchestrator

__all__ = [
    "ManagerAgent",
    "ResearchPlan",
    "ResearchTask",
    "ResearchAgent",
    "DocumentAgent",
    "DocumentLoader",
    "CuratorAgent",
    "WriterAgent",
    "ReviewerAgent",
    "AgentOrchestrator",
]


def __getattr__(name: str):
    if name in ("ManagerAgent", "ResearchPlan", "ResearchTask"):
        from backend.agents import manager_agent
        return getattr(manager_agent, name)
    if name == "ResearchAgent":
        from backend.agents import research_agent
        return research_agent.ResearchAgent
    if name in ("DocumentAgent", "DocumentLoader"):
        from backend.agents import document_agent
        return getattr(document_agent, name)
    if name == "CuratorAgent":
        from backend.agents import curator_agent
        return curator_agent.CuratorAgent
    if name == "WriterAgent":
        from backend.agents import writer_agent
        return writer_agent.WriterAgent
    if name == "ReviewerAgent":
        from backend.agents import reviewer_agent
        return reviewer_agent.ReviewerAgent
    if name == "AgentOrchestrator":
        from backend.agents import orchestrator
        return orchestrator.AgentOrchestrator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
