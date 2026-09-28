"""
Backward compatibility module.
`DocumentLoader` has been refactored into `backend.agents.document_agent`.
This re-exports DocumentLoader so existing imports continue to work without breaking.
"""
from backend.agents.document_agent import DocumentLoader

__all__ = ["DocumentLoader"]
