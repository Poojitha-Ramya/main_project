"""
Backward compatibility module.
`QueryAnalyzer` is defined in `backend.query_planner`.
This re-exports QueryAnalyzer for any legacy callers.
"""
from backend.query_planner import QueryAnalyzer

__all__ = ["QueryAnalyzer"]
