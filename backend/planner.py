"""
Backward compatibility module.
`planner.py` has been refactored to `query_planner.py`.
This re-exports QueryPlanner so existing imports continue to work without breaking.
"""
from backend.query_planner import QueryPlanner

__all__ = ["QueryPlanner"]
