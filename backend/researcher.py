"""
Backward compatibility module.
The multi-agent coordination architecture has been formalized under `backend.agents`.
`ResearchCoordinator` extends `AgentOrchestrator` to preserve 100% backward compatibility.
"""
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.agents.orchestrator import AgentOrchestrator


class ResearchCoordinator(AgentOrchestrator):
    """
    Backward-compatible coordinator extending AgentOrchestrator.
    """

    def __init__(self):
        super().__init__()


__all__ = ["ResearchCoordinator"]
