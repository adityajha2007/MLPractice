"""File-based JSON persistence with metadata envelope.

Stores graph states as JSON files with provenance metadata for audit trails.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sop_to_dag.schemas import GraphState

DEFAULT_STORE_DIR = Path("output/graphs")


class GraphStore:
    """Persist and retrieve GraphState objects as JSON files."""

    def __init__(self, store_dir: Path = DEFAULT_STORE_DIR):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save_graph(
        self,
        graph_state: GraphState,
        source_file: str,
        converter_id: str,
    ) -> Path:
        """Save a graph state with metadata envelope. Returns the file path."""
        timestamp = datetime.now(timezone.utc).isoformat()
        safe_name = Path(source_file).stem
        filename = f"{safe_name}_{converter_id}_{timestamp[:19].replace(':', '-')}.json"
        path = self.store_dir / filename

        envelope: Dict[str, Any] = {
            "metadata": {
                "source": source_file,
                "converter": converter_id,
                "timestamp": timestamp,
                "status": "draft",
            },
            "graph_state": dict(graph_state),
        }
        path.write_text(json.dumps(envelope, indent=2, default=str))
        return path

    def load_graph(self, path: Path) -> GraphState:
        """Deserialize a stored JSON file back into a GraphState."""
        data = json.loads(path.read_text())
        gs = data["graph_state"]
        return GraphState(
            source_text=gs["source_text"],
            nodes=gs["nodes"],
            feedback=gs.get("feedback", ""),
            iteration=gs.get("iteration", 0),
            is_complete=gs.get("is_complete", False),
            converter_id=gs.get("converter_id", ""),
            analysis_report=gs.get("analysis_report", ""),
            enriched_chunks=gs.get("enriched_chunks", []),
            vector_store=None,  # Not serializable; rebuild if needed
            entity_map=gs.get("entity_map", []),
        )

    def load_envelope(self, path: Path) -> Dict[str, Any]:
        """Load the full envelope (metadata + graph_state)."""
        return json.loads(path.read_text())

    def list_graphs(self, status: Optional[str] = None) -> List[Path]:
        """List stored graph files, optionally filtered by status."""
        paths = sorted(self.store_dir.glob("*.json"))
        if status is None:
            return paths
        filtered = []
        for p in paths:
            envelope = json.loads(p.read_text())
            if envelope.get("metadata", {}).get("status") == status:
                filtered.append(p)
        return filtered

    def update_status(self, path: Path, status: str) -> None:
        """Update the status field in a stored graph's metadata."""
        envelope = json.loads(path.read_text())
        envelope["metadata"]["status"] = status
        path.write_text(json.dumps(envelope, indent=2, default=str))
