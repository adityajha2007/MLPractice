"""Research paper comparison converters: BottomUp (PADME) + EdgeVertex (Agent-S).

These produce graphs for evaluation/comparison only — they do NOT go through
the refinement loop. Inline prompts at top of file.
"""

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from sop_to_dag.models import get_model
from sop_to_dag.schemas import ExtractorOutput

# ---------------------------------------------------------------------------
# Bottom-Up (PADME-inspired) prompts
# ---------------------------------------------------------------------------

_CHUNKER_SYSTEM = """\
You are a Document Segmentation Specialist. Split the following SOP text into
logical semantic chunks. Each chunk should represent a coherent section or
sub-procedure of the SOP.

Return a JSON list of objects with:
- "chunk_id": sequential integer
- "title": brief title for the chunk
- "text": the chunk content
"""

_CHUNKER_HUMAN = """\
Split this SOP into semantic chunks:

---
{source_text}
---
"""

_CHUNK_PROCESSOR_SYSTEM = """\
You are a Workflow Graph Builder. Convert the given SOP chunk into workflow nodes.

Rules:
1. Use descriptive snake_case IDs prefixed with the chunk title abbreviation
2. Decision points become "question" nodes with options
3. Actions become "instruction" nodes with next
4. End states become "terminal" nodes
5. If the chunk references steps from previous chunks, create edges pointing
   to those node IDs (provided in the context)
6. The very first node of the FIRST chunk must have id 'start'

## Context from Previous Chunks
{prior_context}
"""

_CHUNK_PROCESSOR_HUMAN = """\
Convert this chunk into workflow nodes:

## Chunk: {chunk_title}
{chunk_text}

Return an ExtractorOutput with reasoning and all_nodes.
"""

_MERGE_SYSTEM = """\
You are a Graph Merge Specialist. You receive nodes from multiple chunks that
were processed independently. Your job is to:

1. Deduplicate any nodes that represent the same step
2. Fix cross-chunk edges to ensure connectivity
3. Ensure there is exactly one 'start' node
4. Ensure all terminal paths end with 'terminal' type nodes
5. Remove orphan nodes that are not reachable from start

Return the final merged set of nodes.
"""

_MERGE_HUMAN = """\
Merge these independently-extracted node sets into a single connected graph:

{all_nodes_json}

Return an ExtractorOutput with reasoning and all_nodes.
"""

# ---------------------------------------------------------------------------
# Edge-Vertex (Agent-S-inspired) prompts
# ---------------------------------------------------------------------------

_VERTEX_SYSTEM = """\
You are an Entity Extraction Specialist. Extract ALL workflow entities (nodes)
from the SOP text with NO edges/connections.

For each entity, determine:
- id: descriptive snake_case identifier
- type: "question" (decision point), "instruction" (action), "terminal" (end),
  or "reference" (external lookup)
- text: the description of what this node represents
- external_ref: if it references an external guide/document, capture it here

Rules:
- The first node must have id 'start'
- Be exhaustive — capture every distinct step, decision, and endpoint
- Do NOT determine connections yet — only identify the nodes
"""

_VERTEX_HUMAN = """\
Extract all workflow entities from this SOP:

---
{source_text}
---

Return a JSON list of objects with: id, type, text, external_ref (optional).
"""

_EDGE_SYSTEM = """\
You are a Relationship Mapping Specialist. Given a list of pre-identified
workflow nodes and the original SOP text, determine all connections (edges).

For each node, determine:
- If it's a "question" type: map its 'options' (answer -> target_node_id)
- If it's an "instruction" or "reference" type: set its 'next' to the
  following node's id
- If it's a "terminal" type: next and options should be null

Rules:
- Every 'next' and option value must reference an existing node id
- Every non-terminal node must have at least one outgoing edge
- Follow the original SOP's flow — don't invent connections
"""

_EDGE_HUMAN = """\
## Nodes (pre-extracted)
{nodes_json}

## Original SOP
{source_text}

Map all edges (next/options) for each node. Return the complete node list
with edges populated as an ExtractorOutput.
"""


# ---------------------------------------------------------------------------
# BottomUpConverter
# ---------------------------------------------------------------------------


class BottomUpConverter:
    """PADME-inspired chunk-based bottom-up conversion with context carryover.

    Uses pre-made enriched chunks when available, falls back to LLM chunking.
    """

    converter_id = "bottom_up"

    def __init__(self):
        self.llm = get_model("bottom_up")

    def convert(
        self,
        source_text: str,
        enriched_chunks: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """Convert SOP via chunking -> per-chunk extraction -> merge."""
        # Use pre-made chunks if available
        if enriched_chunks:
            chunks = [
                {"title": f"Chunk {ec['chunk_id']}", "text": ec["chunk_text"]}
                for ec in enriched_chunks
            ]
        else:
            chunks = self._chunk_text(source_text)

        # Process each chunk sequentially with context carryover
        all_chunk_nodes: List[List[Dict[str, Any]]] = []
        prior_context = "No previous chunks."

        for chunk in chunks:
            chunk_nodes = self._process_chunk(
                chunk_title=chunk["title"],
                chunk_text=chunk["text"],
                prior_context=prior_context,
            )
            all_chunk_nodes.append(chunk_nodes)

            node_ids = [n["id"] for n in chunk_nodes]
            prior_context = (
                f"Previous chunk '{chunk['title']}' produced nodes: {node_ids}"
            )

        return self._merge_nodes(all_chunk_nodes)

    def _chunk_text(self, source_text: str) -> List[Dict[str, str]]:
        """Split SOP text into semantic chunks via LLM."""
        messages = [
            SystemMessage(content=_CHUNKER_SYSTEM),
            HumanMessage(content=_CHUNKER_HUMAN.format(source_text=source_text)),
        ]
        response = self.llm.invoke(messages)

        try:
            chunks = json.loads(response.content)
            return chunks if isinstance(chunks, list) else [{"title": "full", "text": source_text}]
        except (json.JSONDecodeError, AttributeError):
            return [{"title": "full", "text": source_text}]

    def _process_chunk(
        self, chunk_title: str, chunk_text: str, prior_context: str
    ) -> List[Dict[str, Any]]:
        """Process a single chunk into nodes."""
        structured_llm = self.llm.with_structured_output(ExtractorOutput)
        messages = [
            SystemMessage(
                content=_CHUNK_PROCESSOR_SYSTEM.format(prior_context=prior_context)
            ),
            HumanMessage(
                content=_CHUNK_PROCESSOR_HUMAN.format(
                    chunk_title=chunk_title,
                    chunk_text=chunk_text,
                )
            ),
        ]

        try:
            result = structured_llm.invoke(messages)
            return [n.model_dump() for n in result.all_nodes]
        except Exception:
            return []

    def _merge_nodes(
        self, all_chunk_nodes: List[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Merge nodes from all chunks, deduplicating and resolving edges."""
        all_nodes_flat = []
        for chunk_nodes in all_chunk_nodes:
            all_nodes_flat.extend(chunk_nodes)

        structured_llm = self.llm.with_structured_output(ExtractorOutput)
        messages = [
            SystemMessage(content=_MERGE_SYSTEM),
            HumanMessage(
                content=_MERGE_HUMAN.format(
                    all_nodes_json=json.dumps(all_nodes_flat, indent=2)
                )
            ),
        ]

        try:
            result = structured_llm.invoke(messages)
            return {n.id: n.model_dump() for n in result.all_nodes}
        except Exception:
            return {n["id"]: n for n in all_nodes_flat}


# ---------------------------------------------------------------------------
# EdgeVertexConverter
# ---------------------------------------------------------------------------


class EdgeVertexConverter:
    """Agent-S-inspired two-stage: extract vertices first, then add edges."""

    converter_id = "edge_vertex"

    def __init__(self):
        self.llm = get_model("edge_vertex")

    def convert(
        self,
        source_text: str,
        enriched_chunks: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """Convert SOP via vertex extraction -> edge mapping."""
        # enriched_chunks not directly used; source_text should already be
        # the concatenated enriched text from preprocessing
        vertices = self._extract_vertices(source_text)
        return self._map_edges(vertices, source_text)

    def _extract_vertices(self, source_text: str) -> List[Dict[str, Any]]:
        """Stage A: Extract all workflow entities with no connections."""
        messages = [
            SystemMessage(content=_VERTEX_SYSTEM),
            HumanMessage(content=_VERTEX_HUMAN.format(source_text=source_text)),
        ]
        response = self.llm.invoke(messages)

        try:
            vertices = json.loads(response.content)
            return vertices if isinstance(vertices, list) else []
        except (json.JSONDecodeError, AttributeError):
            return []

    def _map_edges(
        self, vertices: List[Dict[str, Any]], source_text: str
    ) -> Dict[str, Any]:
        """Stage B: Add edge mappings to extracted vertices."""
        structured_llm = self.llm.with_structured_output(ExtractorOutput)
        messages = [
            SystemMessage(content=_EDGE_SYSTEM),
            HumanMessage(
                content=_EDGE_HUMAN.format(
                    nodes_json=json.dumps(vertices, indent=2),
                    source_text=source_text,
                )
            ),
        ]

        try:
            result = structured_llm.invoke(messages)
            return {n.id: n.model_dump() for n in result.all_nodes}
        except Exception:
            return {v.get("id", f"node_{i}"): v for i, v in enumerate(vertices)}
