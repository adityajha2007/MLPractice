"""SOP-to-DAG pipeline converter (v4 — graph-first).

Step 1 (Graph Gen):     LLM converts full enriched SOP -> graph JSON directly
Step 2 (Graph Refine):  For each chunk, LLM produces a patch to add/modify/remove nodes

The LLM produces structured graph output from the start — no lossy text-outline
intermediate. This preserves temporal dependencies, decision scope, and branch
history that indentation-based outlines lose.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from sop_to_dag.graph_ops import apply_patch, generate_adjacency_map, get_graph_issues
from sop_to_dag.models import get_model
from sop_to_dag.graph_ops import SchemaValidator
from sop_to_dag.schemas import GraphPatch, InitialGraph, WorkflowNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inline prompts — LLM produces STRUCTURED GRAPH OUTPUT
# ---------------------------------------------------------------------------

_GRAPH_SYSTEM = """\
You are a Process Logic Engineer. Convert an SOP document directly into a \
workflow DAG (directed acyclic graph) represented as a list of nodes.

Each node must follow this EXACT schema:
{
  "id": "snake_case_id",          // 2-4 word snake_case identifier
  "type": "<type>",               // One of: "instruction", "question", "terminal", "reference"
  "text": "Description",          // Self-explanatory action or question text
  "next": "next_node_id" | null,  // For instruction/reference: ID of next node. Null for question/terminal.
  "options": {"Yes": "id", "No": "id"} | null,  // For question nodes ONLY. Null otherwise.
  "external_ref": "Doc Name" | null,  // For reference nodes: name of external document
  "role": "Role Name" | null,     // Who performs this action (if specified in SOP)
  "system": "System Name" | null, // Software or tool used (if specified in SOP)
  "confidence": "high"            // "high" = explicit in SOP, "medium" = inferred, "low" = guess
}

NODE TYPE RULES:
- "instruction": A sequential action step. MUST have "next" pointing to another node ID.
- "question": A decision point. MUST have "options" with Yes/No keys. "next" must be null.
  Question text MUST end with "?".
- "terminal": End of a process path. "next" and "options" must both be null.
- "reference": Like instruction but links to an external document. MUST have "next" and "external_ref".

STRUCTURE RULES:
1. The FIRST node must have id="start".
2. There must be at least one terminal node (typically id="end").
3. Every instruction/reference node must have "next" pointing to a valid node ID.
4. Every question node must have "options" with at least "Yes" and "No" keys.
5. All node IDs referenced in "next" or "options" must exist as node IDs in your output.
6. After a decision where both branches converge, both branches' last nodes should \
point to the same convergence node.

CONTENT RULES:
1. Capture MAXIMUM detail — every specific click, check, data entry, decision point, \
and cross-section dependency mentioned in the SOP. Do NOT summarize or collapse \
multiple actions into one node. Each distinct action gets its own node.
2. Include role and system metadata when the SOP specifies who does what and in which tool.
3. Cross-section references should use type="reference" with the document name in external_ref.
4. Set confidence to "medium" when inferring a connection not explicitly stated, \
"low" when guessing to maintain connectivity.

EXAMPLE (simple decision flow):
Input: "Open the case. If fraud detected, escalate to supervisor. Otherwise, close the case."
Output nodes:
[
  {"id": "start", "type": "instruction", "text": "Open the case", "next": "is_fraud_detected_question", "options": null, "external_ref": null, "role": null, "system": null, "confidence": "high"},
  {"id": "is_fraud_detected_question", "type": "question", "text": "Is fraud detected?", "next": null, "options": {"Yes": "escalate_to_supervisor", "No": "close_the_case"}, "external_ref": null, "role": null, "system": null, "confidence": "high"},
  {"id": "escalate_to_supervisor", "type": "instruction", "text": "Escalate case to supervisor", "next": "end", "options": null, "external_ref": null, "role": null, "system": null, "confidence": "high"},
  {"id": "close_the_case", "type": "instruction", "text": "Close the case", "next": "end", "options": null, "external_ref": null, "role": null, "system": null, "confidence": "high"},
  {"id": "end", "type": "terminal", "text": "End of procedure.", "next": null, "options": null, "external_ref": null, "role": null, "system": null, "confidence": "high"}
]

EXAMPLE (reference node):
{"id": "refer_fraud_guide", "type": "reference", "text": "Refer to Fraud Resolution Guide for escalation procedures", "next": "next_step_id", "options": null, "external_ref": "Fraud Resolution Guide", "role": "Analyst", "system": null, "confidence": "high"}

In your reasoning field, provide a detailed analysis of the SOP structure: \
identify all decision points, branches, convergence points, cross-section \
dependencies, and the overall process flow before producing the nodes.
"""

_GRAPH_HUMAN = """\
Convert this SOP into a workflow graph following the schema exactly. \
Capture every detail — do not summarize or skip steps.

{enriched_sop}
"""

_PATCH_SYSTEM = """\
You are a Graph Refinement Engineer. You compare a specific SOP section/chunk \
against an existing workflow graph to find MISSING or INCORRECT details.

You receive:
- The current graph as an adjacency map (showing connections)
- The current graph as full JSON (showing all node details)
- One specific SOP section/chunk to verify against the graph

Your job:
1. Read the SOP section carefully
2. Check if every action, decision, reference, and detail from that section \
is captured in the graph
3. If anything is MISSING — add it via add_nodes
4. If any existing node is INCORRECT or needs updating — fix it via modify_nodes
5. If any node should be removed — list it in remove_nodes
6. Do NOT remove or restructure nodes that are correct
7. Do NOT add nodes for content not in this SOP section

PATCH RULES:
- When ADDING nodes: ensure their "next"/"options" point to existing node IDs \
or other newly added node IDs. Wire them into the graph correctly.
- When MODIFYING nodes: include ALL fields of the node (the entire node dict \
will be replaced). Match by "id".
- When REMOVING nodes: ensure no remaining nodes reference the removed IDs. \
If they do, include those referencing nodes in modify_nodes with updated \
"next"/"options" values.
- For INSERTING a node between A and B: add the new node N with next=B, \
and modify node A to have next=N.
- For INSERTING a decision that splits a chain A→B: add the question node Q \
and any new branch nodes, modify A to point to Q, and ensure branch ends \
point to B (or wherever they should converge).

If the graph already captures this chunk completely, return empty lists \
for add_nodes, modify_nodes, and remove_nodes.

Node schema reminder:
- instruction: must have "next", "options" must be null
- question: must have "options" (Yes/No), "next" must be null, text ends with "?"
- terminal: "next" and "options" both null
- reference: must have "next" and "external_ref"
"""

_PATCH_HUMAN = """\
## Current Graph (Adjacency Map)
{adjacency_map}

## Current Graph (Full Nodes JSON)
{nodes_json}

## SOP Chunk to Verify
{chunk_text}

Return a GraphPatch with any additions, modifications, or removals needed. \
If the graph already captures this chunk completely, return empty lists.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_call(stage: str, system: str, human: str, **format_kwargs) -> str:
    """Plain-text LLM call (no structured output). Returns raw text response."""
    llm = get_model(stage)
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=human.format(**format_kwargs)),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def _structured_llm_call(
    stage: str, schema, system: str, human: str, retries: int = 2, **format_kwargs
):
    """Structured output LLM call with retry. Returns a Pydantic model instance."""
    llm = get_model(stage)
    structured_llm = llm.with_structured_output(schema)
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=human.format(**format_kwargs)),
    ]
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return structured_llm.invoke(messages)
        except Exception as e:
            last_error = e
            logger.warning(
                "  Structured output attempt %d/%d failed: %s", attempt, retries, e
            )
    raise last_error


def _to_snake_case(text: str) -> str:
    """Convert a text description to a snake_case node ID."""
    words = re.sub(r"[^a-zA-Z0-9\s]", "", text).lower().split()[:4]
    slug = "_".join(words)
    return slug or "node"


def _reassemble_enriched_sop(
    source_text: str,
    enriched_chunks: Optional[List[dict]],
) -> str:
    """Reassemble enriched chunks into one document with cross-references inlined."""
    if not enriched_chunks:
        return source_text

    parts = []
    for ec in enriched_chunks:
        chunk_text = ec.get("chunk_text", "")
        retrieved = ec.get("retrieved_context", "")

        parts.append(chunk_text)
        if retrieved.strip():
            parts.append(f"[Cross-reference context: {retrieved.strip()}]")

    return "\n\n".join(parts)


def _nodes_list_to_dict(nodes: List[WorkflowNode]) -> Dict[str, Dict[str, Any]]:
    """Convert a list of WorkflowNode models to the standard nodes dict."""
    result: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        data = node.model_dump()
        result[data["id"]] = data
    return result


def _ensure_start_node(nodes: Dict[str, Dict[str, Any]]) -> None:
    """Ensure the first node has id='start'. Renames in-place if needed."""
    if "start" in nodes:
        return
    if not nodes:
        return

    # Pick the first node as start
    first_id = next(iter(nodes))
    node_data = nodes.pop(first_id)
    node_data["id"] = "start"
    # Rebuild dict with "start" first
    new_nodes = {"start": node_data}
    new_nodes.update(nodes)
    nodes.clear()
    nodes.update(new_nodes)

    # Update all references to the old ID
    for n in nodes.values():
        if n.get("next") == first_id:
            n["next"] = "start"
        if n.get("options"):
            for k, v in n["options"].items():
                if v == first_id:
                    n["options"][k] = "start"


# ---------------------------------------------------------------------------
# PipelineConverter
# ---------------------------------------------------------------------------


class PipelineConverter:
    """Graph-first pipeline: SOP -> graph JSON -> chunk-by-chunk graph refinement.

    Step 1: LLM produces a workflow graph directly from the full enriched SOP.
    Step 2: Each enriched chunk is used to refine the graph via structured patches.
    """

    converter_id = "pipeline_v4"

    def convert(
        self,
        source_text: str,
        enriched_chunks: Optional[List[dict]] = None,
        dump_dir: Optional[str] = None,
        resume: bool = False,
    ) -> Dict[str, Any]:
        dump_path: Optional[Path] = None
        if dump_dir:
            dump_path = Path(dump_dir)
            dump_path.mkdir(parents=True, exist_ok=True)
            logger.info("[CONVERTER] Stage outputs dir: %s (resume=%s)", dump_path, resume)

        # Reassemble enriched chunks into one document
        enriched_sop = _reassemble_enriched_sop(source_text, enriched_chunks)
        logger.info("[CONVERTER] Enriched SOP: %d chars", len(enriched_sop))

        if dump_path:
            self._dump_stage(dump_path, "enriched_sop", enriched_sop)

        validator = SchemaValidator()

        # ----- Step 1/2: Full SOP -> Graph (structured output) -----
        cached_graph = self._load_cache(dump_path, "initial_graph") if resume else None
        if cached_graph:
            logger.info("[CONVERTER Step 1/2] Loaded initial graph from cache.")
            nodes = json.loads(cached_graph)
        else:
            logger.info(
                "[CONVERTER Step 1/2] Generating graph from SOP (%d chars)...",
                len(enriched_sop),
            )
            result: InitialGraph = _structured_llm_call(
                stage="graph_gen",
                schema=InitialGraph,
                system=_GRAPH_SYSTEM,
                human=_GRAPH_HUMAN,
                enriched_sop=enriched_sop,
            )
            nodes = _nodes_list_to_dict(result.nodes)
            _ensure_start_node(nodes)
            nodes, fixes = validator.validate_and_fix(nodes)
            if fixes:
                logger.info("  Schema fixes applied: %s", fixes)

            # Edge integrity: check all next/options targets exist
            topo_report = get_graph_issues(nodes)
            if topo_report != "Topology Valid.":
                logger.warning("  Initial graph topology issues: %s", topo_report)
            else:
                logger.info("  Initial graph topology: clean")

            logger.info("  Initial graph: %d nodes", len(nodes))

            if dump_path:
                self._dump_stage(dump_path, "initial_graph", json.dumps(nodes, indent=2))

        # ----- Step 2/2: Chunk-by-chunk graph refinement (multi-pass) -----
        cached_refined = self._load_cache(dump_path, "final_graph") if resume else None
        if cached_refined:
            logger.info("[CONVERTER Step 2/2] Loaded refined graph from cache.")
            nodes = json.loads(cached_refined)
        elif enriched_chunks and len(enriched_chunks) > 1:
            num_passes = 2
            total = len(enriched_chunks)
            logger.info(
                "[CONVERTER Step 2/2] Graph refinement — %d chunks × %d passes...",
                total, num_passes,
            )

            for pass_num in range(1, num_passes + 1):
                pass_changes = 0
                logger.info("  --- Pass %d/%d ---", pass_num, num_passes)

                # Resume: check for per-pass checkpoint
                pass_cache_name = f"graph_after_pass_{pass_num}"
                cached_pass = self._load_cache(dump_path, pass_cache_name) if resume else None
                if cached_pass:
                    logger.info("  Loaded pass %d graph from cache.", pass_num)
                    nodes = json.loads(cached_pass)
                    continue

                # Resume: find the latest per-chunk checkpoint within this pass
                start_chunk_idx = 1
                if resume and dump_path:
                    for check_idx in range(total, 0, -1):
                        ckpt = self._load_cache(dump_path, f"graph_p{pass_num}_c{check_idx}")
                        if ckpt:
                            nodes = json.loads(ckpt)
                            start_chunk_idx = check_idx + 1
                            logger.info(
                                "  Resuming pass %d from chunk %d/%d (loaded checkpoint).",
                                pass_num, start_chunk_idx, total,
                            )
                            break

                for idx, ec in enumerate(enriched_chunks, start=1):
                    if idx < start_chunk_idx:
                        continue

                    chunk_text = ec.get("chunk_text", "")
                    ctx = ec.get("retrieved_context", "").strip()
                    if ctx:
                        chunk_text += f"\n\n[Cross-reference context: {ctx}]"

                    logger.info(
                        "  Pass %d, chunk %d/%d (chunk %s)...",
                        pass_num, idx, total, ec.get("chunk_id", idx - 1),
                    )

                    # Snapshot for rollback
                    pre_patch = {nid: dict(data) for nid, data in nodes.items()}
                    pre_patch_count = len(nodes)

                    try:
                        adjacency_map = generate_adjacency_map(nodes)
                        nodes_json = json.dumps(nodes, indent=2)

                        patch: GraphPatch = _structured_llm_call(
                            stage="graph_refine",
                            schema=GraphPatch,
                            system=_PATCH_SYSTEM,
                            human=_PATCH_HUMAN,
                            adjacency_map=adjacency_map,
                            nodes_json=nodes_json,
                            chunk_text=chunk_text,
                        )

                        changes = (
                            len(patch.add_nodes)
                            + len(patch.modify_nodes)
                            + len(patch.remove_nodes)
                        )
                        if changes == 0:
                            logger.info("    No changes needed.")
                            continue

                        logger.info(
                            "    Patch: +%d add, ~%d modify, -%d remove",
                            len(patch.add_nodes),
                            len(patch.modify_nodes),
                            len(patch.remove_nodes),
                        )

                        # Dump patch reasoning for debugging
                        if dump_path and patch.reasoning:
                            reason_name = f"patch_p{pass_num}_c{idx}_reasoning"
                            self._dump_stage(dump_path, reason_name, patch.reasoning)

                        apply_patch(nodes, patch)
                        nodes, fixes = validator.validate_and_fix(nodes)
                        if fixes:
                            logger.info("    Schema fixes: %s", fixes)

                        # Topological validation after patch
                        topo_report = get_graph_issues(nodes)
                        if topo_report != "Topology Valid.":
                            logger.warning("    Post-patch topology issues: %s", topo_report)

                        # Sanity checks
                        if len(nodes) < pre_patch_count * 0.7:
                            logger.warning(
                                "    Patch shrank graph from %d to %d nodes (>30%% loss) — rolling back.",
                                pre_patch_count, len(nodes),
                            )
                            nodes = pre_patch
                            continue

                        if "start" not in nodes:
                            logger.warning("    Patch removed 'start' node — rolling back.")
                            nodes = pre_patch
                            continue

                        pass_changes += changes

                    except Exception as e:
                        logger.warning(
                            "    Patch failed for chunk %d (%s) — keeping previous graph.",
                            idx, e,
                        )
                        nodes = pre_patch

                    # Checkpoint after each chunk so we can resume mid-pass
                    if dump_path:
                        self._dump_stage(
                            dump_path,
                            f"graph_p{pass_num}_c{idx}",
                            json.dumps(nodes, indent=2),
                        )

                logger.info(
                    "  Pass %d complete: %d total changes applied.", pass_num, pass_changes
                )

                # If pass 2 made no changes, the graph is stable
                if pass_num > 1 and pass_changes == 0:
                    logger.info("  No changes in pass %d — graph is stable.", pass_num)
                    break

                # Dump pass-level checkpoint
                if dump_path:
                    self._dump_stage(
                        dump_path,
                        pass_cache_name,
                        json.dumps(nodes, indent=2),
                    )

            if dump_path:
                self._dump_stage(dump_path, "final_graph", json.dumps(nodes, indent=2))
        else:
            logger.info("[CONVERTER Step 2/2] Skipped (single chunk / no chunks).")

        type_counts: Dict[str, int] = {}
        for n in nodes.values():
            t = n.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        logger.info("  Graph: %d nodes — %s", len(nodes), type_counts)
        logger.info("[CONVERTER] Complete.")

        return nodes

    @staticmethod
    def _load_cache(dump_dir: Optional[Path], name: str) -> Optional[str]:
        if dump_dir is None:
            return None
        # Try both .txt and .json extensions
        for ext in (".txt", ".json"):
            path = dump_dir / f"{name}{ext}"
            if path.exists():
                logger.info("  [CACHE HIT] %s <- %s", name, path)
                return path.read_text()
        return None

    @staticmethod
    def _dump_stage(dump_dir: Path, name: str, content: str) -> None:
        # Use .txt for plain text outlines, .json for structured data
        ext = ".json" if content.lstrip().startswith(("{", "[")) else ".txt"
        path = dump_dir / f"{name}{ext}"
        path.write_text(content)
        logger.info("  [DUMP] %s -> %s", name, path)
