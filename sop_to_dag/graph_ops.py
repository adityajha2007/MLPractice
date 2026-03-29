"""Graph operations: analysis, refinement, and the self-refinement loop.

Consolidates all graph-level operations into one module:
  - Topological checks (pure Python)
  - LLM-based quality checks (completeness, context, granularity)
  - Analysis orchestrator
  - Triplet verification, granularity expansion, error resolution
  - Schema validation (deterministic)
  - Refinement orchestrator
  - LangGraph self-refinement loop
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from sop_to_dag.models import get_model, safe_invoke
from sop_to_dag.schemas import (
    GraphPatch,
    GraphState,
    RefineFeedback,
    WorkflowNode,
)
from sop_to_dag.storage import GraphStore

logger = logging.getLogger(__name__)


# ===========================================================================
# Pydantic models (internal to this module)
# ===========================================================================


class ContextFeedback(BaseModel):
    """Structured output for the context adjacency check."""

    is_valid: bool = Field(description="Whether all edges are logically valid.")
    issues: List[str] = Field(
        default_factory=list,
        description="List of logical flow problems found.",
    )


class _NodePatch(BaseModel):
    """Wrapper for structured output: list of repaired nodes."""

    nodes: List[WorkflowNode]


class _TripletResult(BaseModel):
    """Result of verifying a single triplet."""

    triplet_index: int
    is_valid: bool
    explanation: str = ""


class _TripletVerification(BaseModel):
    """Structured output for triplet batch verification."""

    results: List[_TripletResult]


# ===========================================================================
# Prompts — Analysis
# ===========================================================================

_COMPLETENESS_SYSTEM = """\
You are a Process Quality Auditor. Your job is to verify that a workflow graph
completely and accurately represents all steps in the original SOP text.

Compare the graph (provided as an adjacency map) against the original SOP.
Identify:
1. Steps in the SOP that are MISSING from the graph
2. Decision branches that are incomplete (e.g., only Yes branch, no No branch)
3. Any significant information loss

Be precise — cite specific SOP text that is not represented.
"""

_COMPLETENESS_HUMAN = """\
## Original SOP
{source_text}

## Current Graph Nodes
{nodes_compact}

Evaluate completeness. Return RefineFeedback with is_complete and missing_branches.
"""

_CONTEXT_SYSTEM = """\
You are a Process Flow Analyst. Your job is to verify that connected nodes in a
workflow graph are logically adjacent — meaning the flow from one node to the
next makes sense in the context of the original SOP.

For each edge in the graph, evaluate:
1. Does the transition make logical sense?
2. Are there missing intermediate steps?
3. Is the edge direction correct?

Report only genuine issues — do not flag correct transitions.
"""

_CONTEXT_HUMAN = """\
## Original SOP
{source_text}

## Current Graph Nodes
{nodes_compact}

Evaluate logical adjacency of connected nodes. Return a ContextFeedback with
is_valid and issues.
"""


# ===========================================================================
# Prompts — Refinement
# ===========================================================================

_TRIPLET_SYSTEM = """\
You are a Graph Verification Specialist. You verify that decision-node triplets
(source_node, edge_label, target_node) are correct according to the original SOP.

For each triplet, evaluate:
1. Does the source node's question/condition match the SOP?
2. Does the edge label (answer option) correctly represent a valid answer?
3. Does the target node logically follow from that answer?

Mark each triplet as VALID or INVALID with an explanation.
"""

_TRIPLET_HUMAN = """\
## Original SOP
{source_text}

## Triplets to Verify (batch)
{triplets_json}

For each triplet, return a _TripletResult with triplet_index, is_valid, and
explanation.
"""

_PATCH_RESOLVER_SYSTEM = """\
You are a Graph Repair Specialist. You fix issues in workflow graphs by \
producing a structured patch (add/modify/remove operations).

You receive:
1. The full current graph as JSON (all node fields included)
2. The original SOP text
3. A specific list of issues to fix (from the analyser)

Your job: produce a GraphPatch that fixes ALL the listed issues while \
preserving everything that is already correct.

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
- For EXPANDING a coarse node: replace it with multiple sub-step nodes. The \
first replacement must keep the original ID (so incoming edges work). The last \
must point to whatever the original's "next" was.
- For FIXING an invalid triplet: modify the question node's options or the \
target node as needed.
- Do NOT touch nodes that are already correct.
- If no fixes are needed, return empty lists.

Node schema reminder:
- instruction: must have "next", "options" must be null
- question: must have "options" (Yes/No), "next" must be null, text ends with "?"
- terminal: "next" and "options" both null
- reference: must have "next" and "external_ref"
- All IDs must be snake_case
- confidence: "high" = explicit in SOP, "medium" = inferred, "low" = guess
"""

_PATCH_RESOLVER_HUMAN = """\
## Original SOP
{source_text}

## Current Graph (Full Nodes JSON)
{nodes_json}

## Issues to Fix
{feedback}

Return a GraphPatch with the additions, modifications, and removals needed \
to fix all listed issues. If nothing needs fixing, return empty lists.
"""

BATCH_SIZE = 12


# ===========================================================================
# Topological checks (pure Python, no LLM)
# ===========================================================================


def get_graph_issues(nodes: Dict[str, Any]) -> str:
    """Scan for ORPHAN, BROKEN and dead-end nodes.

    Returns a human-readable report string, or 'Topology Valid.' if clean.
    """
    if not nodes:
        return "Graph is empty. Create the start node."

    defined_ids = set(nodes.keys())
    referenced_ids: set = set()

    for _, data in nodes.items():
        if data.get("next"):
            referenced_ids.add(data["next"])
        if data.get("options"):
            for target in data["options"].values():
                referenced_ids.add(target)

    orphans = [
        n
        for n in (defined_ids - referenced_ids)
        if n not in ["start", "START", "root"]
    ]

    dead_ends = [
        n
        for n, d in nodes.items()
        if d.get("type") != "terminal" and not d.get("next") and not d.get("options")
    ]

    broken = list(referenced_ids - defined_ids)

    report: List[str] = []
    if orphans:
        report.append(f"ORPHAN NODES: {orphans}")
    if dead_ends:
        report.append(f"DEAD ENDS: {dead_ends}")
    if broken:
        report.append(f"BROKEN LINKS: {broken}")

    return "\n".join(report) if report else "Topology Valid."


def generate_adjacency_map(nodes: Dict[str, Any]) -> str:
    """Create a simplified text map (Node A --> Node B) for LLM consumption."""
    lines: List[str] = []
    for n_id, data in nodes.items():
        if data.get("type") == "terminal":
            lines.append(f"{n_id} [END]")
        else:
            targets: List[str] = []
            if data.get("next"):
                targets.append(f"--> {data['next']}")
            if data.get("options"):
                for k, v in data["options"].items():
                    targets.append(f"--({k})--> {v}")
            lines.append(f"{n_id} {' '.join(targets)}")

    return "\n".join(lines)


_CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0}
_RANK_TO_CONFIDENCE = {v: k for k, v in _CONFIDENCE_RANK.items()}


def _lower_confidence(a: str, b: str) -> str:
    """Return the lower of two confidence levels."""
    rank = min(_CONFIDENCE_RANK.get(a, 0), _CONFIDENCE_RANK.get(b, 0))
    return _RANK_TO_CONFIDENCE.get(rank, "low")


def _can_merge(
    node_a: Dict[str, Any],
    node_b: Dict[str, Any],
    b_id: str,
    nodes: Dict[str, Any],
) -> bool:
    """Check whether node_a and node_b can be merged (both instruction, same role/system, B has 1 incoming)."""
    if node_b.get("type") != "instruction":
        return False
    if node_a.get("role") != node_b.get("role"):
        return False
    if node_a.get("system") != node_b.get("system"):
        return False
    # B must have exactly one incoming edge
    incoming = sum(
        1 for n in nodes.values()
        if n.get("next") == b_id
        or (n.get("options") and b_id in n["options"].values())
    )
    return incoming == 1


def merge_sequential_instructions(
    nodes: Dict[str, Any],
) -> Tuple[Dict[str, Any], int]:
    """Collapse overly granular sequential instruction chains.

    Merges A→B when both are instructions with the same role/system and B has
    exactly one incoming edge. Chains extend greedily (A→B→C all merge into A).

    Returns (nodes, merge_count).
    """
    merged_count = 0
    visited: set = set()

    for node_id in list(nodes.keys()):
        if node_id in visited or node_id not in nodes:
            continue

        current_id = node_id
        while True:
            current = nodes.get(current_id)
            if not current or current.get("type") != "instruction":
                break

            next_id = current.get("next")
            if not next_id or next_id not in nodes:
                break

            next_node = nodes[next_id]
            if not _can_merge(current, next_node, next_id, nodes):
                break

            # Merge next_node into current
            current["text"] = current["text"] + ". " + next_node["text"]
            current["next"] = next_node.get("next")
            current["confidence"] = _lower_confidence(
                current.get("confidence", "high"),
                next_node.get("confidence", "high"),
            )
            del nodes[next_id]
            visited.add(next_id)
            merged_count += 1

        visited.add(current_id)

    return nodes, merged_count


def compact_nodes_repr(nodes: Dict[str, Any]) -> str:
    """Compact text representation of nodes for analyser LLM calls.

    Includes id, type, text, and connections — drops null fields.
    ~60-70% smaller than full JSON, enough for completeness/context checks.
    Full JSON is only needed by the resolver (which produces patches).
    """
    lines: List[str] = []
    for n_id, data in nodes.items():
        parts = [f"[{n_id}] ({data.get('type', '?')}) \"{data.get('text', '')}\""]
        if data.get("next"):
            parts.append(f"next={data['next']}")
        if data.get("options"):
            opts = ", ".join(f"{k}:{v}" for k, v in data["options"].items())
            parts.append(f"options={{{opts}}}")
        if data.get("role"):
            parts.append(f"role={data['role']}")
        if data.get("system"):
            parts.append(f"system={data['system']}")
        if data.get("external_ref"):
            parts.append(f"ref={data['external_ref']}")
        if data.get("confidence") and data["confidence"] != "high":
            parts.append(f"confidence={data['confidence']}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def get_all_issues_structured(
    nodes: Dict[str, Any],
) -> Tuple[List[str], List[str], List[str]]:
    """Return (orphans, dead_ends, broken_links) as separate lists."""
    if not nodes:
        return [], [], []

    defined_ids = set(nodes.keys())
    referenced_ids: set = set()

    for _, data in nodes.items():
        if data.get("next"):
            referenced_ids.add(data["next"])
        if data.get("options"):
            for target in data["options"].values():
                referenced_ids.add(target)

    orphans = [
        n
        for n in (defined_ids - referenced_ids)
        if n not in ["start", "START", "root"]
    ]
    dead_ends = [
        n
        for n, d in nodes.items()
        if d.get("type") != "terminal" and not d.get("next") and not d.get("options")
    ]
    broken = list(referenced_ids - defined_ids)

    return orphans, dead_ends, broken


# ===========================================================================
# Graph helpers (shared by refiner components)
# ===========================================================================


def _get_neighbors(node_id: str, nodes: Dict[str, Any]) -> List[str]:
    """Get direct neighbors (outgoing + incoming) of a node."""
    neighbors = []
    data = nodes.get(node_id, {})

    if data.get("next"):
        neighbors.append(data["next"])
    if data.get("options"):
        neighbors.extend(data["options"].values())

    for nid, ndata in nodes.items():
        if nid == node_id:
            continue
        if ndata.get("next") == node_id:
            neighbors.append(nid)
        if ndata.get("options") and node_id in ndata["options"].values():
            neighbors.append(nid)

    return list(set(neighbors))


def _get_2hop_neighborhood(
    node_id: str, nodes: Dict[str, Any]
) -> Dict[str, Any]:
    """Get all nodes within 2 hops of the given node."""
    neighborhood: Dict[str, Any] = {}
    hop1_ids = _get_neighbors(node_id, nodes)
    neighborhood[node_id] = nodes[node_id]

    for nid in hop1_ids:
        if nid in nodes:
            neighborhood[nid] = nodes[nid]
            hop2_ids = _get_neighbors(nid, nodes)
            for nid2 in hop2_ids:
                if nid2 in nodes:
                    neighborhood[nid2] = nodes[nid2]

    return neighborhood


# ===========================================================================
# LLM-based analysis checks
# ===========================================================================


def check_completeness(nodes: dict, source_text: str) -> RefineFeedback:
    """LLM check: graph vs source text alignment."""
    nodes_compact = compact_nodes_repr(nodes)

    llm = get_model("completeness")
    structured_llm = llm.with_structured_output(RefineFeedback)
    messages = [
        SystemMessage(content=_COMPLETENESS_SYSTEM),
        HumanMessage(
            content=_COMPLETENESS_HUMAN.format(
                source_text=source_text,
                nodes_compact=nodes_compact,
            )
        ),
    ]
    return safe_invoke(structured_llm, messages, context="completeness_check")


def check_context(nodes: dict, source_text: str) -> Dict[str, Any]:
    """LLM check: logical adjacency of connected nodes."""
    nodes_compact = compact_nodes_repr(nodes)

    llm = get_model("context")
    structured_llm = llm.with_structured_output(ContextFeedback)
    messages = [
        SystemMessage(content=_CONTEXT_SYSTEM),
        HumanMessage(
            content=_CONTEXT_HUMAN.format(
                source_text=source_text,
                nodes_compact=nodes_compact,
            )
        ),
    ]

    result = safe_invoke(structured_llm, messages, context="context_check")
    return {"is_valid": result.is_valid, "issues": result.issues}



# ===========================================================================
# Analysis orchestrator
# ===========================================================================


def analyse(state: GraphState) -> GraphState:
    """Run topological -> completeness -> context -> granularity checks.

    Mutates and returns the state with updated feedback, analysis_report,
    and is_complete fields.
    """
    iteration = state.get("iteration", 0)
    logger.info("[ANALYSER iter %d] Analysing graph with %d nodes...", iteration, len(state["nodes"]))

    nodes = state["nodes"]
    source_text = state["source_text"]
    report_parts = []

    # 1. Topological check (code-based, free)
    logger.info("  [ANALYSER] Check 1/3: Topological (pure Python)...")
    topo_report = get_graph_issues(nodes)
    report_parts.append(f"## Topological Check\n{topo_report}")
    has_topo_issues = topo_report != "Topology Valid."
    if has_topo_issues:
        logger.warning("    Topology issues found: %s", topo_report)
    else:
        logger.info("    Topology: VALID")

    # 2. Completeness check (LLM-based)
    logger.info("  [ANALYSER] Check 2/3: Completeness (LLM)...")
    completeness = check_completeness(nodes, source_text)
    report_parts.append(
        f"## Completeness Check\n"
        f"Complete: {completeness.is_complete}\n"
        f"Missing: {completeness.missing_branches}"
    )
    logger.info("    Complete: %s", completeness.is_complete)
    if completeness.missing_branches:
        logger.info("    Missing branches: %s", completeness.missing_branches)

    # 3. Context check (LLM-based)
    logger.info("  [ANALYSER] Check 3/3: Context adjacency (LLM)...")
    context_result = check_context(nodes, source_text)
    report_parts.append(
        f"## Context Check\n"
        f"Valid: {context_result['is_valid']}\n"
        f"Issues: {context_result['issues']}"
    )
    logger.info("    Valid: %s", context_result["is_valid"])
    if context_result["issues"]:
        logger.info("    Context issues: %s", context_result["issues"])

    # Aggregate: graph is complete only if ALL checks pass
    is_complete = (
        not has_topo_issues
        and completeness.is_complete
        and context_result["is_valid"]
    )

    # Build categorized feedback — each category is a list of individual issues
    categorized_feedback: Dict[str, List[str]] = {}
    if has_topo_issues:
        topo_lines = [line.strip() for line in topo_report.split("\n") if line.strip()]
        categorized_feedback["topological"] = topo_lines
    if not completeness.is_complete:
        categorized_feedback["completeness"] = list(completeness.missing_branches)
    if not context_result["is_valid"]:
        categorized_feedback["context"] = list(context_result["issues"])

    # Flat string for logging
    all_issues = [issue for issues in categorized_feedback.values() for issue in issues]
    flat_feedback = "\n".join(all_issues)

    state["is_complete"] = is_complete
    state["feedback"] = flat_feedback
    state["categorized_feedback"] = categorized_feedback
    state["analysis_report"] = "\n\n".join(report_parts)

    verdict = "PASS — graph is complete" if is_complete else "FAIL — needs refinement"
    logger.info("  [ANALYSER] Verdict: %s", verdict)

    return state


# ===========================================================================
# TripletVerifier
# ===========================================================================


class TripletVerifier:
    """Verify decision-node triplets against source SOP text.

    Prioritizes low-confidence edges first, then medium. High-confidence
    edges are skipped unless flagged by topological analysis.
    """

    def __init__(self):
        self.llm = get_model("triplet")

    def extract_conditional_triplets(
        self, nodes: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Extract all (source, edge_label, target) triplets from question nodes."""
        triplets = []
        for node_id, data in nodes.items():
            if data.get("type") != "question":
                continue
            options = data.get("options", {})
            if not options:
                continue
            for label, target_id in options.items():
                target_data = nodes.get(target_id, {})
                triplets.append(
                    {
                        "source_id": node_id,
                        "source_text": data.get("text", ""),
                        "edge_label": label,
                        "target_id": target_id,
                        "target_text": target_data.get("text", f"[MISSING: {target_id}]"),
                        "confidence": data.get("confidence", "high"),
                    }
                )
        return triplets

    def verify(
        self, nodes: Dict[str, Any], source_text: str
    ) -> List[Dict[str, Any]]:
        """Verify conditional triplets, prioritizing low-confidence edges.

        Returns list of invalid triplets with explanations.
        """
        triplets = self.extract_conditional_triplets(nodes)
        if not triplets:
            return []

        # Sort by confidence: low first, then medium, skip high
        priority_order = {"low": 0, "medium": 1, "high": 2}
        triplets.sort(key=lambda t: priority_order.get(t.get("confidence", "high"), 2))

        # Filter: verify low and medium; skip high unless very few triplets
        to_verify = [t for t in triplets if t.get("confidence") != "high"]
        if not to_verify:
            # All high confidence — verify all as fallback
            to_verify = triplets

        invalid_triplets = []
        for i in range(0, len(to_verify), BATCH_SIZE):
            batch = to_verify[i : i + BATCH_SIZE]
            batch_results = self._verify_batch(batch, source_text)
            invalid_triplets.extend(batch_results)

        return invalid_triplets

    def _verify_batch(
        self, batch: List[Dict[str, str]], source_text: str
    ) -> List[Dict[str, Any]]:
        """Verify a single batch of triplets via LLM."""
        numbered = [{**t, "index": idx} for idx, t in enumerate(batch)]
        structured_llm = self.llm.with_structured_output(_TripletVerification)

        messages = [
            SystemMessage(content=_TRIPLET_SYSTEM),
            HumanMessage(
                content=_TRIPLET_HUMAN.format(
                    source_text=source_text,
                    triplets_json=json.dumps(numbered, indent=2),
                )
            ),
        ]

        invalid = []
        verification = safe_invoke(
            structured_llm, messages, context="triplet_verification"
        )
        for r in verification.results:
            if not r.is_valid and r.triplet_index < len(batch):
                invalid.append(
                    {
                        **batch[r.triplet_index],
                        "explanation": r.explanation,
                    }
                )

        return invalid


# ===========================================================================
# Patch application (shared with converter.py)
# ===========================================================================


def apply_patch(
    nodes: Dict[str, Dict[str, Any]],
    patch: GraphPatch,
) -> Dict[str, Dict[str, Any]]:
    """Apply a GraphPatch to an existing nodes dict.

    Order: add_nodes -> modify_nodes -> remove_nodes.
    Returns the updated nodes dict (mutated in place).
    """
    # 1. Add new nodes
    for node in patch.add_nodes:
        data = node.model_dump()
        nid = data["id"]
        if nid in nodes:
            logger.warning("  [PATCH] Skipping add for '%s' — ID already exists.", nid)
            continue
        nodes[nid] = data
        logger.debug("  [PATCH] Added node '%s'.", nid)

    # 2. Modify existing nodes
    for node in patch.modify_nodes:
        data = node.model_dump()
        nid = data["id"]
        if nid not in nodes:
            logger.warning("  [PATCH] Skipping modify for '%s' — ID not found.", nid)
            continue
        nodes[nid] = data
        logger.debug("  [PATCH] Modified node '%s'.", nid)

    # 3. Remove nodes
    for nid in patch.remove_nodes:
        if nid not in nodes:
            logger.warning("  [PATCH] Skipping remove for '%s' — ID not found.", nid)
            continue
        for other_id, other_data in nodes.items():
            if other_id == nid:
                continue
            if other_data.get("next") == nid:
                logger.warning(
                    "  [PATCH] Node '%s' still references removed node '%s' via next.",
                    other_id, nid,
                )
            if other_data.get("options"):
                for opt_key, opt_val in other_data["options"].items():
                    if opt_val == nid:
                        logger.warning(
                            "  [PATCH] Node '%s' still references removed node '%s' via options.%s.",
                            other_id, nid, opt_key,
                        )
        del nodes[nid]
        logger.debug("  [PATCH] Removed node '%s'.", nid)

    return nodes


# ===========================================================================
# GraphPatchResolver
# ===========================================================================


class GraphPatchResolver:
    """Resolve all analyser-flagged issues via a single full-graph patch.

    Unlike the old ErrorResolver (2-hop window per node) and GranularityExpander
    (2-hop window per coarse node), this resolver sees the FULL graph + FULL
    feedback + source SOP and produces a coordinated GraphPatch that can:
    - Insert multi-node decision branches
    - Restructure sections spanning many hops
    - Expand coarse nodes while rewiring surrounding edges
    - Fix invalid triplets with coordinated option + target changes
    """

    def __init__(self):
        self.llm = get_model("resolver")

    def resolve(
        self,
        nodes: Dict[str, Any],
        feedback: str,
        source_text: str,
    ) -> Tuple[Dict[str, Any], GraphPatch]:
        """Produce and apply a GraphPatch to fix all issues in feedback.

        Returns (updated_nodes, patch_applied).
        """
        if not feedback.strip():
            empty_patch = GraphPatch(reasoning="No issues to fix.")
            return nodes, empty_patch

        nodes_json = json.dumps(nodes, indent=2)

        structured_llm = self.llm.with_structured_output(GraphPatch)
        messages = [
            SystemMessage(content=_PATCH_RESOLVER_SYSTEM),
            HumanMessage(
                content=_PATCH_RESOLVER_HUMAN.format(
                    source_text=source_text,
                    nodes_json=nodes_json,
                    feedback=feedback,
                )
            ),
        ]

        patch = safe_invoke(
            structured_llm, messages, context="patch_resolver"
        )

        changes = (
            len(patch.add_nodes)
            + len(patch.modify_nodes)
            + len(patch.remove_nodes)
        )

        if changes == 0:
            logger.info("  [RESOLVER] No changes needed.")
            return nodes, patch

        logger.info(
            "  [RESOLVER] Patch: +%d add, ~%d modify, -%d remove",
            len(patch.add_nodes),
            len(patch.modify_nodes),
            len(patch.remove_nodes),
        )

        # Snapshot for rollback
        pre_patch = {nid: dict(data) for nid, data in nodes.items()}
        pre_patch_count = len(nodes)

        apply_patch(nodes, patch)

        # Rollback if patch is catastrophic
        if len(nodes) < pre_patch_count * 0.7:
            logger.warning(
                "  [RESOLVER] Patch shrank graph from %d to %d nodes (>30%% loss) — rolling back.",
                pre_patch_count, len(nodes),
            )
            nodes.clear()
            nodes.update(pre_patch)
            return nodes, GraphPatch(reasoning="Rolled back — catastrophic shrinkage.")

        if "start" not in nodes:
            logger.warning("  [RESOLVER] Patch removed 'start' node — rolling back.")
            nodes.clear()
            nodes.update(pre_patch)
            return nodes, GraphPatch(reasoning="Rolled back — start node removed.")

        return nodes, patch


# ===========================================================================
# SchemaValidator
# ===========================================================================


class SchemaValidator:
    """Validate and auto-fix nodes against the WorkflowNode schema."""

    def validate_and_fix(
        self, nodes: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Validate all nodes and apply deterministic fixes.

        Returns (fixed_nodes_dict, list_of_fix_descriptions).
        """
        fixed_nodes: Dict[str, Any] = {}
        fixes: List[str] = []

        for node_id, data in nodes.items():
            node_data = {**data, "id": node_id}
            fixed, fix_msgs = self._fix_single_node(node_data)
            fixed_nodes[fixed["id"]] = fixed
            fixes.extend(fix_msgs)

        return fixed_nodes, fixes

    def _fix_single_node(
        self, data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Attempt to fix a single node to pass Pydantic validation."""
        fixes: List[str] = []
        node_id = data.get("id", "unknown")
        node_type = data.get("type", "instruction")

        # Fix: terminal nodes should not have next/options
        if node_type == "terminal":
            if data.get("next"):
                fixes.append(f"Removed 'next' from terminal node '{node_id}'")
                data["next"] = None
            if data.get("options"):
                fixes.append(f"Removed 'options' from terminal node '{node_id}'")
                data["options"] = None

        # Fix: question nodes must have options
        if node_type == "question" and not data.get("options"):
            fixes.append(
                f"Node '{node_id}' is question-type but has no options — "
                f"converting to instruction"
            )
            data["type"] = "instruction"
            node_type = "instruction"

        # Fix: instruction nodes must have next
        if node_type == "instruction" and not data.get("next"):
            fixes.append(
                f"Node '{node_id}' is instruction-type but has no 'next' — "
                f"needs manual resolution"
            )

        # Fix: question nodes should not have 'next'
        if node_type == "question" and data.get("next"):
            fixes.append(f"Removed 'next' from question node '{node_id}' (use options)")
            data["next"] = None

        try:
            node = WorkflowNode(**data)
            return node.model_dump(), fixes
        except Exception as e:
            fixes.append(f"Validation failed for '{node_id}': {e}")
            return data, fixes


# ===========================================================================
# Refinement orchestrator
# ===========================================================================


def refine(state: GraphState) -> GraphState:
    """Issue-by-issue refinement: each analyser issue gets its own resolver call.

    Order:
      1. Triplet verification → collect invalid triplets as individual issues
      2. LLM-based categories (completeness, context, triplets) — one resolver
         call per issue line, so nothing gets skipped
      3. Fresh topological scan — catches structural issues introduced by
         the LLM patches above
      4. Schema validation (deterministic)

    Mutates and returns the state with updated nodes and incremented iteration.
    """
    iteration = state.get("iteration", 0) + 1
    logger.info("[REFINER iter %d] Starting refinement cycle...", iteration)

    nodes = state["nodes"]
    source_text = state["source_text"]
    categorized_feedback: Dict[str, List[str]] = {
        k: list(v) for k, v in state.get("categorized_feedback", {}).items()
    }

    triplet_verifier = TripletVerifier()
    patch_resolver = GraphPatchResolver()
    schema_validator = SchemaValidator()

    # 1. Triplet verification — add as its own category
    logger.info("  [REFINER iter %d] Step 1: Triplet verification...", iteration)
    invalid_triplets = triplet_verifier.verify(nodes, source_text)
    logger.info("    %d invalid triplets found", len(invalid_triplets))
    for t in invalid_triplets:
        logger.info("    INVALID: %s --(%s)--> %s: %s",
                     t["source_id"], t["edge_label"], t["target_id"],
                     t.get("explanation", "")[:100])

    if invalid_triplets:
        categorized_feedback["triplets"] = [
            f"{t['source_id']} --({t['edge_label']})--> "
            f"{t['target_id']}: {t['explanation']}"
            for t in invalid_triplets
        ]

    # 2. LLM-based categories — one resolver call per issue
    llm_categories = ["completeness", "context", "triplets"]
    total_issues = 0

    for category in llm_categories:
        issues = categorized_feedback.get(category, [])
        if not issues:
            continue
        logger.info(
            "  [REFINER iter %d] Category '%s': %d issues",
            iteration, category, len(issues),
        )
        for i, issue in enumerate(issues):
            logger.info("    [%s %d/%d] %s", category, i + 1, len(issues), issue[:120])
            nodes_before = len(nodes)
            nodes, patch = patch_resolver.resolve(nodes, issue, source_text)
            changes = (
                len(patch.add_nodes) + len(patch.modify_nodes) + len(patch.remove_nodes)
            )
            logger.info(
                "      Nodes: %d -> %d (changes: %d)",
                nodes_before, len(nodes), changes,
            )
            total_issues += 1

    # 3. Fresh topological scan — catches issues introduced by LLM patches
    logger.info("  [REFINER iter %d] Step 3: Fresh topological scan...", iteration)
    topo_report = get_graph_issues(nodes)
    if topo_report != "Topology Valid.":
        topo_lines = [line.strip() for line in topo_report.split("\n") if line.strip()]
        logger.info("    %d topological issues found post-patch", len(topo_lines))
        for i, issue in enumerate(topo_lines):
            logger.info("    [topo %d/%d] %s", i + 1, len(topo_lines), issue)
            nodes, patch = patch_resolver.resolve(
                nodes, f"Fix this topological issue: {issue}", source_text,
            )
            total_issues += 1
    else:
        logger.info("    Topology: VALID")

    # 4. Schema validation (deterministic fixes)
    logger.info("  [REFINER iter %d] Step 4: Schema validation...", iteration)
    nodes, fix_msgs = schema_validator.validate_and_fix(nodes)
    if fix_msgs:
        for msg in fix_msgs:
            logger.info("    FIX: %s", msg)
    else:
        logger.info("    All nodes valid")

    state["nodes"] = nodes
    state["iteration"] = iteration
    if fix_msgs:
        state["feedback"] = (
            state.get("feedback", "") + "\nSchema fixes: " + "; ".join(fix_msgs)
        )

    logger.info(
        "  [REFINER iter %d] Complete. %d nodes in graph. %d issues patched.",
        iteration, len(nodes), total_issues,
    )
    return state


# ===========================================================================
# LangGraph self-refinement loop
# ===========================================================================

MAX_ITERATIONS = 10


def _dump_iteration(dump_dir: Path, iteration: int, state: GraphState) -> None:
    """Dump graph state and analysis report for a single refinement iteration."""
    (dump_dir / f"refine_iter{iteration}_graph.json").write_text(
        json.dumps(state["nodes"], indent=2)
    )
    (dump_dir / f"refine_iter{iteration}_report.txt").write_text(
        state.get("analysis_report", "")
    )
    logger.info("  [DUMP] Refinement iteration %d -> %s", iteration, dump_dir)


def _build_graph(
    store: GraphStore | None = None,
    max_iterations: int = MAX_ITERATIONS,
    dump_dir: Optional[Path] = None,
) -> StateGraph:
    """Build the LangGraph StateGraph for the refinement loop."""

    def analyse_node(state: GraphState) -> GraphState:
        result = analyse(state)
        if dump_dir:
            _dump_iteration(dump_dir, result.get("iteration", 0), result)
        return result

    def refine_node(state: GraphState) -> GraphState:
        state = refine(state)
        if store:
            store.save_graph(
                state,
                source_file=f"refinement_iter_{state['iteration']}",
                converter_id=state.get("converter_id", "unknown"),
            )
        return state

    def should_continue(state: GraphState) -> str:
        """Routing function: decide whether to refine or stop."""
        if state.get("is_complete", False):
            logger.info("[LOOP] Graph is complete — stopping refinement.")
            return "end"
        if state.get("iteration", 0) >= max_iterations:
            logger.warning("[LOOP] Max iterations (%d) reached — stopping refinement.", max_iterations)
            return "end"
        logger.info("[LOOP] Graph incomplete — routing to refiner (iteration %d)...",
                     state.get("iteration", 0) + 1)
        return "refine"

    graph = StateGraph(GraphState)

    graph.add_node("analyse", analyse_node)
    graph.add_node("refine", refine_node)

    graph.set_entry_point("analyse")

    graph.add_conditional_edges(
        "analyse",
        should_continue,
        {
            "refine": "refine",
            "end": END,
        },
    )

    graph.add_edge("refine", "analyse")

    return graph


def run_refinement(
    graph_state: GraphState,
    max_iterations: int = MAX_ITERATIONS,
    store: GraphStore | None = None,
    dump_dir: Optional[str] = None,
    resume: bool = False,
) -> GraphState:
    """Execute the refinement loop on a graph state.

    Args:
        graph_state: Initial graph state (from a converter).
        max_iterations: Safety cap on iterations.
        store: Optional GraphStore for intermediate persistence.
        dump_dir: Optional directory to dump per-iteration state files.
        resume: If True, resume from the latest dumped iteration.

    Returns:
        Final refined GraphState.
    """
    dump_path = Path(dump_dir) if dump_dir else None

    # Resume: load the latest iteration checkpoint
    if resume and dump_path and dump_path.exists():
        for check_iter in range(max_iterations, 0, -1):
            ckpt_path = dump_path / f"refine_iter{check_iter}_graph.json"
            if ckpt_path.exists():
                logger.info(
                    "[LOOP] Resuming from iteration %d checkpoint.", check_iter
                )
                graph_state["nodes"] = json.loads(ckpt_path.read_text())
                graph_state["iteration"] = check_iter
                break

    logger.info("[LOOP] Starting refinement loop (max %d iterations, %d nodes, iter=%d)...",
                 max_iterations, len(graph_state["nodes"]),
                 graph_state.get("iteration", 0))

    graph = _build_graph(store, max_iterations=max_iterations, dump_dir=dump_path)
    app = graph.compile()

    final_state = app.invoke(graph_state)

    logger.info("[LOOP] Refinement loop finished after %d iterations. Complete: %s. Final nodes: %d.",
                 final_state.get("iteration", 0),
                 final_state.get("is_complete", False),
                 len(final_state["nodes"]))

    # Post-refinement: merge overly granular sequential instruction chains
    nodes, merge_count = merge_sequential_instructions(final_state["nodes"])
    if merge_count:
        logger.info("[LOOP] Merged %d sequential instruction pairs post-refinement.", merge_count)
        final_state["nodes"] = nodes
        if dump_path:
            (dump_path / "merged_graph.json").write_text(
                json.dumps(nodes, indent=2)
            )

    return final_state
