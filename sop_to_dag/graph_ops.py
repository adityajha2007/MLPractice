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

from sop_to_dag.models import get_model
from sop_to_dag.schemas import (
    GranularityFeedback,
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

## Graph Adjacency Map
{adjacency_map}

## Current Nodes (JSON)
{nodes_json}

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

## Graph Adjacency Map
{adjacency_map}

## Current Nodes (JSON)
{nodes_json}

Evaluate logical adjacency of connected nodes. Return a ContextFeedback with
is_valid and issues.
"""

_GRANULARITY_SYSTEM = """\
You are a Process Granularity Auditor. Your job is to compare each non-terminal
node in a workflow graph against the original SOP and identify nodes that are
TOO COARSE — i.e., a single node that collapses multiple distinct user actions
into one step.

A node is "coarse" when:
- The corresponding SOP text describes 2 or more SEPARATE actions that a user
  must perform sequentially (e.g., "Open the system, navigate to the tab, and
  enter the data" is 3 actions crammed into one node).
- The node text uses conjunctions like "and", "then", or semicolons to join
  multiple operations.
- The SOP section for this step contains sub-bullets, numbered sub-steps, or
  multiple imperative verbs describing distinct operations.

A node is NOT coarse when:
- It describes a single atomic action (even if the text is long for clarity).
- It is a decision/question node asking one question.
- It is a terminal or reference node.

For each coarse node, estimate how many sub-steps it should be split into.
Be conservative — only flag nodes where the SOP clearly describes multiple
distinct actions. Do not flag nodes just because they could theoretically be
more detailed.
"""

_GRANULARITY_HUMAN = """\
## Original SOP
{source_text}

## Current Nodes (JSON)
{nodes_json}

Identify coarse nodes. Return GranularityFeedback with is_granular and
coarse_nodes (each with node_id, reason, suggested_split).
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
1. The full current graph as an adjacency map and full JSON
2. The original SOP text
3. A detailed list of issues found by the analyser (missing branches, \
invalid triplets, coarse nodes, topological problems, context issues)

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

## Current Graph (Adjacency Map)
{adjacency_map}

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
    adjacency_map = generate_adjacency_map(nodes)
    nodes_json = json.dumps(nodes, indent=2)

    llm = get_model("completeness")
    structured_llm = llm.with_structured_output(RefineFeedback)
    messages = [
        SystemMessage(content=_COMPLETENESS_SYSTEM),
        HumanMessage(
            content=_COMPLETENESS_HUMAN.format(
                source_text=source_text,
                adjacency_map=adjacency_map,
                nodes_json=nodes_json,
            )
        ),
    ]
    return structured_llm.invoke(messages)


def check_context(nodes: dict, source_text: str) -> Dict[str, Any]:
    """LLM check: logical adjacency of connected nodes."""
    adjacency_map = generate_adjacency_map(nodes)
    nodes_json = json.dumps(nodes, indent=2)

    llm = get_model("context")
    structured_llm = llm.with_structured_output(ContextFeedback)
    messages = [
        SystemMessage(content=_CONTEXT_SYSTEM),
        HumanMessage(
            content=_CONTEXT_HUMAN.format(
                source_text=source_text,
                adjacency_map=adjacency_map,
                nodes_json=nodes_json,
            )
        ),
    ]

    try:
        result = structured_llm.invoke(messages)
        return {"is_valid": result.is_valid, "issues": result.issues}
    except Exception as e:
        logger.warning("Context check structured output failed: %s", e)
        return {"is_valid": False, "issues": [f"Context check failed: {e}"]}


def check_granularity(nodes: dict, source_text: str) -> GranularityFeedback:
    """LLM check: are graph nodes granular enough vs the source SOP?"""
    nodes_json = json.dumps(nodes, indent=2)

    llm = get_model("completeness")  # same temp=0.0 as other analyser checks
    structured_llm = llm.with_structured_output(GranularityFeedback)
    messages = [
        SystemMessage(content=_GRANULARITY_SYSTEM),
        HumanMessage(
            content=_GRANULARITY_HUMAN.format(
                source_text=source_text,
                nodes_json=nodes_json,
            )
        ),
    ]

    try:
        return structured_llm.invoke(messages)
    except Exception as e:
        logger.warning("Granularity check failed: %s", e)
        return GranularityFeedback(
            is_granular=False,
            coarse_nodes=[],
        )


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
    logger.info("  [ANALYSER] Check 1/4: Topological (pure Python)...")
    topo_report = get_graph_issues(nodes)
    report_parts.append(f"## Topological Check\n{topo_report}")
    has_topo_issues = topo_report != "Topology Valid."
    if has_topo_issues:
        logger.warning("    Topology issues found: %s", topo_report)
    else:
        logger.info("    Topology: VALID")

    # 2. Completeness check (LLM-based)
    logger.info("  [ANALYSER] Check 2/4: Completeness (LLM)...")
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
    logger.info("  [ANALYSER] Check 3/4: Context adjacency (LLM)...")
    context_result = check_context(nodes, source_text)
    report_parts.append(
        f"## Context Check\n"
        f"Valid: {context_result['is_valid']}\n"
        f"Issues: {context_result['issues']}"
    )
    logger.info("    Valid: %s", context_result["is_valid"])
    if context_result["issues"]:
        logger.info("    Context issues: %s", context_result["issues"])

    # 4. Granularity check (LLM-based)
    logger.info("  [ANALYSER] Check 4/4: Granularity (LLM)...")
    granularity = check_granularity(nodes, source_text)
    report_parts.append(
        f"## Granularity Check\n"
        f"Granular: {granularity.is_granular}\n"
        f"Coarse nodes: {[c.node_id for c in granularity.coarse_nodes]}"
    )
    logger.info("    Granular: %s", granularity.is_granular)
    if granularity.coarse_nodes:
        for c in granularity.coarse_nodes:
            logger.info("    COARSE: %s (split→%d): %s", c.node_id, c.suggested_split, c.reason)

    # Aggregate: graph is complete only if ALL checks pass
    is_complete = (
        not has_topo_issues
        and completeness.is_complete
        and context_result["is_valid"]
        and granularity.is_granular
    )

    # Build categorized feedback for the refiner (one patch per category)
    categorized_feedback: Dict[str, str] = {}
    if has_topo_issues:
        categorized_feedback["topological"] = (
            f"Fix these topological issues: {topo_report}"
        )
    if not completeness.is_complete:
        categorized_feedback["completeness"] = (
            f"Add the following missing branches/steps to the graph: "
            f"{completeness.missing_branches}"
        )
    if not context_result["is_valid"]:
        categorized_feedback["context"] = (
            f"Fix these logical adjacency issues between connected nodes: "
            f"{context_result['issues']}"
        )
    if not granularity.is_granular:
        coarse_desc = "; ".join(
            f"'{c.node_id}' ({c.reason}, split into ~{c.suggested_split} steps)"
            for c in granularity.coarse_nodes
        )
        categorized_feedback["granularity"] = (
            f"Expand these coarse nodes into detailed sub-steps: {coarse_desc}"
        )

    # Also store the flat string for backward compat / logging
    flat_feedback = "\n".join(categorized_feedback.values())

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
        try:
            verification = structured_llm.invoke(messages)
            for r in verification.results:
                if not r.is_valid and r.triplet_index < len(batch):
                    invalid.append(
                        {
                            **batch[r.triplet_index],
                            "explanation": r.explanation,
                        }
                    )
        except Exception as e:
            logger.warning("Triplet verification failed: %s", e)

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

        adjacency_map = generate_adjacency_map(nodes)
        nodes_json = json.dumps(nodes, indent=2)

        structured_llm = self.llm.with_structured_output(GraphPatch)
        messages = [
            SystemMessage(content=_PATCH_RESOLVER_SYSTEM),
            HumanMessage(
                content=_PATCH_RESOLVER_HUMAN.format(
                    source_text=source_text,
                    adjacency_map=adjacency_map,
                    nodes_json=nodes_json,
                    feedback=feedback,
                )
            ),
        ]

        try:
            patch = structured_llm.invoke(messages)
        except Exception as e:
            logger.warning("GraphPatchResolver structured output failed: %s", e)
            try:
                patch = structured_llm.invoke(messages)
            except Exception as e2:
                logger.warning("GraphPatchResolver retry also failed: %s", e2)
                empty_patch = GraphPatch(reasoning=f"Resolution failed: {e2}")
                return nodes, empty_patch

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
    """Run triplet verification -> per-category graph patching -> schema validation.

    Each issue category (topological, completeness, context, granularity, triplets)
    gets its own GraphPatch call with a narrow mandate and independent rollback.
    This prevents a single "fix everything" patch from causing collateral damage.

    Mutates and returns the state with updated nodes and incremented iteration.
    """
    iteration = state.get("iteration", 0) + 1
    logger.info("[REFINER iter %d] Starting refinement cycle...", iteration)

    nodes = state["nodes"]
    source_text = state["source_text"]
    categorized_feedback = dict(state.get("categorized_feedback", {}))

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
        triplet_feedback = "\n".join(
            f"Invalid triplet: {t['source_id']} --({t['edge_label']})--> "
            f"{t['target_id']}: {t['explanation']}"
            for t in invalid_triplets
        )
        categorized_feedback["triplets"] = (
            f"Fix these invalid edge triplets:\n{triplet_feedback}"
        )

    # 2. Per-category graph patching
    category_order = ["topological", "completeness", "context", "granularity", "triplets"]
    applied_categories = []

    for category in category_order:
        cat_feedback = categorized_feedback.get(category)
        if not cat_feedback:
            continue

        logger.info(
            "  [REFINER iter %d] Patching category '%s'...", iteration, category
        )
        nodes_before = len(nodes)
        nodes, patch = patch_resolver.resolve(nodes, cat_feedback, source_text)
        nodes_after = len(nodes)

        changes = (
            len(patch.add_nodes) + len(patch.modify_nodes) + len(patch.remove_nodes)
        )
        logger.info(
            "    [%s] Nodes: %d -> %d (delta: %+d, changes: %d). Reasoning: %s",
            category, nodes_before, nodes_after, nodes_after - nodes_before,
            changes, (patch.reasoning[:150] if patch.reasoning else "N/A"),
        )
        applied_categories.append(category)

    if not applied_categories:
        logger.info("  [REFINER iter %d] No categories to patch — skipping", iteration)

    # 3. Schema validation (deterministic fixes)
    logger.info("  [REFINER iter %d] Schema validation...", iteration)
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
        "  [REFINER iter %d] Complete. %d nodes in graph. Categories patched: %s",
        iteration, len(nodes), applied_categories or "none",
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
    return final_state
