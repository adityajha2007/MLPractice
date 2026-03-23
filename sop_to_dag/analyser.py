"""Analyser: topological (code) + completeness (LLM) + context (LLM) checks.

All three checks in one file. Inline prompts. Cheap checks run first.
"""

import json
import logging
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from sop_to_dag.models import get_model
from sop_to_dag.schemas import GraphState, RefineFeedback

logger = logging.getLogger(__name__)


class ContextFeedback(BaseModel):
    """Structured output for the context adjacency check."""

    is_valid: bool = Field(description="Whether all edges are logically valid.")
    issues: List[str] = Field(
        default_factory=list,
        description="List of logical flow problems found.",
    )

# ---------------------------------------------------------------------------
# Inline prompts
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Topological checks (pure Python, no LLM)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# LLM-based checks
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def analyse(state: GraphState) -> GraphState:
    """Run topological -> completeness -> context checks.

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

    # Build feedback string for the refiner
    feedback_parts = []
    if has_topo_issues:
        feedback_parts.append(f"Topological issues: {topo_report}")
    if not completeness.is_complete:
        feedback_parts.append(
            f"Missing branches: {completeness.missing_branches}"
        )
    if not context_result["is_valid"]:
        feedback_parts.append(
            f"Context issues: {context_result['issues']}"
        )

    state["is_complete"] = is_complete
    state["feedback"] = "\n".join(feedback_parts) if feedback_parts else ""
    state["analysis_report"] = "\n\n".join(report_parts)

    verdict = "PASS — graph is complete" if is_complete else "FAIL — needs refinement"
    logger.info("  [ANALYSER] Verdict: %s", verdict)

    return state
