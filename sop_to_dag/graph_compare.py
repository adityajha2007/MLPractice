"""Standalone graph comparison: auto-generated vs human-created vs SOP document.

Three-way comparison with semantic node alignment, structural edge comparison,
SOP grounding checks, and numerical metrics. Fully self-contained — no imports
from sop_to_dag.

Usage:
    python -m sop_to_dag.graph_compare \
        --auto output/auto_graph.json \
        --human output/human_graph.json \
        --sop input/sop.md \
        --output output/comparison_report.json
"""

import argparse
import json
import logging
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
LLM_MODEL_NAME = "gpt-oss-120b"
SOP_CHUNK_SIZE = 500  # characters per SOP chunk for FAISS indexing
TOP_K_CANDIDATES = 10  # embedding candidates per human node for LLM verification
LLM_BATCH_SIZE = 10  # human nodes per LLM matching call

# Type mapping: human format → normalized
_HUMAN_TYPE_MAP = {
    "activity": "instruction",
    "decision": "question",
    "reference": "reference",
    "terminal": "terminal",
    "start": "instruction",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AlignmentResult:
    """Result of semantic node alignment between two graphs."""

    human_to_auto: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    auto_to_human: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    covered_human: Set[str] = field(default_factory=set)
    grounded_auto: Set[str] = field(default_factory=set)
    unmatched_human: Set[str] = field(default_factory=set)
    unmatched_auto: Set[str] = field(default_factory=set)
    # Similarity matrix: auto_ids (rows) x human_ids (cols)
    sim_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    auto_ids: List[str] = field(default_factory=list)
    human_ids: List[str] = field(default_factory=list)


@dataclass
class EdgeComparisonResult:
    """Result of edge structure comparison."""

    matched_edges: int = 0
    total_auto_edges: int = 0
    total_human_edges: int = 0
    mismatches: List[Dict[str, Any]] = field(default_factory=list)
    type_matches: int = 0
    type_total: int = 0


# ---------------------------------------------------------------------------
# Step 0: Normalize graphs to common format
# ---------------------------------------------------------------------------


def _normalize_human_graph(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Convert human graph format to normalized internal format.

    Human format: {id: {type, action, next (str or dict)}}
    Output:       {id: {type, text, next, options}}
    """
    nodes = {}
    for node_id, data in raw.items():
        human_type = data.get("type", "activity")
        norm_type = _HUMAN_TYPE_MAP.get(human_type, "instruction")

        text = data.get("action", data.get("text", ""))
        raw_next = data.get("next")

        next_id = None
        options = None

        if isinstance(raw_next, dict):
            # Check if it's a single-next wrapper like {"next": "some_id"}
            if list(raw_next.keys()) == ["next"]:
                next_id = raw_next["next"]
            else:
                # Decision node with labeled branches: {"yes": "id", "no": "id"}
                options = dict(raw_next)
        elif isinstance(raw_next, str):
            next_id = raw_next
        elif isinstance(raw_next, list):
            if len(raw_next) == 1:
                next_id = raw_next[0]
            elif len(raw_next) > 1:
                options = {f"option_{i}": v for i, v in enumerate(raw_next)}

        nodes[node_id] = {
            "id": node_id,
            "type": norm_type,
            "text": text,
            "next": next_id,
            "options": options,
        }
    return nodes


def _normalize_auto_graph(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Strip auto graph to common fields."""
    nodes = {}
    for node_id, data in raw.items():
        nodes[node_id] = {
            "id": node_id,
            "type": data.get("type", "instruction"),
            "text": data.get("text", ""),
            "next": data.get("next"),
            "options": data.get("options"),
        }
    return nodes


# ---------------------------------------------------------------------------
# Embeddings helper
# ---------------------------------------------------------------------------


def _get_embeddings_model():
    """Return a HuggingFace embeddings model (local, no API cost)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _embed_texts(model, texts: List[str]) -> np.ndarray:
    """Embed a list of texts and return as numpy matrix."""
    embeddings = model.embed_documents(texts)
    return np.array(embeddings, dtype=np.float32)


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between two embedding matrices.

    Both should be L2-normalized (bge-base outputs are).
    Returns matrix of shape (len(a), len(b)).
    """
    # With normalized vectors, cosine similarity = dot product
    return a @ b.T


# ---------------------------------------------------------------------------
# Step 1: Hybrid node alignment (embeddings + LLM)
# ---------------------------------------------------------------------------

_MATCH_SYSTEM = """\
You are a Workflow Graph Alignment Specialist. You match nodes between two \
workflow graphs that represent the same SOP process.

You receive a batch of HUMAN nodes, each with up to {top_k} AUTO node candidates \
(pre-filtered by text similarity). Your job is to decide which auto nodes \
genuinely represent the same step as each human node.

MATCHING RULES:
1. A human node can match MULTIPLE auto nodes (the auto graph may break one \
   human step into several granular sub-steps).
2. An auto node can match at most ONE human node.
3. Match based on MEANING, not wording. "Enter the dispute code" and \
   "Input the code in the system" are the same step.
4. Do NOT match nodes that describe different actions, even if they share \
   some keywords. "Check fraud score" and "Check dispute code" are different.
5. If none of the candidates match, return an empty list for that human node.
6. Consider the node type — a decision/question node is unlikely to match \
   an activity/instruction node unless the text clearly describes the same step.

Return a JSON object mapping each human node ID to a list of matched auto node IDs.
Example: {{"human_node_1": ["auto_a", "auto_b"], "human_node_2": [], "human_node_3": ["auto_c"]}}
"""

_MATCH_HUMAN = """\
## Human Nodes with Auto Candidates

{batch_text}

Return a JSON object mapping each human_id to a list of matched auto_ids. \
Use only IDs from the candidates provided. Return empty list [] if no match.
"""


def _get_llm():
    """Return a ChatOpenAI instance for matching."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.0)


def _build_candidate_text(
    human_batch: List[Tuple[str, Dict]],
    candidates: Dict[str, List[Tuple[str, str, float]]],
) -> str:
    """Build the text payload for one LLM matching batch."""
    parts = []
    for h_id, h_data in human_batch:
        cands = candidates.get(h_id, [])
        part = f"### Human: `{h_id}` ({h_data['type']})\n> {h_data['text']}\n\nCandidates:\n"
        for a_id, a_text, sim in cands:
            part += f"  - `{a_id}`: {a_text} (sim={sim:.2f})\n"
        if not cands:
            part += "  (no candidates above threshold)\n"
        parts.append(part)
    return "\n".join(parts)


def _llm_match_batch(
    llm,
    human_batch: List[Tuple[str, Dict]],
    candidates: Dict[str, List[Tuple[str, str, float]]],
) -> Dict[str, List[str]]:
    """Send one batch to the LLM and parse the matching result."""
    from langchain_core.messages import HumanMessage, SystemMessage

    batch_text = _build_candidate_text(human_batch, candidates)
    messages = [
        SystemMessage(content=_MATCH_SYSTEM.format(top_k=TOP_K_CANDIDATES)),
        HumanMessage(content=_MATCH_HUMAN.format(batch_text=batch_text)),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        # Extract JSON from response (may be wrapped in ```json ... ```)
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        result = json.loads(content)
        if isinstance(result, dict):
            return {k: v for k, v in result.items() if isinstance(v, list)}
    except Exception as e:
        logger.warning("LLM matching failed for batch: %s — falling back to embedding-only", e)

    # Fallback: use top-1 embedding candidate for each human node
    fallback = {}
    for h_id, _ in human_batch:
        cands = candidates.get(h_id, [])
        fallback[h_id] = [cands[0][0]] if cands else []
    return fallback


def _align_nodes(
    auto_nodes: Dict[str, Dict],
    human_nodes: Dict[str, Dict],
) -> AlignmentResult:
    """Hybrid alignment: embeddings for top-K candidates, LLM for final matching.

    1. Embed all node texts → cosine similarity matrix
    2. For each human node: pick top-K auto candidates by similarity
    3. Send batches to LLM for semantic matching
    4. Build final alignment from LLM decisions
    """
    if not auto_nodes or not human_nodes:
        return AlignmentResult(
            unmatched_human=set(human_nodes.keys()),
            unmatched_auto=set(auto_nodes.keys()),
        )

    emb_model = _get_embeddings_model()

    auto_ids = list(auto_nodes.keys())
    human_ids = list(human_nodes.keys())
    auto_texts = [auto_nodes[nid]["text"] for nid in auto_ids]
    human_texts = [human_nodes[nid]["text"] for nid in human_ids]

    # Phase 1: Embedding-based candidate selection
    logger.info("Phase 1: Embedding %d auto + %d human nodes...", len(auto_ids), len(human_ids))
    auto_emb = _embed_texts(emb_model, auto_texts)
    human_emb = _embed_texts(emb_model, human_texts)
    sim_matrix = _cosine_similarity_matrix(auto_emb, human_emb)

    # For each human node, get top-K auto candidates
    candidates: Dict[str, List[Tuple[str, str, float]]] = {}
    for j, h_id in enumerate(human_ids):
        sims = [(auto_ids[i], auto_nodes[auto_ids[i]]["text"], float(sim_matrix[i][j]))
                for i in range(len(auto_ids))]
        sims.sort(key=lambda x: x[2], reverse=True)
        candidates[h_id] = sims[:TOP_K_CANDIDATES]

    logger.info("Phase 1 complete: top-%d candidates per human node", TOP_K_CANDIDATES)

    # Phase 2: LLM-based matching in batches
    logger.info("Phase 2: LLM matching in batches of %d...", LLM_BATCH_SIZE)
    llm = _get_llm()

    human_items = [(h_id, human_nodes[h_id]) for h_id in human_ids]
    all_matches: Dict[str, List[str]] = {}

    for batch_start in range(0, len(human_items), LLM_BATCH_SIZE):
        batch = human_items[batch_start:batch_start + LLM_BATCH_SIZE]
        batch_num = batch_start // LLM_BATCH_SIZE + 1
        total_batches = (len(human_items) + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE
        logger.info("  Batch %d/%d (%d human nodes)...", batch_num, total_batches, len(batch))

        batch_result = _llm_match_batch(llm, batch, candidates)
        all_matches.update(batch_result)

    # Phase 3: Build AlignmentResult from LLM matches
    logger.info("Phase 3: Building alignment from LLM results...")
    result = AlignmentResult(
        sim_matrix=sim_matrix, auto_ids=auto_ids, human_ids=human_ids,
    )

    # Get embedding similarity for matched pairs (for reporting)
    auto_idx = {a_id: i for i, a_id in enumerate(auto_ids)}
    human_idx = {h_id: j for j, h_id in enumerate(human_ids)}

    for h_id in human_ids:
        matched_auto_ids = all_matches.get(h_id, [])
        # Filter to valid auto IDs
        matched_auto_ids = [a_id for a_id in matched_auto_ids if a_id in auto_nodes]
        if matched_auto_ids:
            pairs = []
            for a_id in matched_auto_ids:
                sim = float(sim_matrix[auto_idx[a_id]][human_idx[h_id]])
                pairs.append((a_id, sim))
            pairs.sort(key=lambda x: x[1], reverse=True)
            result.human_to_auto[h_id] = pairs
            result.covered_human.add(h_id)
        else:
            result.unmatched_human.add(h_id)

    # Build auto_to_human (best human match for each auto node)
    for a_id in auto_ids:
        best_h_id = None
        best_sim = -1.0
        for h_id, pairs in result.human_to_auto.items():
            for matched_a_id, sim in pairs:
                if matched_a_id == a_id and sim > best_sim:
                    best_h_id = h_id
                    best_sim = sim
        if best_h_id is not None:
            result.auto_to_human[a_id] = (best_h_id, best_sim)
            result.grounded_auto.add(a_id)
        else:
            result.unmatched_auto.add(a_id)

    logger.info(
        "Alignment: %d/%d human covered, %d/%d auto grounded, "
        "%d human unmatched, %d auto unmatched",
        len(result.covered_human), len(human_ids),
        len(result.grounded_auto), len(auto_ids),
        len(result.unmatched_human), len(result.unmatched_auto),
    )

    return result


# ---------------------------------------------------------------------------
# Step 2: Structural edge comparison
# ---------------------------------------------------------------------------


def _get_edges(nodes: Dict[str, Dict]) -> Set[Tuple[str, str]]:
    """Extract all directed edges from a graph as (source, target) tuples."""
    edges = set()
    for nid, data in nodes.items():
        if data.get("next"):
            edges.add((nid, data["next"]))
        if data.get("options"):
            for target in data["options"].values():
                edges.add((nid, target))
    return edges


EDGE_LLM_BATCH_SIZE = 12

_EDGE_MATCH_SYSTEM = """\
You are a Workflow Graph Edge Analyst. You compare connections (edges) between \
two workflow graphs that represent the same SOP process.

You receive a batch of human graph edges that were NOT matched by structural \
analysis. For each edge, you also see the surrounding subgraph context from \
both the human graph and the auto graph.

Your job: decide whether the auto graph **semantically preserves** each human \
edge's flow, even if the structure differs.

An edge is PRESERVED if:
1. The auto graph has a path (possibly through intermediate steps) that \
   connects equivalent steps in the same logical order.
2. The branching logic is consistent — if the human edge is a "Yes" branch \
   from a decision, the auto graph should also take the affirmative path.
3. The flow intent is the same even if intermediate steps are added or removed.

An edge is NOT PRESERVED if:
1. The auto graph has no path connecting the equivalent steps.
2. The flow goes through a DIFFERENT decision branch (e.g., human takes "Yes" \
   but auto takes "No" path to reach the target).
3. One or both endpoints have no equivalent auto node at all.

Return a JSON list:
[{{"human_edge": "src → tgt", "preserved": true/false, "reason": "..."}}]
"""

_EDGE_MATCH_HUMAN = """\
## Unmatched Human Edges

{edges_text}

## Auto Graph Context (relevant subgraph)
{auto_context}

For each human edge, determine if the auto graph semantically preserves this \
connection. Return preserved (true/false) and a one-sentence reason.
"""


def _get_node_neighborhood(
    nodes: Dict[str, Dict], node_id: str, hops: int = 2,
) -> Dict[str, Dict]:
    """Get a node and its neighborhood (predecessors + successors) within N hops."""
    neighborhood: Dict[str, Dict] = {}
    if node_id not in nodes:
        return neighborhood

    # Forward traversal
    frontier = [node_id]
    visited = {node_id}
    for _ in range(hops):
        next_frontier = []
        for nid in frontier:
            data = nodes.get(nid, {})
            neighborhood[nid] = data
            successors = []
            if data.get("next"):
                successors.append(data["next"])
            if data.get("options"):
                successors.extend(data["options"].values())
            for s in successors:
                if s not in visited and s in nodes:
                    visited.add(s)
                    next_frontier.append(s)
        frontier = next_frontier

    # Backward traversal (find predecessors)
    predecessors: Dict[str, List[str]] = {}
    for nid, data in nodes.items():
        if data.get("next"):
            predecessors.setdefault(data["next"], []).append(nid)
        if data.get("options"):
            for tgt in data["options"].values():
                predecessors.setdefault(tgt, []).append(nid)

    frontier = [node_id]
    visited_back = {node_id}
    for _ in range(hops):
        next_frontier = []
        for nid in frontier:
            for pred in predecessors.get(nid, []):
                if pred not in visited_back:
                    visited_back.add(pred)
                    neighborhood[pred] = nodes.get(pred, {})
                    next_frontier.append(pred)
        frontier = next_frontier

    return neighborhood


def _format_subgraph(nodes: Dict[str, Dict]) -> str:
    """Format a subgraph for LLM context."""
    parts = []
    for nid, data in sorted(nodes.items()):
        line = f"  `{nid}` ({data.get('type', '?')}): \"{data.get('text', '')[:80]}\""
        if data.get("next"):
            line += f" → `{data['next']}`"
        if data.get("options"):
            opts = ", ".join(f"{k}: `{v}`" for k, v in data["options"].items())
            line += f" → {{{opts}}}"
        parts.append(line)
    return "\n".join(parts)


def _llm_validate_edges(
    llm,
    unmatched_edges: List[Dict],
    auto_nodes: Dict[str, Dict],
    human_nodes: Dict[str, Dict],
    human_to_auto_set: Dict[str, Set[str]],
) -> Dict[str, bool]:
    """LLM validation for edges that failed deterministic matching.

    Returns {edge_key: preserved} for each validated edge.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    results: Dict[str, Dict[str, Any]] = {}

    for batch_start in range(0, len(unmatched_edges), EDGE_LLM_BATCH_SIZE):
        batch = unmatched_edges[batch_start:batch_start + EDGE_LLM_BATCH_SIZE]
        batch_num = batch_start // EDGE_LLM_BATCH_SIZE + 1

        # Build edge descriptions + collect relevant auto subgraph
        edge_parts = []
        auto_neighborhood: Dict[str, Dict] = {}
        for edge_info in batch:
            h_src = edge_info["_h_src"]
            h_tgt = edge_info["_h_tgt"]
            h_src_text = human_nodes.get(h_src, {}).get("text", "?")[:80]
            h_tgt_text = human_nodes.get(h_tgt, {}).get("text", "?")[:80]

            # Show human edge with context
            part = f"- **`{h_src}` → `{h_tgt}`**\n"
            part += f"  Source: \"{h_src_text}\"\n"
            part += f"  Target: \"{h_tgt_text}\"\n"

            # Show which auto nodes are aligned to src/tgt
            a_srcs = human_to_auto_set.get(h_src, set())
            a_tgts = human_to_auto_set.get(h_tgt, set())
            if a_srcs:
                part += f"  Aligned auto sources: {', '.join(f'`{a}`' for a in a_srcs)}\n"
            else:
                part += "  Aligned auto sources: (none — source has no auto match)\n"
            if a_tgts:
                part += f"  Aligned auto targets: {', '.join(f'`{a}`' for a in a_tgts)}\n"
            else:
                part += "  Aligned auto targets: (none — target has no auto match)\n"

            edge_parts.append(part)

            # Collect neighborhoods of aligned auto nodes for context
            for a_id in a_srcs | a_tgts:
                auto_neighborhood.update(_get_node_neighborhood(auto_nodes, a_id, hops=2))

        edges_text = "\n".join(edge_parts)
        auto_context = _format_subgraph(auto_neighborhood) if auto_neighborhood else "(no aligned auto nodes)"

        messages = [
            SystemMessage(content=_EDGE_MATCH_SYSTEM),
            HumanMessage(content=_EDGE_MATCH_HUMAN.format(
                edges_text=edges_text, auto_context=auto_context,
            )),
        ]

        try:
            response = llm.invoke(messages)
            content = response.content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            verdicts = json.loads(content)

            for v in verdicts:
                edge_key = v.get("human_edge", "")
                results[edge_key] = {
                    "preserved": bool(v.get("preserved", False)),
                    "reason": v.get("reason", ""),
                }
            logger.info("  Edge batch %d: %d/%d preserved by LLM",
                       batch_num,
                       sum(1 for v in verdicts if v.get("preserved")),
                       len(batch))
        except Exception as e:
            logger.warning("Edge LLM validation failed for batch %d: %s", batch_num, e)
            for edge_info in batch:
                edge_key = edge_info["human_edge"]
                if edge_key not in results:
                    results[edge_key] = {
                        "preserved": False,
                        "reason": f"LLM validation failed: {e}",
                    }

    return results


def _compare_edges(
    auto_nodes: Dict[str, Dict],
    human_nodes: Dict[str, Dict],
    alignment: AlignmentResult,
) -> EdgeComparisonResult:
    """Hybrid edge comparison: deterministic path-check + LLM validation.

    Phase 1 (deterministic): Check if a path exists in auto graph between
    aligned nodes (up to 3 hops). Free, catches obvious matches.

    Phase 2 (LLM): For edges that fail Phase 1, send to LLM with subgraph
    context. LLM checks semantic flow preservation, branching logic, etc.
    """
    auto_edges = _get_edges(auto_nodes)
    human_edges = _get_edges(human_nodes)

    # Build reverse lookup: auto_id → set of human_ids it aligns to
    auto_to_human_set: Dict[str, Set[str]] = {}
    for a_id, (h_id, _) in alignment.auto_to_human.items():
        auto_to_human_set.setdefault(a_id, set()).add(h_id)

    # Build reverse: human_id → set of auto_ids
    human_to_auto_set: Dict[str, Set[str]] = {}
    for h_id, matches in alignment.human_to_auto.items():
        for a_id, _ in matches:
            human_to_auto_set.setdefault(h_id, set()).add(a_id)

    # Phase 1: Deterministic path check
    det_matched = 0
    det_mismatches = []

    for h_src, h_tgt in human_edges:
        a_srcs = human_to_auto_set.get(h_src, set())
        a_tgts = human_to_auto_set.get(h_tgt, set())

        found = False
        for a_src in a_srcs:
            for a_tgt in a_tgts:
                if (a_src, a_tgt) in auto_edges:
                    found = True
                    break
                if _has_path(auto_nodes, a_src, a_tgt, max_hops=3):
                    found = True
                    break
            if found:
                break

        if found:
            det_matched += 1
        else:
            det_mismatches.append({
                "human_edge": f"{h_src} → {h_tgt}",
                "human_src_text": human_nodes.get(h_src, {}).get("text", "?")[:80],
                "human_tgt_text": human_nodes.get(h_tgt, {}).get("text", "?")[:80],
                "auto_candidates_src": list(a_srcs)[:3],
                "auto_candidates_tgt": list(a_tgts)[:3],
                "_h_src": h_src,
                "_h_tgt": h_tgt,
            })

    logger.info("  Phase 1 (deterministic): %d/%d human edges matched, %d unmatched",
                det_matched, len(human_edges), len(det_mismatches))

    # Phase 2: LLM validation — only for DECISION edges (question/decision nodes)
    # Sequential instruction→instruction edges are trivial; decision branches are
    # where the real logic lives and where structural differences matter.
    decision_types = {"question", "decision"}
    decision_mismatches = [
        e for e in det_mismatches
        if human_nodes.get(e["_h_src"], {}).get("type") in decision_types
    ]
    non_decision_mismatches = [
        e for e in det_mismatches
        if human_nodes.get(e["_h_src"], {}).get("type") not in decision_types
    ]

    logger.info("  Phase 2: %d decision edges to validate by LLM (%d non-decision skipped)",
                len(decision_mismatches), len(non_decision_mismatches))

    llm_recovered = 0
    final_mismatches = []

    # Non-decision edges go straight to mismatches (no LLM needed)
    for edge_info in non_decision_mismatches:
        clean = {k: v for k, v in edge_info.items() if not k.startswith("_")}
        clean["llm_reason"] = "Skipped — non-decision edge"
        final_mismatches.append(clean)

    if decision_mismatches:
        llm = _get_llm()
        llm_results = _llm_validate_edges(
            llm, decision_mismatches, auto_nodes, human_nodes, human_to_auto_set,
        )

        for edge_info in decision_mismatches:
            edge_key = edge_info["human_edge"]
            llm_verdict = llm_results.get(edge_key, {})

            if llm_verdict.get("preserved", False):
                llm_recovered += 1
            else:
                clean = {k: v for k, v in edge_info.items() if not k.startswith("_")}
                clean["llm_reason"] = llm_verdict.get("reason", "Not evaluated")
                final_mismatches.append(clean)

        logger.info("  Phase 2 (LLM): recovered %d/%d decision edges, %d truly unmatched",
                    llm_recovered, len(decision_mismatches),
                    len(final_mismatches) - len(non_decision_mismatches))

    total_matched = det_matched + llm_recovered

    # Type comparison for grounded auto nodes
    type_matches = 0
    type_total = 0
    for a_id, (h_id, _) in alignment.auto_to_human.items():
        a_type = auto_nodes[a_id]["type"]
        h_type = human_nodes[h_id]["type"]
        type_total += 1
        if a_type == h_type:
            type_matches += 1

    return EdgeComparisonResult(
        matched_edges=total_matched,
        total_auto_edges=len(auto_edges),
        total_human_edges=len(human_edges),
        mismatches=final_mismatches,
        type_matches=type_matches,
        type_total=type_total,
    )


def _has_path(
    nodes: Dict[str, Dict], src: str, tgt: str, max_hops: int = 3
) -> bool:
    """Check if there's a directed path from src to tgt within max_hops."""
    if src == tgt:
        return True
    visited = {src}
    frontier = [src]
    for _ in range(max_hops):
        next_frontier = []
        for nid in frontier:
            data = nodes.get(nid, {})
            successors = []
            if data.get("next"):
                successors.append(data["next"])
            if data.get("options"):
                successors.extend(data["options"].values())
            for s in successors:
                if s == tgt:
                    return True
                if s not in visited and s in nodes:
                    visited.add(s)
                    next_frontier.append(s)
        frontier = next_frontier
        if not frontier:
            break
    return False


# ---------------------------------------------------------------------------
# Step 3: SOP grounding check
# ---------------------------------------------------------------------------


def _chunk_sop(sop_text: str, chunk_size: int = SOP_CHUNK_SIZE) -> List[str]:
    """Split SOP text into overlapping chunks for FAISS indexing."""
    words = sop_text.split()
    chunks = []
    # Approximate chunk_size in characters → split by words
    current: List[str] = []
    current_len = 0
    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            # Keep last 20% as overlap
            overlap_start = max(0, len(current) - len(current) // 5)
            current = current[overlap_start:]
            current_len = sum(len(w) + 1 for w in current)
    if current:
        chunks.append(" ".join(current))
    return chunks


_GROUNDING_SYSTEM = """\
You are an SOP Verification Specialist. You determine whether workflow graph \
nodes are grounded in (i.e., explicitly stated or directly implied by) the \
original SOP document.

You receive a batch of graph nodes and the most relevant SOP excerpts for each.

For EACH node, decide:
- "grounded": The SOP explicitly describes this step, action, or decision.
- "not_grounded": The SOP does NOT mention this. It may be inferred, \
  hallucinated, or added from domain knowledge — but it's not in the document.

RULES:
1. Be strict. "Grounded" means the SOP TEXT says it, not that it makes sense.
2. A node is grounded even if the wording differs, as long as the SOP \
   describes the same action/decision.
3. A node is NOT grounded if it's a reasonable step that the SOP simply \
   doesn't mention.
4. Provide a brief reason for each verdict (1 sentence).

Return a JSON list:
[{{"id": "node_id", "grounded": true/false, "reason": "..."}}]
"""

_GROUNDING_HUMAN = """\
## Graph Nodes to Verify

{nodes_text}

## Source SOP Document
{sop_text}

For each node, return grounded (true/false) and a one-sentence reason.
"""

SOP_GROUNDING_BATCH_SIZE = 15


def _check_sop_grounding(
    unmatched_nodes: Dict[str, Dict[str, Any]],
    sop_text: str,
) -> Dict[str, Dict[str, Any]]:
    """LLM-validated SOP grounding for unmatched nodes.

    Uses embeddings to find relevant SOP excerpts per node, then LLM
    validates whether each node is actually stated in the SOP.

    Returns {node_id: {text, grounded, reason, relevant_sop_excerpt}}.
    """
    if not unmatched_nodes or not sop_text:
        return {
            nid: {"text": data["text"], "grounded": False, "reason": "No SOP provided",
                  "relevant_sop_excerpt": ""}
            for nid, data in unmatched_nodes.items()
        }

    # Phase 1: Embedding retrieval — find top-3 relevant SOP chunks per node
    emb_model = _get_embeddings_model()
    chunks = _chunk_sop(sop_text)

    if not chunks:
        return {
            nid: {"text": data["text"], "grounded": False, "reason": "SOP empty",
                  "relevant_sop_excerpt": ""}
            for nid, data in unmatched_nodes.items()
        }

    logger.info("SOP grounding: %d nodes, retrieving relevant SOP chunks...",
                len(unmatched_nodes))

    chunk_emb = _embed_texts(emb_model, chunks)
    node_ids = list(unmatched_nodes.keys())
    node_texts = [unmatched_nodes[nid]["text"] for nid in node_ids]
    node_emb = _embed_texts(emb_model, node_texts)
    sim_matrix = _cosine_similarity_matrix(node_emb, chunk_emb)

    # Top-3 SOP chunks per node
    node_excerpts: Dict[str, str] = {}
    for i, nid in enumerate(node_ids):
        top_indices = np.argsort(sim_matrix[i])[-3:][::-1]
        top_chunks = [chunks[j] for j in top_indices]
        node_excerpts[nid] = "\n---\n".join(top_chunks)

    # Phase 2: LLM validation in batches
    logger.info("SOP grounding: LLM validation in batches of %d...",
                SOP_GROUNDING_BATCH_SIZE)
    llm = _get_llm()
    results: Dict[str, Dict[str, Any]] = {}

    for batch_start in range(0, len(node_ids), SOP_GROUNDING_BATCH_SIZE):
        batch_ids = node_ids[batch_start:batch_start + SOP_GROUNDING_BATCH_SIZE]
        batch_num = batch_start // SOP_GROUNDING_BATCH_SIZE + 1

        nodes_text_parts = []
        for nid in batch_ids:
            nodes_text_parts.append(
                f"- `{nid}`: \"{unmatched_nodes[nid]['text']}\""
            )
        nodes_text = "\n".join(nodes_text_parts)

        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=_GROUNDING_SYSTEM),
            HumanMessage(content=_GROUNDING_HUMAN.format(
                nodes_text=nodes_text, sop_text=sop_text,
            )),
        ]

        try:
            response = llm.invoke(messages)
            content = response.content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            verdicts = json.loads(content)

            for v in verdicts:
                nid = v.get("id", "")
                if nid in unmatched_nodes:
                    results[nid] = {
                        "text": unmatched_nodes[nid]["text"],
                        "grounded": bool(v.get("grounded", False)),
                        "reason": v.get("reason", ""),
                        "relevant_sop_excerpt": node_excerpts.get(nid, "")[:300],
                    }
            logger.info("  Batch %d: %d/%d nodes grounded",
                       batch_num,
                       sum(1 for nid in batch_ids if results.get(nid, {}).get("grounded")),
                       len(batch_ids))
        except Exception as e:
            logger.warning("SOP grounding LLM failed for batch %d: %s — marking as unknown", batch_num, e)
            for nid in batch_ids:
                if nid not in results:
                    results[nid] = {
                        "text": unmatched_nodes[nid]["text"],
                        "grounded": False,
                        "reason": f"LLM validation failed: {e}",
                        "relevant_sop_excerpt": node_excerpts.get(nid, "")[:300],
                    }

    # Fill any missing
    for nid in node_ids:
        if nid not in results:
            results[nid] = {
                "text": unmatched_nodes[nid]["text"],
                "grounded": False,
                "reason": "Not evaluated",
                "relevant_sop_excerpt": "",
            }

    grounded_count = sum(1 for r in results.values() if r["grounded"])
    logger.info("SOP grounding complete: %d/%d nodes grounded by LLM", grounded_count, len(results))

    return results


# ---------------------------------------------------------------------------
# Step 4: Compute metrics
# ---------------------------------------------------------------------------


def _compute_single_graph_metrics(nodes: Dict[str, Dict]) -> Dict[str, Any]:
    """Compute basic metrics for a single graph."""
    if not nodes:
        return {"node_count": 0, "edge_count": 0}

    edge_count = 0
    type_dist: Dict[str, int] = {}
    for data in nodes.values():
        t = data.get("type", "unknown")
        type_dist[t] = type_dist.get(t, 0) + 1
        if data.get("next"):
            edge_count += 1
        if data.get("options"):
            edge_count += len(data["options"])

    # Topological checks
    defined = set(nodes.keys())
    referenced: Set[str] = set()
    for data in nodes.values():
        if data.get("next"):
            referenced.add(data["next"])
        if data.get("options"):
            referenced.update(data["options"].values())

    orphans = [n for n in (defined - referenced) if n not in ("start", "START", "root")]
    dead_ends = [
        n for n, d in nodes.items()
        if d.get("type") != "terminal" and not d.get("next") and not d.get("options")
    ]
    broken = list(referenced - defined)

    return {
        "node_count": len(nodes),
        "edge_count": edge_count,
        "type_distribution": type_dist,
        "orphan_count": len(orphans),
        "dead_end_count": len(dead_ends),
        "broken_link_count": len(broken),
        "has_start_node": "start" in nodes,
        "terminal_count": type_dist.get("terminal", 0),
    }


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def compute_comparison_metrics(
    auto_nodes: Dict[str, Dict],
    human_nodes: Dict[str, Dict],
    alignment: AlignmentResult,
    edge_result: EdgeComparisonResult,
    auto_grounding: Optional[Dict] = None,
    human_grounding: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Compute all comparison metrics."""

    total_auto = len(auto_nodes)
    total_human = len(human_nodes)

    # Node metrics
    node_recall = _safe_div(len(alignment.covered_human), total_human)
    node_precision = _safe_div(len(alignment.grounded_auto), total_auto)
    node_f1 = _f1(node_precision, node_recall)

    # Type accuracy
    type_accuracy = _safe_div(edge_result.type_matches, edge_result.type_total)

    # Edge metrics
    edge_recall = _safe_div(edge_result.matched_edges, edge_result.total_human_edges)
    edge_precision = _safe_div(edge_result.matched_edges, edge_result.total_auto_edges)
    edge_f1 = _f1(edge_precision, edge_recall)

    # Average alignment score
    all_sims = [sim for _, sim in alignment.auto_to_human.values()]
    avg_alignment = float(np.mean(all_sims)) if all_sims else 0.0

    # Granularity ratio
    granularity_ratio = _safe_div(total_auto, len(alignment.covered_human))

    # Structural score
    structural_score = (node_f1 + edge_f1) / 2

    # Similarity distribution buckets
    sim_buckets = {"high_90_100": 0, "good_80_90": 0, "fair_70_80": 0}
    for sim in all_sims:
        if sim >= 0.9:
            sim_buckets["high_90_100"] += 1
        elif sim >= 0.8:
            sim_buckets["good_80_90"] += 1
        else:
            sim_buckets["fair_70_80"] += 1

    # Type disagreement breakdown
    type_disagree: Dict[str, int] = {}
    for a_id, (h_id, _) in alignment.auto_to_human.items():
        a_type = auto_nodes[a_id]["type"]
        h_type = human_nodes[h_id]["type"]
        if a_type != h_type:
            key = f"{a_type} vs {h_type}"
            type_disagree[key] = type_disagree.get(key, 0) + 1

    # Many-to-one stats: how many human nodes map to >1 auto node
    multi_match_human = {
        h_id: len(matches)
        for h_id, matches in alignment.human_to_auto.items()
        if len(matches) > 1
    }
    avg_fan_out = (
        float(np.mean([len(m) for m in alignment.human_to_auto.values()]))
        if alignment.human_to_auto else 1.0
    )

    metrics = {
        # Node alignment
        "node_recall": round(node_recall, 4),
        "node_precision": round(node_precision, 4),
        "node_f1": round(node_f1, 4),
        # Edge alignment
        "edge_recall": round(edge_recall, 4),
        "edge_precision": round(edge_precision, 4),
        "edge_f1": round(edge_f1, 4),
        # Type
        "type_accuracy": round(type_accuracy, 4),
        "type_disagreements": type_disagree,
        # Similarity
        "avg_alignment_score": round(avg_alignment, 4),
        "min_alignment_score": round(float(min(all_sims)), 4) if all_sims else 0.0,
        "max_alignment_score": round(float(max(all_sims)), 4) if all_sims else 0.0,
        "similarity_distribution": sim_buckets,
        # Structure
        "granularity_ratio": round(granularity_ratio, 2),
        "structural_score": round(structural_score, 4),
        "avg_fan_out": round(avg_fan_out, 2),
        "multi_match_human_count": len(multi_match_human),
        # Counts
        "total_auto_nodes": total_auto,
        "total_human_nodes": total_human,
        "covered_human_nodes": len(alignment.covered_human),
        "grounded_auto_nodes": len(alignment.grounded_auto),
        "unmatched_human_nodes": len(alignment.unmatched_human),
        "unmatched_auto_nodes": len(alignment.unmatched_auto),
        "edge_mismatches_count": edge_result.total_human_edges - edge_result.matched_edges,
    }

    # SOP grounding metrics (if available)
    if auto_grounding:
        auto_advantages = sum(1 for r in auto_grounding.values() if r["grounded"])
        metrics["auto_only_sop_grounded"] = auto_advantages
        metrics["auto_only_hallucinated"] = len(auto_grounding) - auto_advantages
        metrics["auto_advantage_rate"] = round(
            _safe_div(auto_advantages, len(auto_grounding)), 4
        )

    if human_grounding:
        true_gaps = sum(1 for r in human_grounding.values() if r["grounded"])
        metrics["human_only_sop_grounded"] = true_gaps
        metrics["human_only_extrapolated"] = len(human_grounding) - true_gaps
        metrics["true_gap_rate"] = round(
            _safe_div(true_gaps, len(human_grounding)), 4
        )

    return metrics


# ---------------------------------------------------------------------------
# Step 5: Generate full report
# ---------------------------------------------------------------------------


def generate_report(
    auto_graph: Dict[str, Any],
    human_graph: Dict[str, Any],
    sop_text: Optional[str] = None,
    auto_format: str = "auto",
    human_format: str = "human",
) -> Dict[str, Any]:
    """Full three-way comparison: auto graph vs human graph vs SOP.

    Args:
        auto_graph: Auto-generated graph (our format or raw).
        human_graph: Human-created graph (their format or raw).
        sop_text: Original SOP document text (optional, enables grounding check).
        auto_format: "auto" (our WorkflowNode format) or "human" (if same format).
        human_format: "human" (their format) or "auto" (if same format).

    Returns:
        Full comparison report dict.
    """
    logger.info("=== Graph Comparison Report ===")

    # Step 0: Normalize
    logger.info("Step 0: Normalizing graphs...")
    if auto_format == "human":
        auto_nodes = _normalize_human_graph(auto_graph)
    else:
        auto_nodes = _normalize_auto_graph(auto_graph)

    if human_format == "auto":
        human_nodes = _normalize_auto_graph(human_graph)
    else:
        human_nodes = _normalize_human_graph(human_graph)

    logger.info("  Auto: %d nodes, Human: %d nodes", len(auto_nodes), len(human_nodes))

    # Step 1: Semantic alignment
    logger.info("Step 1: Semantic node alignment...")
    alignment = _align_nodes(auto_nodes, human_nodes)

    # Step 2: Edge comparison
    logger.info("Step 2: Structural edge comparison...")
    edge_result = _compare_edges(auto_nodes, human_nodes, alignment)
    logger.info(
        "  Edges: %d/%d human edges matched, %d mismatches",
        edge_result.matched_edges, edge_result.total_human_edges,
        len(edge_result.mismatches),
    )

    # Step 3: SOP grounding (if SOP provided)
    auto_grounding = None
    human_grounding = None
    if sop_text and alignment.unmatched_auto:
        logger.info("Step 3a: SOP grounding for %d unmatched auto nodes...", len(alignment.unmatched_auto))
        unmatched_auto_data = {nid: auto_nodes[nid] for nid in alignment.unmatched_auto}
        auto_grounding = _check_sop_grounding(unmatched_auto_data, sop_text)

    if sop_text and alignment.unmatched_human:
        logger.info("Step 3b: SOP grounding for %d unmatched human nodes...", len(alignment.unmatched_human))
        unmatched_human_data = {nid: human_nodes[nid] for nid in alignment.unmatched_human}
        human_grounding = _check_sop_grounding(unmatched_human_data, sop_text)

    # Step 4: Compute metrics
    logger.info("Step 4: Computing metrics...")
    metrics = compute_comparison_metrics(
        auto_nodes, human_nodes, alignment, edge_result,
        auto_grounding, human_grounding,
    )

    # Build matched pairs detail
    matched_pairs = []
    for a_id, (h_id, sim) in alignment.auto_to_human.items():
        matched_pairs.append({
            "auto_id": a_id,
            "auto_text": auto_nodes[a_id]["text"],
            "human_id": h_id,
            "human_text": human_nodes[h_id]["text"],
            "similarity": round(sim, 4),
            "type_match": auto_nodes[a_id]["type"] == human_nodes[h_id]["type"],
            "auto_type": auto_nodes[a_id]["type"],
            "human_type": human_nodes[h_id]["type"],
        })
    matched_pairs.sort(key=lambda x: x["similarity"])

    # Build many-to-one detail: human nodes that split into multiple auto nodes
    many_to_one = []
    for h_id, matches in alignment.human_to_auto.items():
        if len(matches) > 1:
            many_to_one.append({
                "human_id": h_id,
                "human_text": human_nodes[h_id]["text"],
                "auto_matches": [
                    {"id": a_id, "text": auto_nodes[a_id]["text"], "similarity": round(s, 4)}
                    for a_id, s in matches
                ],
                "fan_out": len(matches),
            })
    many_to_one.sort(key=lambda x: x["fan_out"], reverse=True)

    # Build auto-only detail
    auto_only = []
    for nid in alignment.unmatched_auto:
        entry = {
            "id": nid,
            "text": auto_nodes[nid]["text"],
            "type": auto_nodes[nid]["type"],
        }
        if auto_grounding and nid in auto_grounding:
            g = auto_grounding[nid]
            entry["verdict"] = "auto_advantage" if g["grounded"] else "hallucinated"
            entry["reason"] = g.get("reason", "")
            entry["relevant_sop_excerpt"] = g.get("relevant_sop_excerpt", "")[:200]
        else:
            entry["verdict"] = "unknown"
            entry["reason"] = "SOP not provided for grounding check"
        auto_only.append(entry)

    # Build human-only detail
    human_only = []
    for nid in alignment.unmatched_human:
        entry = {
            "id": nid,
            "text": human_nodes[nid]["text"],
            "type": human_nodes[nid]["type"],
        }
        if human_grounding and nid in human_grounding:
            g = human_grounding[nid]
            entry["verdict"] = "gap" if g["grounded"] else "human_extrapolation"
            entry["reason"] = g.get("reason", "")
            entry["relevant_sop_excerpt"] = g.get("relevant_sop_excerpt", "")[:200]
        else:
            entry["verdict"] = "unknown"
            entry["reason"] = "SOP not provided for grounding check"
        human_only.append(entry)

    # Graph-level metrics
    auto_metrics = _compute_single_graph_metrics(auto_nodes)
    human_metrics = _compute_single_graph_metrics(human_nodes)

    # Build similarity matrix table (auto_id × human_id → cosine sim)
    sim_matrix_table = None
    if alignment.sim_matrix is not None:
        sim_matrix_table = {
            "auto_ids": alignment.auto_ids,
            "human_ids": alignment.human_ids,
            "matrix": [
                [round(float(alignment.sim_matrix[i][j]), 4)
                 for j in range(len(alignment.human_ids))]
                for i in range(len(alignment.auto_ids))
            ],
        }

    report = {
        "summary": metrics,
        "auto_graph_metrics": auto_metrics,
        "human_graph_metrics": human_metrics,
        "matched_pairs": matched_pairs,
        "many_to_one": many_to_one,
        "auto_only": auto_only,
        "human_only": human_only,
        "edge_mismatches": edge_result.mismatches[:50],
        "similarity_matrix": sim_matrix_table,
    }

    logger.info("=== Comparison complete ===")
    _print_summary(metrics, auto_only, human_only)

    return report


def _print_summary(
    metrics: Dict,
    auto_only: List, human_only: List,
) -> None:
    """Print a brief console summary."""
    print(f"\nNode F1: {metrics['node_f1']:.1%}  |  "
          f"Edge F1: {metrics['edge_f1']:.1%}  |  "
          f"Structural: {metrics['structural_score']:.1%}  |  "
          f"Auto: {metrics['total_auto_nodes']} nodes  |  "
          f"Human: {metrics['total_human_nodes']} nodes")
    if auto_only:
        print(f"  Auto-only: {len(auto_only)} nodes")
    if human_only:
        print(f"  Human-only: {len(human_only)} nodes")


def generate_markdown_report(report: Dict[str, Any]) -> str:
    """Generate a detailed markdown report from the comparison results."""
    m = report["summary"]
    auto_m = report["auto_graph_metrics"]
    human_m = report["human_graph_metrics"]
    lines: List[str] = []

    def _add(text: str = "") -> None:
        lines.append(text)

    # --- Header ---
    _add("# Graph Comparison Report")
    _add()
    _add("Auto-generated graph vs Human-curated graph, evaluated against the source SOP.")
    _add()

    # --- Overview ---
    _add("## 1. Overview")
    _add()
    _add("| | Auto Graph | Human Graph |")
    _add("|---|---|---|")
    _add(f"| Total Nodes | {m['total_auto_nodes']} | {m['total_human_nodes']} |")
    _add(f"| Total Edges | {auto_m['edge_count']} | {human_m['edge_count']} |")
    _add(f"| Orphan Nodes | {auto_m['orphan_count']} | {human_m['orphan_count']} |")
    _add(f"| Dead Ends | {auto_m['dead_end_count']} | {human_m['dead_end_count']} |")
    _add(f"| Broken Links | {auto_m['broken_link_count']} | {human_m['broken_link_count']} |")
    _add(f"| Terminal Nodes | {auto_m['terminal_count']} | {human_m['terminal_count']} |")
    _add()

    # Type distribution
    all_types = sorted(set(
        list(auto_m.get("type_distribution", {}).keys())
        + list(human_m.get("type_distribution", {}).keys())
    ))
    _add("**Node type distribution:**")
    _add()
    _add("| Type | Auto | Human |")
    _add("|---|---|---|")
    for t in all_types:
        _add(f"| {t} | {auto_m.get('type_distribution', {}).get(t, 0)} | {human_m.get('type_distribution', {}).get(t, 0)} |")
    _add()

    # --- Core Metrics ---
    _add("## 2. Core Metrics")
    _add()
    _add("| Metric | Score | Interpretation |")
    _add("|---|---|---|")
    _add(f"| **Node Recall** | {m['node_recall']:.1%} | {_interpret_recall(m['node_recall'])} |")
    _add(f"| **Node Precision** | {m['node_precision']:.1%} | {_interpret_precision(m['node_precision'])} |")
    _add(f"| **Node F1** | {m['node_f1']:.1%} | Combined node coverage score |")
    _add(f"| **Edge Recall** | {m['edge_recall']:.1%} | {_interpret_recall(m['edge_recall'], 'edge')} |")
    _add(f"| **Edge Precision** | {m['edge_precision']:.1%} | {_interpret_precision(m['edge_precision'], 'edge')} |")
    _add(f"| **Edge F1** | {m['edge_f1']:.1%} | Combined edge coverage score |")
    _add(f"| **Type Accuracy** | {m['type_accuracy']:.1%} | Agreement on node types (question vs instruction etc.) |")
    _add(f"| **Structural Score** | {m['structural_score']:.1%} | Overall structural similarity (avg of node F1 + edge F1) |")
    _add()

    # --- Alignment Quality ---
    _add("## 3. Alignment Quality")
    _add()
    _add(f"- **Average similarity** of matched pairs: {m['avg_alignment_score']:.4f}")
    _add(f"- **Best match**: {m['max_alignment_score']:.4f}")
    _add(f"- **Weakest match**: {m['min_alignment_score']:.4f}")
    _add(f"- **Granularity ratio**: {m['granularity_ratio']:.2f}x "
         f"({'auto graph is more granular' if m['granularity_ratio'] > 1.1 else 'similar granularity' if m['granularity_ratio'] > 0.9 else 'human graph is more granular'})")
    _add(f"- **Avg fan-out**: {m['avg_fan_out']:.2f} auto nodes per human node")
    _add(f"- **Human nodes with multiple auto matches**: {m['multi_match_human_count']}")
    _add()

    sim_dist = m.get("similarity_distribution", {})
    _add("**Similarity distribution:**")
    _add()
    _add("| Range | Count | Quality |")
    _add("|---|---|---|")
    _add(f"| 0.90 - 1.00 | {sim_dist.get('high_90_100', 0)} | Near-identical wording |")
    _add(f"| 0.80 - 0.90 | {sim_dist.get('good_80_90', 0)} | Same step, different wording |")
    _add(f"| 0.70 - 0.80 | {sim_dist.get('fair_70_80', 0)} | Likely same step, loosely worded |")
    _add()

    # --- Type Disagreements ---
    type_disagree = m.get("type_disagreements", {})
    if type_disagree:
        _add("## 4. Type Disagreements")
        _add()
        _add("These matched node pairs disagree on node type:")
        _add()
        _add("| Auto Type vs Human Type | Count |")
        _add("|---|---|")
        for k, v in sorted(type_disagree.items(), key=lambda x: x[1], reverse=True):
            _add(f"| {k} | {v} |")
        _add()

    # --- Matched Pairs ---
    matched = report.get("matched_pairs", [])
    _add(f"## 5. Matched Node Pairs ({len(matched)} total)")
    _add()
    if matched:
        _add("Sorted by similarity (weakest matches first — most likely to be wrong):")
        _add()
        _add("| Sim | Auto ID | Auto Text | Human ID | Human Text | Type Match |")
        _add("|---|---|---|---|---|---|")
        for p in matched:
            type_icon = "Y" if p["type_match"] else f"N ({p['auto_type']}/{p['human_type']})"
            _add(f"| {p['similarity']:.2f} | `{p['auto_id']}` | {p['auto_text'][:60]} | `{p['human_id']}` | {p['human_text'][:60]} | {type_icon} |")
        _add()

    # --- Many-to-One ---
    many = report.get("many_to_one", [])
    if many:
        _add(f"## 6. Many-to-One Matches ({len(many)} human nodes)")
        _add()
        _add("These human nodes are broad — our auto graph breaks them into multiple granular steps:")
        _add()
        for item in many:
            _add(f"### Human: `{item['human_id']}` ({item['fan_out']} auto nodes)")
            _add(f"> {item['human_text']}")
            _add()
            _add("| Auto ID | Auto Text | Similarity |")
            _add("|---|---|---|")
            for am in item["auto_matches"]:
                _add(f"| `{am['id']}` | {am['text'][:80]} | {am['similarity']:.2f} |")
            _add()

    # --- SOP Grounding ---
    auto_only = report.get("auto_only", [])
    human_only = report.get("human_only", [])
    has_grounding = any(n.get("verdict") not in ("unknown", None) for n in auto_only + human_only)

    section_num = 7 if many else 6

    if has_grounding:
        _add(f"## {section_num}. SOP Grounding Analysis")
        _add()
        _add("For nodes that exist in only one graph, we check whether the content is actually in the SOP document.")
        _add()

        # Auto advantages
        auto_advantages = [n for n in auto_only if n.get("verdict") == "auto_advantage"]
        auto_hallucinated = [n for n in auto_only if n.get("verdict") == "hallucinated"]

        if auto_advantages:
            _add(f"### {section_num}.1 Auto Advantages ({len(auto_advantages)} nodes)")
            _add()
            _add("Our auto graph caught these SOP details that the human graph missed:")
            _add()
            for n in auto_advantages:
                _add(f"- **`{n['id']}`**: {n['text']}")
                _add(f"  - *Evidence*: {n.get('reason', 'N/A')}")
                _add()

        if auto_hallucinated:
            _add(f"### {section_num}.2 Potential Hallucinations ({len(auto_hallucinated)} nodes)")
            _add()
            _add("These auto nodes are NOT grounded in the SOP (LLM-verified):")
            _add()
            for n in auto_hallucinated:
                _add(f"- **`{n['id']}`**: {n['text']}")
                _add(f"  - *Reason*: {n.get('reason', 'N/A')}")
                _add()

        # True gaps
        true_gaps = [n for n in human_only if n.get("verdict") == "gap"]
        human_extrap = [n for n in human_only if n.get("verdict") == "human_extrapolation"]

        if true_gaps:
            _add(f"### {section_num}.3 True Gaps ({len(true_gaps)} nodes)")
            _add()
            _add("Our auto graph is missing these steps that ARE in the SOP (LLM-verified):")
            _add()
            for n in true_gaps:
                _add(f"- **`{n['id']}`**: {n['text']}")
                _add(f"  - *Evidence*: {n.get('reason', 'N/A')}")
                _add()

        if human_extrap:
            _add(f"### {section_num}.4 Human Extrapolations ({len(human_extrap)} nodes)")
            _add()
            _add("The human graph added these steps that are NOT in the SOP (LLM-verified):")
            _add()
            for n in human_extrap:
                _add(f"- **`{n['id']}`**: {n['text']}")
                _add(f"  - *Reason*: {n.get('reason', 'N/A')}")
                _add()

        # Summary counts
        _add(f"### {section_num}.5 SOP Grounding Summary")
        _add()
        _add("| Category | Count | Meaning |")
        _add("|---|---|---|")
        _add(f"| Auto Advantages | {m.get('auto_only_sop_grounded', 0)} | SOP content we caught, human missed |")
        _add(f"| Hallucinated | {m.get('auto_only_hallucinated', 0)} | Auto nodes not in SOP |")
        _add(f"| True Gaps | {m.get('human_only_sop_grounded', 0)} | SOP content we missed |")
        _add(f"| Human Extrapolations | {m.get('human_only_extrapolated', 0)} | Human added beyond SOP |")
        _add()

        section_num += 1
    else:
        # No SOP grounding, just list unmatched nodes
        if auto_only:
            _add(f"## {section_num}. Auto-Only Nodes ({len(auto_only)})")
            _add()
            _add("These nodes exist only in the auto graph (no SOP provided for grounding check):")
            _add()
            _add("| ID | Type | Text |")
            _add("|---|---|---|")
            for n in auto_only:
                _add(f"| `{n['id']}` | {n['type']} | {n['text'][:80]} |")
            _add()
            section_num += 1

        if human_only:
            _add(f"## {section_num}. Human-Only Nodes ({len(human_only)})")
            _add()
            _add("These nodes exist only in the human graph:")
            _add()
            _add("| ID | Type | Text |")
            _add("|---|---|---|")
            for n in human_only:
                _add(f"| `{n['id']}` | {n['type']} | {n['text'][:80]} |")
            _add()
            section_num += 1

    # --- Edge Mismatches ---
    edge_mm = report.get("edge_mismatches", [])
    if edge_mm:
        # Split into decision vs non-decision edges
        decision_mm = [e for e in edge_mm if e.get("llm_reason", "") != "Skipped — non-decision edge"]
        non_decision_mm = [e for e in edge_mm if e.get("llm_reason", "") == "Skipped — non-decision edge"]

        _add(f"## {section_num}. Edge Mismatches ({len(edge_mm)} unmatched human edges)")
        _add()

        if decision_mm:
            _add(f"### {section_num}.1 Decision Edge Mismatches ({len(decision_mm)} — LLM validated)")
            _add()
            _add("These decision branches in the human graph are NOT preserved in the auto graph:")
            _add()
            for e in decision_mm[:30]:
                _add(f"- **`{e['human_edge']}`**")
                _add(f"  - Source: {e['human_src_text']}")
                _add(f"  - Target: {e['human_tgt_text']}")
                _add(f"  - *Reason*: {e.get('llm_reason', 'N/A')}")
                _add()

        if non_decision_mm:
            _add(f"### {section_num}.2 Sequential Edge Mismatches ({len(non_decision_mm)})")
            _add()
            _add("These non-decision edges have no corresponding auto path (structural only, not LLM-validated):")
            _add()
            _add("| Human Edge | Source Text | Target Text |")
            _add("|---|---|---|")
            for e in non_decision_mm[:30]:
                _add(f"| `{e['human_edge']}` | {e['human_src_text']} | {e['human_tgt_text']} |")
            if len(non_decision_mm) > 30:
                _add(f"| ... | *{len(non_decision_mm) - 30} more* | |")
            _add()

        section_num += 1

    # --- Similarity Matrix ---
    sim_data = report.get("similarity_matrix")
    if sim_data:
        _add(f"## {section_num}. Embedding Similarity Matrix")
        _add()
        _add("Cosine similarity between every auto node (rows) and human node (columns).")
        _add("Cells show similarity scores. **Bold** = LLM-confirmed match.")
        _add()

        h_ids = sim_data["human_ids"]
        a_ids = sim_data["auto_ids"]
        matrix = sim_data["matrix"]

        # Build set of LLM-confirmed pairs for bolding
        confirmed_pairs: Set[Tuple[str, str]] = set()
        for p in report.get("matched_pairs", []):
            confirmed_pairs.add((p["auto_id"], p["human_id"]))

        # Header row — use short IDs (last 20 chars)
        h_short = [h[:20] for h in h_ids]
        header = "| Auto \\\\ Human | " + " | ".join(f"`{h}`" for h in h_short) + " |"
        sep = "|---|" + "|".join("---" for _ in h_ids) + "|"
        _add(header)
        _add(sep)

        for i, a_id in enumerate(a_ids):
            cells = []
            for j, h_id in enumerate(h_ids):
                val = matrix[i][j]
                cell = f"{val:.2f}"
                if (a_id, h_id) in confirmed_pairs:
                    cell = f"**{cell}**"
                elif val >= 0.70:
                    cell = f"*{cell}*"
                cells.append(cell)
            _add(f"| `{a_id[:20]}` | " + " | ".join(cells) + " |")

        _add()
        _add("Legend: **bold** = LLM-confirmed match, *italic* = above embedding threshold but not matched by LLM")
        _add()
        section_num += 1

    # --- Key Takeaways ---
    _add(f"## {section_num}. Key Takeaways")
    _add()
    takeaways = _generate_takeaways(m)
    for t in takeaways:
        _add(f"- {t}")
    _add()

    return "\n".join(lines)


def _interpret_recall(score: float, kind: str = "node") -> str:
    if score >= 0.9:
        return f"Excellent — auto graph captures almost all human {kind}s"
    if score >= 0.75:
        return f"Good — most human {kind}s are represented"
    if score >= 0.5:
        return f"Fair — significant {kind}s missing from auto graph"
    return f"Poor — auto graph misses many human {kind}s"


def _interpret_precision(score: float, kind: str = "node") -> str:
    if score >= 0.9:
        return f"Excellent — almost all auto {kind}s are real"
    if score >= 0.75:
        return f"Good — most auto {kind}s correspond to human {kind}s"
    if score >= 0.5:
        return f"Fair — many auto {kind}s have no human counterpart"
    return f"Poor — auto graph has many ungrounded {kind}s"


def _generate_takeaways(m: Dict) -> List[str]:
    takeaways = []

    # Overall quality
    score = m["structural_score"]
    if score >= 0.8:
        takeaways.append(f"**Strong overall match** (structural score {score:.1%}). The auto graph closely mirrors the human graph.")
    elif score >= 0.6:
        takeaways.append(f"**Moderate match** (structural score {score:.1%}). The auto graph captures the main flow but diverges on details.")
    else:
        takeaways.append(f"**Significant divergence** (structural score {score:.1%}). The graphs differ substantially in structure.")

    # Granularity
    gr = m["granularity_ratio"]
    if gr > 1.3:
        takeaways.append(f"Auto graph is **{gr:.1f}x more granular** than human graph — it breaks steps into finer sub-steps.")
    elif gr < 0.8:
        takeaways.append(f"Human graph is more granular (ratio {gr:.1f}x) — auto graph may be grouping steps too broadly.")

    # Coverage
    if m["node_recall"] >= 0.9:
        takeaways.append("Auto graph has **excellent coverage** of human graph content.")
    elif m["node_recall"] < 0.7:
        takeaways.append(f"Auto graph **misses {m['unmatched_human_nodes']} human nodes** — check the True Gaps section for SOP content we missed.")

    # Precision
    if m["node_precision"] < 0.7:
        takeaways.append(f"**{m['unmatched_auto_nodes']} auto-only nodes** — check if these are real SOP content (advantages) or hallucinations.")

    # Type accuracy
    if m["type_accuracy"] < 0.85:
        disagree = m.get("type_disagreements", {})
        top = max(disagree.items(), key=lambda x: x[1]) if disagree else None
        msg = f"Type accuracy is {m['type_accuracy']:.0%}"
        if top:
            msg += f" — most common mismatch: {top[0]} ({top[1]} cases)"
        takeaways.append(msg)

    # SOP grounding insights
    if "auto_advantage_rate" in m:
        adv = m.get("auto_only_sop_grounded", 0)
        hall = m.get("auto_only_hallucinated", 0)
        if adv > 0:
            takeaways.append(f"**{adv} auto advantages** — our graph caught SOP content the human missed.")
        if hall > 0:
            takeaways.append(f"**{hall} potential hallucinations** — auto nodes not grounded in the SOP.")

    if "true_gap_rate" in m:
        gaps = m.get("human_only_sop_grounded", 0)
        extrap = m.get("human_only_extrapolated", 0)
        if gaps > 0:
            takeaways.append(f"**{gaps} true gaps** — SOP content present in human graph but missing from ours.")
        if extrap > 0:
            takeaways.append(f"**{extrap} human extrapolations** — human added content beyond the SOP.")

    # Edge coverage
    if m["edge_recall"] < m["node_recall"] - 0.15:
        takeaways.append("Edge recall is notably lower than node recall — the auto graph captures the right steps but connects them differently.")

    return takeaways


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _load_graph_json(path: str) -> Dict[str, Any]:
    """Load a graph from JSON file. Handles both raw and envelope formats."""
    data = json.loads(Path(path).read_text())

    # If it's our envelope format, extract the nodes
    if "graph_state" in data:
        return data["graph_state"].get("nodes", {})
    if "nodes" in data and isinstance(data["nodes"], dict):
        return data["nodes"]
    # Assume the top-level dict IS the graph
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Compare auto-generated and human-created workflow graphs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m sop_to_dag.graph_compare --auto auto.json --human human.json
              python -m sop_to_dag.graph_compare --auto auto.json --human human.json --sop sop.md
              python -m sop_to_dag.graph_compare --auto auto.json --human human.json --sop sop.md --output report.json
        """),
    )
    parser.add_argument("--auto", required=True, help="Path to auto-generated graph JSON")
    parser.add_argument("--human", required=True, help="Path to human-created graph JSON")
    parser.add_argument("--sop", help="Path to original SOP document (enables grounding check)")
    parser.add_argument("--output", "-o", help="Path to save full report JSON")
    parser.add_argument("--md", help="Path to save markdown report (e.g. comparison_report.md)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    auto_graph = _load_graph_json(args.auto)
    human_graph = _load_graph_json(args.human)

    sop_text = None
    if args.sop:
        sop_text = Path(args.sop).read_text()

    report = generate_report(
        auto_graph, human_graph,
        sop_text=sop_text,
    )

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"\nFull report saved to: {args.output}")

    # Generate markdown report
    md_path = args.md
    if not md_path:
        # Default: same dir as auto graph, named comparison_report.md
        md_path = str(Path(args.auto).parent / "comparison_report.md")
    md_content = generate_markdown_report(report)
    Path(md_path).write_text(md_content)
    print(f"Markdown report saved to: {md_path}")


if __name__ == "__main__":
    main()
