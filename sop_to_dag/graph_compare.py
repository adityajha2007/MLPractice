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
ALIGNMENT_THRESHOLD = 0.70
SOP_GROUNDING_THRESHOLD = 0.72
SOP_CHUNK_SIZE = 500  # characters per SOP chunk for FAISS indexing

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
# Step 1: Semantic node alignment (many-to-one)
# ---------------------------------------------------------------------------


def _align_nodes(
    auto_nodes: Dict[str, Dict],
    human_nodes: Dict[str, Dict],
    threshold: float = ALIGNMENT_THRESHOLD,
) -> AlignmentResult:
    """Semantic many-to-one alignment via cosine similarity.

    A single broad human node can match multiple granular auto nodes.
    """
    if not auto_nodes or not human_nodes:
        return AlignmentResult(
            unmatched_human=set(human_nodes.keys()),
            unmatched_auto=set(auto_nodes.keys()),
        )

    model = _get_embeddings_model()

    auto_ids = list(auto_nodes.keys())
    human_ids = list(human_nodes.keys())
    auto_texts = [auto_nodes[nid]["text"] for nid in auto_ids]
    human_texts = [human_nodes[nid]["text"] for nid in human_ids]

    logger.info("Embedding %d auto nodes and %d human nodes...", len(auto_ids), len(human_ids))
    auto_emb = _embed_texts(model, auto_texts)
    human_emb = _embed_texts(model, human_texts)

    sim_matrix = _cosine_similarity_matrix(auto_emb, human_emb)
    # sim_matrix[i][j] = similarity between auto_ids[i] and human_ids[j]

    result = AlignmentResult()

    # For each human node: find ALL auto nodes above threshold
    for j, h_id in enumerate(human_ids):
        matches = []
        for i, a_id in enumerate(auto_ids):
            if sim_matrix[i][j] >= threshold:
                matches.append((a_id, float(sim_matrix[i][j])))
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            result.human_to_auto[h_id] = matches
            result.covered_human.add(h_id)
        else:
            result.unmatched_human.add(h_id)

    # For each auto node: find best human node above threshold
    for i, a_id in enumerate(auto_ids):
        best_j = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i][best_j])
        if best_sim >= threshold:
            result.auto_to_human[a_id] = (human_ids[best_j], best_sim)
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


def _compare_edges(
    auto_nodes: Dict[str, Dict],
    human_nodes: Dict[str, Dict],
    alignment: AlignmentResult,
) -> EdgeComparisonResult:
    """Compare edge structure between aligned graphs.

    An auto edge (a1→a2) matches a human edge (h1→h2) if:
    - a1 is aligned to h1 (or h1 is aligned to a1)
    - a2 is aligned to h2 (or h2 is aligned to a2)
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

    # Check which human edges are covered by auto edges
    matched = 0
    mismatches = []

    for h_src, h_tgt in human_edges:
        # Find all auto nodes aligned to h_src and h_tgt
        a_srcs = human_to_auto_set.get(h_src, set())
        a_tgts = human_to_auto_set.get(h_tgt, set())

        # Check if any auto edge connects an aligned src to an aligned tgt
        found = False
        for a_src in a_srcs:
            for a_tgt in a_tgts:
                if (a_src, a_tgt) in auto_edges:
                    found = True
                    break
                # Also check indirect paths: a_src → ... → a_tgt within 3 hops
                if _has_path(auto_nodes, a_src, a_tgt, max_hops=3):
                    found = True
                    break
            if found:
                break

        if found:
            matched += 1
        else:
            mismatches.append({
                "human_edge": f"{h_src} → {h_tgt}",
                "human_src_text": human_nodes.get(h_src, {}).get("text", "?")[:60],
                "human_tgt_text": human_nodes.get(h_tgt, {}).get("text", "?")[:60],
                "auto_candidates_src": list(a_srcs)[:3],
                "auto_candidates_tgt": list(a_tgts)[:3],
            })

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
        matched_edges=matched,
        total_auto_edges=len(auto_edges),
        total_human_edges=len(human_edges),
        mismatches=mismatches,
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


def _check_sop_grounding(
    unmatched_nodes: Dict[str, Dict[str, Any]],
    sop_text: str,
    threshold: float = SOP_GROUNDING_THRESHOLD,
) -> Dict[str, Dict[str, Any]]:
    """For each unmatched node, check if its content exists in the SOP.

    Returns {node_id: {text, grounded: bool, best_sop_chunk, similarity}}.
    """
    if not unmatched_nodes or not sop_text:
        return {
            nid: {"text": data["text"], "grounded": False, "best_sop_chunk": "", "similarity": 0.0}
            for nid, data in unmatched_nodes.items()
        }

    model = _get_embeddings_model()
    chunks = _chunk_sop(sop_text)

    if not chunks:
        return {
            nid: {"text": data["text"], "grounded": False, "best_sop_chunk": "", "similarity": 0.0}
            for nid, data in unmatched_nodes.items()
        }

    logger.info("SOP grounding: %d unmatched nodes against %d SOP chunks", len(unmatched_nodes), len(chunks))

    chunk_emb = _embed_texts(model, chunks)
    node_ids = list(unmatched_nodes.keys())
    node_texts = [unmatched_nodes[nid]["text"] for nid in node_ids]
    node_emb = _embed_texts(model, node_texts)

    sim_matrix = _cosine_similarity_matrix(node_emb, chunk_emb)

    results = {}
    for i, nid in enumerate(node_ids):
        best_j = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i][best_j])
        results[nid] = {
            "text": unmatched_nodes[nid]["text"],
            "grounded": best_sim >= threshold,
            "best_sop_chunk": chunks[best_j][:200],
            "similarity": round(best_sim, 4),
        }

    grounded_count = sum(1 for r in results.values() if r["grounded"])
    logger.info("SOP grounding: %d/%d nodes grounded", grounded_count, len(results))

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

    metrics = {
        "node_recall": round(node_recall, 4),
        "node_precision": round(node_precision, 4),
        "node_f1": round(node_f1, 4),
        "type_accuracy": round(type_accuracy, 4),
        "edge_recall": round(edge_recall, 4),
        "edge_precision": round(edge_precision, 4),
        "edge_f1": round(edge_f1, 4),
        "avg_alignment_score": round(avg_alignment, 4),
        "granularity_ratio": round(granularity_ratio, 2),
        "structural_score": round(structural_score, 4),
        "total_auto_nodes": total_auto,
        "total_human_nodes": total_human,
        "covered_human_nodes": len(alignment.covered_human),
        "grounded_auto_nodes": len(alignment.grounded_auto),
        "unmatched_human_nodes": len(alignment.unmatched_human),
        "unmatched_auto_nodes": len(alignment.unmatched_auto),
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
    threshold: float = ALIGNMENT_THRESHOLD,
) -> Dict[str, Any]:
    """Full three-way comparison: auto graph vs human graph vs SOP.

    Args:
        auto_graph: Auto-generated graph (our format or raw).
        human_graph: Human-created graph (their format or raw).
        sop_text: Original SOP document text (optional, enables grounding check).
        auto_format: "auto" (our WorkflowNode format) or "human" (if same format).
        human_format: "human" (their format) or "auto" (if same format).
        threshold: Cosine similarity threshold for alignment.

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
    alignment = _align_nodes(auto_nodes, human_nodes, threshold=threshold)

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
            "auto_text": auto_nodes[a_id]["text"][:100],
            "human_id": h_id,
            "human_text": human_nodes[h_id]["text"][:100],
            "similarity": round(sim, 4),
            "type_match": auto_nodes[a_id]["type"] == human_nodes[h_id]["type"],
            "auto_type": auto_nodes[a_id]["type"],
            "human_type": human_nodes[h_id]["type"],
        })
    matched_pairs.sort(key=lambda x: x["similarity"])

    # Build auto-only detail
    auto_only = []
    for nid in alignment.unmatched_auto:
        entry = {
            "id": nid,
            "text": auto_nodes[nid]["text"][:150],
            "type": auto_nodes[nid]["type"],
        }
        if auto_grounding and nid in auto_grounding:
            g = auto_grounding[nid]
            entry["verdict"] = "auto_advantage" if g["grounded"] else "hallucinated"
            entry["sop_similarity"] = g["similarity"]
            entry["best_sop_chunk"] = g["best_sop_chunk"][:100]
        else:
            entry["verdict"] = "unknown"
        auto_only.append(entry)

    # Build human-only detail
    human_only = []
    for nid in alignment.unmatched_human:
        entry = {
            "id": nid,
            "text": human_nodes[nid]["text"][:150],
            "type": human_nodes[nid]["type"],
        }
        if human_grounding and nid in human_grounding:
            g = human_grounding[nid]
            entry["verdict"] = "gap" if g["grounded"] else "human_extrapolation"
            entry["sop_similarity"] = g["similarity"]
            entry["best_sop_chunk"] = g["best_sop_chunk"][:100]
        else:
            entry["verdict"] = "unknown"
        human_only.append(entry)

    # Graph-level metrics
    auto_metrics = _compute_single_graph_metrics(auto_nodes)
    human_metrics = _compute_single_graph_metrics(human_nodes)

    report = {
        "summary": metrics,
        "auto_graph_metrics": auto_metrics,
        "human_graph_metrics": human_metrics,
        "matched_pairs": matched_pairs,
        "auto_only": auto_only,
        "human_only": human_only,
        "edge_mismatches": edge_result.mismatches[:50],  # Cap for readability
    }

    logger.info("=== Comparison complete ===")
    _print_summary(metrics, auto_metrics, human_metrics, auto_only, human_only)

    return report


def _print_summary(
    metrics: Dict,
    auto_only: List, human_only: List,
) -> None:
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 70)
    print("GRAPH COMPARISON REPORT")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    print(f"{'Auto nodes':<30} {metrics['total_auto_nodes']:>10}")
    print(f"{'Human nodes':<30} {metrics['total_human_nodes']:>10}")
    print(f"{'Granularity ratio':<30} {metrics['granularity_ratio']:>10.2f}x")
    print()
    print(f"{'Node Recall':<30} {metrics['node_recall']:>10.1%}")
    print(f"{'Node Precision':<30} {metrics['node_precision']:>10.1%}")
    print(f"{'Node F1':<30} {metrics['node_f1']:>10.1%}")
    print()
    print(f"{'Edge Recall':<30} {metrics['edge_recall']:>10.1%}")
    print(f"{'Edge Precision':<30} {metrics['edge_precision']:>10.1%}")
    print(f"{'Edge F1':<30} {metrics['edge_f1']:>10.1%}")
    print()
    print(f"{'Type Accuracy':<30} {metrics['type_accuracy']:>10.1%}")
    print(f"{'Avg Alignment Score':<30} {metrics['avg_alignment_score']:>10.4f}")
    print(f"{'Structural Score':<30} {metrics['structural_score']:>10.1%}")

    if "auto_advantage_rate" in metrics:
        print(f"\n{'--- SOP Grounding ---':<30}")
        print(f"{'Auto advantages':<30} {metrics.get('auto_only_sop_grounded', 0):>10}")
        print(f"{'Auto hallucinated':<30} {metrics.get('auto_only_hallucinated', 0):>10}")
        print(f"{'True gaps (we missed)':<30} {metrics.get('human_only_sop_grounded', 0):>10}")
        print(f"{'Human extrapolations':<30} {metrics.get('human_only_extrapolated', 0):>10}")

    if auto_only:
        print(f"\n--- Auto-only nodes ({len(auto_only)}) ---")
        for n in auto_only[:10]:
            verdict = n.get("verdict", "?")
            print(f"  [{verdict:>18}] {n['id']}: {n['text'][:70]}")
        if len(auto_only) > 10:
            print(f"  ... and {len(auto_only) - 10} more")

    if human_only:
        print(f"\n--- Human-only nodes ({len(human_only)}) ---")
        for n in human_only[:10]:
            verdict = n.get("verdict", "?")
            print(f"  [{verdict:>18}] {n['id']}: {n['text'][:70]}")
        if len(human_only) > 10:
            print(f"  ... and {len(human_only) - 10} more")

    print("\n" + "=" * 70)


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
    parser.add_argument("--threshold", type=float, default=ALIGNMENT_THRESHOLD,
                        help=f"Alignment similarity threshold (default: {ALIGNMENT_THRESHOLD})")

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
        threshold=args.threshold,
    )

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()
