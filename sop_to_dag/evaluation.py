"""Evaluation metrics: node count, edge coverage, structural similarity.

Provides quantitative metrics for comparing converter outputs.
"""

from typing import Any, Dict

from sop_to_dag.analyser import get_all_issues_structured


def compute_metrics(nodes: Dict[str, Any]) -> Dict[str, Any]:
    """Compute evaluation metrics for a graph.

    Returns dict with node_count, edge_count, type_distribution,
    topological_validity, orphan_count, dead_end_count, broken_link_count,
    has_start_node, terminal_count.
    """
    if not nodes:
        return {"node_count": 0, "edge_count": 0, "topological_validity": False}

    edge_count = 0
    type_dist: Dict[str, int] = {}
    for data in nodes.values():
        node_type = data.get("type", "unknown")
        type_dist[node_type] = type_dist.get(node_type, 0) + 1
        if data.get("next"):
            edge_count += 1
        if data.get("options"):
            edge_count += len(data["options"])

    orphans, dead_ends, broken = get_all_issues_structured(nodes)

    return {
        "node_count": len(nodes),
        "edge_count": edge_count,
        "type_distribution": type_dist,
        "topological_validity": not (orphans or dead_ends or broken),
        "orphan_count": len(orphans),
        "dead_end_count": len(dead_ends),
        "broken_link_count": len(broken),
        "has_start_node": "start" in nodes,
        "terminal_count": type_dist.get("terminal", 0),
    }


def structural_similarity(
    nodes_a: Dict[str, Any], nodes_b: Dict[str, Any]
) -> Dict[str, float]:
    """Compare two graphs for structural similarity.

    Returns node_overlap (Jaccard), type_similarity (cosine), edge_count_ratio.
    """
    ids_a = set(nodes_a.keys())
    ids_b = set(nodes_b.keys())

    intersection = len(ids_a & ids_b)
    union = len(ids_a | ids_b)
    node_overlap = intersection / union if union > 0 else 0.0

    types = set()
    dist_a: Dict[str, int] = {}
    dist_b: Dict[str, int] = {}
    for d in nodes_a.values():
        t = d.get("type", "unknown")
        types.add(t)
        dist_a[t] = dist_a.get(t, 0) + 1
    for d in nodes_b.values():
        t = d.get("type", "unknown")
        types.add(t)
        dist_b[t] = dist_b.get(t, 0) + 1

    dot = sum(dist_a.get(t, 0) * dist_b.get(t, 0) for t in types)
    mag_a = sum(v**2 for v in dist_a.values()) ** 0.5
    mag_b = sum(v**2 for v in dist_b.values()) ** 0.5
    type_similarity = dot / (mag_a * mag_b) if (mag_a and mag_b) else 0.0

    edges_a = sum(
        1 for d in nodes_a.values() if d.get("next")
    ) + sum(
        len(d.get("options", {})) for d in nodes_a.values()
    )
    edges_b = sum(
        1 for d in nodes_b.values() if d.get("next")
    ) + sum(
        len(d.get("options", {})) for d in nodes_b.values()
    )
    edge_count_ratio = (
        min(edges_a, edges_b) / max(edges_a, edges_b)
        if max(edges_a, edges_b) > 0
        else 0.0
    )

    return {
        "node_overlap": node_overlap,
        "type_similarity": type_similarity,
        "edge_count_ratio": edge_count_ratio,
    }
