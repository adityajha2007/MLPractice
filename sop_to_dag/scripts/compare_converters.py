"""CLI: Run all three converters on the same SOP, compare outputs.

Alternatives (BottomUp, EdgeVertex) do NOT go through refinement.

Usage:
    python -m sop_to_dag.scripts.compare_converters <sop_file_path>
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from sop_to_dag.alternatives import BottomUpConverter, EdgeVertexConverter
from sop_to_dag.graph_ops import get_graph_issues
from sop_to_dag.converter import PipelineConverter
from sop_to_dag.evaluation import compute_metrics
from sop_to_dag.preprocessing import cached_preprocessing
from sop_to_dag.schemas import GraphState
from sop_to_dag.storage import GraphStore


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Compare all three converter approaches."
    )
    parser.add_argument("sop_file", type=str, help="Path to the SOP markdown file.")
    parser.add_argument(
        "--output-dir", type=str, default="output/comparison", help="Output directory."
    )
    args = parser.parse_args()

    sop_path = Path(args.sop_file)
    if not sop_path.exists():
        print(f"Error: File not found: {sop_path}")
        sys.exit(1)

    source_text = sop_path.read_text()
    store = GraphStore(store_dir=Path(args.output_dir))

    # Preprocessing (shared across all converters, content-based cache)
    print("=== PREPROCESSING ===")
    prep_state = cached_preprocessing(source_text, rebuild_faiss=False)
    enriched_chunks = prep_state["enriched_chunks"]
    print(f"Chunks: {len(prep_state['chunks'])}")
    print(f"Entity mappings: {len(prep_state['entity_map'])}")

    converters = [
        PipelineConverter(),
        BottomUpConverter(),
        EdgeVertexConverter(),
    ]

    results = {}

    for converter in converters:
        cid = converter.converter_id
        print(f"\n{'='*60}")
        print(f"Running: {cid}")
        print(f"{'='*60}")

        try:
            nodes = converter.convert(source_text, enriched_chunks)
            topo_report = get_graph_issues(nodes)
            metrics = compute_metrics(nodes)

            results[cid] = {
                "node_count": len(nodes),
                "topology": topo_report,
                "metrics": metrics,
            }

            state = GraphState(
                source_text=source_text,
                nodes=nodes,
                feedback="",
                iteration=0,
                is_complete=False,
                converter_id=cid,
                analysis_report="",
                enriched_chunks=enriched_chunks,
                vector_store=None,
                entity_map=prep_state["entity_map"],
            )
            path = store.save_graph(state, str(sop_path), cid)

            print(f"  Nodes: {len(nodes)}")
            print(f"  Topology: {topo_report}")
            print(f"  Metrics: {json.dumps(metrics, indent=2)}")
            print(f"  Saved: {path}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results[cid] = {"error": str(e)}

    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    for cid, data in results.items():
        print(f"\n{cid}:")
        if "error" in data:
            print(f"  Failed: {data['error']}")
        else:
            print(f"  Nodes: {data['node_count']}")
            print(f"  Topology: {data['topology']}")


if __name__ == "__main__":
    main()
