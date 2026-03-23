"""CLI: Preprocessing -> Convert -> Refine end-to-end.

Usage:
    python -m sop_to_dag.scripts.run_full_pipeline <sop_file_path>
"""

import argparse
import sys
from pathlib import Path

from sop_to_dag.converter import PipelineConverter
from sop_to_dag.loop import run_refinement
from sop_to_dag.preprocessing import run_preprocessing
from sop_to_dag.schemas import GraphState
from sop_to_dag.storage import GraphStore


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: SOP -> Preprocess -> JSON DAG -> Refined DAG."
    )
    parser.add_argument("sop_file", type=str, help="Path to the SOP markdown file.")
    parser.add_argument(
        "--output-dir", type=str, default="output/graphs", help="Output directory."
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10, help="Max refinement iterations."
    )
    args = parser.parse_args()

    sop_path = Path(args.sop_file)
    if not sop_path.exists():
        print(f"Error: File not found: {sop_path}")
        sys.exit(1)

    source_text = sop_path.read_text()
    store = GraphStore(store_dir=Path(args.output_dir))

    # Phase 0: Preprocessing
    print("=== PREPROCESSING ===")
    prep_state = run_preprocessing(source_text)
    print(f"Chunks: {len(prep_state['chunks'])}")
    print(f"Enriched chunks: {len(prep_state['enriched_chunks'])}")
    print(f"Entity mappings: {len(prep_state['entity_map'])}")

    # Phase 1: Convert
    print(f"\n=== CONVERSION ===")
    print(f"Converting: {sop_path.name}")
    converter = PipelineConverter()
    nodes = converter.convert(source_text, prep_state["enriched_chunks"])
    print(f"Conversion complete. Nodes: {len(nodes)}")

    # Save draft
    state = GraphState(
        source_text=source_text,
        nodes=nodes,
        feedback="",
        iteration=0,
        is_complete=False,
        converter_id=converter.converter_id,
        analysis_report="",
        enriched_chunks=prep_state["enriched_chunks"],
        vector_store=prep_state["vector_store"],
        entity_map=prep_state["entity_map"],
    )
    draft_path = store.save_graph(state, str(sop_path), converter.converter_id)
    print(f"Draft saved: {draft_path}")

    # Phase 2: Refine
    print(f"\n=== REFINEMENT ===")
    print(f"Starting refinement loop (max {args.max_iterations} iterations)...")
    final_state = run_refinement(state, max_iterations=args.max_iterations, store=store)

    print(f"\nRefinement complete!")
    print(f"Iterations: {final_state['iteration']}")
    print(f"Is complete: {final_state['is_complete']}")
    print(f"Final nodes: {len(final_state['nodes'])}")

    final_path = store.save_graph(
        final_state, str(sop_path), f"{converter.converter_id}_refined"
    )
    store.update_status(final_path, "final")
    print(f"Final graph saved: {final_path}")


if __name__ == "__main__":
    main()
