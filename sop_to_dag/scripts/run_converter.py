"""CLI: SOP file -> preprocessing -> JSON DAG using the 3-stage pipeline.

Usage:
    python -m sop_to_dag.scripts.run_converter <sop_file_path>
"""

import argparse
import sys
from pathlib import Path

from sop_to_dag.converter import PipelineConverter
from sop_to_dag.preprocessing import run_preprocessing
from sop_to_dag.schemas import GraphState
from sop_to_dag.storage import GraphStore


def main():
    parser = argparse.ArgumentParser(
        description="Convert an SOP file to a JSON DAG."
    )
    parser.add_argument("sop_file", type=str, help="Path to the SOP markdown file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/graphs",
        help="Directory to store output JSON files.",
    )
    args = parser.parse_args()

    sop_path = Path(args.sop_file)
    if not sop_path.exists():
        print(f"Error: File not found: {sop_path}")
        sys.exit(1)

    source_text = sop_path.read_text()
    store = GraphStore(store_dir=Path(args.output_dir))

    # Preprocessing: chunk + RAG enrich + entity resolution
    print(f"Preprocessing: {sop_path.name}")
    prep_state = run_preprocessing(source_text)
    print(f"  Chunks: {len(prep_state['chunks'])}")
    print(f"  Entity mappings: {len(prep_state['entity_map'])}")

    # Conversion: 3-stage pipeline
    print("Running 3-stage pipeline (TopDown -> CodeBased -> GraphBased)...")
    converter = PipelineConverter()
    nodes = converter.convert(source_text, prep_state["enriched_chunks"])

    # Build GraphState and save
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

    output_path = store.save_graph(state, str(sop_path), converter.converter_id)
    print(f"\nGraph saved to: {output_path}")
    print(f"Nodes: {len(nodes)}")
    print(f"Node IDs: {list(nodes.keys())}")


if __name__ == "__main__":
    main()
