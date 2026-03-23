"""CLI: SOP file -> preprocessing -> JSON DAG using the 3-stage pipeline.

Usage:
    python -m sop_to_dag.scripts.run_converter <sop_file_path>
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from sop_to_dag.converter import PipelineConverter
from sop_to_dag.preprocessing import run_preprocessing
from sop_to_dag.schemas import GraphState
from sop_to_dag.storage import GraphStore


def _create_run_dir(base_dir: str, sop_name: str) -> Path:
    """Create a unique timestamped directory for this run's stage dumps."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{sop_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _dump_preprocessing(run_dir: Path, prep_state: dict) -> None:
    """Dump preprocessing outputs (chunks, enriched chunks, entity map)."""
    (run_dir / "prep_chunks.json").write_text(
        json.dumps(prep_state["chunks"], indent=2)
    )
    (run_dir / "prep_enriched_chunks.json").write_text(
        json.dumps(prep_state["enriched_chunks"], indent=2)
    )
    (run_dir / "prep_entity_map.json").write_text(
        json.dumps(prep_state["entity_map"], indent=2)
    )


def _setup_logging(run_dir: Path) -> None:
    """Configure logging to both console and a log file in the run directory."""
    log_format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    date_format = "%H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
    )

    file_handler = logging.FileHandler(run_dir / "run.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.getLogger().addHandler(file_handler)


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
    parser.add_argument(
        "--dump-stages",
        type=str,
        default="output/stage_dumps",
        help="Base directory for stage dumps (a timestamped subdirectory is created per run).",
    )
    args = parser.parse_args()

    sop_path = Path(args.sop_file)
    if not sop_path.exists():
        print(f"Error: File not found: {sop_path}")
        sys.exit(1)

    source_text = sop_path.read_text()
    store = GraphStore(store_dir=Path(args.output_dir))

    # Create a unique run directory for this invocation
    run_dir = _create_run_dir(args.dump_stages, sop_path.stem)
    _setup_logging(run_dir)
    print(f"Stage dumps: {run_dir}")

    # Preprocessing: chunk + RAG enrich + entity resolution
    print(f"Preprocessing: {sop_path.name}")
    prep_state = run_preprocessing(source_text)
    print(f"  Chunks: {len(prep_state['chunks'])}")
    print(f"  Entity mappings: {len(prep_state['entity_map'])}")
    _dump_preprocessing(run_dir, prep_state)

    # Conversion: 3-stage pipeline
    print("Running 3-stage pipeline (Overview -> Pseudocode -> Merge -> Graph)...")
    converter = PipelineConverter()
    nodes = converter.convert(source_text, prep_state["enriched_chunks"],
                              dump_dir=str(run_dir))

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
    print(f"All stage dumps: {run_dir}")


if __name__ == "__main__":
    main()
