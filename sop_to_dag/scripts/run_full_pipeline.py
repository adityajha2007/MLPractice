"""CLI: Preprocessing -> Convert -> Refine end-to-end.

Usage:
    python -m sop_to_dag.scripts.run_full_pipeline <sop_file_path>
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from sop_to_dag.converter import PipelineConverter
from sop_to_dag.loop import run_refinement
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

    # Console handler
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
    )

    # File handler — saves ALL log output to run.log
    file_handler = logging.FileHandler(run_dir / "run.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.getLogger().addHandler(file_handler)


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

    # Phase 0: Preprocessing
    print("=== PREPROCESSING ===")
    prep_state = run_preprocessing(source_text)
    print(f"Chunks: {len(prep_state['chunks'])}")
    print(f"Enriched chunks: {len(prep_state['enriched_chunks'])}")
    print(f"Entity mappings: {len(prep_state['entity_map'])}")
    _dump_preprocessing(run_dir, prep_state)

    # Phase 1: Convert
    print(f"\n=== CONVERSION ===")
    print(f"Converting: {sop_path.name}")
    converter = PipelineConverter()
    nodes = converter.convert(source_text, prep_state["enriched_chunks"],
                              dump_dir=str(run_dir))
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

    # Phase 2: Refine (per-iteration dumps happen inside the loop)
    print(f"\n=== REFINEMENT ===")
    print(f"Starting refinement loop (max {args.max_iterations} iterations)...")
    final_state = run_refinement(state, max_iterations=args.max_iterations,
                                 store=store, dump_dir=str(run_dir))

    print(f"\nRefinement complete!")
    print(f"Iterations: {final_state['iteration']}")
    print(f"Is complete: {final_state['is_complete']}")
    print(f"Final nodes: {len(final_state['nodes'])}")

    final_path = store.save_graph(
        final_state, str(sop_path), f"{converter.converter_id}_refined"
    )
    store.update_status(final_path, "final")
    print(f"Final graph saved: {final_path}")
    print(f"All stage dumps: {run_dir}")


if __name__ == "__main__":
    main()
