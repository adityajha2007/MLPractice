"""CLI: Pick up a stored graph -> run refinement loop.

Usage:
    python -m sop_to_dag.scripts.run_refinement <stored_graph_path>
"""

import argparse
import sys
from pathlib import Path

from sop_to_dag.loop import run_refinement
from sop_to_dag.storage import GraphStore


def main():
    parser = argparse.ArgumentParser(
        description="Run the refinement loop on a stored graph."
    )
    parser.add_argument("graph_path", type=str, help="Path to the stored graph JSON.")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum refinement iterations.",
    )
    args = parser.parse_args()

    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        print(f"Error: File not found: {graph_path}")
        sys.exit(1)

    store = GraphStore(store_dir=graph_path.parent)
    state = store.load_graph(graph_path)

    print(f"Loaded graph: {graph_path.name}")
    print(f"Nodes: {len(state['nodes'])}")
    print(f"Starting refinement loop (max {args.max_iterations} iterations)...")

    final_state = run_refinement(state, max_iterations=args.max_iterations, store=store)

    print(f"\nRefinement complete!")
    print(f"Iterations: {final_state['iteration']}")
    print(f"Is complete: {final_state['is_complete']}")
    print(f"Nodes: {len(final_state['nodes'])}")

    output_path = store.save_graph(
        final_state,
        source_file=f"refined_{graph_path.stem}",
        converter_id=final_state.get("converter_id", "unknown"),
    )
    store.update_status(output_path, "refined")
    print(f"Refined graph saved to: {output_path}")


if __name__ == "__main__":
    main()
