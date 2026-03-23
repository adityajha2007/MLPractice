"""LangGraph StateGraph: Analyser <-> Refiner self-refinement loop.

Flow:
    START -> analyse -> conditional:
      if is_complete OR iteration >= MAX_ITERATIONS -> END
      else -> refine -> analyse (loop)

Only used for the main pipeline converter, NOT for alternatives.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from langgraph.graph import END, StateGraph

from sop_to_dag.analyser import analyse
from sop_to_dag.refiner import refine
from sop_to_dag.schemas import GraphState
from sop_to_dag.storage import GraphStore

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10


def _dump_iteration(dump_dir: Path, iteration: int, state: GraphState) -> None:
    """Dump graph state and analysis report for a single refinement iteration."""
    (dump_dir / f"refine_iter{iteration}_graph.json").write_text(
        json.dumps(state["nodes"], indent=2)
    )
    (dump_dir / f"refine_iter{iteration}_report.txt").write_text(
        state.get("analysis_report", "")
    )
    logger.info("  [DUMP] Refinement iteration %d -> %s", iteration, dump_dir)


def _build_graph(
    store: GraphStore | None = None,
    max_iterations: int = MAX_ITERATIONS,
    dump_dir: Optional[Path] = None,
) -> StateGraph:
    """Build the LangGraph StateGraph for the refinement loop."""

    def analyse_node(state: GraphState) -> GraphState:
        result = analyse(state)
        if dump_dir:
            _dump_iteration(dump_dir, result.get("iteration", 0), result)
        return result

    def refine_node(state: GraphState) -> GraphState:
        state = refine(state)
        if store:
            store.save_graph(
                state,
                source_file=f"refinement_iter_{state['iteration']}",
                converter_id=state.get("converter_id", "unknown"),
            )
        return state

    def should_continue(state: GraphState) -> str:
        """Routing function: decide whether to refine or stop."""
        if state.get("is_complete", False):
            logger.info("[LOOP] Graph is complete — stopping refinement.")
            return "end"
        if state.get("iteration", 0) >= max_iterations:
            logger.warning("[LOOP] Max iterations (%d) reached — stopping refinement.", max_iterations)
            return "end"
        logger.info("[LOOP] Graph incomplete — routing to refiner (iteration %d)...",
                     state.get("iteration", 0) + 1)
        return "refine"

    graph = StateGraph(GraphState)

    graph.add_node("analyse", analyse_node)
    graph.add_node("refine", refine_node)

    graph.set_entry_point("analyse")

    graph.add_conditional_edges(
        "analyse",
        should_continue,
        {
            "refine": "refine",
            "end": END,
        },
    )

    graph.add_edge("refine", "analyse")

    return graph


def run_refinement(
    graph_state: GraphState,
    max_iterations: int = MAX_ITERATIONS,
    store: GraphStore | None = None,
    dump_dir: Optional[str] = None,
) -> GraphState:
    """Execute the refinement loop on a graph state.

    Args:
        graph_state: Initial graph state (from a converter).
        max_iterations: Safety cap on iterations.
        store: Optional GraphStore for intermediate persistence.
        dump_dir: Optional directory to dump per-iteration state files.

    Returns:
        Final refined GraphState.
    """
    logger.info("[LOOP] Starting refinement loop (max %d iterations, %d nodes)...",
                 max_iterations, len(graph_state["nodes"]))

    dump_path = Path(dump_dir) if dump_dir else None
    graph = _build_graph(store, max_iterations=max_iterations, dump_dir=dump_path)
    app = graph.compile()

    final_state = app.invoke(graph_state)

    logger.info("[LOOP] Refinement loop finished after %d iterations. Complete: %s. Final nodes: %d.",
                 final_state.get("iteration", 0),
                 final_state.get("is_complete", False),
                 len(final_state["nodes"]))
    return final_state
