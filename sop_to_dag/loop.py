"""LangGraph StateGraph: Analyser <-> Refiner self-refinement loop.

Flow:
    START -> analyse -> conditional:
      if is_complete OR iteration >= MAX_ITERATIONS -> END
      else -> refine -> analyse (loop)

Only used for the main pipeline converter, NOT for alternatives.
"""

from langgraph.graph import END, StateGraph

from sop_to_dag.analyser import analyse
from sop_to_dag.refiner import refine
from sop_to_dag.schemas import GraphState
from sop_to_dag.storage import GraphStore

MAX_ITERATIONS = 10


def _build_graph(store: GraphStore | None = None) -> StateGraph:
    """Build the LangGraph StateGraph for the refinement loop."""

    def analyse_node(state: GraphState) -> GraphState:
        return analyse(state)

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
            return "end"
        if state.get("iteration", 0) >= MAX_ITERATIONS:
            return "end"
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
) -> GraphState:
    """Execute the refinement loop on a graph state.

    Args:
        graph_state: Initial graph state (from a converter).
        max_iterations: Safety cap on iterations.
        store: Optional GraphStore for intermediate persistence.

    Returns:
        Final refined GraphState.
    """
    global MAX_ITERATIONS
    MAX_ITERATIONS = max_iterations

    graph = _build_graph(store)
    app = graph.compile()

    final_state = app.invoke(graph_state)
    return final_state
