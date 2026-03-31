"""ALL Pydantic models + TypedDicts for the SOP-to-DAG system.

Sections:
  - Graph models (WorkflowNode, ExtractorOutput, RefineFeedback)
  - Intermediate models (ProcedureOverview, PseudocodeBlock, etc.)
  - RAG models (DocumentChunks, DependencyQueries, EnrichedChunk, etc.)
  - Entity resolution models (EntityMapping, EntityMap)
  - State dicts (GraphState, RAGPrepState)
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Graph models
# ---------------------------------------------------------------------------


class WorkflowNode(BaseModel):
    """A single node in the SOP workflow DAG."""

    id: str = Field(description="Snake_case ID. Example: 'check_fraud_score'.")
    type: Literal["question", "instruction", "terminal", "reference"] = Field(...)
    text: str = Field(...)
    next: Optional[str] = Field(default=None, description="ID of the next node.")
    options: Optional[Dict[str, str]] = Field(
        default=None, description="Map of answer->next_node_id."
    )
    external_ref: Optional[str] = Field(default=None)
    role: Optional[str] = Field(default=None, description="Who performs this action.")
    system: Optional[str] = Field(default=None, description="Software or tool used.")
    confidence: Literal["high", "medium", "low"] = Field(
        default="high",
        description=(
            "Confidence in this node's outgoing edges. "
            "'high' = directly stated in SOP, "
            "'medium' = inferred from context, "
            "'low' = guess to maintain connectivity."
        ),
    )

    @model_validator(mode="after")
    def check_question_has_options(self) -> "WorkflowNode":
        if self.type == "question" and not self.options:
            raise ValueError(
                f"Node '{self.id}': question-type nodes must have 'options'."
            )
        return self

    @model_validator(mode="after")
    def check_terminal_has_no_next(self) -> "WorkflowNode":
        if self.type == "terminal":
            self.next = None
            self.options = None
        return self

    @model_validator(mode="after")
    def check_instruction_has_next(self) -> "WorkflowNode":
        if self.type == "instruction" and not self.next:
            raise ValueError(
                f"Node '{self.id}': instruction-type nodes must have 'next'."
            )
        return self


class ExtractorOutput(BaseModel):
    """LLM output from a converter: reasoning chain + extracted nodes."""

    reasoning: str
    all_nodes: List[WorkflowNode]


class SectionCoverage(BaseModel):
    """Coverage audit for a single SOP section/step."""

    sop_text: str = Field(description="The SOP step or instruction being checked.")
    node_ids: List[str] = Field(
        default_factory=list,
        description="Graph node IDs that cover this SOP step. Empty if missing.",
    )
    status: Literal["covered", "missing"] = Field(
        description="Whether this SOP step is represented in the graph.",
    )
    gap_description: str = Field(
        default="",
        description="If missing: what specifically is not represented. Empty if covered.",
    )


class RefineFeedback(BaseModel):
    """Section-by-section completeness audit of the graph against the SOP."""

    sections: List[SectionCoverage] = Field(
        description="One entry per SOP section/step, showing coverage status.",
    )
    is_complete: bool = Field(
        description="True only if every section has status='covered'.",
    )
    missing_branches: List[str] = Field(
        description="List of specific missing items — one per gap found.",
    )


class InitialGraph(BaseModel):
    """LLM output for zero-shot SOP-to-graph conversion."""

    reasoning: str = Field(
        description=(
            "Detailed analysis of the SOP structure, decision points, branches, "
            "convergence points, and cross-section dependencies."
        )
    )
    nodes: List[WorkflowNode]


class GraphPatch(BaseModel):
    """A patch to apply to an existing workflow graph."""

    reasoning: str = Field(
        description="What this chunk adds/changes and why."
    )
    add_nodes: List[WorkflowNode] = Field(
        default_factory=list,
        description="New nodes to insert into the graph.",
    )
    modify_nodes: List[WorkflowNode] = Field(
        default_factory=list,
        description=(
            "Existing nodes with updated fields (matched by id). "
            "Include ALL fields of the node, not just changed ones."
        ),
    )
    remove_nodes: List[str] = Field(
        default_factory=list,
        description="IDs of nodes to delete from the graph.",
    )


# ---------------------------------------------------------------------------
# Intermediate models (pipeline stages)
# ---------------------------------------------------------------------------


class ProcedureOverview(BaseModel):
    """Lightweight overview of the SOP — global context for per-chunk conversion."""

    goal: str = Field(description="The overarching goal of the SOP.")
    phase_names: List[str] = Field(
        description="Ordered list of major phase/section names."
    )
    cross_phase_decisions: List[str] = Field(
        default_factory=list,
        description=(
            "Decision gates that span multiple phases "
            "(e.g., 'if fraud detected in intake, escalate in resolution')."
        ),
    )


class ActionStep(BaseModel):
    """A single sequential action."""

    id: Optional[str] = Field(
        default=None,
        description=(
            "Short snake_case ID for this step (2-4 words). "
            "Example: 'start_processing', 'update_status_fraud', 'mark_fraud_confirmed'."
        ),
    )
    action: str = Field(
        description=(
            "Self-explanatory description of what to do. Must include the system, "
            "entity, or document name directly — a reader should understand the full "
            "action from this text alone. Example: 'Update case status to Fraud "
            "Confirmed in the Case Management System'."
        )
    )
    target: Optional[str] = Field(
        default=None,
        description="Optional: entity or system acted upon (for metadata only, not displayed).",
    )


class ConditionalBlock(BaseModel):
    """An IF/ELSE decision block."""

    id: Optional[str] = Field(
        default=None,
        description=(
            "Short snake_case ID for this decision (2-4 words, ending with '_question'). "
            "Example: 'check_dispute_code_question', 'check_fraud_indicators_question'."
        ),
    )
    condition: str
    if_true: List["StepItem"] = Field(description="Steps when condition is true.")
    if_false: List["StepItem"] = Field(
        default_factory=list, description="Steps when condition is false."
    )


class StepItem(BaseModel):
    """Union wrapper: either an action or a conditional block."""

    action_step: Optional[ActionStep] = None
    conditional: Optional[ConditionalBlock] = None


class Procedure(BaseModel):
    """A named procedure with ordered steps."""

    name: str
    steps: List[StepItem]


class PseudocodeBlock(BaseModel):
    """Stage 2 output: structured pseudocode representation."""

    procedures: List[Procedure]


# Rebuild forward refs for recursive model
ConditionalBlock.model_rebuild()
StepItem.model_rebuild()


# ---------------------------------------------------------------------------
# RAG models (preprocessing)
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """A single semantic chunk of an SOP document."""

    chunk_id: int
    title: str
    text: str


class DocumentChunks(BaseModel):
    """Output of the agentic chunking step."""

    chunks: List[Chunk]


class Dependency(BaseModel):
    """A single dependency/reference query for a chunk."""

    query: str = Field(description="Search query for a dangling reference.")
    reference_text: str = Field(
        default="",
        description="The original text snippet that triggered this query.",
    )


class DependencyQueries(BaseModel):
    """Queries generated for a chunk's dangling references."""

    queries: List[Dependency]


class RetrievalGrade(BaseModel):
    """Grade for a single retrieval result."""

    is_relevant: bool
    reasoning: str


class DependencyReview(BaseModel):
    """Graded retrieval results for a chunk's dependencies."""

    grades: List[RetrievalGrade]


class EnrichedChunk(BaseModel):
    """A chunk enriched with RAG-retrieved context."""

    chunk_id: int
    chunk_text: str
    retrieved_context: str = Field(
        default="", description="Synthesized relevant context from FAISS retrieval."
    )
    generated_queries: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Entity resolution models
# ---------------------------------------------------------------------------


class EntityMapping(BaseModel):
    """Maps a canonical entity name to its aliases."""

    canonical: str = Field(description="Standardized entity name.")
    aliases: List[str] = Field(description="Variant names for this entity.")


class EntityMap(BaseModel):
    """Collection of entity mappings for the entire SOP."""

    mappings: List[EntityMapping]


# ---------------------------------------------------------------------------
# State dicts (LangGraph TypedDicts)
# ---------------------------------------------------------------------------


class GraphState(TypedDict):
    """LangGraph state flowing through the refinement loop."""

    source_text: str
    nodes: Dict[str, Any]
    feedback: str
    categorized_feedback: Dict[str, List[str]]
    iteration: int
    is_complete: bool
    converter_id: str
    analysis_report: str
    enriched_chunks: List[dict]
    vector_store: Any
    entity_map: List[dict]
    resolved_issues: List[str]
    verified_triplets: List[str]


class RAGPrepState(TypedDict):
    """LangGraph state for the preprocessing pipeline."""

    document: str
    chunks: List[dict]
    vector_store: Any
    enriched_chunks: List[dict]
    entity_map: List[dict]
