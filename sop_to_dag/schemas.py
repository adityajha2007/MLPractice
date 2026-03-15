"""ALL Pydantic models + TypedDicts for the SOP-to-DAG system.

Sections:
  - Graph models (WorkflowNode, ExtractorOutput, RefineFeedback)
  - Intermediate models (ProcedureCard, PseudocodeBlock, etc.)
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


class RefineFeedback(BaseModel):
    """Feedback from the analyser about graph completeness."""

    is_complete: bool
    missing_branches: List[str] = Field(
        description="List of specific 'If X then Y' rules missing from the graph."
    )


# ---------------------------------------------------------------------------
# Intermediate models (pipeline stages)
# ---------------------------------------------------------------------------


class DecisionGate(BaseModel):
    """A decision point identified in the SOP."""

    condition: str = Field(description="The condition being evaluated.")
    true_branch: str = Field(description="What happens when condition is true.")
    false_branch: str = Field(description="What happens when condition is false.")


class Phase(BaseModel):
    """A major phase or section of the SOP."""

    name: str
    description: str
    decision_points: List[str] = Field(
        default_factory=list,
        description="Key decisions within this phase.",
    )
    sub_steps: List[str] = Field(
        default_factory=list,
        description="Ordered sub-steps within this phase.",
    )


class ProcedureCard(BaseModel):
    """Stage 1 output: high-level skeleton of the SOP."""

    title: str
    goal: str = Field(description="The overarching goal of this SOP.")
    major_phases: List[Phase]
    decision_gates: List[DecisionGate] = Field(
        description="All decision points across the entire SOP."
    )


class ActionStep(BaseModel):
    """A single sequential action."""

    action: str = Field(description="What to do.")
    target: Optional[str] = Field(
        default=None, description="Entity or system acted upon."
    )


class ConditionalBlock(BaseModel):
    """An IF/ELSE decision block."""

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
    """A named procedure with pre/postconditions and ordered steps."""

    name: str
    preconditions: List[str] = Field(default_factory=list)
    steps: List[StepItem]
    postconditions: List[str] = Field(default_factory=list)


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
    iteration: int
    is_complete: bool
    converter_id: str
    analysis_report: str
    enriched_chunks: List[dict]
    vector_store: Any
    entity_map: List[dict]


class RAGPrepState(TypedDict):
    """LangGraph state for the preprocessing pipeline."""

    document: str
    chunks: List[dict]
    vector_store: Any
    enriched_chunks: List[dict]
    entity_map: List[dict]
