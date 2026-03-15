"""Main 3-stage pipeline converter: TopDown -> CodeBased -> GraphBased.

Inline prompts + _llm_extract helper to deduplicate LLM wrapper boilerplate.
Graph-based stage assigns confidence labels to edges.
"""

from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from sop_to_dag.models import get_model
from sop_to_dag.schemas import (
    ExtractorOutput,
    ProcedureCard,
    PseudocodeBlock,
)

# ---------------------------------------------------------------------------
# Inline prompts
# ---------------------------------------------------------------------------

_TOP_DOWN_SYSTEM = """\
You are a Senior Process Architect specializing in Standard Operating Procedures.

Your task is to extract a high-level macro-skeleton from the provided SOP text.
Identify:
1. The overarching GOAL of the procedure
2. The MAJOR PHASES in sequential order
3. All DECISION GATES (if/then branching points)

For each phase, identify:
- A clear name and description
- Key decision points within the phase
- Ordered sub-steps

For each decision gate, identify:
- The exact condition being evaluated
- What happens when the condition is TRUE
- What happens when the condition is FALSE

Be exhaustive — capture every decision point and phase mentioned in the SOP.
Do NOT invent steps that are not in the source text.
"""

_TOP_DOWN_HUMAN = """\
Extract the macro-skeleton from this SOP:

---
{source_text}
---

Return a ProcedureCard with title, goal, major_phases, and decision_gates.
"""

_CODE_BASED_SYSTEM = """\
You are a Process Logic Engineer. You translate structured procedure descriptions
into precise pseudocode representations.

Given a ProcedureCard (high-level skeleton) and the original SOP text, produce
a PseudocodeBlock containing one or more Procedures.

Rules:
1. Every sequential action becomes an ActionStep
2. Every decision point becomes a ConditionalBlock with IF/ELSE branches
3. Preserve the exact conditions from the source text — do not simplify or rephrase
4. Each Procedure should have clear preconditions and postconditions
5. Nested conditionals are allowed (ConditionalBlock within ConditionalBlock)
6. Cover ALL paths — every branch must lead somewhere
7. Reference the original SOP text to ensure no steps are missed
"""

_CODE_BASED_HUMAN = """\
Convert this procedure skeleton into structured pseudocode.

## Procedure Card
{procedure_card}

## Original SOP Text (for reference)
{source_text}

Return a PseudocodeBlock with all procedures, their steps, and conditions.
"""

_GRAPH_BASED_SYSTEM = """\
You are a Senior Process Architect. Convert the provided pseudocode into a
precise JSON Workflow Graph.

### NODE IDENTITY & STRUCTURE
- **nodeIds**: Use descriptive, snake_case IDs (e.g., 'check_fraud_score').
  NEVER use generic IDs like 'node_1'.
- **Granularity**: Break complex steps into single logical units.
- **Strict Connectivity**: Every 'next' or option value must match exactly
  one 'id' present in your node list.

### NODE TYPES & SCHEMA RULES
Your output must strictly adhere to these types:

- **"question"**: Use for decision points (Yes/No, Branch A/B).
  - REQUIRED: 'options' dictionary mapping "answer_label" -> "target_node_id".
  - CONSTRAINTS: Do NOT use the 'next' field for questions.

- **"instruction"**: Use for actions, tasks, or linear steps.
  - REQUIRED: 'next' field containing exactly one target node ID.
  - CONSTRAINTS: Never use lists or pipe-separated strings in 'next'.

- **"terminal"**: Use for the absolute end of a flow or a handoff.
  - CONSTRAINTS: Both 'next' and 'options' SHOULD be null.

- **"reference"**: Use for cross-references or external lookups.
  - REQUIRED: 'next' field pointing to the node after the lookup.
  - CONSTRAINTS: 'text' field pointing to where the flow resumes after
    looking up information.

### CONFIDENCE LABELS
For EACH node, assign a 'confidence' level for its outgoing edges:
- **"high"**: Edge directly stated in SOP text ("If X, then do Y")
- **"medium"**: Edge inferred from context but not explicitly stated
- **"low"**: Edge is a guess to maintain connectivity (e.g., filling a gap)

### EXTERNAL REFERENCES
If the SOP mentions an external guide (e.g., 'Refer to the XYZ Guide'),
extract that guide name into the 'external_ref' field.

### ENTRY POINT
The very first node in the workflow MUST have the id 'start'.

### MAPPING RULES
- ActionStep -> "instruction" node
- ConditionalBlock -> "question" node (options map to branches)
- Terminal/end states -> "terminal" node
- External lookups -> "reference" node
"""

_GRAPH_BASED_HUMAN = """\
Convert this pseudocode into a JSON Workflow Graph.

## Pseudocode
{pseudocode}

## Original SOP Text (for completeness verification)
{source_text}

{enrichment_context}

Return an ExtractorOutput with reasoning and all_nodes.
Assign a confidence level (high/medium/low) to each node based on how
directly its outgoing edges are supported by the SOP text.
"""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _llm_extract(
    stage: str,
    system: str,
    human: str,
    output_schema: type[BaseModel],
    **format_kwargs,
) -> BaseModel:
    """Deduplicated LLM call: create messages, invoke structured output."""
    llm = get_model(stage)
    structured_llm = llm.with_structured_output(output_schema)
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=human.format(**format_kwargs)),
    ]
    return structured_llm.invoke(messages)


# ---------------------------------------------------------------------------
# PipelineConverter
# ---------------------------------------------------------------------------


class PipelineConverter:
    """Three-stage converter: TopDown -> CodeBased -> GraphBased.

    Produces a nodes dict with confidence labels on each node.
    """

    converter_id = "pipeline_3stage"

    def convert(
        self,
        source_text: str,
        enriched_chunks: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """Run the full 3-stage pipeline.

        Args:
            source_text: Raw SOP text (or concatenated enriched chunks).
            enriched_chunks: Optional enriched chunks from preprocessing.

        Returns:
            Dict mapping node_id to node data dicts.
        """
        # Build enrichment context if available
        enrichment_context = ""
        if enriched_chunks:
            context_parts = []
            for ec in enriched_chunks:
                if ec.get("retrieved_context"):
                    context_parts.append(
                        f"## Context for Chunk {ec['chunk_id']}\n"
                        f"{ec['retrieved_context']}"
                    )
            if context_parts:
                enrichment_context = (
                    "## Cross-Reference Context (from RAG enrichment)\n"
                    + "\n\n".join(context_parts)
                )

        # Stage 1: Raw text -> macro-skeleton
        procedure_card: ProcedureCard = _llm_extract(
            stage="top_down",
            system=_TOP_DOWN_SYSTEM,
            human=_TOP_DOWN_HUMAN,
            output_schema=ProcedureCard,
            source_text=source_text,
        )

        # Stage 2: Skeleton + text -> pseudocode
        pseudocode: PseudocodeBlock = _llm_extract(
            stage="code_based",
            system=_CODE_BASED_SYSTEM,
            human=_CODE_BASED_HUMAN,
            output_schema=PseudocodeBlock,
            procedure_card=procedure_card.model_dump_json(indent=2),
            source_text=source_text,
        )

        # Stage 3: Pseudocode + text -> JSON graph (with confidence labels)
        extractor_output: ExtractorOutput = _llm_extract(
            stage="graph_based",
            system=_GRAPH_BASED_SYSTEM,
            human=_GRAPH_BASED_HUMAN,
            output_schema=ExtractorOutput,
            pseudocode=pseudocode.model_dump_json(indent=2),
            source_text=source_text,
            enrichment_context=enrichment_context,
        )

        # Convert to nodes dict (node_id -> node_data)
        nodes: Dict[str, Any] = {}
        for node in extractor_output.all_nodes:
            nodes[node.id] = node.model_dump()

        return nodes
