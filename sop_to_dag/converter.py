"""Chunk-driven 2-tier pipeline converter.

Step 0 (Overview):       LLM extracts lightweight ProcedureOverview (goal, phases, gates)
Step 1 (Per-Chunk):      LLM converts each enriched chunk -> Procedure with carryover
Step 2 (Merge):          LLM reconciles assembled pseudocode with original SOP text
Step 3 (Graph Build):    DETERMINISTIC walk of merged pseudocode -> WorkflowNodes

The LLM's job ends after the merge. Step 3 is pure compilation — no hallucination.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from sop_to_dag.models import get_model
from sop_to_dag.schemas import (
    ActionStep,
    ConditionalBlock,
    Procedure,
    ProcedureOverview,
    PseudocodeBlock,
    StepItem,
)

# ---------------------------------------------------------------------------
# Inline prompts
# ---------------------------------------------------------------------------

_OVERVIEW_SYSTEM = """\
You are a Senior Process Architect. Extract a LIGHTWEIGHT overview from the
provided SOP text. This is NOT a full skeleton — just the global map.

Identify ONLY:
1. The overarching GOAL of the procedure (one sentence)
2. The ordered list of MAJOR PHASE / SECTION NAMES
3. Any CROSS-PHASE DECISION GATES — decisions in one phase that affect the
   flow in a later phase (e.g., "if fraud detected in intake, escalate in
   resolution")

Do NOT list sub-steps, do NOT enumerate every decision point.
Keep the output minimal — it serves as a table of contents for downstream steps.
"""

_OVERVIEW_HUMAN = """\
Extract a lightweight overview from this SOP:

---
{source_text}
---

Return a ProcedureOverview with goal, phase_names, and cross_phase_decisions.
"""

_CHUNK_TO_PSEUDOCODE_SYSTEM = """\
You are a Process Logic Engineer. You convert a SINGLE section/chunk of an SOP
into a precise Procedure (structured pseudocode).

You receive:
- An overview of the full SOP (goal, phase list, cross-phase decisions) for context
- The enriched chunk text to convert
- Cross-reference context retrieved for this chunk (may be empty)
- The previous chunk's Procedure output (may be empty for the first chunk)

Rules:
1. Every sequential action becomes an ActionStep
2. Every decision point becomes a ConditionalBlock with IF/ELSE branches
3. Preserve the EXACT conditions from the source text — do not simplify or rephrase
4. The Procedure should have clear preconditions and postconditions
5. Nested conditionals are allowed (ConditionalBlock within ConditionalBlock)
6. Cover ALL paths — every branch must lead somewhere
7. Use cross-reference context to resolve dangling references, but do NOT
   invent steps that are not in the chunk text or its cross-references
8. Ensure continuity with the previous chunk's output — if the prior procedure
   ends with an action or branch that leads into this chunk, set appropriate
   preconditions
"""

_CHUNK_TO_PSEUDOCODE_HUMAN = """\
Convert this SOP chunk into a structured Procedure.

## SOP Overview (global context)
{overview}

## Chunk Text
{chunk_text}

## Cross-Reference Context
{retrieved_context}

## Previous Chunk's Procedure (for continuity)
{prior_procedure}

Return a single Procedure with name, preconditions, steps, and postconditions.
"""

_MERGE_SYSTEM = """\
You are a Detail Reconciliation Specialist. Your job is to compare a structured
pseudocode representation against the ORIGINAL SOP document and ensure nothing
is missing.

You receive:
1. A PseudocodeBlock (structured procedures with steps and conditions)
2. The original SOP text

Your task: produce a MERGED PseudocodeBlock that contains EVERYTHING from the
original pseudocode PLUS any granular details from the SOP that were missed.

Rules:
1. NEVER remove or simplify existing steps — only ADD missing details
2. If the SOP mentions a specific system, form, team, threshold, or reference
   that the pseudocode glossed over, add it as an ActionStep in the right place
3. If a conditional branch in the SOP has more detail than the pseudocode
   captures (e.g., specific codes, team names, document references), expand it
4. If the SOP mentions external references (guides, documents), ensure they
   appear as ActionSteps with the target field set to the reference name
5. Preserve the exact wording from the SOP for conditions and actions — the
   graph builder downstream depends on this text for node descriptions
6. Every piece of information in the original SOP MUST appear somewhere in
   the output pseudocode. Missing even one detail is a failure.
"""

_MERGE_HUMAN = """\
## Current Pseudocode
{pseudocode}

## Original SOP Text (the ground truth — NOTHING here should be lost)
{source_text}

{enrichment_context}

Compare line by line. Add any missing granular details to produce the final
merged PseudocodeBlock. If nothing is missing, return the pseudocode as-is.
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


def _to_snake_case(text: str) -> str:
    """Convert a text description to a snake_case node ID."""
    # Take first ~6 words, lowercase, replace non-alphanum with underscores
    words = re.sub(r"[^a-zA-Z0-9\s]", "", text).lower().split()[:6]
    slug = "_".join(words)
    return slug or "node"


# ---------------------------------------------------------------------------
# Stage 3: Deterministic graph builder
# ---------------------------------------------------------------------------

# Terminal detection: expanded keyword list for robust matching
_TERMINAL_KEYWORDS = [
    "end processing", "end of process", "end of procedure",
    "end the process", "end the procedure", "process complete",
    "processing complete", "procedure complete", "no further action",
    "stop processing", "workflow complete", "close the case",
    "mark as complete", "mark as done", "mark onboarding as complete",
    "end of onboarding",
]

# Result type: (head_node_id, tail_node_ids_needing_wiring)
_BuildResult = Tuple[Optional[str], List[str]]


class _GraphBuilder:
    """Deterministic walker: PseudocodeBlock -> Dict[str, node_data].

    Walks the pseudocode tree and emits WorkflowNodes:
      ActionStep       -> instruction node (or reference/terminal via detection)
      ConditionalBlock -> question node (options map to branch heads)

    Key design: every emit function returns (head_id, tail_ids).
      - head_id:  the entry point of the emitted subgraph
      - tail_ids: nodes whose `next` is None and needs to be wired by the caller

    This lets branches CONVERGE: after a conditional, both branches' tails get
    wired to whatever step comes next in the parent list. No orphans.

    No LLM calls — this is pure compilation.
    """

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self._id_counter: Dict[str, int] = {}

    def _unique_id(self, base: str) -> str:
        """Generate a unique snake_case ID, appending _2, _3, etc. on clash."""
        if base not in self.nodes and base not in self._id_counter:
            self._id_counter[base] = 1
            return base
        self._id_counter.setdefault(base, 1)
        self._id_counter[base] += 1
        return f"{base}_{self._id_counter[base]}"

    # ------------------------------------------------------------------
    # Top-level entry
    # ------------------------------------------------------------------

    def build(self, pseudocode: PseudocodeBlock) -> Dict[str, Any]:
        """Walk all procedures and return the complete nodes dict."""
        # Build a unified step list with preconditions/postconditions
        all_steps: List[StepItem] = []
        for proc in pseudocode.procedures:
            # Emit preconditions as action steps
            for pre in proc.preconditions:
                all_steps.append(
                    StepItem(action_step=ActionStep(action=f"Precondition: {pre}"))
                )
            all_steps.extend(proc.steps)
            # Emit postconditions as action steps
            for post in proc.postconditions:
                all_steps.append(
                    StepItem(action_step=ActionStep(action=f"Postcondition: {post}"))
                )

        if not all_steps:
            self._add_terminal("end", "No steps found in procedure.")
            return self.nodes

        _, tails = self._walk_steps(all_steps, force_first_id="start")

        # Any remaining tails at the top level need a terminal
        self._terminate_tails(tails)

        # Safety net: catch any dangling instruction/reference nodes
        self._ensure_terminals()

        return self.nodes

    # ------------------------------------------------------------------
    # Core: walk a step list, returning (head, tails)
    # ------------------------------------------------------------------

    def _walk_steps(
        self,
        steps: List[StepItem],
        force_first_id: Optional[str] = None,
    ) -> _BuildResult:
        """Walk a list of StepItems, chaining them sequentially.

        Returns (head_id, tail_ids):
          - head_id:  ID of the first node (entry point of this subgraph)
          - tail_ids: IDs of nodes whose `next` is still None (exit points)
        """
        if not steps:
            return None, []

        step_results: List[_BuildResult] = []

        for i, step in enumerate(steps):
            use_id = force_first_id if (i == 0 and force_first_id) else None

            if step.conditional:
                result = self._emit_conditional(step.conditional, forced_id=use_id)
            elif step.action_step:
                result = self._emit_action(step.action_step, forced_id=use_id)
            else:
                continue

            step_results.append(result)

        if not step_results:
            return None, []

        # Chain: wire each step's tails -> next step's head
        for j in range(len(step_results) - 1):
            _, prev_tails = step_results[j]
            next_head, _ = step_results[j + 1]
            if next_head is None:
                continue
            for tail_id in prev_tails:
                node = self.nodes[tail_id]
                if node["type"] in ("instruction", "reference") and node["next"] is None:
                    node["next"] = next_head

        overall_head = step_results[0][0]
        overall_tails = step_results[-1][1]
        return overall_head, overall_tails

    # ------------------------------------------------------------------
    # Emit: ActionStep -> instruction / reference / terminal
    # ------------------------------------------------------------------

    def _emit_action(
        self, action: ActionStep, forced_id: Optional[str] = None
    ) -> _BuildResult:
        """Emit an instruction node for an ActionStep.

        Returns (node_id, [node_id]) — the node is its own tail.
        Terminals return (node_id, []) — nothing to wire after them.
        """
        base_id = forced_id or _to_snake_case(action.action)
        node_id = forced_id or self._unique_id(base_id)

        # Build node text, incorporating target metadata if present
        text = action.action
        if action.target:
            text = f"{action.action} (target: {action.target})"

        # Detect external references
        external_ref = None
        node_type = "instruction"

        ref_match = re.search(
            r"[Rr]efer(?:\s+to)?[:\s]+[\"']?([^\"'\n.]+)", action.action
        )
        if ref_match:
            external_ref = ref_match.group(1).strip()
            node_type = "reference"

        # Detect terminal-like actions
        lower = action.action.lower()
        if any(kw in lower for kw in _TERMINAL_KEYWORDS):
            tid = self._add_terminal(node_id, text)
            return tid, []  # terminal has no tails

        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "text": text,
            "next": None,  # wired by the caller
            "options": None,
            "external_ref": external_ref,
            "confidence": "high",
        }
        return node_id, [node_id]

    # ------------------------------------------------------------------
    # Emit: ConditionalBlock -> question node + branch subgraphs
    # ------------------------------------------------------------------

    def _emit_conditional(
        self, cond: ConditionalBlock, forced_id: Optional[str] = None
    ) -> _BuildResult:
        """Emit a question node + recursively walk branches.

        Returns (question_id, true_tails + false_tails) so that both
        branches' exit points can be wired to whatever comes AFTER
        this conditional in the parent step list.
        """
        base_id = forced_id or _to_snake_case(cond.condition)
        node_id = forced_id or self._unique_id(base_id)

        # Walk true branch
        true_head, true_tails = self._walk_steps(cond.if_true)
        if true_head is None:
            true_head = self._add_terminal(
                self._unique_id("end_true"), "End (true branch)."
            )
            true_tails = []

        # Walk false branch
        if cond.if_false:
            false_head, false_tails = self._walk_steps(cond.if_false)
        else:
            false_head, false_tails = None, []
        if false_head is None:
            false_head = self._add_terminal(
                self._unique_id("end_false"), "End (false branch)."
            )
            false_tails = []

        self.nodes[node_id] = {
            "id": node_id,
            "type": "question",
            "text": cond.condition,
            "next": None,
            "options": {"Yes": true_head, "No": false_head},
            "external_ref": None,
            "confidence": "high",
        }

        # Tails = union of both branches' tails (for convergence wiring)
        return node_id, true_tails + false_tails

    # ------------------------------------------------------------------
    # Terminal helpers
    # ------------------------------------------------------------------

    def _add_terminal(self, node_id: str, text: str) -> str:
        """Add a terminal node. Returns its ID."""
        nid = self._unique_id(node_id) if node_id in self.nodes else node_id
        self.nodes[nid] = {
            "id": nid,
            "type": "terminal",
            "text": text,
            "next": None,
            "options": None,
            "external_ref": None,
            "confidence": "high",
        }
        return nid

    def _terminate_tails(self, tails: List[str]) -> None:
        """Wire remaining tail nodes to a terminal (low confidence — inferred)."""
        if not tails:
            return
        dangling = [
            t for t in tails
            if self.nodes[t]["type"] in ("instruction", "reference")
            and self.nodes[t].get("next") is None
        ]
        if dangling:
            term_id = self._add_terminal(self._unique_id("end"), "End of procedure.")
            self.nodes[term_id]["confidence"] = "low"
            for nid in dangling:
                self.nodes[nid]["next"] = term_id

    def _ensure_terminals(self):
        """Safety net: any instruction/reference with next=None gets a terminal."""
        dangling = [
            nid
            for nid, data in self.nodes.items()
            if data["type"] in ("instruction", "reference") and data.get("next") is None
        ]
        if dangling:
            term_id = self._add_terminal(
                self._unique_id("end"), "End of procedure."
            )
            self.nodes[term_id]["confidence"] = "low"
            for nid in dangling:
                self.nodes[nid]["next"] = term_id


# ---------------------------------------------------------------------------
# PipelineConverter
# ---------------------------------------------------------------------------


class PipelineConverter:
    """Chunk-driven pipeline: Overview -> Per-Chunk Pseudocode -> Merge -> Graph.

    Steps 0-2 use LLMs. Step 3 is deterministic compilation.
    """

    converter_id = "pipeline_3stage"

    def convert(
        self,
        source_text: str,
        enriched_chunks: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """Run the full pipeline.

        Args:
            source_text: Raw SOP text.
            enriched_chunks: Optional enriched chunks from preprocessing.

        Returns:
            Dict mapping node_id to node data dicts.
        """
        # Step 0: Lightweight overview extraction
        overview: ProcedureOverview = _llm_extract(
            stage="top_down",
            system=_OVERVIEW_SYSTEM,
            human=_OVERVIEW_HUMAN,
            output_schema=ProcedureOverview,
            source_text=source_text,
        )

        # Step 1: Sequential per-chunk pseudocode with carryover
        procedures: List[Procedure] = []
        prior_procedure_json = ""

        chunks_to_process = enriched_chunks if enriched_chunks else [
            {"chunk_id": 0, "chunk_text": source_text, "retrieved_context": ""}
        ]

        for ec in chunks_to_process:
            procedure: Procedure = _llm_extract(
                stage="code_based",
                system=_CHUNK_TO_PSEUDOCODE_SYSTEM,
                human=_CHUNK_TO_PSEUDOCODE_HUMAN,
                output_schema=Procedure,
                overview=overview.model_dump_json(indent=2),
                chunk_text=ec["chunk_text"],
                retrieved_context=ec.get("retrieved_context", ""),
                prior_procedure=prior_procedure_json,
            )
            procedures.append(procedure)
            prior_procedure_json = procedure.model_dump_json(indent=2)

        pseudocode = PseudocodeBlock(procedures=procedures)

        # Step 2: Merge with original SOP for granular detail guarantee
        enrichment_context = ""
        if enriched_chunks:
            context_parts = [
                f"## Context for Chunk {ec['chunk_id']}\n{ec['retrieved_context']}"
                for ec in enriched_chunks
                if ec.get("retrieved_context")
            ]
            if context_parts:
                enrichment_context = (
                    "## Cross-Reference Context\n"
                    + "\n\n".join(context_parts)
                )

        merged_pseudocode: PseudocodeBlock = _llm_extract(
            stage="code_based",
            system=_MERGE_SYSTEM,
            human=_MERGE_HUMAN,
            output_schema=PseudocodeBlock,
            pseudocode=pseudocode.model_dump_json(indent=2),
            source_text=source_text,
            enrichment_context=enrichment_context,
        )

        # Step 3: Deterministic compilation -> WorkflowNode graph
        builder = _GraphBuilder()
        return builder.build(merged_pseudocode)
