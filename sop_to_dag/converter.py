"""Chunk-driven 2-tier pipeline converter.

Step 0 (Overview):       LLM extracts lightweight ProcedureOverview (goal, phases, gates)
Step 1 (Per-Chunk):      LLM converts each enriched chunk -> Procedure with carryover
Step 2 (Merge):          LLM reconciles assembled pseudocode with original SOP text
Step 3 (Graph Build):    DETERMINISTIC walk of merged pseudocode -> WorkflowNodes

The LLM's job ends after the merge. Step 3 is pure compilation — no hallucination.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from sop_to_dag.models import get_model

logger = logging.getLogger(__name__)

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
1. Every sequential action becomes an ActionStep with a short, descriptive snake_case `id`
   (2-4 words). Examples: 'start_processing', 'mark_fraud_confirmed', 'update_status_fraud'
2. Every decision point becomes a ConditionalBlock with a short `id` ending in '_question'.
   Examples: 'check_dispute_code_question', 'check_fraud_indicators_question'
3. Keep steps at a BROAD, meaningful level — each step should represent one
   significant business action or decision, NOT micro-operations. Do NOT split
   a single SOP step into multiple sub-steps unless there is a genuine decision branch.
4. Do NOT create separate nodes for preconditions, postconditions, or context-setting
   statements. Fold any relevant context into the action text of the step itself.
5. Nested conditionals are allowed (ConditionalBlock within ConditionalBlock)
6. Cover ALL paths — every branch must lead somewhere
7. Use cross-reference context to resolve dangling references, but do NOT
   invent steps that are not in the chunk text or its cross-references
8. Ensure continuity with the previous chunk's output
9. CRITICAL: If the chunk references steps/procedures from OTHER chunks (e.g.,
   "proceed to fraud resolution", "go to Step 5"), include those referenced
   steps as ActionSteps INSIDE the relevant branch of the conditional. Use the
   cross-reference context and overview to understand what those steps entail.
   Do NOT leave branches with just a "go to X" placeholder — expand them with
   the actual actions from the referenced step/section.
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

Return a single Procedure with name and steps. Keep steps broad — one step per
meaningful action or decision. Every ActionStep and ConditionalBlock MUST have a
short, descriptive `id` field.
"""

_MERGE_SYSTEM = """\
You are a Process Quality Reviewer. Your job is to compare a structured
pseudocode representation against the ORIGINAL SOP document and ensure
correctness, completeness, and proper cross-procedure linking.

You receive:
1. A PseudocodeBlock with MULTIPLE procedures (one per SOP chunk)
2. The original SOP text

Your PRIMARY task: produce a FINAL PseudocodeBlock that merges all procedures
into ONE UNIFIED PROCEDURE with a single connected flow. The downstream graph
builder chains procedures sequentially, so if the SOP has non-linear branches
(e.g., "if fraud → go to resolution step"), these MUST be represented as nested
ConditionalBlocks, NOT as separate procedures.

Rules:
1. MERGE all procedures into ideally ONE procedure (or as few as possible).
   If procedure A's branch says "go to fraud resolution" and procedure C has
   the fraud resolution steps, those steps MUST appear INSIDE procedure A's
   conditional branch — not as a separate procedure.
2. Verify that every DECISION PATH in the SOP is represented — no missing branches
3. Verify that key references to external guides/documents are captured
4. CONSOLIDATE redundant or overly granular steps — merge micro-steps into
   broader meaningful actions.
5. Do NOT add new sub-steps, preconditions, postconditions, or context-setting
   nodes. Stay at the same level of abstraction as the SOP.
6. Keep every ActionStep and ConditionalBlock `id` short and descriptive
   (2-4 words, snake_case). Question IDs should end with '_question'.
7. Remove any steps that are purely informational context rather than actions.
"""

_MERGE_HUMAN = """\
## Current Pseudocode (one procedure per chunk — needs merging)
{pseudocode}

## Original SOP Text (the ground truth)
{source_text}

{enrichment_context}

IMPORTANT: Merge all procedures into ONE unified procedure. If a branch in one
procedure references steps from another procedure, inline those steps into the
branch. The graph builder processes procedures sequentially — non-linear jumps
will break unless they are nested inside conditionals. Return the final
PseudocodeBlock — simpler is better as long as all paths are correctly connected.
"""

_SUMMARIZE_CONTEXT_SYSTEM = """\
You are a concise technical summarizer. Summarize the cross-reference context
below into a compact form that preserves ALL:
- Entity names, system names, team names, and document/guide references
- Decision-relevant details (codes, thresholds, conditions)
- Cross-chunk relationships (which chunks reference which other chunks)

Remove redundancy, verbose descriptions, and filler. Keep it factual and dense.
Output plain text, not structured data.
"""

_SUMMARIZE_CONTEXT_HUMAN = """\
Summarize this cross-reference context concisely:

{context}
"""

_PAIRWISE_MERGE_SYSTEM = """\
You are a Process Linker. You receive TWO procedures and must merge them into
ONE unified procedure that preserves the complete flow.

Rules:
1. Identify how the two procedures connect — look for branches in Procedure A
   that reference content from Procedure B, or sequential flow from A into B.
2. If Procedure A has a branch that should lead into Procedure B's steps,
   INLINE those steps into the branch's step list.
3. If the flow is purely sequential (A finishes, then B starts), concatenate
   A's steps followed by B's steps into one procedure.
4. Preserve ALL decision branches and actions — do not drop any paths.
5. CONSOLIDATE any duplicate or redundant steps that appear in both procedures.
6. Keep every ActionStep and ConditionalBlock `id` short and descriptive
   (2-4 words, snake_case). Question IDs should end with '_question'.
"""

_PAIRWISE_MERGE_HUMAN = """\
## Procedure A (already merged from earlier chunks)
{procedure_a}

## Procedure B (next chunk to integrate)
{procedure_b}

## SOP Overview (for context on how these connect)
{overview}

Merge these two procedures into ONE procedure. Inline Procedure B's steps into
the correct location within Procedure A's flow. Return a PseudocodeBlock with
exactly ONE procedure.
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
    """Convert a text description to a snake_case node ID (fallback)."""
    # Take first ~4 words, lowercase, replace non-alphanum with underscores
    words = re.sub(r"[^a-zA-Z0-9\s]", "", text).lower().split()[:4]
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
        all_steps: List[StepItem] = []
        for proc in pseudocode.procedures:
            all_steps.extend(proc.steps)

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
        base_id = forced_id or action.id or _to_snake_case(action.action)
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
        base_id = forced_id or cond.id or _to_snake_case(cond.condition)
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
        dump_dir: Optional[str] = None,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """Run the full pipeline.

        Args:
            source_text: Raw SOP text.
            enriched_chunks: Optional enriched chunks from preprocessing.
            dump_dir: If provided, dump each stage's output to files in this
                      directory for inspection.
            resume: If True AND dump_dir has cached stage files, load from
                    cache instead of re-running LLM calls (saves API credits).

        Returns:
            Dict mapping node_id to node data dicts.
        """
        # Set up stage dump directory
        dump_path: Optional[Path] = None
        if dump_dir:
            dump_path = Path(dump_dir)
            dump_path.mkdir(parents=True, exist_ok=True)
            logger.info("[CONVERTER] Stage outputs dir: %s (resume=%s)", dump_path, resume)

        # Step 0: Lightweight overview extraction
        cached_overview = self._load_cache(dump_path, "stage0_overview") if resume else None
        if cached_overview:
            logger.info("[CONVERTER Stage 0/3] Loaded overview from cache.")
            overview = ProcedureOverview.model_validate_json(cached_overview)
        else:
            logger.info("[CONVERTER Stage 0/3] Extracting overview (goal, phases, gates)...")
            overview = _llm_extract(
                stage="top_down",
                system=_OVERVIEW_SYSTEM,
                human=_OVERVIEW_HUMAN,
                output_schema=ProcedureOverview,
                source_text=source_text,
            )
            if dump_path:
                self._dump_stage(dump_path, "stage0_overview", overview.model_dump_json(indent=2))

        logger.info("  Goal: %s", overview.goal)
        logger.info("  Phases: %s", overview.phase_names)
        logger.info("  Cross-phase decisions: %d", len(overview.cross_phase_decisions))

        # Step 1: Sequential per-chunk pseudocode with carryover
        cached_all_pseudo = self._load_cache(dump_path, "stage1_all_pseudocode") if resume else None
        if cached_all_pseudo:
            logger.info("[CONVERTER Stage 1/3] Loaded all pseudocode from cache.")
            pseudocode = PseudocodeBlock.model_validate_json(cached_all_pseudo)
        else:
            procedures: List[Procedure] = []
            prior_procedure_json = ""

            chunks_to_process = enriched_chunks if enriched_chunks else [
                {"chunk_id": 0, "chunk_text": source_text, "retrieved_context": ""}
            ]

            total_chunks = len(chunks_to_process)
            logger.info("[CONVERTER Stage 1/3] Per-chunk pseudocode — %d chunks to process...", total_chunks)

            for idx, ec in enumerate(chunks_to_process):
                logger.info("  Chunk %d/%d (id=%s): converting to pseudocode...",
                             idx + 1, total_chunks, ec.get("chunk_id", idx))
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
                logger.info("    Procedure '%s': %d steps", procedure.name, len(procedure.steps))

                if dump_path:
                    self._dump_stage(dump_path, f"stage1_chunk{idx}_pseudocode",
                                     procedure.model_dump_json(indent=2))

            pseudocode = PseudocodeBlock(procedures=procedures)

            if dump_path:
                self._dump_stage(dump_path, "stage1_all_pseudocode",
                                 pseudocode.model_dump_json(indent=2))

        # Step 2: Merge / consolidate
        cached_merged = self._load_cache(dump_path, "stage2_merged_pseudocode") if resume else None
        if cached_merged:
            logger.info("[CONVERTER Stage 2/3] Loaded merged pseudocode from cache.")
            merged_pseudocode = PseudocodeBlock.model_validate_json(cached_merged)
        else:
            logger.info("[CONVERTER Stage 2/3] Consolidating pseudocode with original SOP...")

            # Summarize enrichment context instead of truncating — preserves
            # key details (entity names, codes, cross-references) while fitting
            # within the context window.
            enrichment_context = ""
            if enriched_chunks:
                context_parts = [
                    f"## Context for Chunk {ec['chunk_id']}\n{ec['retrieved_context']}"
                    for ec in enriched_chunks
                    if ec.get("retrieved_context")
                ]
                if context_parts:
                    full_context = "\n\n".join(context_parts)
                    if len(full_context) > 4000:
                        logger.info("  Enrichment context is %d chars — summarizing with LLM...",
                                    len(full_context))
                        full_context = self._summarize_context(full_context)
                        logger.info("  Summarized to %d chars.", len(full_context))
                    enrichment_context = "## Cross-Reference Context\n" + full_context

            num_procedures = len(pseudocode.procedures)

            if num_procedures <= 3:
                # Small SOP — single-shot merge is safe
                merged_pseudocode = self._single_merge(
                    pseudocode, source_text, enrichment_context
                )
            else:
                # Large SOP — pairwise incremental merge to stay within token limits
                logger.info("  %d procedures — using pairwise merge strategy.", num_procedures)
                merged_pseudocode = self._pairwise_merge(
                    pseudocode, overview, source_text, enrichment_context
                )

            if dump_path:
                self._dump_stage(dump_path, "stage2_merged_pseudocode",
                                 merged_pseudocode.model_dump_json(indent=2))

        total_steps = sum(len(p.steps) for p in merged_pseudocode.procedures)
        logger.info("  Merged: %d procedures, %d total steps",
                     len(merged_pseudocode.procedures), total_steps)

        # Step 3: Deterministic compilation -> WorkflowNode graph
        logger.info("[CONVERTER Stage 3/3] Deterministic graph build (no LLM)...")
        builder = _GraphBuilder()
        nodes = builder.build(merged_pseudocode)

        # Log graph summary
        type_counts: Dict[str, int] = {}
        for n in nodes.values():
            t = n.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        logger.info("  Graph built: %d nodes — %s", len(nodes), type_counts)
        logger.info("[CONVERTER] Complete.")

        if dump_path:
            self._dump_stage(dump_path, "stage3_final_graph", json.dumps(nodes, indent=2))
            logger.info("  All stage outputs saved to: %s", dump_path)

        return nodes

    @staticmethod
    def _summarize_context(context: str) -> str:
        """Use LLM to summarize enrichment context, preserving key details."""
        llm = get_model("enrichment")
        messages = [
            SystemMessage(content=_SUMMARIZE_CONTEXT_SYSTEM),
            HumanMessage(content=_SUMMARIZE_CONTEXT_HUMAN.format(context=context)),
        ]
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.warning("  Context summarization failed (%s), using first 4000 chars.", e)
            return context[:4000] + "\n... (truncated due to summarization failure)"

    @staticmethod
    def _single_merge(
        pseudocode: PseudocodeBlock,
        source_text: str,
        enrichment_context: str,
    ) -> PseudocodeBlock:
        """Single-shot merge: all procedures + SOP in one LLM call."""
        try:
            return _llm_extract(
                stage="code_based",
                system=_MERGE_SYSTEM,
                human=_MERGE_HUMAN,
                output_schema=PseudocodeBlock,
                pseudocode=pseudocode.model_dump_json(indent=2),
                source_text=source_text,
                enrichment_context=enrichment_context,
            )
        except Exception as e:
            logger.warning("[CONVERTER Stage 2/3] Single merge failed (%s). "
                           "Falling back to Stage 1 pseudocode.", e)
            return pseudocode

    @staticmethod
    def _pairwise_merge(
        pseudocode: PseudocodeBlock,
        overview: ProcedureOverview,
        source_text: str,
        enrichment_context: str,
    ) -> PseudocodeBlock:
        """Incremental pairwise merge: merge procedures two at a time.

        This keeps each LLM call within token limits even for large SOPs.
        The overview provides cross-chunk context so the LLM knows how
        procedures relate to each other.
        """
        procedures = pseudocode.procedures
        if len(procedures) == 0:
            return pseudocode
        if len(procedures) == 1:
            return pseudocode

        # Start with the first procedure and incrementally merge each next one
        accumulated = procedures[0]
        overview_json = overview.model_dump_json(indent=2)

        for i, next_proc in enumerate(procedures[1:], start=2):
            logger.info("  Pairwise merge: integrating procedure %d/%d ('%s')...",
                         i, len(procedures), next_proc.name)
            try:
                merged_block: PseudocodeBlock = _llm_extract(
                    stage="code_based",
                    system=_PAIRWISE_MERGE_SYSTEM,
                    human=_PAIRWISE_MERGE_HUMAN,
                    output_schema=PseudocodeBlock,
                    procedure_a=accumulated.model_dump_json(indent=2),
                    procedure_b=next_proc.model_dump_json(indent=2),
                    overview=overview_json,
                )
                if merged_block.procedures:
                    accumulated = merged_block.procedures[0]
                else:
                    logger.warning("    Pairwise merge returned empty — keeping previous.")
            except Exception as e:
                logger.warning("    Pairwise merge failed (%s) — appending sequentially.", e)
                # Fallback: just concatenate steps (sequential chaining)
                accumulated = Procedure(
                    name=accumulated.name,
                    steps=accumulated.steps + next_proc.steps,
                )

        # Final quality pass: compare merged result against original SOP
        logger.info("  Final quality pass: verifying merged procedure against SOP...")
        final_pseudocode = PseudocodeBlock(procedures=[accumulated])
        try:
            final_pseudocode = _llm_extract(
                stage="code_based",
                system=_MERGE_SYSTEM,
                human=_MERGE_HUMAN,
                output_schema=PseudocodeBlock,
                pseudocode=final_pseudocode.model_dump_json(indent=2),
                source_text=source_text,
                enrichment_context=enrichment_context,
            )
        except Exception as e:
            logger.warning("  Final quality pass failed (%s) — using pairwise result.", e)

        return final_pseudocode

    @staticmethod
    def _load_cache(dump_dir: Optional[Path], name: str) -> Optional[str]:
        """Load a cached stage output. Returns file content or None."""
        if dump_dir is None:
            return None
        path = dump_dir / f"{name}.json"
        if path.exists():
            logger.info("  [CACHE HIT] %s <- %s", name, path)
            return path.read_text()
        return None

    @staticmethod
    def _dump_stage(dump_dir: Path, name: str, content: str) -> None:
        """Write a stage's output to a file for inspection."""
        path = dump_dir / f"{name}.json"
        path.write_text(content)
        logger.info("  [DUMP] %s -> %s", name, path)
