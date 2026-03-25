"""SOP-to-DAG pipeline converter.

Step 1 (Outline):      LLM converts full enriched SOP -> plain-text outline
Step 2 (Detail pass):  For each chunk, LLM compares against outline and fills gaps
Step 3 (Graph Build):  Deterministic parser+builder: text outline -> WorkflowNodes

The LLM produces plain text only — no structured output wrestling. Step 3
is pure compilation with no hallucination.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from sop_to_dag.models import get_model

logger = logging.getLogger(__name__)

from sop_to_dag.schemas import (
    ActionStep,
    ConditionalBlock,
    Procedure,
    PseudocodeBlock,
    StepItem,
)

# ---------------------------------------------------------------------------
# Inline prompts — LLM produces PLAIN TEXT, not structured output
# ---------------------------------------------------------------------------

_OUTLINE_SYSTEM = """\
You are a Process Logic Engineer. Convert an SOP document into a plain-text
numbered outline that captures the COMPLETE workflow.

FORMAT RULES (follow exactly):
- Sequential actions: numbered lines
  Example: 1. Begin processing the indirect dispute
- Decision points: prefix with "DECISION:" and phrase as a YES/NO question
  Example: 2. DECISION: Is the dispute code 183 or 186?
- Branches: indented YES: / NO: blocks under decisions
  Example:
    2. DECISION: Is the dispute code 183 or 186?
      YES:
        3. Check if borrower mentions fraud indicators
      NO:
        4. Check force memo for fraud keywords
- Nested decisions: just indent further
- Steps AFTER a decision (where both branches converge): un-indent back
  Example:
    2. DECISION: Is the item in stock?
      YES:
        3. Ship the item
      NO:
        4. Notify the customer of backorder
    5. Update order status  ← both branches lead here
- External references: include "Refer to: <document name>" in the step text

CONTENT RULES:
1. Each step must be self-explanatory — include the system, entity, or document
   name directly. A reader should understand the full action without any context.
2. Keep steps BROAD — one step per meaningful business action or decision.
   Do NOT split a single SOP step into micro-operations.
3. Cover ALL decision paths — every branch must lead somewhere.
4. Cross-section references (e.g., "proceed to fraud resolution") must be
   inlined as actual steps inside the relevant branch.
5. Preserve ALL detail from the SOP — every decision, action, and reference.
6. Do NOT add preamble, headers, or commentary. Output ONLY the numbered outline.
"""

_OUTLINE_HUMAN = """\
Convert this SOP into a numbered outline following the format rules exactly.

{enriched_sop}
"""

_DETAIL_PASS_SYSTEM = """\
You are a Detail Verification Engineer. You compare a single SOP section against
an existing workflow outline to find MISSING details.

You receive:
- The current outline (covers the full SOP so far)
- One specific SOP section/chunk to verify against the outline

Your job:
1. Read the SOP section carefully
2. Check if every action, decision, reference, and detail from that section
   is captured in the outline
3. If anything is MISSING — add it in the correct location
4. If the section references external documents/guides not in the outline — add them
5. Do NOT remove or restructure existing steps
6. Do NOT add steps that aren't in the SOP section

Return the COMPLETE updated outline (all existing steps + any additions).
If nothing is missing, return the outline unchanged.

Use the same format: numbered steps, DECISION: prefix for questions, indented
YES:/NO: blocks, self-explanatory step text. Output ONLY the outline.
"""

_DETAIL_PASS_HUMAN = """\
## Current Outline
{current_outline}

## SOP Section to Verify
{chunk_text}

Return the complete updated outline with any missing details added.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_call(stage: str, system: str, human: str, **format_kwargs) -> str:
    """Plain-text LLM call (no structured output). Returns raw text response."""
    llm = get_model(stage)
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=human.format(**format_kwargs)),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def _to_snake_case(text: str) -> str:
    """Convert a text description to a snake_case node ID."""
    words = re.sub(r"[^a-zA-Z0-9\s]", "", text).lower().split()[:4]
    slug = "_".join(words)
    return slug or "node"


def _reassemble_enriched_sop(
    source_text: str,
    enriched_chunks: Optional[List[dict]],
) -> str:
    """Reassemble enriched chunks into one document with cross-references inlined."""
    if not enriched_chunks:
        return source_text

    parts = []
    for ec in enriched_chunks:
        chunk_text = ec.get("chunk_text", "")
        retrieved = ec.get("retrieved_context", "")

        parts.append(chunk_text)
        if retrieved.strip():
            parts.append(f"[Cross-reference context: {retrieved.strip()}]")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Outline parser: plain text -> PseudocodeBlock
# ---------------------------------------------------------------------------


def parse_outline(text: str) -> PseudocodeBlock:
    """Parse a plain-text numbered outline into a PseudocodeBlock.

    Handles:
      "3. Do something"              -> ActionStep
      "4. DECISION: Is it valid?"    -> ConditionalBlock
      "  YES:" / "  NO:"             -> branches (indented content)
      Nested decisions               -> recursive parsing

    Returns a PseudocodeBlock with one Procedure.
    """
    lines = text.strip().split("\n")
    steps, _ = _parse_lines(lines, 0, base_indent=0)
    return PseudocodeBlock(procedures=[Procedure(name="SOP", steps=steps)])


def _get_indent(line: str) -> int:
    """Count leading spaces."""
    return len(line) - len(line.lstrip())


def _parse_lines(
    lines: List[str], start: int, base_indent: int
) -> Tuple[List[StepItem], int]:
    """Recursively parse lines at a given indentation level.

    Returns (steps, next_line_index).
    """
    steps: List[StepItem] = []
    i = start

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip blank lines
        if not stripped:
            i += 1
            continue

        indent = _get_indent(line)

        # If we've de-indented past our level, we're done with this block
        if indent < base_indent:
            break

        # Branch markers (YES: / NO:) are handled by the decision parser
        if stripped.upper() in ("YES:", "NO:"):
            break

        # Try to parse a numbered step
        step_match = re.match(r"^(\d+)\.\s*(.*)", stripped)
        if not step_match:
            # Non-numbered line — skip (could be stray text)
            i += 1
            continue

        step_text = step_match.group(2).strip()

        # Check if this is a DECISION
        decision_match = re.match(r"^DECISION:\s*(.*)", step_text, re.IGNORECASE)
        if decision_match:
            condition = decision_match.group(1).strip()
            cond_id = _to_snake_case(condition)
            if not cond_id.endswith("_question"):
                cond_id += "_question"

            i += 1  # move past the DECISION line

            # Parse YES: and NO: branches
            if_true, i = _parse_branch(lines, i, "YES:", indent)
            if_false, i = _parse_branch(lines, i, "NO:", indent)

            steps.append(StepItem(conditional=ConditionalBlock(
                id=cond_id,
                condition=condition,
                if_true=if_true,
                if_false=if_false,
            )))
        else:
            # Regular action step
            step_id = _to_snake_case(step_text)
            steps.append(StepItem(action_step=ActionStep(
                id=step_id,
                action=step_text,
            )))
            i += 1

    return steps, i


def _parse_branch(
    lines: List[str], start: int, marker: str, parent_indent: int
) -> Tuple[List[StepItem], int]:
    """Parse a YES: or NO: branch block.

    Looks for the marker line, then parses indented content under it.
    Returns (steps, next_line_index).
    """
    i = start

    # Skip blank lines looking for the branch marker
    while i < len(lines) and not lines[i].strip():
        i += 1

    if i >= len(lines):
        return [], i

    stripped = lines[i].strip()
    if stripped.upper() != marker.upper():
        # No branch marker found — return empty branch
        return [], i

    i += 1  # move past the marker

    # Determine the indentation of the branch content.
    # Content must be indented deeper than the parent (DECISION line).
    # If the first non-blank line is at or before parent indent, the branch is empty.
    content_indent = None
    for j in range(i, min(i + 5, len(lines))):
        if lines[j].strip():
            candidate = _get_indent(lines[j])
            if candidate > parent_indent:
                content_indent = candidate
            break

    if content_indent is None:
        return [], i

    steps, i = _parse_lines(lines, i, base_indent=content_indent)
    return steps, i


# ---------------------------------------------------------------------------
# Direct text-to-graph parser (no PseudocodeBlock intermediate)
# ---------------------------------------------------------------------------


def parse_outline_to_graph(text: str) -> Dict[str, Any]:
    """Parse a plain-text outline directly into graph nodes.

    Combines parsing and graph building in one pass — no intermediate
    PseudocodeBlock. Uses `_GraphBuilder` for node emission / convergence wiring.
    """
    builder = _GraphBuilder()
    lines = text.strip().split("\n")

    if not lines or not text.strip():
        builder._add_terminal("end", "No steps found in procedure.")
        return builder.nodes

    results, _ = _parse_and_emit(builder, lines, 0, base_indent=0)
    _, tails = _chain_results(builder, results)
    if results:
        first_head = results[0][0]
        # Re-emit the first node with "start" id if it wasn't already
        if first_head and first_head != "start" and "start" not in builder.nodes:
            # Rename the first node to "start"
            node_data = builder.nodes.pop(first_head)
            node_data["id"] = "start"
            builder.nodes["start"] = node_data
            # Update any references to the old id
            for n in builder.nodes.values():
                if n.get("next") == first_head:
                    n["next"] = "start"
                if n.get("options"):
                    for k, v in n["options"].items():
                        if v == first_head:
                            n["options"][k] = "start"
            # Update tails list
            tails = ["start" if t == first_head else t for t in tails]

    builder._terminate_tails(tails)
    builder._ensure_terminals()
    return builder.nodes


def _parse_and_emit(
    builder: "_GraphBuilder",
    lines: List[str],
    start: int,
    base_indent: int,
    force_first_id: Optional[str] = None,
) -> Tuple[List[_BuildResult], int]:
    """Recursively parse lines and emit graph nodes directly.

    Returns (list of _BuildResult, next_line_index).
    """
    results: List[_BuildResult] = []
    i = start

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        indent = _get_indent(line)

        if indent < base_indent:
            break

        if stripped.upper() in ("YES:", "NO:"):
            break

        step_match = re.match(r"^(\d+)\.\s*(.*)", stripped)
        if not step_match:
            i += 1
            continue

        step_text = step_match.group(2).strip()
        use_id = force_first_id if (not results and force_first_id) else None

        decision_match = re.match(r"^DECISION:\s*(.*)", step_text, re.IGNORECASE)
        if decision_match:
            condition = decision_match.group(1).strip()
            cond_id = _to_snake_case(condition)
            if not cond_id.endswith("_question"):
                cond_id += "_question"

            i += 1

            true_results, i = _parse_branch_and_emit(builder, lines, i, "YES:", indent)
            false_results, i = _parse_branch_and_emit(builder, lines, i, "NO:", indent)

            true_result = _chain_results(builder, true_results)
            false_result = _chain_results(builder, false_results)

            result = builder._emit_conditional(
                condition=condition,
                cond_id=cond_id,
                true_result=true_result,
                false_result=false_result,
                forced_id=use_id,
            )
            results.append(result)
        else:
            step_id = _to_snake_case(step_text)
            result = builder._emit_action(
                action_text=step_text,
                action_id=step_id,
                forced_id=use_id,
            )
            results.append(result)
            i += 1

    return results, i


def _parse_branch_and_emit(
    builder: "_GraphBuilder",
    lines: List[str],
    start: int,
    marker: str,
    parent_indent: int,
) -> Tuple[List[_BuildResult], int]:
    """Parse a YES:/NO: branch and emit nodes directly.

    Returns (list of _BuildResult, next_line_index).
    """
    i = start

    while i < len(lines) and not lines[i].strip():
        i += 1

    if i >= len(lines):
        return [], i

    stripped = lines[i].strip()
    if stripped.upper() != marker.upper():
        return [], i

    i += 1

    content_indent = None
    for j in range(i, min(i + 5, len(lines))):
        if lines[j].strip():
            candidate = _get_indent(lines[j])
            if candidate > parent_indent:
                content_indent = candidate
            break

    if content_indent is None:
        return [], i

    results, i = _parse_and_emit(builder, lines, i, base_indent=content_indent)
    return results, i


# ---------------------------------------------------------------------------
# Deterministic graph builder
# ---------------------------------------------------------------------------

_TERMINAL_KEYWORDS = [
    "end processing", "end of process", "end of procedure",
    "end the process", "end the procedure", "process complete",
    "processing complete", "procedure complete", "no further action",
    "stop processing", "workflow complete", "close the case",
    "mark as complete", "mark as done", "mark onboarding as complete",
    "end of onboarding",
]

_BuildResult = Tuple[Optional[str], List[str]]


def _chain_results(builder: "_GraphBuilder", step_results: List[_BuildResult]) -> _BuildResult:
    """Wire a sequence of _BuildResults: each step's tails → next step's head.

    Returns the overall (first_head, last_tails).
    """
    if not step_results:
        return None, []

    for j in range(len(step_results) - 1):
        _, prev_tails = step_results[j]
        next_head, _ = step_results[j + 1]
        if next_head is None:
            continue
        for tail_id in prev_tails:
            node = builder.nodes[tail_id]
            if node["type"] in ("instruction", "reference") and node["next"] is None:
                node["next"] = next_head

    return step_results[0][0], step_results[-1][1]


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
    """

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self._id_counter: Dict[str, int] = {}
        self._shared_terminal_id: Optional[str] = None

    def _unique_id(self, base: str) -> str:
        """Generate a unique snake_case ID, appending _2, _3, etc. on clash."""
        if base not in self.nodes and base not in self._id_counter:
            self._id_counter[base] = 1
            return base
        self._id_counter.setdefault(base, 1)
        self._id_counter[base] += 1
        return f"{base}_{self._id_counter[base]}"

    def build(self, pseudocode: PseudocodeBlock) -> Dict[str, Any]:
        """Walk all procedures and return the complete nodes dict."""
        all_steps: List[StepItem] = []
        for proc in pseudocode.procedures:
            all_steps.extend(proc.steps)

        if not all_steps:
            self._add_terminal("end", "No steps found in procedure.")
            return self.nodes

        _, tails = self._walk_steps(all_steps, force_first_id="start")
        self._terminate_tails(tails)
        self._ensure_terminals()
        return self.nodes

    def _walk_steps(
        self,
        steps: List[StepItem],
        force_first_id: Optional[str] = None,
    ) -> _BuildResult:
        if not steps:
            return None, []

        step_results: List[_BuildResult] = []

        for i, step in enumerate(steps):
            use_id = force_first_id if (i == 0 and force_first_id) else None

            if step.conditional:
                cond = step.conditional
                true_result = self._walk_steps(cond.if_true)
                false_result = self._walk_steps(cond.if_false) if cond.if_false else (None, [])
                result = self._emit_conditional(
                    condition=cond.condition,
                    cond_id=cond.id,
                    true_result=true_result,
                    false_result=false_result,
                    forced_id=use_id,
                )
            elif step.action_step:
                a = step.action_step
                result = self._emit_action(
                    action_text=a.action,
                    action_id=a.id,
                    target=a.target,
                    forced_id=use_id,
                )
            else:
                continue

            step_results.append(result)

        return _chain_results(self, step_results)

    def _emit_action(
        self,
        action_text: str,
        action_id: str | None = None,
        target: str | None = None,
        forced_id: str | None = None,
    ) -> _BuildResult:
        base_id = forced_id or action_id or _to_snake_case(action_text)
        node_id = forced_id or self._unique_id(base_id)

        external_ref = None
        node_type = "instruction"

        ref_match = re.search(
            r"[Rr]efer(?:\s+to)?[:\s]+[\"']?([^\"'\n.]+)", action_text
        )
        if ref_match:
            external_ref = ref_match.group(1).strip()
            node_type = "reference"
        elif target:
            external_ref = target

        lower = action_text.lower()
        if any(kw in lower for kw in _TERMINAL_KEYWORDS):
            tid = self._get_shared_terminal()
            return tid, []

        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "text": action_text,
            "next": None,
            "options": None,
            "external_ref": external_ref,
            "confidence": "high",
        }
        return node_id, [node_id]

    def _emit_conditional(
        self,
        condition: str,
        cond_id: str | None = None,
        true_result: _BuildResult = (None, []),
        false_result: _BuildResult = (None, []),
        forced_id: str | None = None,
    ) -> _BuildResult:
        base_id = forced_id or cond_id or _to_snake_case(condition)
        node_id = forced_id or self._unique_id(base_id)

        true_head, true_tails = true_result
        if true_head is None:
            true_head = self._get_shared_terminal()
            true_tails = []

        false_head, false_tails = false_result
        if false_head is None:
            false_head = self._get_shared_terminal()
            false_tails = []

        question_text = condition.strip()
        if not question_text.endswith("?"):
            question_text += "?"

        self.nodes[node_id] = {
            "id": node_id,
            "type": "question",
            "text": question_text,
            "next": None,
            "options": {"Yes": true_head, "No": false_head},
            "external_ref": None,
            "confidence": "high",
        }
        return node_id, true_tails + false_tails

    def _add_terminal(self, node_id: str, text: str) -> str:
        nid = self._unique_id(node_id) if node_id in self.nodes else node_id
        self.nodes[nid] = {
            "id": nid, "type": "terminal", "text": text,
            "next": None, "options": None, "external_ref": None,
            "confidence": "high",
        }
        return nid

    def _get_shared_terminal(self) -> str:
        if self._shared_terminal_id and self._shared_terminal_id in self.nodes:
            return self._shared_terminal_id
        self._shared_terminal_id = self._add_terminal("end", "End of procedure.")
        return self._shared_terminal_id

    def _terminate_tails(self, tails: List[str]) -> None:
        if not tails:
            return
        dangling = [
            t for t in tails
            if self.nodes[t]["type"] in ("instruction", "reference")
            and self.nodes[t].get("next") is None
        ]
        if dangling:
            term_id = self._get_shared_terminal()
            for nid in dangling:
                self.nodes[nid]["next"] = term_id

    def _ensure_terminals(self):
        dangling = [
            nid for nid, data in self.nodes.items()
            if data["type"] in ("instruction", "reference") and data.get("next") is None
        ]
        if dangling:
            term_id = self._get_shared_terminal()
            for nid in dangling:
                self.nodes[nid]["next"] = term_id


# ---------------------------------------------------------------------------
# PipelineConverter
# ---------------------------------------------------------------------------


class PipelineConverter:
    """Plain-text pipeline: SOP -> outline -> detail pass -> graph.

    Step 1: LLM produces a plain-text numbered outline (no structured output).
    Step 2: Chunk-by-chunk detail pass fills in missing details.
    Step 3: Direct text-to-graph build (no intermediate PseudocodeBlock).
    """

    converter_id = "pipeline_v3"

    def convert(
        self,
        source_text: str,
        enriched_chunks: Optional[List[dict]] = None,
        dump_dir: Optional[str] = None,
        resume: bool = False,
    ) -> Dict[str, Any]:
        dump_path: Optional[Path] = None
        if dump_dir:
            dump_path = Path(dump_dir)
            dump_path.mkdir(parents=True, exist_ok=True)
            logger.info("[CONVERTER] Stage outputs dir: %s (resume=%s)", dump_path, resume)

        # Reassemble enriched chunks into one document
        enriched_sop = _reassemble_enriched_sop(source_text, enriched_chunks)
        logger.info("[CONVERTER] Enriched SOP: %d chars", len(enriched_sop))

        if dump_path:
            self._dump_stage(dump_path, "enriched_sop", enriched_sop)

        # Step 1: Single-shot outline
        cached_outline = self._load_cache(dump_path, "outline") if resume else None
        if cached_outline:
            logger.info("[CONVERTER Step 1/3] Loaded outline from cache.")
            outline = cached_outline
        else:
            logger.info("[CONVERTER Step 1/3] Generating outline (%d chars)...", len(enriched_sop))
            outline = _llm_call(
                stage="code_based",
                system=_OUTLINE_SYSTEM,
                human=_OUTLINE_HUMAN,
                enriched_sop=enriched_sop,
            )
            logger.info("  Outline: %d lines", outline.count("\n") + 1)

            if dump_path:
                self._dump_stage(dump_path, "outline", outline)

        # Step 2: Chunk-by-chunk detail pass
        cached_refined = self._load_cache(dump_path, "outline_refined") if resume else None
        if cached_refined:
            logger.info("[CONVERTER Step 2/3] Loaded refined outline from cache.")
            outline = cached_refined
        elif enriched_chunks and len(enriched_chunks) > 1:
            logger.info("[CONVERTER Step 2/3] Detail pass — %d chunks to verify...",
                        len(enriched_chunks))
            outline = self._detail_pass(outline, enriched_chunks)
            logger.info("  Refined outline: %d lines", outline.count("\n") + 1)

            if dump_path:
                self._dump_stage(dump_path, "outline_refined", outline)
        else:
            logger.info("[CONVERTER Step 2/3] Skipped (single chunk / no chunks).")

        # Step 3: Direct text-to-graph build
        logger.info("[CONVERTER Step 3/3] Parsing outline and building graph...")
        nodes = parse_outline_to_graph(outline)

        type_counts: Dict[str, int] = {}
        for n in nodes.values():
            t = n.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        logger.info("  Graph: %d nodes — %s", len(nodes), type_counts)
        logger.info("[CONVERTER] Complete.")

        if dump_path:
            self._dump_stage(dump_path, "final_graph", json.dumps(nodes, indent=2))

        return nodes

    @staticmethod
    def _detail_pass(outline: str, enriched_chunks: List[dict]) -> str:
        """Chunk-by-chunk detail verification and gap-filling.

        For each enriched chunk, asks the LLM: "Is anything from this chunk
        missing in the outline?" If yes, the LLM adds it. The outline grows
        more detailed with each pass while maintaining its structure.
        """
        current = outline
        total = len(enriched_chunks)

        for idx, ec in enumerate(enriched_chunks, start=1):
            chunk_text = ec.get("chunk_text", "")
            ctx = ec.get("retrieved_context", "").strip()
            if ctx:
                chunk_text += f"\n\n[Cross-reference context: {ctx}]"

            logger.info("  Detail pass %d/%d (chunk %s)...",
                        idx, total, ec.get("chunk_id", idx - 1))
            try:
                updated = _llm_call(
                    stage="code_based",
                    system=_DETAIL_PASS_SYSTEM,
                    human=_DETAIL_PASS_HUMAN,
                    current_outline=current,
                    chunk_text=chunk_text,
                )
                # Basic sanity: the updated outline shouldn't be drastically shorter
                if len(updated.strip().split("\n")) >= len(current.strip().split("\n")) * 0.7:
                    current = updated
                else:
                    logger.warning("    Detail pass returned much shorter outline — keeping previous.")
            except Exception as e:
                logger.warning("    Detail pass failed for chunk %d (%s) — keeping previous.", idx, e)

        return current

    @staticmethod
    def _load_cache(dump_dir: Optional[Path], name: str) -> Optional[str]:
        if dump_dir is None:
            return None
        # Try both .txt and .json extensions
        for ext in (".txt", ".json"):
            path = dump_dir / f"{name}{ext}"
            if path.exists():
                logger.info("  [CACHE HIT] %s <- %s", name, path)
                return path.read_text()
        return None

    @staticmethod
    def _dump_stage(dump_dir: Path, name: str, content: str) -> None:
        # Use .txt for plain text outlines, .json for structured data
        ext = ".json" if content.lstrip().startswith(("{", "[")) else ".txt"
        path = dump_dir / f"{name}{ext}"
        path.write_text(content)
        logger.info("  [DUMP] %s -> %s", name, path)
