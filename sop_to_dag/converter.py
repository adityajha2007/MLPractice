"""SOP-to-graph pipeline converter (v4 — graph-first).

Step 1 (Graph Gen):     LLM converts full enriched SOP -> graph JSON directly
Step 2 (Graph Refine):  For each chunk, LLM produces a patch to add/modify/remove nodes

The LLM produces structured graph output from the start — no lossy text-outline
intermediate. This preserves temporal dependencies, decision scope, and branch
history that indentation-based outlines lose.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from sop_to_dag.graph_ops import apply_patch, generate_adjacency_map, get_graph_issues
from sop_to_dag.models import LLMStopError, get_model, safe_invoke
from sop_to_dag.graph_ops import SchemaValidator
from sop_to_dag.schemas import GraphPatch, InitialGraph, WorkflowNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inline prompts — LLM produces STRUCTURED GRAPH OUTPUT
# ---------------------------------------------------------------------------

_GRAPH_SYSTEM = """\
You are a Senior Process Architect. Convert the provided procedural text into \
a precise JSON Workflow Graph.

### 1. NODE IDENTITY & STRUCTURE
- **IDs**: You MUST use descriptive, snake_case IDs (e.g., 'check_fraud_score', \
'update_case_status'). NEVER use generic IDs like 'node_1', 'step_2'.
- **Granularity**: Break complex steps into single logical units. "Open System X, \
navigate to Tab Y, click Z" = 3 separate nodes, not 1.
- **Strict Connectivity**: Every 'next' or option value must match exactly one \
'id' present in your output list.
- **Loops allowed**: The graph may contain cycles. If the SOP says "go back to \
step X", "retry", or "repeat", point 'next' or an option back to the earlier \
node ID. Do NOT avoid back-edges.

### 2. NODE TYPES & SCHEMA RULES
Your output must strictly adhere to these types:

* **"question"**
  * Use for decision points. All decisions MUST be binary Yes/No.
  * **REQUIRED**: 'options' with exactly two keys: {{"Yes": "id", "No": "id"}}.
  * **Constraint**: Do NOT use the 'next' field for questions. It must be null.
  * **Constraint**: Question text MUST end with "?".
  * **Multi-way decisions**: If the SOP has 3+ options, decompose into a chain of \
binary Yes/No questions. See the PATTERN GUIDE in the input for examples.

* **"instruction"**
  * Use for linear steps or actions.
  * **REQUIRED**: 'next' field containing exactly **one** string ID.
  * **Constraint**: Never use lists or pipe-separated strings in 'next'.

* **"terminal"**
  * Use for the absolute end of a flow or a hand-off to a different department.
  * **Constraint**: 'next' and 'options' should be null.

* **"reference"**
  * Use for static data lookups (e.g., "See Response Codes Table").
  * **REQUIRED**: 'next' field pointing to where the flow resumes after lookup.
  * **REQUIRED**: 'external_ref' field with the document/table name.

### 3. STRUCTURE RULES
- First node must have id="start". At least one terminal node (typically id="end").
- After a decision where branches converge, both branches' last nodes point to \
the same convergence node.
- If the text mentions an external guide, use type="reference" with 'external_ref'.
- Preserve numbered step ordering. Do NOT reorder or skip steps.

### 4. METADATA
- **role**: Who performs this action (if specified in SOP).
- **system**: Software or tool used (if specified in SOP).
- **confidence**: "high" = stated in SOP, "medium" = inferred, "low" = guess.

### 5. CONTENT DETAIL & GRANULARITY
- Capture MAXIMUM detail — every click, check, data entry, decision point, \
field name, code, threshold, and cross-section dependency.
- Do NOT summarize or collapse multiple actions into one node.
- **Split**: Sequential actions, system interactions, decisions = separate nodes.
- **Keep as one**: A list of fields to validate/populate = one node listing all fields.
- When copying data between systems, each system interaction = separate node.
- Email/escalation actions: capture full template (recipients, subject, body).

### 6. LOOPS & REPEATED PROCEDURES
- "Go back to step X" / "retry" = back-edge to existing node.
- "Repeat steps N-M for another section" = back-edge to first node of that \
sequence. Do NOT duplicate nodes for repeated sub-procedures.

In your reasoning field, provide a detailed analysis of the SOP structure: \
identify all decision points, branches, convergence points, cross-section \
dependencies, and the overall process flow before producing the nodes.
"""

_GRAPH_PATTERN_GUIDE = """\
## PATTERN GUIDE — follow these modeling patterns exactly.

### P1. Multi-option → chain of binary Yes/No questions
SOP: "If dispute code is 103, do X. If 1047, do Y. Otherwise do Z."
  {{"id": "is_code_103_question", "type": "question", "text": "Is the dispute code 103?", "next": null, "options": {{"Yes": "handle_103", "No": "is_code_1047_question"}}}},
  {{"id": "is_code_1047_question", "type": "question", "text": "Is the dispute code 1047?", "next": null, "options": {{"Yes": "handle_1047", "No": "handle_other_codes"}}}}
WRONG: {{"options": {{"103": "...", "1047": "...", "Other": "..."}}}}

### P2. Nested conditionals → chained question nodes
SOP: "If account is special, check if data auto-populated. If not, populate. Then check for error note."
  {{"id": "is_special_account_question", "type": "question", "text": "Is the account type special?", "next": null, "options": {{"Yes": "did_data_populate_question", "No": "next_section"}}}},
  {{"id": "did_data_populate_question", "type": "question", "text": "Did the data auto-populate?", "next": null, "options": {{"Yes": "verify_data", "No": "manually_populate"}}}},
  {{"id": "manually_populate", "type": "instruction", "text": "Manually populate required fields", "next": "check_error_note_question"}},
  {{"id": "check_error_note_question", "type": "question", "text": "Is there an error note?", "next": null, "options": {{"Yes": "correct_error", "No": "next_section"}}}}

### P3. Cross-system data flow → separate node per system interaction
SOP: "Copy the note from System A, paste into System B on Transfer tab."
  {{"id": "copy_note_system_a", "type": "instruction", "text": "Copy the activity note from System A", "next": "paste_note_system_b", "system": "System A"}},
  {{"id": "paste_note_system_b", "type": "instruction", "text": "Paste the activity note into System B on the Transfer tab", "next": "next_step", "system": "System B"}}

### P4. Auto-populate / manual fallback → action → question → converge
SOP: "Click Populate Data. If not populated, escalate. Verify fields."
  {{"id": "click_populate_data", "type": "instruction", "text": "Click the Populate Data button", "next": "did_data_populate_question", "system": "System A"}},
  {{"id": "did_data_populate_question", "type": "question", "text": "Did the data auto-populate?", "next": null, "options": {{"Yes": "verify_fields", "No": "escalate_to_supervisor"}}}},
  {{"id": "escalate_to_supervisor", "type": "instruction", "text": "Escalate to supervisor for manual data population", "next": "verify_fields", "role": "Supervisor"}},
  {{"id": "verify_fields", "type": "instruction", "text": "Verify all required fields are populated", "next": "next_step"}}
This pattern may repeat for different tabs/sections — use distinct node IDs each time.

### P5. Field validation list → ONE node listing all fields
SOP: "Validate: Account Status, Payment Rating, Date of Last Payment, Balance, Days Past Due."
  {{"id": "validate_required_fields", "type": "instruction", "text": "Validate the following fields: Account Status, Payment Rating, Date of Last Payment, Balance, Days Past Due", "next": "next_step"}}
WRONG: separate node per field.

### P6. Email/escalation → capture full template
SOP: "Email docs to team@company.com. Subject: Account # - Docs. Body: Include viewer link."
  {{"id": "email_docs_to_team", "type": "instruction", "text": "Email documents to team@company.com. Subject: 'Account # - Docs'. Body: 'Include viewer link'", "next": "screenshot_email"}},
  {{"id": "screenshot_email", "type": "instruction", "text": "Add screenshot of sent email to the case system", "next": "next_step", "system": "Case System"}}

### P7. Retry loop → back-edge
SOP: "Submit request. If it fails, wait 1 day and retry."
  {{"id": "submit_request", "type": "instruction", "text": "Submit the request", "next": "did_request_succeed_question"}},
  {{"id": "did_request_succeed_question", "type": "question", "text": "Did the request succeed?", "next": null, "options": {{"Yes": "next_step", "No": "wait_one_day"}}}},
  {{"id": "wait_one_day", "type": "instruction", "text": "Wait 1 business day", "next": "submit_request"}}

### P8. Simple decision flow
SOP: "Open case. If fraud, escalate. Otherwise close."
  {{"id": "start", "type": "instruction", "text": "Open the case", "next": "is_fraud_detected_question"}},
  {{"id": "is_fraud_detected_question", "type": "question", "text": "Is fraud detected?", "next": null, "options": {{"Yes": "escalate_to_supervisor", "No": "close_case"}}}},
  {{"id": "escalate_to_supervisor", "type": "instruction", "text": "Escalate case to supervisor", "next": "end"}},
  {{"id": "close_case", "type": "instruction", "text": "Close the case", "next": "end"}},
  {{"id": "end", "type": "terminal", "text": "End of procedure", "next": null, "options": null}}
"""

_GRAPH_HUMAN = """\
{pattern_guide}

---

Convert this SOP into a workflow graph following the schema and patterns above. \
Capture every detail — do not summarize or skip steps.

{enriched_sop}
"""

_PATCH_SYSTEM = """\
You are a Graph Refinement Engineer performing a LINE-BY-LINE audit of an SOP \
section against an existing workflow graph. Your goal is to catch EVERY missing \
detail, no matter how small.

You receive:
- The current graph as an adjacency map (showing connections)
- The current graph as full JSON (showing all node details)
- One specific SOP section/chunk to audit against the graph

AUDIT PROCESS (follow this exactly in your reasoning):

Step 1 — EXTRACT: Read the SOP chunk sentence by sentence. For each sentence, \
list every discrete detail:
  - Specific actions (clicks, data entry, navigation steps, lookups)
  - Decision points and their conditions (thresholds, codes, statuses)
  - System/tool names where actions are performed
  - Role assignments (who does what)
  - Specific field names, values, codes, or thresholds mentioned
  - References to other documents, SOPs, or procedures
  - Temporal dependencies ("after X", "before Y", "within N days")
  - Exception/error handling paths

Step 2 — MATCH: For each extracted detail, find the graph node(s) that capture \
it. Mark each detail as:
  ✓ COVERED — a node captures this detail accurately
  ~ PARTIAL — a node mentions it but is missing specifics (wrong system name, \
missing threshold value, vague text that loses the original detail)
  ✗ MISSING — no node captures this detail

Step 3 — PATCH: For every MISSING or PARTIAL detail, produce the appropriate \
add/modify operations. Be specific:
  - MISSING action → add a new instruction node, wire it into the correct \
position in the flow
  - MISSING decision → add a question node with correct options
  - PARTIAL node → modify it to include the missing specifics (system name, \
field name, threshold, role)
  - MISSING reference → add a reference node with external_ref
  - MISSING role/system metadata → modify existing node to add role/system fields

WHAT TO LOOK FOR (common gaps Step 1 misses):
- Sub-steps within a bullet point ("Open System X, navigate to Tab Y, click Z" \
= 3 separate nodes, not 1)
- Conditional paths that the graph collapsed into a single branch
- Specific system names or tool names that the graph replaced with generic text
- Exact field names, codes, or threshold values that the graph omitted
- Role assignments ("the analyst does X, the supervisor does Y")
- Time constraints ("within 2 business days", "before end of shift")
- Error/exception paths ("if the system is unavailable, do X instead")

PATCH RULES:
- When ADDING nodes: ensure their "next"/"options" point to existing node IDs \
or other newly added node IDs. Wire them into the graph correctly.
- When MODIFYING nodes: include ALL fields of the node (the entire node dict \
will be replaced). Match by "id".
- When REMOVING nodes: ensure no remaining nodes reference the removed IDs. \
If they do, include those referencing nodes in modify_nodes with updated \
"next"/"options" values.
- For INSERTING a node between A and B: add the new node N with next=B, \
and modify node A to have next=N.
- For INSERTING a decision that splits a chain A→B: add the question node Q \
and any new branch nodes, modify A to point to Q, and ensure branch ends \
point to B (or wherever they should converge).
- Do NOT remove or restructure nodes that are correct.
- Do NOT add nodes for content not in this SOP section.

If (and ONLY if) every detail is ✓ COVERED, return empty lists.

Node schema reminder:
- instruction: must have "next", "options" must be null
- question: must have "options" (Yes/No), "next" must be null, text ends with "?"
- terminal: "next" and "options" both null
- reference: must have "next" and "external_ref"
"""

_PATCH_HUMAN = """\
## Current Graph (Adjacency Map)
{adjacency_map}

## Current Graph (Full Nodes JSON)
{nodes_json}

## SOP Chunk to Audit
{chunk_text}

Follow the 3-step audit process (Extract → Match → Patch) in your reasoning. \
List every detail you extract and its coverage status before producing the patch.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _structured_llm_call(
    stage: str, schema, system: str, human: str, **format_kwargs
):
    """Structured output LLM call. Returns a Pydantic model instance.

    Halts the pipeline on non-200 responses (via safe_invoke) so the
    checkpoint system can handle recovery.
    """
    llm = get_model(stage)
    structured_llm = llm.with_structured_output(schema)
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=human.format(**format_kwargs)),
    ]
    return safe_invoke(structured_llm, messages, context=f"structured/{stage}")


def _reassemble_sop(
    source_text: str,
    enriched_chunks: Optional[List[dict]],
    include_context: bool = False,
) -> str:
    """Reassemble chunks into one document.

    Args:
        source_text: Raw SOP text (fallback if no enriched chunks).
        enriched_chunks: Entity-resolved chunks with optional cross-ref context.
        include_context: If True, append cross-reference context notes.
            Step 1 sets this to False (LLM sees the full doc, notes are noise).
            Step 2 would set True, but Step 2 sends chunks individually instead.
    """
    if not enriched_chunks:
        return source_text

    parts = []
    for ec in enriched_chunks:
        chunk_text = ec.get("chunk_text", "")
        parts.append(chunk_text)

        if include_context:
            retrieved = ec.get("retrieved_context", "")
            if retrieved.strip():
                parts.append(f"[Cross-reference context: {retrieved.strip()}]")

    return "\n\n".join(parts)


def _nodes_list_to_dict(nodes: List[WorkflowNode]) -> Dict[str, Dict[str, Any]]:
    """Convert a list of WorkflowNode models to the standard nodes dict."""
    result: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        data = node.model_dump()
        result[data["id"]] = data
    return result


def _ensure_start_node(nodes: Dict[str, Dict[str, Any]]) -> None:
    """Ensure the first node has id='start'. Renames in-place if needed."""
    if "start" in nodes:
        return
    if not nodes:
        return

    # Pick the first node as start
    first_id = next(iter(nodes))
    node_data = nodes.pop(first_id)
    node_data["id"] = "start"
    # Rebuild dict with "start" first
    new_nodes = {"start": node_data}
    new_nodes.update(nodes)
    nodes.clear()
    nodes.update(new_nodes)

    # Update all references to the old ID
    for n in nodes.values():
        if n.get("next") == first_id:
            n["next"] = "start"
        if n.get("options"):
            for k, v in n["options"].items():
                if v == first_id:
                    n["options"][k] = "start"


# ---------------------------------------------------------------------------
# PipelineConverter
# ---------------------------------------------------------------------------


class PipelineConverter:
    """Graph-first pipeline: SOP -> graph JSON -> chunk-by-chunk graph refinement.

    Step 1: LLM produces a workflow graph directly from the full enriched SOP.
    Step 2: Each enriched chunk is used to refine the graph via structured patches.
    """

    converter_id = "pipeline_v4"

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
        # Step 1 gets entity-resolved text WITHOUT cross-ref context notes.
        # The LLM sees the full doc — cross-ref notes are redundant noise.
        enriched_sop = _reassemble_sop(source_text, enriched_chunks, include_context=False)
        logger.info("[CONVERTER] Enriched SOP: %d chars", len(enriched_sop))

        if dump_path:
            self._dump_stage(dump_path, "enriched_sop", enriched_sop)

        validator = SchemaValidator()

        # ----- Step 1/2: Full SOP -> Graph (structured output) -----
        cached_graph = self._load_cache(dump_path, "initial_graph") if resume else None
        if cached_graph:
            logger.info("[CONVERTER Step 1/2] Loaded initial graph from cache.")
            nodes = json.loads(cached_graph)
        else:
            logger.info(
                "[CONVERTER Step 1/2] Generating graph from SOP (%d chars)...",
                len(enriched_sop),
            )
            result: InitialGraph = _structured_llm_call(
                stage="graph_gen",
                schema=InitialGraph,
                system=_GRAPH_SYSTEM,
                human=_GRAPH_HUMAN,
                pattern_guide=_GRAPH_PATTERN_GUIDE,
                enriched_sop=enriched_sop,
            )
            nodes = _nodes_list_to_dict(result.nodes)
            _ensure_start_node(nodes)
            nodes, fixes = validator.validate_and_fix(nodes)
            if fixes:
                logger.info("  Schema fixes applied: %s", fixes)

            # Edge integrity: check all next/options targets exist
            topo_report = get_graph_issues(nodes)
            if topo_report != "Topology Valid.":
                logger.warning("  Initial graph topology issues: %s", topo_report)
            else:
                logger.info("  Initial graph topology: clean")

            logger.info("  Initial graph: %d nodes", len(nodes))

            if dump_path:
                self._dump_stage(dump_path, "initial_graph", json.dumps(nodes, indent=2))

        # ----- Step 2/2: Chunk-by-chunk graph refinement (multi-pass) -----
        cached_refined = self._load_cache(dump_path, "final_graph") if resume else None
        if cached_refined:
            logger.info("[CONVERTER Step 2/2] Loaded refined graph from cache.")
            nodes = json.loads(cached_refined)
        elif enriched_chunks and len(enriched_chunks) > 1:
            num_passes = 2
            total = len(enriched_chunks)
            logger.info(
                "[CONVERTER Step 2/2] Graph refinement — %d chunks × %d passes...",
                total, num_passes,
            )

            for pass_num in range(1, num_passes + 1):
                pass_changes = 0
                logger.info("  --- Pass %d/%d ---", pass_num, num_passes)

                # Resume: check for per-pass checkpoint
                pass_cache_name = f"graph_after_pass_{pass_num}"
                cached_pass = self._load_cache(dump_path, pass_cache_name) if resume else None
                if cached_pass:
                    logger.info("  Loaded pass %d graph from cache.", pass_num)
                    nodes = json.loads(cached_pass)
                    continue

                # Resume: find the latest per-chunk checkpoint within this pass
                start_chunk_idx = 1
                if resume and dump_path:
                    for check_idx in range(total, 0, -1):
                        ckpt = self._load_cache(dump_path, f"graph_p{pass_num}_c{check_idx}")
                        if ckpt:
                            nodes = json.loads(ckpt)
                            start_chunk_idx = check_idx + 1
                            logger.info(
                                "  Resuming pass %d from chunk %d/%d (loaded checkpoint).",
                                pass_num, start_chunk_idx, total,
                            )
                            break

                for idx, ec in enumerate(enriched_chunks, start=1):
                    if idx < start_chunk_idx:
                        continue

                    chunk_text = ec.get("chunk_text", "")
                    ctx = ec.get("retrieved_context", "").strip()
                    if ctx:
                        chunk_text += f"\n\n[Cross-reference context: {ctx}]"

                    logger.info(
                        "  Pass %d, chunk %d/%d (chunk %s)...",
                        pass_num, idx, total, ec.get("chunk_id", idx - 1),
                    )

                    # Snapshot for rollback
                    pre_patch = {nid: dict(data) for nid, data in nodes.items()}
                    pre_patch_count = len(nodes)

                    try:
                        adjacency_map = generate_adjacency_map(nodes)
                        nodes_json = json.dumps(nodes, indent=2)

                        patch: GraphPatch = _structured_llm_call(
                            stage="graph_refine",
                            schema=GraphPatch,
                            system=_PATCH_SYSTEM,
                            human=_PATCH_HUMAN,
                            adjacency_map=adjacency_map,
                            nodes_json=nodes_json,
                            chunk_text=chunk_text,
                        )

                        changes = (
                            len(patch.add_nodes)
                            + len(patch.modify_nodes)
                            + len(patch.remove_nodes)
                        )
                        if changes == 0:
                            logger.info("    No changes needed.")
                            continue

                        logger.info(
                            "    Patch: +%d add, ~%d modify, -%d remove",
                            len(patch.add_nodes),
                            len(patch.modify_nodes),
                            len(patch.remove_nodes),
                        )

                        # Dump patch reasoning for debugging
                        if dump_path and patch.reasoning:
                            reason_name = f"patch_p{pass_num}_c{idx}_reasoning"
                            self._dump_stage(dump_path, reason_name, patch.reasoning)

                        apply_patch(nodes, patch)
                        nodes, fixes = validator.validate_and_fix(nodes)
                        if fixes:
                            logger.info("    Schema fixes: %s", fixes)

                        # Topological validation after patch
                        topo_report = get_graph_issues(nodes)
                        if topo_report != "Topology Valid.":
                            logger.warning("    Post-patch topology issues: %s", topo_report)

                        # Sanity checks
                        if len(nodes) < pre_patch_count * 0.7:
                            logger.warning(
                                "    Patch shrank graph from %d to %d nodes (>30%% loss) — rolling back.",
                                pre_patch_count, len(nodes),
                            )
                            nodes = pre_patch
                            continue

                        if "start" not in nodes:
                            logger.warning("    Patch removed 'start' node — rolling back.")
                            nodes = pre_patch
                            continue

                        pass_changes += changes

                    except LLMStopError:
                        # Non-200 API response (rate limit, server error) — stop
                        # the pipeline. Checkpoint was saved for the previous chunk,
                        # so --resume will pick up from here.
                        raise
                    except Exception as e:
                        logger.warning(
                            "    Patch failed for chunk %d (%s) — keeping previous graph.",
                            idx, e,
                        )
                        nodes = pre_patch

                    # Checkpoint after each chunk so we can resume mid-pass
                    if dump_path:
                        self._dump_stage(
                            dump_path,
                            f"graph_p{pass_num}_c{idx}",
                            json.dumps(nodes, indent=2),
                        )

                logger.info(
                    "  Pass %d complete: %d total changes applied.", pass_num, pass_changes
                )

                # If pass 2 made no changes, the graph is stable
                if pass_num > 1 and pass_changes == 0:
                    logger.info("  No changes in pass %d — graph is stable.", pass_num)
                    break

                # Dump pass-level checkpoint
                if dump_path:
                    self._dump_stage(
                        dump_path,
                        pass_cache_name,
                        json.dumps(nodes, indent=2),
                    )

            if dump_path:
                self._dump_stage(dump_path, "final_graph", json.dumps(nodes, indent=2))
        else:
            logger.info("[CONVERTER Step 2/2] Skipped (single chunk / no chunks).")

        type_counts: Dict[str, int] = {}
        for n in nodes.values():
            t = n.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        logger.info("  Graph: %d nodes — %s", len(nodes), type_counts)
        logger.info("[CONVERTER] Complete.")

        return nodes

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
