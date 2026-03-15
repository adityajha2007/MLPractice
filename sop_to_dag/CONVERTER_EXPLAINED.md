# How the Converter Works

This document explains the full SOP-to-JSON-DAG conversion pipeline — from raw text to a fully-connected workflow graph.

## The Big Picture

```
Raw SOP Text
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  PREPROCESSING (preprocessing.py)                       │
│  LangGraph pipeline — runs before conversion            │
│                                                         │
│  1. Agentic Chunking    → semantic chunks               │
│  2. FAISS Indexing       → vector store                  │
│  3. RAG Enrichment       → chunks + cross-ref context    │
│  4. Entity Resolution    → canonical term mapping        │
└──────────────────────────┬──────────────────────────────┘
                           │
              enriched_chunks + entity_map
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  CONVERTER (converter.py) — 3.5 Stages                  │
│                                                         │
│  Stage 1   [LLM]  Raw text → ProcedureCard              │
│  Stage 2   [LLM]  ProcedureCard → PseudocodeBlock       │
│  Stage 2.5 [LLM]  Merge pseudocode ↔ original doc       │
│  Stage 3   [CODE] Deterministic compile → JSON DAG       │
│                                                         │
│  LLM's job ends after Stage 2.5. Stage 3 is pure code. │
└──────────────────────────┬──────────────────────────────┘
                           │
                      JSON DAG (nodes dict)
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  REFINEMENT LOOP (loop.py)                              │
│  LangGraph StateGraph: Analyser ↔ Refiner               │
│  Cycles until all checks pass or max iterations          │
└─────────────────────────────────────────────────────────┘
```

---

## Stage 1: TopDown — Extract the Macro Skeleton

**Input:** Raw SOP text
**Output:** `ProcedureCard` (Pydantic model)
**Method:** LLM with structured output

The LLM reads the entire SOP and produces a high-level skeleton:

```
ProcedureCard
├── title: "Fraud Dispute Resolution"
├── goal: "Resolve credit bureau disputes..."
├── major_phases:
│   ├── Phase("Intake", description="...", sub_steps=[...], decision_points=[...])
│   ├── Phase("Investigation", ...)
│   └── Phase("Resolution", ...)
└── decision_gates:
    ├── DecisionGate(condition="Is dispute code 183 or 186?", true_branch="...", false_branch="...")
    └── DecisionGate(condition="Is there a force memo?", ...)
```

**Why this stage exists:** The SOP is unstructured natural language. This stage forces the LLM to identify the structural backbone — phases, decision gates, goals — without worrying about granular details yet. It's a top-down decomposition: forest before trees.

**Prompt role:** `Senior Process Architect` — focuses on exhaustive extraction of phases and decision gates. Explicitly told not to invent steps.

---

## Stage 2: CodeBased — Translate to Structured Pseudocode

**Input:** `ProcedureCard` + original SOP text
**Output:** `PseudocodeBlock` (Pydantic model)
**Method:** LLM with structured output

The LLM translates the skeleton into a code-like structure with two primitive types:

| Pseudocode primitive | What it represents |
|---|---|
| `ActionStep` | A sequential action ("Send email to team", "Update record in HR System") |
| `ConditionalBlock` | An IF/ELSE decision with `if_true` and `if_false` branch step lists |

These are wrapped in `StepItem` (a union type) and organized into `Procedure` objects:

```
PseudocodeBlock
└── procedures:
    └── Procedure("Fraud Dispute Resolution")
        ├── preconditions: ["Agent must be logged in", "Dispute ticket exists"]
        ├── steps:
        │   ├── StepItem(action_step=ActionStep("Receive dispute notification"))
        │   ├── StepItem(conditional=ConditionalBlock(
        │   │       condition="Is dispute code 183 or 186?",
        │   │       if_true=[
        │   │           StepItem(action_step=ActionStep("Route to fraud team")),
        │   │           StepItem(conditional=ConditionalBlock(  ← nesting allowed
        │   │               condition="Is borrower's account frozen?",
        │   │               if_true=[...],
        │   │               if_false=[...]
        │   │           ))
        │   │       ],
        │   │       if_false=[
        │   │           StepItem(action_step=ActionStep("Route to standard disputes"))
        │   │       ]
        │   │   ))
        │   └── StepItem(action_step=ActionStep("End processing"))
        └── postconditions: ["Dispute ticket marked as resolved"]
```

**Why this stage exists:** The ProcedureCard is descriptive ("Phase: Investigation"). The pseudocode is prescriptive — it defines exact control flow with IF/ELSE branches that map 1:1 to graph structure. This is the representation that Stage 3 can mechanically compile.

**Key rules the LLM follows:**
- Every decision → `ConditionalBlock` with both branches
- Every action → `ActionStep` (with optional `target` field for the system being acted on)
- Nesting is allowed (conditionals inside conditionals)
- Exact conditions from the SOP text are preserved verbatim

---

## Stage 2.5: Merge — Reconcile with Original Document

**Input:** `PseudocodeBlock` + original SOP text + (optional) RAG enrichment context
**Output:** `PseudocodeBlock` (merged, more detailed)
**Method:** LLM with structured output

This is the **granularity guarantee** stage. The LLM compares the pseudocode line-by-line against the original SOP and adds anything that was missed:

- Specific system names, form numbers, team names
- Threshold values, codes, reference IDs
- External document references (guides, policies)
- Implicit steps the skeleton glossed over

**Rules:**
1. NEVER remove or simplify existing steps — only ADD
2. Preserve exact wording from SOP (the graph builder depends on this text)
3. Every piece of information in the original SOP MUST appear somewhere in the output

**Why this stage exists:** Stage 1+2 produce a structurally correct but potentially lossy skeleton. The merge stage is a "diff and patch" — it ensures no granular detail from the source document is lost before we lock the structure in Stage 3.

If RAG enrichment is available (from preprocessing), it's injected here as additional context — cross-references between chunks that the LLM might not have seen together.

---

## Stage 3: Deterministic Graph Build — Pure Compilation

**Input:** Merged `PseudocodeBlock`
**Output:** `Dict[str, node_data]` — the JSON DAG
**Method:** Pure Python tree walk. No LLM. Zero hallucination risk.

This is the core of `_GraphBuilder`. It walks the pseudocode tree and emits `WorkflowNode` dicts:

### Mapping Rules

| Pseudocode element | Graph node type | Description |
|---|---|---|
| `ActionStep` | `instruction` | A step to perform. Has `next` pointing to the following node. |
| `ActionStep` with "Refer to..." | `reference` | Same as instruction but includes `external_ref` field. |
| `ActionStep` with terminal keyword | `terminal` | End state. No outgoing edges. |
| `ConditionalBlock` | `question` | Decision node. Has `options: {"Yes": node_id, "No": node_id}`. |
| Empty branch / dangling end | `terminal` | Auto-generated end node. |

### The Head/Tails Convergence Pattern

This is the key algorithm that makes the graph fully connected. Every emit function returns:

```
(head_id, tail_ids)
```

- **head_id**: The entry point of the emitted subgraph (the first node)
- **tail_ids**: Nodes whose `next` is still `None` — they need to be wired to whatever comes after

#### How it works for each type:

**ActionStep → instruction:**
```
Returns (node_id, [node_id])
         ↑ entry    ↑ this node's `next` is None, caller will wire it
```

**ActionStep → terminal:**
```
Returns (node_id, [])
         ↑ entry    ↑ empty — nothing comes after a terminal
```

**ConditionalBlock → question:**
```
                    ┌── Yes branch ──→ [steps...] → tail_A (next=None)
  question_node ──┤
                    └── No branch  ──→ [steps...] → tail_B (next=None)

Returns (question_id, [tail_A, tail_B])
         ↑ entry       ↑ BOTH branches' exits need wiring
```

#### Chaining a step list:

When `_walk_steps` processes `[step1, step2, step3]`:

1. Emit step1 → get `(head1, tails1)`
2. Emit step2 → get `(head2, tails2)`
3. Emit step3 → get `(head3, tails3)`
4. Wire: `tails1.next → head2`, `tails2.next → head3`
5. Return: `(head1, tails3)` — first node is entry, last step's tails are exits

This is what makes **branch convergence** work:

```
Step list: [ConditionalBlock, ActionStep("Continue")]

1. Emit ConditionalBlock:
   question → Yes → handle_valid (tail)
            → No  → handle_invalid (tail)
   Returns: (question, [handle_valid, handle_invalid])

2. Emit ActionStep("Continue"):
   Returns: (continue, [continue])

3. Chain: handle_valid.next → continue
          handle_invalid.next → continue    ← CONVERGENCE!

4. Return: (question, [continue])
```

Both branches merge into the same downstream node. This works recursively for nested conditionals at any depth.

### Walk-through Example

Given this pseudocode:
```
Procedure: Dispute Resolution
  preconditions: ["Agent logged in"]
  steps:
    1. ActionStep("Receive dispute")
    2. ConditionalBlock("Is code 183?")
         if_true:  ActionStep("Route to fraud")
         if_false: ActionStep("Route to standard")
    3. ActionStep("Send confirmation email")
    4. ActionStep("End processing")
  postconditions: ["Ticket resolved"]
```

The builder produces:

```
start ──────────────────→ receive_dispute ──→ is_code_183
(Precondition:                                    │
 Agent logged in)                         ┌───Yes─┴──No───┐
                                          ▼               ▼
                                   route_to_fraud   route_to_standard
                                          │               │
                                          └───────┬───────┘
                                                  ▼
                                      send_confirmation_email
                                                  │
                                                  ▼
                                          end_processing ← TERMINAL (keyword detected)
                                                  │
                                          (postcondition node)
                                                  │
                                                  ▼
                                              end ← TERMINAL (auto)
```

### Node Detection

**Terminal detection** — if `action.lower()` contains any of these keywords, the node becomes a terminal:
- "end processing", "end of process", "end of procedure"
- "process complete", "no further action", "workflow complete"
- "close the case", "mark as complete", "mark as done"
- (and more — 16 keywords total)

**Reference detection** — regex match on `Refer to: <name>` or `Refer <name>`:
- Sets `node_type = "reference"` and populates `external_ref`

**Target metadata** — if `ActionStep.target` is set (e.g., "HR System"):
- Appended to node text: `"Update record (target: HR System)"`

### Confidence Labels

| Confidence | Meaning | When assigned |
|---|---|---|
| `high` | Directly stated in SOP | All nodes from explicit pseudocode steps |
| `low` | Inferred for connectivity | Auto-generated terminals from `_terminate_tails` or `_ensure_terminals` |

Low-confidence nodes are prioritized during the refinement loop — the `TripletVerifier` checks these edges first.

### Safety Nets

After the main walk, two safety passes run:

1. **`_terminate_tails(tails)`** — Any tail nodes still dangling at the top level get wired to a shared terminal node (marked `confidence: "low"`)

2. **`_ensure_terminals()`** — Final sweep: any `instruction` or `reference` node with `next=None` gets a terminal. This catches edge cases the main walk might miss (e.g., unreachable subgraphs from unusual pseudocode structure)

---

## Output Format

The final output is a Python dict mapping node IDs to node data:

```json
{
  "start": {
    "id": "start",
    "type": "instruction",
    "text": "Precondition: Agent logged in",
    "next": "receive_dispute",
    "options": null,
    "external_ref": null,
    "confidence": "high"
  },
  "receive_dispute": {
    "id": "receive_dispute",
    "type": "instruction",
    "text": "Receive dispute notification",
    "next": "is_code_183",
    "options": null,
    "external_ref": null,
    "confidence": "high"
  },
  "is_code_183": {
    "id": "is_code_183",
    "type": "question",
    "text": "Is dispute code 183 or 186?",
    "next": null,
    "options": {"Yes": "route_to_fraud", "No": "route_to_standard"},
    "external_ref": null,
    "confidence": "high"
  }
}
```

### Node Types

| Type | `next` | `options` | Description |
|---|---|---|---|
| `instruction` | Required (node_id) | null | A step to execute |
| `question` | null | Required (`{"Yes": id, "No": id}`) | A decision point |
| `terminal` | null | null | End state |
| `reference` | Required (node_id) | null | Like instruction, but references external doc |

---

## What Makes This Design Work

1. **LLM narrows ambiguity, code compiles structure.** Stages 1–2.5 use the LLM for what it's good at (understanding natural language, identifying decision logic). Stage 3 uses deterministic code for what it's good at (building a correct, connected graph with no hallucination).

2. **The merge stage (2.5) is the granularity guarantee.** Without it, the skeleton from Stages 1–2 might miss specific codes, team names, or threshold values. The merge forces a line-by-line reconciliation with the source document.

3. **Head/tails pattern prevents orphaned nodes.** The old builder had a critical bug: steps after a ConditionalBlock were unreachable because the chaining logic only wired `instruction` nodes. The head/tails pattern makes convergence automatic — both branches' exit points bubble up to the parent, which wires them to the next step.

4. **Confidence labels drive refinement priority.** The refinement loop knows which edges to verify first — low-confidence auto-terminals are checked before high-confidence SOP-stated edges.
