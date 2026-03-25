"""Refiner: triplet verification + granularity expansion + error resolution + schema validation.

All components in one file. Inline prompts. TripletVerifier prioritizes
low-confidence edges first. GranularityExpander breaks coarse nodes into
sub-steps. ErrorResolver can use FAISS for surgical fixes.
"""

import json
import logging
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from pydantic import BaseModel

from sop_to_dag.models import get_model
from sop_to_dag.schemas import GraphState, WorkflowNode

logger = logging.getLogger(__name__)


class _NodePatch(BaseModel):
    """Wrapper for structured output: list of repaired nodes."""

    nodes: List[WorkflowNode]


class _TripletResult(BaseModel):
    """Result of verifying a single triplet."""

    triplet_index: int
    is_valid: bool
    explanation: str = ""


class _TripletVerification(BaseModel):
    """Structured output for triplet batch verification."""

    results: List[_TripletResult]

# ---------------------------------------------------------------------------
# Inline prompts
# ---------------------------------------------------------------------------

_TRIPLET_SYSTEM = """\
You are a Graph Verification Specialist. You verify that decision-node triplets
(source_node, edge_label, target_node) are correct according to the original SOP.

For each triplet, evaluate:
1. Does the source node's question/condition match the SOP?
2. Does the edge label (answer option) correctly represent a valid answer?
3. Does the target node logically follow from that answer?

Mark each triplet as VALID or INVALID with an explanation.
"""

_TRIPLET_HUMAN = """\
## Original SOP
{source_text}

## Triplets to Verify (batch)
{triplets_json}

For each triplet, return a _TripletResult with triplet_index, is_valid, and
explanation.
"""

_RESOLVER_SYSTEM = """\
You are a Graph Repair Specialist. You fix specific issues in workflow graphs
by making SURGICAL EDITS — never regenerate the entire graph.

You receive:
1. A flagged node and its 2-hop neighborhood (nearby connected nodes)
2. The relevant section of the original SOP
3. A description of the issue to fix

Rules:
- Only modify the flagged node and its immediate connections
- Preserve all other nodes and edges exactly as they are
- If a new node is needed, give it a descriptive snake_case ID
- Ensure all edits maintain schema validity (question->options, instruction->next, terminal->null)
"""

_RESOLVER_HUMAN = """\
## Issue Description
{issue_description}

## Flagged Node
{flagged_node_json}

## 2-Hop Neighborhood
{neighborhood_json}

## Relevant SOP Section
{sop_section}

Return the corrected node(s) as a JSON list of WorkflowNode objects.
Only include nodes that were modified or newly created.
"""

_EXPANDER_SYSTEM = """\
You are a Graph Expansion Specialist. You expand a COARSE workflow node into
multiple detailed sub-steps while maintaining graph connectivity.

You receive:
1. The coarse node and its 2-hop neighborhood (nearby connected nodes)
2. The relevant section of the original SOP
3. The reason why the node is considered coarse

Rules:
- Break the coarse node into sequential sub-step nodes that each describe ONE
  atomic user action
- The FIRST replacement node must keep the same ID as the original coarse node
  (so incoming edges still work)
- The LAST replacement node must have its 'next' set to whatever the original
  node's 'next' was (so outgoing edges still work)
- If the original node was pointed to by a question node's options, the first
  replacement inherits that connection automatically via the same ID
- Each new intermediate node needs a descriptive snake_case ID
- Preserve role and system metadata from the original node on all sub-steps
  (unless the SOP indicates different roles/systems for different sub-steps)
- All replacement nodes must be valid: instruction nodes need 'next',
  question nodes need 'options', terminals need nothing
"""

_EXPANDER_HUMAN = """\
## Reason for Expansion
{reason}

## Coarse Node
{coarse_node_json}

## 2-Hop Neighborhood
{neighborhood_json}

## Relevant SOP Section
{sop_section}

Return the replacement nodes as a JSON list of WorkflowNode objects.
The first node MUST have id="{original_id}" to preserve incoming edges.
The last node's 'next' MUST be "{original_next}" to preserve outgoing edges.
"""

BATCH_SIZE = 12


# ---------------------------------------------------------------------------
# TripletVerifier
# ---------------------------------------------------------------------------


class TripletVerifier:
    """Verify decision-node triplets against source SOP text.

    Prioritizes low-confidence edges first, then medium. High-confidence
    edges are skipped unless flagged by topological analysis.
    """

    def __init__(self):
        self.llm = get_model("triplet")

    def extract_conditional_triplets(
        self, nodes: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Extract all (source, edge_label, target) triplets from question nodes."""
        triplets = []
        for node_id, data in nodes.items():
            if data.get("type") != "question":
                continue
            options = data.get("options", {})
            if not options:
                continue
            for label, target_id in options.items():
                target_data = nodes.get(target_id, {})
                triplets.append(
                    {
                        "source_id": node_id,
                        "source_text": data.get("text", ""),
                        "edge_label": label,
                        "target_id": target_id,
                        "target_text": target_data.get("text", f"[MISSING: {target_id}]"),
                        "confidence": data.get("confidence", "high"),
                    }
                )
        return triplets

    def verify(
        self, nodes: Dict[str, Any], source_text: str
    ) -> List[Dict[str, Any]]:
        """Verify conditional triplets, prioritizing low-confidence edges.

        Returns list of invalid triplets with explanations.
        """
        triplets = self.extract_conditional_triplets(nodes)
        if not triplets:
            return []

        # Sort by confidence: low first, then medium, skip high
        priority_order = {"low": 0, "medium": 1, "high": 2}
        triplets.sort(key=lambda t: priority_order.get(t.get("confidence", "high"), 2))

        # Filter: verify low and medium; skip high unless very few triplets
        to_verify = [t for t in triplets if t.get("confidence") != "high"]
        if not to_verify:
            # All high confidence — verify all as fallback
            to_verify = triplets

        invalid_triplets = []
        for i in range(0, len(to_verify), BATCH_SIZE):
            batch = to_verify[i : i + BATCH_SIZE]
            batch_results = self._verify_batch(batch, source_text)
            invalid_triplets.extend(batch_results)

        return invalid_triplets

    def _verify_batch(
        self, batch: List[Dict[str, str]], source_text: str
    ) -> List[Dict[str, Any]]:
        """Verify a single batch of triplets via LLM."""
        numbered = [{**t, "index": idx} for idx, t in enumerate(batch)]
        structured_llm = self.llm.with_structured_output(_TripletVerification)

        messages = [
            SystemMessage(content=_TRIPLET_SYSTEM),
            HumanMessage(
                content=_TRIPLET_HUMAN.format(
                    source_text=source_text,
                    triplets_json=json.dumps(numbered, indent=2),
                )
            ),
        ]

        invalid = []
        try:
            verification = structured_llm.invoke(messages)
            for r in verification.results:
                if not r.is_valid and r.triplet_index < len(batch):
                    invalid.append(
                        {
                            **batch[r.triplet_index],
                            "explanation": r.explanation,
                        }
                    )
        except Exception as e:
            logger.warning("Triplet verification failed: %s", e)

        return invalid


# ---------------------------------------------------------------------------
# Shared graph helpers
# ---------------------------------------------------------------------------


def _get_neighbors(node_id: str, nodes: Dict[str, Any]) -> List[str]:
    """Get direct neighbors (outgoing + incoming) of a node."""
    neighbors = []
    data = nodes.get(node_id, {})

    if data.get("next"):
        neighbors.append(data["next"])
    if data.get("options"):
        neighbors.extend(data["options"].values())

    for nid, ndata in nodes.items():
        if nid == node_id:
            continue
        if ndata.get("next") == node_id:
            neighbors.append(nid)
        if ndata.get("options") and node_id in ndata["options"].values():
            neighbors.append(nid)

    return list(set(neighbors))


def _get_2hop_neighborhood(
    node_id: str, nodes: Dict[str, Any]
) -> Dict[str, Any]:
    """Get all nodes within 2 hops of the given node."""
    neighborhood: Dict[str, Any] = {}
    hop1_ids = _get_neighbors(node_id, nodes)
    neighborhood[node_id] = nodes[node_id]

    for nid in hop1_ids:
        if nid in nodes:
            neighborhood[nid] = nodes[nid]
            hop2_ids = _get_neighbors(nid, nodes)
            for nid2 in hop2_ids:
                if nid2 in nodes:
                    neighborhood[nid2] = nodes[nid2]

    return neighborhood


# ---------------------------------------------------------------------------
# ErrorResolver
# ---------------------------------------------------------------------------


class ErrorResolver:
    """Resolve flagged issues via surgical graph edits.

    Can use FAISS vector_store (from GraphState) for retrieving relevant
    SOP sections, enhancing the 2-hop neighborhood approach.
    """

    def __init__(self):
        self.llm = get_model("resolver")

    def resolve(
        self,
        nodes: Dict[str, Any],
        feedback: str,
        source_text: str,
        vector_store: Any = None,
    ) -> Dict[str, Any]:
        """Apply surgical fixes based on feedback. Returns updated nodes."""
        flagged_ids = self._extract_flagged_ids(feedback, nodes)

        if not flagged_ids:
            return nodes

        for node_id in flagged_ids:
            if node_id not in nodes:
                continue

            neighborhood = _get_2hop_neighborhood(node_id, nodes)
            issue = self._get_issue_for_node(node_id, feedback)

            # Enhance SOP section with RAG retrieval if available
            sop_section = source_text
            if vector_store is not None:
                try:
                    relevant_docs = vector_store.similarity_search(
                        nodes[node_id].get("text", ""), k=2
                    )
                    if relevant_docs:
                        sop_section = "\n\n".join(
                            d.page_content for d in relevant_docs
                        )
                except Exception:
                    pass

            fixes = self._resolve_single(
                node_id=node_id,
                node_data=nodes[node_id],
                neighborhood=neighborhood,
                issue=issue,
                sop_section=sop_section,
            )

            for fix in fixes:
                nodes[fix["id"]] = fix

        return nodes

    def _resolve_single(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        neighborhood: Dict[str, Any],
        issue: str,
        sop_section: str,
    ) -> List[Dict[str, Any]]:
        """Resolve a single flagged node."""
        messages = [
            SystemMessage(content=_RESOLVER_SYSTEM),
            HumanMessage(
                content=_RESOLVER_HUMAN.format(
                    issue_description=issue,
                    flagged_node_json=json.dumps(node_data, indent=2),
                    neighborhood_json=json.dumps(neighborhood, indent=2),
                    sop_section=sop_section,
                )
            ),
        ]

        structured_llm = self.llm.with_structured_output(_NodePatch)

        try:
            result = structured_llm.invoke(messages)
            return [n.model_dump() for n in result.nodes]
        except Exception as e:
            logger.warning("Resolution failed for node '%s': %s", node_id, e)
            return []

    def _extract_flagged_ids(
        self, feedback: str, nodes: Dict[str, Any]
    ) -> List[str]:
        """Extract node IDs mentioned in feedback."""
        flagged = []
        for node_id in nodes:
            if node_id in feedback:
                flagged.append(node_id)
        if not flagged:
            logger.info("No node IDs found in feedback, skipping resolution")
        return flagged

    def _get_issue_for_node(self, node_id: str, feedback: str) -> str:
        """Extract the specific issue description for a node from feedback."""
        return f"Issues related to node '{node_id}':\n{feedback}"


# ---------------------------------------------------------------------------
# GranularityExpander
# ---------------------------------------------------------------------------


class GranularityExpander:
    """Expand coarse graph nodes into detailed sub-step sequences.

    For each flagged coarse node:
      1. Retrieves 2-hop neighborhood for context
      2. Optionally retrieves relevant SOP section via vector store
      3. LLM expands the node into multiple sub-steps
      4. Replaces the original node, preserving graph connectivity
    """

    def __init__(self):
        self.llm = get_model("resolver")  # same temp=0.1 as error resolution

    def expand(
        self,
        nodes: Dict[str, Any],
        feedback: str,
        source_text: str,
        vector_store: Any = None,
    ) -> Tuple[Dict[str, Any], int]:
        """Expand coarse nodes identified in feedback.

        Returns (updated_nodes, count_of_expanded_nodes).
        """
        coarse_ids = self._extract_coarse_ids(feedback, nodes)
        if not coarse_ids:
            return nodes, 0

        expanded_count = 0
        for node_id, reason in coarse_ids:
            if node_id not in nodes:
                continue

            neighborhood = _get_2hop_neighborhood(node_id, nodes)

            # Retrieve relevant SOP section via vector store if available
            sop_section = source_text
            if vector_store is not None:
                try:
                    relevant_docs = vector_store.similarity_search(
                        nodes[node_id].get("text", ""), k=2
                    )
                    if relevant_docs:
                        sop_section = "\n\n".join(
                            d.page_content for d in relevant_docs
                        )
                except Exception:
                    pass

            original_next = nodes[node_id].get("next")
            replacement = self._expand_single(
                node_id=node_id,
                node_data=nodes[node_id],
                neighborhood=neighborhood,
                reason=reason,
                sop_section=sop_section,
                original_next=original_next,
            )

            if replacement:
                # Remove original node and insert replacements
                nodes.pop(node_id, None)
                for rn in replacement:
                    nodes[rn["id"]] = rn
                expanded_count += 1

        return nodes, expanded_count

    def _expand_single(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        neighborhood: Dict[str, Any],
        reason: str,
        sop_section: str,
        original_next: str | None,
    ) -> List[Dict[str, Any]]:
        """Expand a single coarse node into sub-steps."""
        messages = [
            SystemMessage(content=_EXPANDER_SYSTEM),
            HumanMessage(
                content=_EXPANDER_HUMAN.format(
                    reason=reason,
                    coarse_node_json=json.dumps(node_data, indent=2),
                    neighborhood_json=json.dumps(neighborhood, indent=2),
                    sop_section=sop_section,
                    original_id=node_id,
                    original_next=original_next or "null",
                )
            ),
        ]

        structured_llm = self.llm.with_structured_output(_NodePatch)

        try:
            result = structured_llm.invoke(messages)
            replacement = [n.model_dump() for n in result.nodes]

            # Validate: first node must keep original ID
            if replacement and replacement[0]["id"] != node_id:
                replacement[0]["id"] = node_id

            # Validate: last non-terminal node must point to original_next
            if replacement and original_next:
                last = replacement[-1]
                if last["type"] in ("instruction", "reference") and last.get("next") != original_next:
                    last["next"] = original_next

            return replacement
        except Exception as e:
            logger.warning("Expansion failed for node '%s': %s", node_id, e)
            return []

    def _extract_coarse_ids(
        self, feedback: str, nodes: Dict[str, Any]
    ) -> List[Tuple[str, str]]:
        """Extract (node_id, reason) pairs for coarse nodes from feedback.

        Looks for the pattern: 'node_id' (reason, split into ~N steps)
        """
        import re

        pairs = []
        # Match the format produced by the analyser:
        # 'node_id' (reason, split into ~N steps)
        pattern = r"'([^']+)'\s*\(([^)]+)\)"
        for match in re.finditer(pattern, feedback):
            nid = match.group(1)
            reason = match.group(2)
            if nid in nodes:
                pairs.append((nid, reason))

        # Fallback: if no structured matches, check for any node IDs in feedback
        if not pairs:
            for nid in nodes:
                if nid in feedback and "coarse" in feedback.lower():
                    pairs.append((nid, "Flagged as coarse by analyser"))

        return pairs


# ---------------------------------------------------------------------------
# SchemaValidator
# ---------------------------------------------------------------------------


class SchemaValidator:
    """Validate and auto-fix nodes against the WorkflowNode schema."""

    def validate_and_fix(
        self, nodes: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Validate all nodes and apply deterministic fixes.

        Returns (fixed_nodes_dict, list_of_fix_descriptions).
        """
        fixed_nodes: Dict[str, Any] = {}
        fixes: List[str] = []

        for node_id, data in nodes.items():
            node_data = {**data, "id": node_id}
            fixed, fix_msgs = self._fix_single_node(node_data)
            fixed_nodes[fixed["id"]] = fixed
            fixes.extend(fix_msgs)

        return fixed_nodes, fixes

    def _fix_single_node(
        self, data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Attempt to fix a single node to pass Pydantic validation."""
        fixes: List[str] = []
        node_id = data.get("id", "unknown")
        node_type = data.get("type", "instruction")

        # Fix: terminal nodes should not have next/options
        if node_type == "terminal":
            if data.get("next"):
                fixes.append(f"Removed 'next' from terminal node '{node_id}'")
                data["next"] = None
            if data.get("options"):
                fixes.append(f"Removed 'options' from terminal node '{node_id}'")
                data["options"] = None

        # Fix: question nodes must have options
        if node_type == "question" and not data.get("options"):
            fixes.append(
                f"Node '{node_id}' is question-type but has no options — "
                f"converting to instruction"
            )
            data["type"] = "instruction"
            node_type = "instruction"

        # Fix: instruction nodes must have next
        if node_type == "instruction" and not data.get("next"):
            fixes.append(
                f"Node '{node_id}' is instruction-type but has no 'next' — "
                f"needs manual resolution"
            )

        # Fix: question nodes should not have 'next'
        if node_type == "question" and data.get("next"):
            fixes.append(f"Removed 'next' from question node '{node_id}' (use options)")
            data["next"] = None

        try:
            node = WorkflowNode(**data)
            return node.model_dump(), fixes
        except Exception as e:
            fixes.append(f"Validation failed for '{node_id}': {e}")
            return data, fixes


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def refine(state: GraphState) -> GraphState:
    """Run triplet verification -> granularity expansion -> error resolution -> schema validation.

    Mutates and returns the state with updated nodes and incremented iteration.
    """
    iteration = state.get("iteration", 0) + 1
    logger.info("[REFINER iter %d] Starting refinement cycle...", iteration)

    nodes = state["nodes"]
    source_text = state["source_text"]
    feedback = state["feedback"]
    vector_store = state.get("vector_store")

    triplet_verifier = TripletVerifier()
    granularity_expander = GranularityExpander()
    error_resolver = ErrorResolver()
    schema_validator = SchemaValidator()

    # 1. Triplet verification (conditionals only, low-confidence first)
    logger.info("  [REFINER iter %d] Step 1/4: Triplet verification...", iteration)
    invalid_triplets = triplet_verifier.verify(nodes, source_text)
    logger.info("    %d invalid triplets found", len(invalid_triplets))
    for t in invalid_triplets:
        logger.info("    INVALID: %s --(%s)--> %s: %s",
                     t["source_id"], t["edge_label"], t["target_id"],
                     t.get("explanation", "")[:100])

    if invalid_triplets:
        triplet_feedback = "\n".join(
            f"Invalid triplet: {t['source_id']} --({t['edge_label']})--> "
            f"{t['target_id']}: {t['explanation']}"
            for t in invalid_triplets
        )
        feedback = f"{feedback}\n\nTriplet Issues:\n{triplet_feedback}"

    # 2. Granularity expansion (break coarse nodes into sub-steps)
    logger.info("  [REFINER iter %d] Step 2/4: Granularity expansion...", iteration)
    if "coarse" in feedback.lower():
        nodes_before = len(nodes)
        nodes, expanded_count = granularity_expander.expand(
            nodes, feedback, source_text, vector_store
        )
        nodes_after = len(nodes)
        logger.info("    Expanded %d coarse nodes. Nodes: %d -> %d (delta: %+d)",
                     expanded_count, nodes_before, nodes_after, nodes_after - nodes_before)
    else:
        logger.info("    No coarse nodes flagged — skipping")

    # 3. Error resolution (surgical LLM edits, with optional RAG)
    logger.info("  [REFINER iter %d] Step 3/4: Error resolution...", iteration)
    if feedback.strip():
        nodes_before = len(nodes)
        nodes = error_resolver.resolve(nodes, feedback, source_text, vector_store)
        nodes_after = len(nodes)
        logger.info("    Nodes: %d -> %d (delta: %+d)", nodes_before, nodes_after, nodes_after - nodes_before)
    else:
        logger.info("    No feedback to resolve — skipping")

    # 4. Schema validation (deterministic fixes)
    logger.info("  [REFINER iter %d] Step 4/4: Schema validation...", iteration)
    nodes, fix_msgs = schema_validator.validate_and_fix(nodes)
    if fix_msgs:
        for msg in fix_msgs:
            logger.info("    FIX: %s", msg)
    else:
        logger.info("    All nodes valid")

    state["nodes"] = nodes
    state["iteration"] = iteration
    if fix_msgs:
        state["feedback"] = (
            state.get("feedback", "") + "\nSchema fixes: " + "; ".join(fix_msgs)
        )

    logger.info("  [REFINER iter %d] Complete. %d nodes in graph.", iteration, len(nodes))
    return state
