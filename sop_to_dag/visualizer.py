"""Standalone Streamlit app: visualize & edit SOP workflow graphs via chat.

Run:
    streamlit run visualizer/app.py

Requires:
    pip install streamlit pyvis langchain-openai
"""

import json
import os
import re
import tempfile
from copy import deepcopy
from pathlib import Path

import streamlit as st
from pyvis.network import Network

st.set_page_config(page_title="Graph Visualizer", layout="wide")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODE_COLORS = {
    "instruction": "#4A90D9",
    "question": "#F5A623",
    "terminal": "#D0021B",
    "reference": "#7ED321",
}

NODE_SHAPES = {
    "instruction": "box",
    "question": "diamond",
    "terminal": "dot",
    "reference": "box",
}

_EDITOR_SYSTEM = """\
You are a Graph Editor. You make precise, surgical modifications to workflow \
graphs based on user instructions.

GRAPH FORMAT (each key is a node ID):
{
  "node_id": {
    "type": "instruction|question|terminal|reference",
    "text": "description of what this node does",
    "next": "next_node_id" or null,
    "options": {"Yes": "node_id", "No": "node_id"} or null,
    "role": "who performs this" or null,
    "system": "software used" or null,
    "confidence": "high|medium|low",
    "external_ref": "url or doc name" or null
  }
}

NODE RULES:
- "instruction": must have "next", "options" must be null
- "question": must have "options" with exactly {"Yes": "id", "No": "id"}, \
"next" must be null, text should end with "?"
- "terminal": both "next" and "options" must be null
- "reference": must have "next" and "external_ref"
- All IDs must be descriptive snake_case

SURGICAL EDITS — modify_nodes should ONLY contain the fields being changed, \
not the entire node. For example, to change just the "next" pointer of a node:
  "modify_nodes": {"some_node": {"next": "new_target"}}
Do NOT include unchanged fields like "type", "text", "role" etc.

RESPOND WITH ONLY a JSON object (no markdown fences, no extra text):
{
  "explanation": "brief description of what will change",
  "add_nodes": {"new_node_id": {"type": "...", "text": "...", ...}, ...},
  "modify_nodes": {"existing_id": {"field_to_change": "new_value"}, ...},
  "remove_nodes": ["id1", "id2"]
}

If the user asks a question about the graph (not a modification), respond with:
{"explanation": "your answer here", "add_nodes": {}, "modify_nodes": {}, "remove_nodes": []}
"""

# ---------------------------------------------------------------------------
# Graph rendering
# ---------------------------------------------------------------------------


def render_graph(graph: dict) -> str:
    """Render graph dict as interactive pyvis HTML string."""
    net = Network(
        height="700px",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="white",
    )

    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "hierarchicalRepulsion": {
                "centralGravity": 0.0,
                "springLength": 200,
                "springConstant": 0.01,
                "nodeDistance": 180
            },
            "solver": "hierarchicalRepulsion"
        },
        "layout": {
            "hierarchical": {
                "enabled": true,
                "direction": "UD",
                "sortMethod": "hubsize",
                "levelSeparation": 150,
                "nodeSpacing": 200
            }
        },
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
            "color": {"color": "#848484", "highlight": "#ffffff"},
            "smooth": {"type": "cubicBezier", "roundness": 0.4},
            "font": {"size": 11, "color": "#cccccc", "strokeWidth": 0}
        },
        "nodes": {
            "font": {"size": 12, "face": "arial"},
            "borderWidth": 2,
            "shadow": true
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true
        }
    }
    """)

    # Add nodes
    for node_id, data in graph.items():
        node_type = data.get("type", "instruction")
        color = NODE_COLORS.get(node_type, "#999999")
        shape = NODE_SHAPES.get(node_type, "box")

        label = f"{node_id}\n{data.get('text', '')[:60]}"
        title_parts = [
            f"ID: {node_id}",
            f"Type: {node_type}",
            f"Text: {data.get('text', '')}",
        ]
        if data.get("role"):
            title_parts.append(f"Role: {data['role']}")
        if data.get("system"):
            title_parts.append(f"System: {data['system']}")
        if data.get("confidence", "high") != "high":
            title_parts.append(f"Confidence: {data['confidence']}")
        title = "\n".join(title_parts)

        net.add_node(
            node_id,
            label=label,
            title=title,
            color=color,
            shape=shape,
            size=25 if node_type == "question" else 20,
        )

    # Add edges
    for node_id, data in graph.items():
        next_id = data.get("next")
        if next_id and next_id in graph:
            net.add_edge(node_id, next_id, color="#848484", width=1.5)

        options = data.get("options")
        if options:
            for label, target_id in options.items():
                if target_id in graph:
                    edge_color = "#7ED321" if label == "Yes" else "#D0021B"
                    net.add_edge(
                        node_id,
                        target_id,
                        label=label,
                        color=edge_color,
                        width=2,
                    )

    # Write to temp file and read HTML
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        tmp_path = f.name

    html = Path(tmp_path).read_text()
    os.unlink(tmp_path)
    return html


# ---------------------------------------------------------------------------
# LLM chat for graph editing
# ---------------------------------------------------------------------------


def get_llm():
    """Get ChatOpenAI instance from sidebar config."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=st.session_state.get("model_name", "gpt-oss-120b"),
        temperature=0.1,
        base_url=st.session_state.get("api_base", None) or None,
        api_key=st.session_state.get("api_key", None) or "not-set",
    )


def apply_patch(graph: dict, patch: dict) -> dict:
    """Apply an add/modify/remove patch to the graph. Modify is a merge."""
    g = deepcopy(graph)

    # Remove nodes
    for node_id in patch.get("remove_nodes", []):
        g.pop(node_id, None)

    # Add nodes (full node dicts for new nodes)
    for node_id, node_data in patch.get("add_nodes", {}).items():
        g[node_id] = node_data

    # Modify nodes — surgical merge, only update provided fields
    for node_id, changed_fields in patch.get("modify_nodes", {}).items():
        if node_id in g:
            g[node_id].update(changed_fields)

    return g


def format_patch_preview(patch: dict, graph: dict) -> str:
    """Format a patch as a human-readable diff for confirmation."""
    lines = []

    add_nodes = patch.get("add_nodes", {})
    modify_nodes = patch.get("modify_nodes", {})
    remove_nodes = patch.get("remove_nodes", [])

    if add_nodes:
        lines.append("**Add nodes:**")
        for node_id, data in add_nodes.items():
            ntype = data.get("type", "?")
            text = data.get("text", "")[:80]
            lines.append(f"- `{node_id}` ({ntype}): {text}")
            if data.get("next"):
                lines.append(f"  next -> `{data['next']}`")
            if data.get("options"):
                for lbl, tgt in data["options"].items():
                    lines.append(f"  {lbl} -> `{tgt}`")

    if modify_nodes:
        lines.append("**Modify nodes:**")
        for node_id, changed in modify_nodes.items():
            current = graph.get(node_id, {})
            lines.append(f"- `{node_id}`:")
            for field, new_val in changed.items():
                old_val = current.get(field)
                if old_val != new_val:
                    old_display = json.dumps(old_val) if not isinstance(old_val, str) else old_val
                    new_display = json.dumps(new_val) if not isinstance(new_val, str) else new_val
                    lines.append(f"  `{field}`: ~~{old_display}~~ -> {new_display}")
                else:
                    lines.append(f"  `{field}`: {new_val} (unchanged)")

    if remove_nodes:
        lines.append("**Remove nodes:**")
        for node_id in remove_nodes:
            text = graph.get(node_id, {}).get("text", "")[:60]
            lines.append(f"- `{node_id}`: {text}")

    return "\n".join(lines) if lines else "No changes."


def get_patch_from_llm(user_message: str, graph: dict) -> tuple[dict, str]:
    """Send user instruction + graph to LLM, return (patch_dict, explanation)."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    llm = get_llm()

    graph_json = json.dumps(graph, indent=2)
    human_content = (
        f"## Current Graph ({len(graph)} nodes)\n"
        f"```json\n{graph_json}\n```\n\n"
        f"## User Instruction\n{user_message}"
    )

    messages = [
        SystemMessage(content=_EDITOR_SYSTEM),
    ]

    # Include recent chat history for context (only last 6 messages)
    for msg in st.session_state.get("chat_history", [])[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=human_content))

    response = llm.invoke(messages)
    response_text = response.content.strip()

    # Parse JSON from response (handle markdown fences)
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        return {}, f"Could not parse LLM response: {response_text[:200]}"

    try:
        patch = json.loads(json_match.group())
    except json.JSONDecodeError:
        return {}, f"Invalid JSON in response: {response_text[:200]}"

    explanation = patch.get("explanation", "")
    return patch, explanation


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------


def graph_stats(graph: dict) -> dict:
    """Compute basic graph statistics."""
    types = {}
    edge_count = 0
    orphans = []
    terminals = []

    all_targets = set()
    for data in graph.values():
        if data.get("next"):
            all_targets.add(data["next"])
        if data.get("options"):
            all_targets.update(data["options"].values())

    for node_id, data in graph.items():
        t = data.get("type", "unknown")
        types[t] = types.get(t, 0) + 1
        if data.get("next"):
            edge_count += 1
        if data.get("options"):
            edge_count += len(data["options"])
        if t == "terminal":
            terminals.append(node_id)
        if node_id not in all_targets:
            orphans.append(node_id)

    return {
        "nodes": len(graph),
        "edges": edge_count,
        "types": types,
        "orphans": orphans,
        "terminals": terminals,
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def main():
    # Session state init
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "graph_history" not in st.session_state:
        st.session_state.graph_history = []
    if "pending_patch" not in st.session_state:
        st.session_state.pending_patch = None
    if "pending_explanation" not in st.session_state:
        st.session_state.pending_explanation = None

    # Sidebar: config & file upload
    with st.sidebar:
        st.header("Settings")

        st.text_input(
            "Model Name",
            value="gpt-oss-120b",
            key="model_name",
        )
        st.text_input(
            "API Base URL (optional)",
            value="",
            key="api_base",
            type="default",
        )
        st.text_input(
            "API Key",
            value=os.environ.get("OPENAI_API_KEY", ""),
            key="api_key",
            type="password",
        )

        st.divider()
        st.header("Load Graph")

        # Option 1: File path on disk
        file_path = st.text_input(
            "Graph JSON path",
            placeholder="e.g. output/auto_graph.json",
        )
        if file_path and st.button("Load from path"):
            p = Path(file_path).expanduser()
            if p.exists():
                try:
                    graph = json.loads(p.read_text())
                    st.session_state.graph = graph
                    st.session_state.graph_history = [deepcopy(graph)]
                    st.session_state.chat_history = []
                    st.session_state.pending_patch = None
                    st.session_state.pending_explanation = None
                    st.success(f"Loaded {len(graph)} nodes from {p.name}")
                except json.JSONDecodeError:
                    st.error("Invalid JSON file")
            else:
                st.error(f"File not found: {p}")

        # Option 2: File upload
        uploaded = st.file_uploader("Or upload graph JSON", type=["json"])
        if uploaded:
            try:
                graph = json.loads(uploaded.read())
                st.session_state.graph = graph
                st.session_state.graph_history = [deepcopy(graph)]
                st.session_state.chat_history = []
                st.session_state.pending_patch = None
                st.session_state.pending_explanation = None
                st.success(f"Loaded {len(graph)} nodes")
            except json.JSONDecodeError:
                st.error("Invalid JSON file")

        st.divider()

        # Graph stats
        if st.session_state.graph:
            stats = graph_stats(st.session_state.graph)
            st.header("Graph Stats")
            st.metric("Nodes", stats["nodes"])
            st.metric("Edges", stats["edges"])
            for t, count in sorted(stats["types"].items()):
                st.caption(f"{t}: {count}")
            if len(stats["orphans"]) <= 3:
                st.caption(f"Entry points: {', '.join(stats['orphans'])}")
            else:
                st.caption(f"Entry points: {len(stats['orphans'])}")

            st.divider()

            # Undo / Download
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Undo", disabled=len(st.session_state.graph_history) <= 1):
                    st.session_state.graph_history.pop()
                    st.session_state.graph = deepcopy(
                        st.session_state.graph_history[-1]
                    )
                    st.session_state.pending_patch = None
                    st.session_state.pending_explanation = None
                    st.rerun()
            with col2:
                st.download_button(
                    "Download",
                    data=json.dumps(st.session_state.graph, indent=2),
                    file_name="graph_edited.json",
                    mime="application/json",
                )

            # Legend
            st.divider()
            st.header("Legend")
            for ntype, color in NODE_COLORS.items():
                st.markdown(
                    f'<span style="color:{color}">&#9632;</span> {ntype}',
                    unsafe_allow_html=True,
                )

    # Main area
    if not st.session_state.graph:
        st.title("Graph Visualizer")
        st.info("Upload a graph JSON file from the sidebar to get started.")
        return

    # Layout: graph (left) + chat (right)
    graph_col, chat_col = st.columns([7, 3])

    with graph_col:
        st.subheader(f"Workflow Graph ({len(st.session_state.graph)} nodes)")
        html = render_graph(st.session_state.graph)
        st.components.v1.html(html, height=720, scrolling=True)

    with chat_col:
        st.subheader("Graph Editor Chat")

        # Chat history display
        chat_container = st.container(height=480)
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Pending patch confirmation
        if st.session_state.pending_patch is not None:
            patch = st.session_state.pending_patch
            explanation = st.session_state.pending_explanation

            has_changes = (
                patch.get("add_nodes")
                or patch.get("modify_nodes")
                or patch.get("remove_nodes")
            )

            if has_changes:
                preview = format_patch_preview(patch, st.session_state.graph)
                st.info(f"**Proposed changes:**\n\n{preview}")

                col_apply, col_reject = st.columns(2)
                with col_apply:
                    if st.button("Apply", type="primary", use_container_width=True):
                        updated = apply_patch(st.session_state.graph, patch)
                        st.session_state.graph = updated
                        st.session_state.graph_history.append(deepcopy(updated))
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"{explanation}\n\n*Applied — {len(updated)} nodes*",
                        })
                        st.session_state.pending_patch = None
                        st.session_state.pending_explanation = None
                        st.rerun()
                with col_reject:
                    if st.button("Reject", use_container_width=True):
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"~~{explanation}~~\n\n*Rejected — no changes made.*",
                        })
                        st.session_state.pending_patch = None
                        st.session_state.pending_explanation = None
                        st.rerun()
            else:
                # No actual changes — just a question answer
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": explanation,
                })
                st.session_state.pending_patch = None
                st.session_state.pending_explanation = None
                st.rerun()

        # Chat input (disabled while a patch is pending confirmation)
        user_input = st.chat_input(
            "e.g. 'Connect node_a to node_b'"
            if st.session_state.pending_patch is None
            else "Apply or reject the pending change first",
            disabled=st.session_state.pending_patch is not None,
        )

        if user_input:
            # Add user message
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            # Process with LLM
            with st.spinner("Thinking..."):
                try:
                    patch, explanation = get_patch_from_llm(
                        user_input, st.session_state.graph
                    )
                    st.session_state.pending_patch = patch
                    st.session_state.pending_explanation = explanation
                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Error: {e}",
                    })

            st.rerun()


if __name__ == "__main__":
    main()
