"""
Ask My Codebase -- Streamlit UI.

Wraps the same pipeline as app.py (CLI), reusing every already-tested
component from earlier phases. This file only adds a chat interface on top
-- it doesn't contain any new ingestion, retrieval, or agent logic.

Run with: streamlit run streamlit_app.py
"""
from pathlib import Path

import streamlit as st

from ingestion.clone_repo import clone_repository
from ingestion.read_files import walk_repository
from ingestion.parse_git_log import build_commit_database
from splitters.python_code_splitter import split_python_file
from splitters.text_structure_splitter import split_markdown_file
from vectorstore.build_index import build_index

DEFAULT_REPO_URL = "https://github.com/tiangolo/fastapi"
DB_PATH = "data/commits.db"
INDEX_DIR = "vectorstore/data"

st.set_page_config(page_title="Ask My Codebase", page_icon="🔍", layout="centered")


def is_non_english_doc(path: Path) -> bool:
    """Skip translated docs (docs/<lang>/docs/...), keep docs/en/... and
    anything not under docs/ at all (like the root README)."""
    parts = path.parts
    if "docs" in parts:
        idx = parts.index("docs")
        if idx + 1 < len(parts):
            candidate = parts[idx + 1]
            if candidate != "en":
                return True
    return False


def run_ingestion(repo_url: str) -> Path:
    """Full pipeline: clone -> walk -> split -> embed -> build SQL db.
    Identical logic to app.py's run_ingestion -- same functions, same order."""
    repo_path = clone_repository(repo_url)

    records = walk_repository(repo_path)
    py_records = [r for r in records if r.extension == ".py"]
    md_records = [r for r in records if r.extension in (".md", ".rst", ".txt")]
    md_records = [r for r in md_records if not is_non_english_doc(r.path)]

    code_chunks = []
    for r in py_records:
        code_chunks.extend(split_python_file(r))

    text_chunks = []
    for r in md_records:
        text_chunks.extend(split_markdown_file(r))

    build_index(
        code_chunks,
        text_chunks,
        output_dir=INDEX_DIR,
        cooldown_every=1000,
        cooldown_seconds=5.0,
    )
    build_commit_database(repo_path, DB_PATH)

    return repo_path


@st.cache_resource(show_spinner=False)
def get_graph():
    """
    Cached across reruns. Streamlit re-executes this whole script top-to-
    bottom on every single user interaction (every chat message) -- without
    this cache, ingestion would try to re-run and the embedding model would
    reload on every message. @st.cache_resource makes this run exactly once
    per server session, not once per interaction.
    """
    index_file = Path(INDEX_DIR) / "index.faiss"
    db_file = Path(DB_PATH)

    if not (index_file.exists() and db_file.exists()):
        run_ingestion(DEFAULT_REPO_URL)

    # Imported here, not at module top-level, for the same reason as app.py:
    # agents/rag_agent.py calls load_index() at ITS import time, which
    # requires the index to already exist on disk. This import must happen
    # after ingestion is confirmed done.
    from agents.graph import graph

    return graph


# --- UI ---

st.title("🔍 Ask My Codebase")
st.caption(
    "Multi-agent supervisor routing questions to a RAG agent (code & docs) "
    "or a SQL agent (commit history)."
)

with st.spinner("Setting up (first run only -- can take a few minutes)..."):
    graph = get_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Try asking")
    st.markdown(
        "**RAG (code & docs)**\n"
        "- How does dependency injection work?\n"
        "- Explain the routing decorator\n\n"
        "**SQL (commit history)**\n"
        "- How many commits touched routing.py?\n"
        "- Who are the top contributors?"
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("route"):
            badge = "blue" if msg["route"] == "rag" else "orange"
            st.markdown(f":{badge}[routed to: {msg['route']}]  \n*{msg['reasoning']}*")
        st.markdown(msg["content"])

question = st.chat_input("Ask about the codebase...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        route, reasoning, answer = None, None, None
        with st.spinner("Thinking..."):
            try:
                state = graph.invoke({"question": question})
                route = state["route"].destination
                reasoning = state["route"].reasoning
                answer = state["final_answer"]
            except Exception as e:
                answer = f"Something went wrong: {e}"

        if route:
            badge = "blue" if route == "rag" else "orange"
            st.markdown(f":{badge}[routed to: {route}]  \n*{reasoning}*")
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "route": route,
            "reasoning": reasoning,
        }
    )