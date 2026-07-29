"""
Ask My Codebase -- Streamlit UI, multi-repo capable.

Reuses every already-tested component from earlier phases. Each processed
repo gets its own index/db files (keyed by repo name), so switching repos
never silently answers using a different repo's data.

Run with: streamlit run streamlit_app.py
"""

from dotenv import load_dotenv
load_dotenv()
import uuid
from pathlib import Path

import streamlit as st

from ingestion.clone_repo import clone_repository
from ingestion.read_files import walk_repository
from ingestion.parse_git_log import build_commit_database
from splitters.python_code_splitter import split_python_file
from splitters.text_structure_splitter import split_markdown_file
from vectorstore.build_index import build_index, load_index
import agents.sql_agent as sql_agent_module

st.set_page_config(page_title="Repoza", page_icon="🔍", layout="centered")


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


def paths_for_repo(repo_name: str) -> tuple[str, str]:
    """Every repo gets its own index dir and db file, keyed by repo name.
    This is what prevents switching repos from silently reusing another
    repo's data -- each has a completely separate on-disk footprint."""
    index_dir = f"vectorstore/data_{repo_name}"
    db_path = f"data/commits_{repo_name}.db"
    return index_dir, db_path


def process_repo(repo_url: str) -> dict:
    """
    Ensures a repo is ingested (skips the slow pipeline if it already was),
    then activates it as the currently-loaded repo for question answering.
    Returns a dict describing the active repo, stored in session_state.
    """
    repo_path = clone_repository(repo_url)
    repo_name = repo_path.name
    index_dir, db_path = paths_for_repo(repo_name)

    index_file = Path(index_dir) / "index.faiss"
    db_file = Path(db_path)

    if not (index_file.exists() and db_file.exists()):
        st.write(f"First time processing **{repo_name}** -- this can take a few minutes.")

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

        st.write(f"{len(code_chunks)} code chunks, {len(text_chunks)} text chunks")

        build_index(
            code_chunks,
            text_chunks,
            output_dir=index_dir,
            cooldown_every=1000,
            cooldown_seconds=5.0,
        )
        build_commit_database(repo_path, db_path)

    # Activate this repo: point the shared FAISS singleton and the SQL
    # agent's DATABASE_PATH at THIS repo's files. This is the step that
    # actually makes switching repos correct -- without it, a previously
    # loaded repo's data would stay active in memory regardless of what's
    # on disk for the new one.
    load_index(output_dir=index_dir)
    sql_agent_module.DATABASE_PATH = db_path

    return {"repo_name": repo_name, "repo_url": repo_url, "index_dir": index_dir, "db_path": db_path}


@st.cache_resource(show_spinner=False)
def get_graph():
    """Graph construction (agents, LLM clients) is repo-independent -- only
    the underlying data (FAISS index, SQL db) needs to change per repo, and
    that's handled by process_repo()'s activation step, not here. Safe to
    build this exactly once per server session."""
    from agents.graph import graph

    return graph


# --- UI ---

st.title("💬 Repoza - AI-powered repository intelligence")
# st.caption(
#     "Multi-agent supervisor routing questions to a RAG agent (code & docs) "
#     "or a SQL agent (commit history)."
# )

if "active_repo" not in st.session_state:
    st.session_state.active_repo = None
if "messages" not in st.session_state:
    st.session_state.messages = {}  # keyed by repo_name, so each repo has its own chat history
if "thread_ids" not in st.session_state:
    # keyed by repo_name, same idea as messages above -- but this is what
    # LangGraph's checkpointer actually uses to look up conversation state.
    # `messages` above is purely for rendering the chat UI; this is what
    # gives the agents real memory of prior turns. Two different browser
    # tabs/sessions get separate st.session_state, so separate thread_ids,
    # so separate conversations, even though the compiled `graph` object
    # (with its one InMemorySaver) is shared across the whole server via
    # @st.cache_resource below.
    st.session_state.thread_ids = {}

with st.sidebar:
    st.subheader("Repository")
    repo_url = st.text_input(
        "GitHub URL",
        value="https://github.com/PurvaTonde28/rag-eval-pipeline",
        help="Any public GitHub repo. Already-processed repos load instantly on repeat runs.",
    )
    process_clicked = st.button("Process repository", type="primary")

    if st.session_state.active_repo:
        st.success(f"Active: {st.session_state.active_repo['repo_name']}")

    st.divider()
    st.subheader("Try asking")
    st.markdown(
        "**RAG (code & docs)**\n"
        "- How does dependency injection work?\n"
        "- Explain the routing decorator\n\n"
        "**SQL (commit history)**\n"
        "- How many commits touched routing.py?\n"
        "- Who are the top contributors?"
    )

    if st.session_state.active_repo and st.button("Clear this repo's conversation"):
        repo_name = st.session_state.active_repo["repo_name"]
        st.session_state.messages[repo_name] = []
        # Regenerate, don't just clear -- InMemorySaver has no delete-thread
        # API we can rely on across versions, so the simplest, most robust
        # fix is to abandon the old thread_id and start a new one. The old
        # thread's messages just become unreachable; harmless, since this
        # is in-memory only anyway.
        st.session_state.thread_ids[repo_name] = str(uuid.uuid4())
        st.rerun()

if process_clicked:
    if not repo_url.strip():
        st.sidebar.error("Enter a GitHub URL first.")
    else:
        with st.spinner("Processing repository..."):
            try:
                active = process_repo(repo_url.strip())
                st.session_state.active_repo = active
                st.session_state.messages.setdefault(active["repo_name"], [])
                st.session_state.thread_ids.setdefault(
                    active["repo_name"], str(uuid.uuid4())
                )
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed to process repo: {e}")

if not st.session_state.active_repo:
    st.info("Enter a GitHub repo URL in the sidebar and click **Process repository** to get started.")
    st.stop()

graph = get_graph()
repo_name = st.session_state.active_repo["repo_name"]
history = st.session_state.messages.setdefault(repo_name, [])

for msg in history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("route"):
            badge = "blue" if msg["route"] == "rag" else "orange"
            st.markdown(f":{badge}[routed to: {msg['route']}]  \n*{msg['reasoning']}*")
        st.markdown(msg["content"])

question = st.chat_input(f"Ask about {repo_name}...")

if question:
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        route, reasoning, answer = None, None, None
        with st.spinner("Thinking..."):
            try:
                # Re-activate in case another repo was processed since this
                # session started -- cheap (just points the singleton/global
                # at the right files again), and guarantees correctness even
                # if st.session_state.active_repo changed between reruns.
                active = st.session_state.active_repo
                load_index(output_dir=active["index_dir"])
                sql_agent_module.DATABASE_PATH = active["db_path"]

                thread_id = st.session_state.thread_ids[repo_name]
                config = {"configurable": {"thread_id": thread_id}}

                state = graph.invoke({"question": question}, config=config)
                route = state["route"]["destination"]
                reasoning = state["route"]["reasoning"]
                answer = state["final_answer"]
            except Exception as e:
                answer = f"Something went wrong: {e}"

        if route:
            badge = "blue" if route == "rag" else "orange"
            st.markdown(f":{badge}[routed to: {route}]  \n*{reasoning}*")
        st.markdown(answer)

    history.append(
        {
            "role": "assistant",
            "content": answer,
            "route": route,
            "reasoning": reasoning,
        }
    )