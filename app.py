"""
Ask My Codebase -- CLI entry point.

First run: clones the target repo, splits code + docs, builds the FAISS
index and commit-history database (one-time setup, several minutes on CPU).
Every run after that: skips straight to the question loop.
"""
import uuid
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from ingestion.clone_repo import clone_repository
from ingestion.read_files import walk_repository
from ingestion.parse_git_log import build_commit_database
from ingestion.filters import is_non_english_doc
from splitters.python_code_splitter import split_python_file
from splitters.text_structure_splitter import split_markdown_file
from vectorstore.build_index import build_index, load_index

console = Console()

DEFAULT_REPO_URL = "https://github.com/PurvaTonde28/rag-eval-pipeline"
DB_PATH = "data/commits.db"
INDEX_DIR = "vectorstore/data"


def run_ingestion(repo_url: str) -> Path:
    """Full pipeline: clone -> walk -> split -> embed -> build SQL db.
    Every step here reuses an already-tested function from an earlier
    phase -- this function only orchestrates, it doesn't parse or embed
    anything itself."""
    console.print(
        Panel(
            "Setting up for the first time: cloning the repo, splitting code "
            "and docs, building embeddings and the commit-history database.\n"
            "This can take several minutes on CPU -- it only happens once.",
            title="First-time setup",
            style="yellow",
        )
    )

    repo_path = clone_repository(repo_url)
    console.print(f"[dim]Cloned to {repo_path}[/dim]")

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

    console.print(f"[dim]{len(code_chunks)} code chunks, {len(text_chunks)} text chunks[/dim]")

    build_index(
        code_chunks,
        text_chunks,
        output_dir=INDEX_DIR,
        cooldown_every=1000,
        cooldown_seconds=5.0,
    )
    build_commit_database(repo_path, DB_PATH)

    console.print("[green]Setup complete.[/green]")
    return repo_path


def ensure_ingested(repo_url: str) -> None:
    index_file = Path(INDEX_DIR) / "index.faiss"
    db_file = Path(DB_PATH)

    if index_file.exists() and db_file.exists():
        console.print("[dim]Existing index and commit database found -- skipping setup.[/dim]")
    else:
        run_ingestion(repo_url)

    load_index(output_dir=INDEX_DIR)


def print_welcome() -> None:
    welcome_text = (
        "[bold]Ask My Codebase[/bold]\n\n"
        "A multi-agent assistant that answers questions about a codebase by "
        "routing to either a RAG agent (code & docs) or a SQL agent (commit history).\n\n"
        "Try asking:\n"
        "  - how does dependency injection work\n"
        "  - how many commits touched routing.py\n"
        "  - explain the routing decorator\n\n"
        "Follow-ups work now -- try asking something, then 'give me an "
        "example of that'.\n"
        "Type 'reset' to start a new conversation, 'exit' to quit."
    )
    console.print(Panel(welcome_text, title="Welcome", border_style="cyan"))


def print_answer(state: dict) -> None:
    route = state["route"]["destination"]
    reasoning = state["route"]["reasoning"]
    answer = state["final_answer"]

    route_style = "cyan" if route == "rag" else "yellow"
    console.print(f"[{route_style}]routed to: {route}[/{route_style}] [dim]({reasoning})[/dim]")
    console.print(Panel(Markdown(answer), border_style=route_style))


def main() -> None:
    ensure_ingested(DEFAULT_REPO_URL)

    # Imported here, not at module top-level: agents/rag_agent.py calls
    # load_index() at ITS import time, which would crash with
    # FileNotFoundError if the index doesn't exist yet. Importing graph only
    # after ensure_ingested() guarantees the index is already built by the
    # time this import (and rag_agent's load_index call inside it) runs.
    from agents.graph import graph

    # One thread_id = one conversation as far as the checkpointer is
    # concerned. Regenerating it (via 'reset') is how we start a fresh
    # conversation without restarting the process -- the old thread's
    # messages just become unreachable, nothing needs to be deleted since
    # InMemorySaver only lives for this process anyway.
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print_welcome()

    while True:
        try:
            question = console.input("\n[bold]>[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            console.print("[dim]Goodbye.[/dim]")
            break
        if question.lower() == "reset":
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            console.print("[dim]Conversation reset.[/dim]")
            continue

        try:
            state = graph.invoke({"question": question}, config=config)
            print_answer(state)
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye.[/dim]")
            break
        except Exception as e:
            console.print(f"[red]Something went wrong: {e}[/red]")


if __name__ == "__main__":
    main()