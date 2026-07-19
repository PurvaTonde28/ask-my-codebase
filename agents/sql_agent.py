from pathlib import Path
import sqlite3
import time

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from groq import BadRequestError as GroqBadRequestError

# Must match wherever Phase 6's build_commit_database() actually wrote the
# database. Getting this wrong doesn't raise a clean error -- sqlite3.connect()
# silently creates an empty db at a bad path, and every query then fails with
# a misleading "no such table" error that looks like a bad-SQL mistake, not a
# wrong-path mistake. Double check this against your real Phase 6 output path.
DATABASE_PATH = "data/commits.db"

SYSTEM_PROMPT = """
You are an SQL assistant.

The database has two tables.

commits(
    hash TEXT PRIMARY KEY,
    author TEXT,
    email TEXT,
    date TEXT,
    message TEXT
)

file_changes(
    hash TEXT,
    file_path TEXT,
    insertions INTEGER,
    deletions INTEGER
)

Rules:
- Only generate SELECT queries.
- Never modify the database.
- Use the run_sql_query tool.
- If the tool returns an SQL error, fix your query and try again.
- Summarize the results in plain English.
- Do not dump raw SQL rows unless the user explicitly asks.
"""


@tool
def run_sql_query(query: str) -> str:
    """Run a read-only SQL SELECT query against the commit history database
    (commits, file_changes tables) and return up to 50 result rows as text."""
    query = query.strip()

    if not query.lower().startswith("select"):
        return "Only SELECT queries are allowed."

    if not Path(DATABASE_PATH).exists():
        return (
            f"Database not found at '{DATABASE_PATH}'. "
            "Run build_commit_database() first, or check DATABASE_PATH is correct."
        )

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    try:
        cursor.execute(query)
        rows = cursor.fetchmany(50)
        if not rows:
            return "Query returned no rows."
        return "\n".join(str(row) for row in rows)

    except sqlite3.Error as e:
        return f"SQL Error: {e}"

    finally:
        connection.close()


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

sql_agent = create_agent(
    model=llm,
    tools=[run_sql_query],
    system_prompt=SYSTEM_PROMPT,
)


def sql_node(question: str, max_retries: int = 2) -> str:
    """
    Entry point for the SQL agent.
    This will later be called by the LangGraph supervisor.

    Retries on groq.BadRequestError: Groq's Llama models occasionally emit
    malformed tool-call syntax (e.g. literal "<function=...>" text instead of
    a proper structured tool call), especially on unusual/complex questions.
    This is a known model-level flakiness, not something a fixed prompt
    reliably prevents -- retrying the same call is the practical fix. Same
    pattern as rag_node(), since both agents share this same exposure.
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = sql_agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question,
                        }
                    ]
                }
            )
            return response["messages"][-1].content

        except GroqBadRequestError as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1)
                continue

    return (
        f"I ran into a temporary error generating a response after "
        f"{max_retries + 1} attempts. Please try rephrasing your question. "
        f"(details: {last_error})"
    )