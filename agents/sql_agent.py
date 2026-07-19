from pathlib import Path
import sqlite3

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent

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


def sql_node(question: str) -> str:
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