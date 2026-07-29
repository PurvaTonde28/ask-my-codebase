from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from groq import BadRequestError as GroqBadRequestError
import time

from agents.memory import trim_history
from vectorstore.build_index import search, load_index
# from vectorstore.build_index import  load_index

# search() relies on a module-level cache (_index, _metadata) inside
# build_index.py. That cache only lives in memory for the process that
# populated it -- since this is a fresh process, the persisted index has to
# be explicitly loaded back in here before any search() call can work. This
# happens once, at import time, not per-question.

# load_index(output_dir="vectorstore/data")


SYSTEM_PROMPT = """
You are a RAG assistant for a source code repository.

Rules:
- Always use the search_codebase tool before answering.
- You may call the tool multiple times if needed.
- Answer ONLY from the retrieved code or documentation.
- Never answer from your own general knowledge if the retrieved information is insufficient.
- If the retrieved content does not answer the question, clearly say so.
- Always cite the file path and line numbers for every claim.
- Summarize the retrieved information in plain English instead of copying large chunks verbatim.
- You may see earlier turns of this conversation before the latest question.
  Use them to resolve references like "that", "it", or "the previous one" --
  but still call search_codebase for the CURRENT question. Never answer a
  follow-up purely from what was said earlier without a fresh search; the
  earlier answer's tool results are not repeated in what you see now.
"""


@tool
def search_codebase(
    query: str,
    k: int = 5,
) -> str:
    """
    Search the indexed codebase using semantic search.
    """

    try:
        results = search(
            query=query,
            k=k,
        )

    except RuntimeError as e:
        return str(e)

    if not results:
        return "No relevant chunks were found."

    formatted_results = []

    for chunk in results:
        formatted_results.append(
            f"""File: {chunk['file_path']}
Lines: {chunk['start_line']}-{chunk['end_line']}

{chunk['content']}
"""
        )

    return "\n\n".join(formatted_results)


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)


rag_agent = create_agent(
    model=llm,
    tools=[search_codebase],
    system_prompt=SYSTEM_PROMPT,
)


def rag_node(messages: list[BaseMessage], max_retries: int = 2) -> str:
    """
    Entry point for the RAG agent.
    Called by rag_graph_node() in graph.py.

    `messages` is the full running conversation (prior turns + the current
    question as the last HumanMessage), built from GraphState["messages"].
    Trimmed here before being sent to the agent so a long session can't grow
    the prompt -- and Groq token usage -- unbounded.

    Retries on groq.BadRequestError: Groq's Llama models occasionally emit
    malformed tool-call syntax (e.g. literal "<function=...>" text instead of
    a proper structured tool call), especially on unusual/out-of-scope
    questions. This is a known model-level flakiness, not something a fixed
    prompt reliably prevents -- retrying the same call is the practical fix.
    """
    last_error = None
    trimmed_messages = trim_history(messages)

    for attempt in range(max_retries + 1):
        try:
            response = rag_agent.invoke({"messages": trimmed_messages})
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