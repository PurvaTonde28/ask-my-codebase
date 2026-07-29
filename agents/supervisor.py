import time
from typing import Literal

from groq import BadRequestError as GroqBadRequestError
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ValidationError

from agents.memory import trim_history

from dotenv import load_dotenv
load_dotenv()

class RouteDecision(BaseModel):
    destination: Literal["rag", "sql"] = Field(
        description=(
            "Choose 'rag' for conceptual questions about how the codebase "
            "works (examples: 'How does dependency injection work?', "
            "'Explain the routing decorator'). "
            "Choose 'sql' for questions about git history, contributors, "
            "commit counts, or file history (examples: "
            "'Who modified routing.py the most?', "
            "'How many commits touched auth.py?')."
        )
    )

    reasoning: str = Field(
        description=(
            "One short sentence explaining why this destination was chosen."
        )
    )


SYSTEM_PROMPT = """
You are the routing supervisor for a multi-agent codebase assistant.

Your only task is deciding which agent should answer the user's question.

Choose:

rag
- Questions about how the code works.
- Documentation.
- APIs.
- Classes.
- Functions.
- Architecture.
- Configuration.

Examples:
- How does dependency injection work?
- Explain APIRouter.
- What does this middleware do?
- How is authentication implemented?

sql
- Questions about git history.
- Commit counts.
- Contributors.
- File history.
- Repository activity.
- Code churn.

Examples:
- Who modified auth.py the most?
- How many commits touched routing.py?
- Which files changed last month?
- Who are the top contributors?

If the question is ambiguous, prefer "rag".

You may also see earlier turns of this conversation before the current
question. Use them ONLY to resolve references in the current question --
words like "that", "it", "the previous one", or "again" -- to figure out
what the current question is actually asking about. Route based on what
the CURRENT question needs once resolved, not automatically on whichever
agent handled the previous turn.

Return only the RouteDecision object.
"""


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

structured_llm = llm.with_structured_output(RouteDecision)


def supervisor_node(
    question: str,
    history: list[BaseMessage] | None = None,
    max_retries: int = 2,
) -> RouteDecision:
    """
    Decide whether a user question should be answered by the
    RAG agent or the SQL agent.

    Returns a RouteDecision object that LangGraph will later
    use for conditional routing.

    `history` is the conversation SO FAR -- prior turns only, not including
    `question` itself (that's appended separately below). Passing it lets
    the router resolve follow-ups like "give me an example of that" instead
    of treating every question as if it arrived with zero context. Trimmed
    the same way the RAG/SQL agents trim it, so a long session doesn't grow
    the routing prompt (and Groq token usage) unbounded.

    Two distinct failure modes are retried here, not just one:
    - GroqBadRequestError: the model emits malformed tool-call syntax
      (same flakiness seen in Phase 8's rag_node/sql_node).
    - pydantic.ValidationError: the tool call is structurally fine, but its
      arguments don't match RouteDecision's schema -- e.g. the model returns
      destination="both", which isn't a valid Literal value here. This is a
      genuinely different failure than the Groq one and needs its own catch,
      confirmed by testing: GroqBadRequestError alone does NOT catch this.
    """

    last_error = None

    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(trim_history(history or []))
    messages.append(HumanMessage(content=question))

    for attempt in range(max_retries + 1):
        try:
            decision = structured_llm.invoke(messages)

            return decision

        except (GroqBadRequestError, ValidationError) as e:
            last_error = e

            if attempt < max_retries:
                time.sleep(1)
                continue

    return RouteDecision(
        destination="rag",
        reasoning=(
            f"Routing failed after {max_retries + 1} attempts "
            f"({last_error}). Defaulting to the RAG agent."
        ),
    )