import time
from typing import Literal

from groq import BadRequestError as GroqBadRequestError
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ValidationError


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

Return only the RouteDecision object.
"""


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

structured_llm = llm.with_structured_output(RouteDecision)


def supervisor_node(
    question: str,
    max_retries: int = 2,
) -> RouteDecision:
    """
    Decide whether a user question should be answered by the
    RAG agent or the SQL agent.

    Returns a RouteDecision object that LangGraph will later
    use for conditional routing.

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

    for attempt in range(max_retries + 1):
        try:
            decision = structured_llm.invoke(
                [
                    (
                        "system",
                        SYSTEM_PROMPT,
                    ),
                    (
                        "human",
                        question,
                    ),
                ]
            )

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