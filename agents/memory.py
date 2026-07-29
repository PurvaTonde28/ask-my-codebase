"""
Shared conversation-memory helpers for the supervisor/RAG/SQL agents.
"""

from langchain_core.messages import BaseMessage

# ~4 prior turns (1 turn = 1 human + 1 ai message). Keeps token usage/cost
# bounded on a long session instead of resending the entire conversation to
# Groq on every question -- older turns are just not sent to the LLM
# anymore, they are NOT deleted from the checkpointed conversation, so the
# UI can still display the full history.
MAX_HISTORY_MESSAGES = 8


def trim_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Keep only the most recent MAX_HISTORY_MESSAGES entries."""
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    return messages[-MAX_HISTORY_MESSAGES:]