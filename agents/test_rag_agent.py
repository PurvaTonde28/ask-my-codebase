# agents/test_rag_agent.py
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage

from agents.rag_agent import rag_node

# rag_node() now takes the running conversation (a list of messages), not a
# bare string -- this is a single-turn conversation, so a one-item list.

# something the codebase actually covers
print(rag_node([HumanMessage(content="how does dependency injection work in this codebase?")]))
print()
print("---")
print()

# something it does NOT cover -- the real test of whether it hallucinates
print(rag_node([HumanMessage(content="how does this codebase implement blockchain consensus?")]))