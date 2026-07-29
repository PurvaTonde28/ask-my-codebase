# agents/test_graph.py
#
# Manual smoke test, not pytest. Run with: python -m agents.test_graph
#
# Two questions on the SAME thread_id, where the second is a follow-up
# that only makes sense with memory of the first ("that one" has no
# antecedent otherwise). This is the actual behavior being verified --
# before the multi-turn memory fix, turn 2 would have no idea what "that
# one" refers to.
from dotenv import load_dotenv
load_dotenv()

import uuid

from agents.graph import graph

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

result = graph.invoke({"question": "how does dependency injection work?"}, config=config)
print("Route:", result["route"]["destination"], "-", result["route"]["reasoning"])
print("Answer:", result["final_answer"][:300], "...")
print()

result2 = graph.invoke({"question": "can you show me the specific file that one lives in?"}, config=config)
print("Route:", result2["route"]["destination"], "-", result2["route"]["reasoning"])
print("Answer:", result2["final_answer"])
print()

# Sanity check for the memory itself, independent of what the LLM says:
# by turn 2 the checkpointed conversation should hold 4 messages (Q1, A1,
# Q2, A2), not just the 2 that this single invoke() call added.
message_count = len(result2["messages"])
assert message_count == 4, (
    "expected 4 accumulated messages by turn 2, got "
    + str(message_count)
    + " -- multi-turn memory is not wired correctly"
)
print("PASS: conversation history accumulated across both turns.")