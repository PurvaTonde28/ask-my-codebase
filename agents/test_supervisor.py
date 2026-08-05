# test_supervisor.py
from dotenv import load_dotenv
load_dotenv()

from agents.supervisor import supervisor_node

questions = [
    "how does dependency injection work",
    "how many commits touched routing.py",
    "who are the top contributors",
    "explain the routing decorator",
    "tell me about the auth module",  # deliberately ambiguous
]

for q in questions:
    result = supervisor_node(q)
    print(f"{result.destination:5} | {q}")
    print(f"      reasoning: {result.reasoning}")
    print()