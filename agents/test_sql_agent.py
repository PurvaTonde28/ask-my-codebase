# agents/test_sql_agent.py
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage

from agents.sql_agent import sql_node

# sql_node() now takes the running conversation (a list of messages), not a
# bare string -- this is a single-turn conversation, so a one-item list.

# answer = sql_node([HumanMessage(content="how many commits are in the database?")])
# print(answer)

print(sql_node([HumanMessage(content="who are the top 5 contributors by commit count?")]))
print(sql_node([HumanMessage(content="how many commits touched routing.py?")]))