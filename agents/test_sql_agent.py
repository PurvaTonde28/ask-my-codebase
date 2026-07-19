# test_sql_agent.py
from dotenv import load_dotenv
load_dotenv()

from agents.sql_agent import sql_node

# answer = sql_node("how many commits are in the database?")
# print(answer)

print(sql_node("who are the top 5 contributors by commit count?"))
print(sql_node("how many commits touched routing.py?"))