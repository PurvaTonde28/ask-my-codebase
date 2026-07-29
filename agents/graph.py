from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from agents.state import GraphState
from agents.supervisor import supervisor_node
from agents.rag_agent import rag_node
from agents.sql_agent import sql_node


def supervisor_graph_node(state: GraphState) -> dict:
    question = state["question"]
    prior_messages = state.get("messages", [])

    # prior_messages does NOT include `question` yet -- that keeps the
    # supervisor's own "history vs. current question" distinction explicit
    # rather than having it parse its own last message back out of the list.
    decision = supervisor_node(question, history=prior_messages)

    return {
        # Plain dict, not the RouteDecision pydantic object -- see RouteInfo
        # in state.py for why. `decision` itself is still a real
        # RouteDecision here; only what gets checkpointed changes shape.
        "route": {"destination": decision.destination, "reasoning": decision.reasoning},
        # Appended (not overwritten) thanks to the add_messages reducer on
        # GraphState["messages"] -- this is what the checkpointer persists
        # under this thread_id for the next invoke() call.
        "messages": [HumanMessage(content=question)],
    }


def rag_graph_node(state: GraphState) -> dict:
    # state["messages"] already includes the HumanMessage supervisor_graph_node
    # just added -- LangGraph merges each node's output into state before the
    # next node in the same run sees it.
    result = rag_node(state["messages"])

    return {
        "rag_result": result,
        "final_answer": result,
        "messages": [AIMessage(content=result)],
    }


def sql_graph_node(state: GraphState) -> dict:
    result = sql_node(state["messages"])

    return {
        "sql_result": result,
        "final_answer": result,
        "messages": [AIMessage(content=result)],
    }


def route_edge(state: GraphState) -> str:
    return state["route"]["destination"]


workflow = StateGraph(GraphState)

workflow.add_node(
    "supervisor",
    supervisor_graph_node,
)

workflow.add_node(
    "rag",
    rag_graph_node,
)

workflow.add_node(
    "sql",
    sql_graph_node,
)

workflow.add_edge(
    START,
    "supervisor",
)

workflow.add_conditional_edges(
    "supervisor",
    route_edge,
    {
        "rag": "rag",
        "sql": "sql",
    },
)

workflow.add_edge(
    "rag",
    END,
)

workflow.add_edge(
    "sql",
    END,
)

# In-memory checkpointer: gives graph.invoke() conversational memory keyed
# by a thread_id passed in at call time (config={"configurable":
# {"thread_id": ...}}). One process (one CLI run, one Streamlit server) =
# one InMemorySaver instance = memory lives only as long as that process.
# That's the right tradeoff for a portfolio project: real multi-turn memory
# within a session, zero new infra, nothing to persist to disk or clean up.
#
# IMPORTANT: once a checkpointer is compiled in, EVERY graph.invoke() call
# MUST include a thread_id, e.g.:
#   graph.invoke({"question": q}, config={"configurable": {"thread_id": t}})
# Omitting it doesn't silently fall back to stateless behavior -- LangGraph
# raises ValueError("Checkpointer requires ... thread_id ..."). Verified
# this directly rather than assuming it. Both app.py and streamlit_app.py
# were updated to always pass one.
#
# Upgrade path if you ever want memory to survive a process restart:
# langgraph.checkpoint.sqlite.SqliteSaver, same thread_id contract, just
# backed by a file instead of a Python dict.
checkpointer = InMemorySaver()

graph = workflow.compile(checkpointer=checkpointer)