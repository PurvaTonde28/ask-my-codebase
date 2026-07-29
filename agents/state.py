from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class RouteInfo(TypedDict):
    """
    Plain-dict mirror of supervisor.RouteDecision -- deliberately NOT the
    pydantic model itself.

    GraphState now gets checkpointed on every graph.invoke() call (see the
    InMemorySaver in graph.py), and LangGraph's checkpoint serializer only
    has first-class, forward-compatible support for built-in JSON-safe
    types. Storing the raw RouteDecision pydantic object here worked, but
    logged: "Deserializing unregistered type agents.supervisor.RouteDecision
    from checkpoint. This will be blocked in a future version." -- caught by
    running an end-to-end test against the real graph, not by inspection.
    Left as-is it would have silently broken on some future
    `pip install --upgrade langgraph`. supervisor_graph_node() converts the
    real RouteDecision to this shape before it's returned into state.
    """
    destination: str
    reasoning: str


class GraphState(TypedDict, total=False):
    # Input
    question: str

    # Running conversation across turns. The add_messages reducer means each
    # node's returned "messages" list is APPENDED to what's already here,
    # never overwritten. Combined with a checkpointer + thread_id on
    # graph.invoke() (see graph.py), this is what makes follow-ups like
    # "give me an example of that" work -- LangGraph reloads this list
    # automatically on every new invoke() call for the same thread_id.
    messages: Annotated[list, add_messages]

    # Added by supervisor -- see RouteInfo above for why this is a plain
    # dict shape rather than the RouteDecision pydantic object.
    route: RouteInfo

    # Added by RAG branch
    rag_result: Optional[str]

    # Added by SQL branch
    sql_result: Optional[str]

    # Final output
    final_answer: Optional[str]