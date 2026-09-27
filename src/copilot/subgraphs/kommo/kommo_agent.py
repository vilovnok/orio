from langgraph.graph import StateGraph, START
from langsmith import traceable

from src.copilot.subgraphs.kommo.nodes import (
    retrieve_kommo_node, kommo_agent_node
)
from src.copilot.utils.state import CopilotState


@traceable(type="subgraph", name="kommo_agent")
def build_kommo_agent():
    g = StateGraph(state_schema=CopilotState)

    g.add_node("retrieve_kommo", retrieve_kommo_node)
    g.add_node("kommo_agent", kommo_agent_node)

    g.add_edge(START, 'retrieve_kommo')

    return g.compile()


kommo_agent = build_kommo_agent()
