from langgraph.graph import StateGraph, START, END

from langsmith import traceable

from src.copilot.subgraphs.global_.nodes import searching_node
from src.copilot.utils.state import CopilotState


@traceable(type="subgraph", name="global_agent")
def build_global_agent():
    g = StateGraph(state_schema=CopilotState)

    g.add_node("search-preview", searching_node)

    g.add_edge(START, 'search-preview')
    g.add_edge('search-preview', END)

    return g.compile()


global_agent = build_global_agent()
