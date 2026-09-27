from langgraph.graph import StateGraph, START, END
from langsmith import traceable

from src.copilot.utils.state import CopilotState
from src.copilot.subgraphs.small_talk.nodes import small_talk_llm_node


@traceable(type="subgraph", name="small_talk_agent")
def build_small_talk_agent():
    g = StateGraph(state_schema=CopilotState)

    g.add_node("small_talk_llm", small_talk_llm_node)

    g.add_edge(START, 'small_talk_llm')
    g.add_edge('small_talk_llm', END)

    return g.compile()


small_talk_agent = build_small_talk_agent()
