import logging
from typing import List

from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage

from src.copilot.utils.state import CopilotState, get_contexts_and_plugs
from src.copilot.utils.memory import checkpointer

from src.app.schemes.agent import GraphOutput, StateInputData

from src.copilot.nodes.supervizor import supervizor_node
from src.copilot.nodes.executor import ExecutorNode
from src.copilot.nodes.validate_answer import validate_answers_node

from src.copilot.subgraphs.small_talk.st_agent import small_talk_agent
from src.copilot.subgraphs.kommo.kommo_agent import kommo_agent
from src.copilot.subgraphs.global_.global_agent import global_agent

logger = logging.getLogger(__name__)


workflow = StateGraph(CopilotState)

# Nodes
workflow.add_node("supervizor", supervizor_node)
workflow.add_node("executor", ExecutorNode())
workflow.add_node("validate", validate_answers_node)

# Subgraphs
workflow.add_node("small_talk", small_talk_agent)
workflow.add_node("kommo", kommo_agent)
workflow.add_node("global", global_agent)

# Edges
workflow.add_edge(START, "supervizor")
workflow.add_edge("supervizor", "executor")
workflow.add_edge("small_talk", "validate")
workflow.add_edge("kommo", "validate")
workflow.add_edge("global", "validate")

workflow.add_edge("validate", END)


graph = workflow.compile(checkpointer=checkpointer)


class CopilotAgent:
    """Wrapper around compiled LangGraph.

    Main method:
        ainvoke(state_input: StateInputData) -> GraphOutput"""

    def __init__(self, graph: CompiledStateGraph):
        self.graph = graph

    # ─────────────────────────────── Main Method ────────────────────────── #
    async def ainvoke(self, state_input: StateInputData) -> GraphOutput:
        """Async invoke of compiled graph."""
        _, plugs = get_contexts_and_plugs(state_input)

        if not self._is_messages_ok(state_input.messages):
            logger.error("Invalid messages format: %s", state_input.messages)
            return GraphOutput(answers=[plugs.unable_to_use()], usage={})

        try:
            # 1. Build start state
            start_state = self._build_start_state(state_input)

            # 2. Build config
            config = self._build_config(state_input)

            # 3. Invoke graph
            response = await self.graph.ainvoke(start_state, config=config)
            logger.debug("Graph raw response: %s", response)

            # 4. Prepare output
            return self._to_graph_output(response)

        except Exception as exc:
            logger.error(
                "Error during graph execution: %s",
                exc, exc_info=True
            )
            return GraphOutput(
                answers=[plugs.unable_to_use()],
                usage={}
            )

    # ─────────────────────────────── Helpers ─────────────────────────────── #
    @staticmethod
    def _is_messages_ok(messages: List[BaseMessage]) -> bool:
        """Validate messages format"""
        return bool(messages and isinstance(messages, list))


    @staticmethod
    def _build_start_state(state_input: StateInputData) -> CopilotState:
        """Builds initial state of graph from input data."""
        return CopilotState(
            messages=state_input.messages,
            contexts=state_input.contexts,
            answers=[],
            usage={},
            answer="",
            retrieve_kommo_context="",
            kommo_articles_links=[],
            kommo_queries=[],
            kommo_tool_calls=[],
            search_in_web_context="",
            global_queries=[],
            loop_count=0,
        )


    @staticmethod
    def _build_config(state_input: StateInputData) -> RunnableConfig:
        """Meta-config, passed to each node in Graph."""
        return RunnableConfig(
            configurable={"thread_id": str(state_input.thread_id)},
            metadata={"thread_id": state_input.thread_id},
        )


    @staticmethod
    def _to_graph_output(raw_state: dict) -> GraphOutput:
        """Preparing public GraphOutput schema."""
        return GraphOutput(
            answers=raw_state.get("answers", []),
            usage=raw_state.get("usage", {}),
        )


copilot_agent = CopilotAgent(graph)
