import asyncio
import logging
from typing import List, Dict, Tuple, Literal, Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.app.schemes.agent import CopilotAnswersScheme

from src.copilot.utils.state import (
    CopilotState,
    merge_state_updates,
    get_contexts_and_plugs
)
from src.copilot.utils.schemes import SubgraphResult

from src.copilot.subgraphs.kommo.kommo_agent import build_kommo_agent
from src.copilot.subgraphs.global_.global_agent import build_global_agent
from src.copilot.subgraphs.small_talk.st_agent import build_small_talk_agent

from src.copilot.subgraphs import KOMMO, GLOBAL, SMALL_TALK


logger = logging.getLogger(__name__)


class ExecutorNode:
    def __init__(self) -> None:
        self.subgraphs = {
            KOMMO: build_kommo_agent(),
            GLOBAL: build_global_agent(),
            SMALL_TALK: build_small_talk_agent()
        }

    async def __call__(
        self,
        state: CopilotState,
        config: RunnableConfig
    ) -> Command[Literal[SMALL_TALK, KOMMO, GLOBAL, 'validate']]:  # type: ignore # noqa: E501
        """Gathering all SubGraphs from state["to_run"]"""
        _, plugs = get_contexts_and_plugs(state)

        to_run: List[str] = state.get("to_run", [])
        tasks: List[Tuple[str, asyncio.Task]] = []

        logger.info('to_run: %s', to_run)
        # Fan-out: creating tasks for each SubGraph
        try:
            for name in to_run:
                subg = self.subgraphs.get(name)
                if not subg:
                    logger.warning("SubGraph %s not found", name)
                    continue

                task = asyncio.create_task(subg.ainvoke(state, config=config))
                tasks.append((name, task))
        except Exception:
            logger.error("error in creating tasks", exc_info=True)
            return Command(
                goto="validate",
                update={"answers": [plugs.unable_to_use()]}
            )

        logger.info('%d tasks to run', len(tasks))

        if not tasks:
            logger.error('no tasks to run')
            return Command(
                goto="validate",
                update={
                    "answers": [plugs.unable_to_use()]
                }
            )

        # Gather SubGraphs
        results: List[SubgraphResult | Exception] = await asyncio.gather(
            *(t[1] for t in tasks),
            return_exceptions=True
        )

        answers: List[CopilotAnswersScheme] = []
        subgraph_results: Dict[str, SubgraphResult] = {}
        state_updates: Dict[str, Any] = {}

        for (name, task), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error('%s error: %s', name, result, exc_info=result)
                subgraph_result = SubgraphResult(
                    answers=[plugs.unable_to_use()],
                    state_updates={}
                )
                subgraph_results[name] = subgraph_result
            else:
                subgraph_result = SubgraphResult(**result)

                subgraph_results[name] = subgraph_result
                answers.extend(subgraph_result.answers)
                state_updates = merge_state_updates(
                    acc=state_updates,
                    update=subgraph_result.state_updates
                )

        return Command(
            goto="validate",
            update={
                "answers": answers,
                **state_updates
            }
        )
