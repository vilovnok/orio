import logging

from langsmith import traceable
from src.copilot.tools.web_search import web_search # gpt_searching

from src.app.schemes.agent import CopilotAnswersScheme
from src.copilot.subgraphs.global_.prompt import prompt_builder
from src.copilot.utils.state import (
    CopilotState,
    get_contexts_and_plugs
)
from src.copilot.utils.schemes import SubgraphResult

logger = logging.getLogger(__name__)


@traceable(type="llm", name="searching")
async def searching_node(
    state: CopilotState
) -> SubgraphResult:
    """Ищет информацию в интернете с использованием GPT-4o-mini"""
    contexts, plugs = get_contexts_and_plugs(state)

    # prompt = prompt_builder.get_global_full_prompt(
    #     contexts=contexts,
    #     messages=state.get('messages', []),
    #     global_queries=state.get('global_queries', []),
    # )

    prompt = state.get('global_queries', [])[-1]

    logger.info('searching_node prompt: %s', prompt)

    try:
        response = await web_search(prompt)
    except Exception:
        logger.error('error in gpt_searching', exc_info=True)
        return SubgraphResult(
            answers=[plugs.unable_to_use()],
            state_updates={}
        )

    answer_text = "\n".join([f"{item.title}: [{item.url}]({item.url})" for item in response.results])

    answer = CopilotAnswersScheme(
        text=answer_text,
        is_global=True,
        action='unknown'
    )

    logger.warning('gpt_searching_node answer: %s', answer)
    return SubgraphResult(
        answers=[answer],
        state_updates={
            'global_queries': []
        }
    ).model_dump()
