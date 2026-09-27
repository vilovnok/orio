import logging

from langsmith import traceable

from langchain_core.messages import AIMessage

from src.app.schemes.agent import CopilotAnswersScheme

from src.copilot.utils.state import (
    CopilotState,
    get_contexts_and_plugs
)
from src.copilot.utils.schemes import SubgraphResult
from src.copilot.subgraphs.small_talk.prompt import prompt_builder

from src.utils.config import inited_config as config
from src.copilot.provider.llm import LLMProvider

logger = logging.getLogger(__name__)



llm = LLMProvider(**config.llm.providers.openai.model_dump())._llm


@traceable(type="llm", name="small_talk_llm")
async def small_talk_llm_node(
    state: CopilotState
) -> SubgraphResult:
    """Small talk llm node."""
    contexts, plugs = get_contexts_and_plugs(state)
    messages = state.get('messages', [])

    small_talk_prompt = prompt_builder.get_small_talk_prompt_template(
        contexts=contexts,
        messages=messages
    )
    logger.info('small_talk_prompt: %s', small_talk_prompt)

    try:
        response = await llm.ainvoke(small_talk_prompt)
    except Exception as e:
        logger.error('small_talk_llm_node error: %s', e)
        return SubgraphResult(
            answers=[plugs.unable_to_use()],
            state_updates={}
        )
    logger.info('small_talk_response: %s', response)

    answers = state.get('answers', [])
    answers.append(CopilotAnswersScheme(text=response.content))
    logger.debug(
        'small_talk_llm_node (%d) answers: %s',
        len(answers), answers
    )

    return SubgraphResult(
        answers=answers,
        state_updates={
            'messages': [AIMessage(content=response.content)]
            if isinstance(response, AIMessage) and response.tool_calls
            else []
        }
    ).model_dump()
