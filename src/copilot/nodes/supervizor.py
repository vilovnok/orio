import logging

from typing import Literal, List

from langgraph.types import Command
from langsmith import traceable

from src.copilot.utils.schemes import QueryList, SmallTalk

from src.copilot.utils.state import (
    CopilotState,
    get_contexts_and_plugs
)
from src.copilot.utils.prompts import supervizor_prompt_template

from src.copilot.services.usage.counters import count_tokens

from src.utils.config import inited_config as config

from src.copilot.provider.llm import LLMProvider

logger = logging.getLogger(__name__)


# supervizor_llm = LLMProvider(**config.llm.providers.openai.model_dump())._llm
# supervizor_chain = supervizor_prompt_template | supervizor_llm

supervizor_llm = LLMProvider(
    **config.llm.providers.openai.model_dump()
)._llm

supervizor_chain = (
    supervizor_prompt_template
    | supervizor_llm.with_structured_output(QueryList)
)



@traceable
async def supervizor_node(
    state: CopilotState
) -> Command[Literal['__end__']]:
    """Split the question into the appropriate tools."""
    messages = state.get("messages", [])
    _, plugs = get_contexts_and_plugs(state)

    usage = {
        "gpt-4o-mini": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0
        }
    }

    full_prompt = supervizor_prompt_template.format(messages=messages)
    prompt_tokens = count_tokens(full_prompt)
    usage["gpt-4o-mini"]["prompt_tokens"] += prompt_tokens
    logger.info("prompt (%d tokens): %s", prompt_tokens, full_prompt)

    try:
        q_list: QueryList = await supervizor_chain.ainvoke(
            input={"messages": messages}
        )
    except Exception:
        logger.error("Error while calling supervizor_chain", exc_info=True)
        return Command(
            goto="validate",
            update={'answers': [plugs.unable_to_use()]}
        )

    completion_tokens = count_tokens(str(q_list))
    usage["gpt-4o-mini"]["completion_tokens"] += completion_tokens
    logger.info("completion (%d tokens): %s", completion_tokens, q_list)

    
    kommo_q, global_q, small_q = [], [], []
    for q in q_list.q_list:
        match q.route:
            case "kommo":  kommo_q.append(q.text)
            case "global": global_q.append(q.text)
            case "small_talk": small_q.append(q.text)

    small_talks: List[SmallTalk] = []
    if kommo_q or global_q:
        target = "kommo" if kommo_q else "global"
        for q in small_q:
            small_talks.append(SmallTalk(text=q, target=target))
    else:
        for q in small_q:
            small_talks.append(SmallTalk(text=q, target="small_talk"))

    # TODO: Make it more concise
    to_run = []
    if kommo_q:
        to_run.append("kommo")

    if global_q:
        to_run.append("global")

    if not any([kommo_q, global_q]):
        if small_talks:
            to_run.append("small_talk")
        else:
            return Command(
                goto="validate",
                update={
                    'answers': [plugs.unable_to_use()]
                }
            )

    if not any([kommo_q, global_q, small_talks]):
        return Command(
            goto="validate",
            update={'answers': [plugs.unable_to_use()]}
        )

    logger.info('supervizor_node to_run: %s', to_run)
    logger.info('queries: %s, %s, %s', kommo_q, global_q, small_talks)
    
    return Command(
        goto="executor",
        update={
            'kommo_queries': kommo_q,
            'kommo_context': '',
            'kommo_tool_calls': [],
            'global_queries': global_q,
            'global_context': '',
            'global_tool_calls': [],
            'small_talk_queries': small_talks,
            'to_run': to_run,
            'usage': usage,
            'answers': [],
            'messages': messages
        }
    )
