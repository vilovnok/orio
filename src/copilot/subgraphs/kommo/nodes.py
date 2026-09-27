import asyncio
import logging
import traceback
import uuid
from typing import Literal, Union, List

from src.app.schemes.agent import CopilotAnswersScheme
from src.copilot.utils.schemes import ToolResult

from langchain_core.messages import ToolMessage

from langsmith import traceable

from langgraph.types import Command
from src.copilot.subgraphs.kommo.prompt import prompt_builder

from src.copilot.services.usage.counters import count_tokens
from src.copilot.tools.kommo_retriever import retrieve_kommo

from src.copilot.utils.state import (
    CopilotState,
    get_contexts_and_plugs
)
from src.copilot.utils.schemes import SubgraphResult

from src.utils.config import inited_config as config
from src.copilot.provider.llm import LLMProvider

from .models import QuotedAnswer
from .utils import format_answer


logger = logging.getLogger(__name__)



llm = LLMProvider(**config.llm.providers.openai.model_dump())._llm
llm_with_struct_output = llm.with_structured_output(QuotedAnswer)


async def kommo_agent_node(
    state: CopilotState
) -> Union[SubgraphResult, Command[Literal['retrieve_kommo', '__end__']]]:
    """Агентный узел Kommo."""
    try:
        contexts, plugs = get_contexts_and_plugs(state)

        kommo_articles_links = state.get('kommo_articles_links', [])
        kommo_context = state.get('retrieve_kommo_context', '')
        kommo_queries = state.get('kommo_queries', [])
        messages = state.get('messages', [])

        # Update kommo prompt with small talks
        small_talks = state.get('small_talks', [])
        addoption_queries = [
            small_talk.get('query')
            for small_talk in small_talks
            if small_talk.get('target') == 'kommo'
        ]
        logger.info('small talks for addoption: %s', addoption_queries)
        kommo_queries.extend(addoption_queries)
        logger.info('extended kommo_queries: %s', kommo_queries)
        
        kommo_prompt = await prompt_builder.get_kommo_full_prompt(
            contexts=contexts,
            messages=messages,
            kommo_queries=kommo_queries,
            kommo_context=kommo_context if kommo_context else '',
        )

        logger.info('kommo_prompt: (%d tokens) %s',
                    count_tokens(kommo_prompt), kommo_prompt)

        kommo_single_relevant_prompt = await prompt_builder.get_single_relevant_article_link_prompt(
            query=kommo_queries[-1],
            kommo_articles_links=kommo_articles_links
        )

        logger.info('kommo_single_relevant_prompt: (%d tokens) %s',
                    count_tokens(kommo_single_relevant_prompt), kommo_single_relevant_prompt)

        try:
            response_kommo_prompt = await llm_with_struct_output.ainvoke(kommo_prompt)
            response_kommo_single_relevant_prompt = await llm_with_struct_output.ainvoke(kommo_single_relevant_prompt)
            
        except Exception as e:
            logger.error('Error in kommo_agent_node: %s', e, exc_info=True)
            return SubgraphResult(
                answers=[plugs.unable_to_use()],
                state_updates={}
            )
        logger.info(
            "response_kommo_prompt: %s | response_kommo_single_relevant_prompt: %s",
            response_kommo_prompt,
            response_kommo_single_relevant_prompt
        )

        # Update answers
        try:
            print('\n\n')
            print('*'*100)
            # print(type(response_kommo_single_relevant_prompt.answer))
            # print('\n\n')
            try:
            #     print(len(response_kommo_single_relevant_prompt.answer))
            #     print(response_kommo_single_relevant_prompt.answer[0])
                print(response_kommo_single_relevant_prompt.answer[0].reasoning.output)
            #     print(dir(response_kommo_single_relevant_prompt.answer[0]))
            except Exception as e:
                print(e)
            print('*'*100)
            print('\n\n')

            answer = CopilotAnswersScheme(
                text=format_answer(response_kommo_prompt, [response_kommo_single_relevant_prompt.answer[0].reasoning.output]),
                is_knowledge_base=True if response_kommo_single_relevant_prompt else False
            )
        except Exception as e:
            logger.error('Error in kommo_agent_node: %s', e, exc_info=True)
            return SubgraphResult(
                answers=[plugs.unable_to_use()],
                state_updates={}
            )
        logger.info('kommo_llm_node answer: %s', answer)

        return SubgraphResult(
            answers=[answer],
            state_updates={}
        )
    except Exception as e:
        logger.error('Error in kommo_agent_node:')
        traceback.print_exc()
        raise e


# --| Retriever Node |--
@traceable(type="tool", name='retrieve_kommo_node')
async def retrieve_kommo_node(
    state: CopilotState
) -> Command[Literal['kommo_agent']]:
    """Узел получения данных Kommo."""
    logger.info('start retrieve_kommo_node with query state: %s', state)
    kommo_queries = state.get('kommo_queries', [])
    query = kommo_queries[0] if kommo_queries else state.get('messages')[-1].content

    tasks = [
        asyncio.create_task(
            retrieve_kommo.ainvoke(query)
        )
    ]

    logger.info('retrieve_kommo_node %d tasks: %s', len(tasks), tasks)

    results: List[Union[ToolResult, Exception]] = await asyncio.gather(
        *tasks,
        return_exceptions=True
    )
    logger.debug('retrieve_kommo_node got (%d) results', len(results))

    kommo_articles_links = []
    retrieve_kommo_context = ''

    tool_messages: List[ToolMessage] = []
    if isinstance(results[0], Exception):
        logger.error("Error in retrieve_kommo: %s", results[0])
        tool_messages.append(
            ToolMessage(
                content=f"error in tool_call: {results[0]}",
                name='retrieve_kommo',
                tool_call_id=str(uuid.uuid4())
            )
        )
    else:
        updates = results[0].state_updates
        kommo_articles_links.extend(updates['kommo_articles_links'])

        raw_contexts = updates['retrieve_kommo_context']
        formatted_contexts = []
        if isinstance(raw_contexts, list):
            for i, ctx in enumerate(raw_contexts, start=1):
                source = ctx.get("source", "unknown")
                text = ctx.get("text", "")
                formatted_contexts.append(f"[{i}] Source: {source}\n{text}")
            retrieve_kommo_context = "\n\n".join(formatted_contexts)
        else:
            retrieve_kommo_context = raw_contexts

    state_updates = {
        'kommo_articles_links': kommo_articles_links,
        'retrieve_kommo_context': retrieve_kommo_context,
        'kommo_tool_calls': [],
        'messages': tool_messages
    }

    return Command(
        goto='kommo_agent',
        update=state_updates
    )
