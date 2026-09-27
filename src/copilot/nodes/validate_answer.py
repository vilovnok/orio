import logging

from typing import Literal, List
from pydantic import BaseModel

from src.app.schemes.agent import CopilotAnswersScheme

from src.copilot.utils.state import (
    CopilotState,
    get_contexts_and_plugs
)
from langchain_openai import ChatOpenAI

from langsmith import traceable

from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage

from src.copilot.services.usage.counters import count_tokens
from src.copilot.utils.prompts import validate_prompt_template, validate_prompt_builder

from src.copilot.provider.llm import LLMProvider
from src.utils.config import inited_config as config

logger = logging.getLogger(__name__)


llm = LLMProvider(**config.llm.providers.openai.model_dump())._llm



class ValidateAnswerOutputScheme(BaseModel):
    """Make sure that llm answer has correct format."""
    is_support: bool


class LanguageOutputScheme(BaseModel):
    """Language detection output schema."""
    language: str


class TranslationOutputScheme(BaseModel):
    """Translation output schema."""
    translation: str


chain = (
    validate_prompt_template | llm.with_structured_output(
        ValidateAnswerOutputScheme
    )
)


@traceable
async def validate_answers_node(
    state: CopilotState,
    config: RunnableConfig
) -> Command[Literal['__end__']]:
    """Validate answers"""
    messages = state.get('messages', [])
    answers = state.get('answers', [])
    usage = state.get('usage', {})
    

    # Getting contexts
    _, plugs = get_contexts_and_plugs(state)

    # Getting last HumanMessage
    last_human_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_message = msg
            break

    # Generating from 1 to 2 answers at a time
    # TODO: Research the way to optimize this (gather or async batch)
    valid_answers: List[CopilotAnswersScheme] = []
    for answer in answers:
        # Forming dialog
        if last_human_message:
            dialog = [last_human_message, AIMessage(content=answer.text)]
        else:
            dialog = [AIMessage(content=answer.text)]

        logger.debug('dialog for validate_answers: %s', dialog)

        # Detecting language of answer.text
        language_prompt_template = validate_prompt_builder.get_language_prompt(answer.text)
        language_chain = (
            language_prompt_template | llm.with_structured_output(LanguageOutputScheme)
        )

        language_messages = language_prompt_template.format_messages()
        language_full_prompt = '\n'.join(str(msg.content) for msg in language_messages)
        language_prompt_tokens = count_tokens(language_full_prompt)
        usage['gpt-4o-mini']['prompt_tokens'] += language_prompt_tokens
        logger.debug('language detection prompt (%d tokens): %s',
                     language_prompt_tokens, language_full_prompt)

        print('\n\n')
        print('language_full_prompt'.upper())
        print(language_full_prompt)
        print('\n\n')

        language_response: LanguageOutputScheme = await language_chain.ainvoke(
            {},
            config=config
        )

        logger.info('detected language: %s', language_response.language)

        # Language detection completion tokens counting
        language_completion_tokens = count_tokens(language_response.language)
        usage['gpt-4o-mini']['completion_tokens'] += language_completion_tokens

        # Translating answer.text to the detected language
        translate_prompt_template = validate_prompt_builder.get_translate_prompt(
            answer.text,
            language_response.language
        )


        translate_chain = (
            translate_prompt_template | llm.with_structured_output(TranslationOutputScheme)
        )
        
        # Translation prompt tokens counting
        # query and language are already filled via .partial(), so we can format without arguments
        translate_messages = translate_prompt_template.format_messages()
        translate_full_prompt = '\n'.join(str(msg.content) for msg in translate_messages)
        translate_prompt_tokens = count_tokens(translate_full_prompt)
        usage['gpt-4o-mini']['prompt_tokens'] += translate_prompt_tokens
        logger.debug('translation prompt (%d tokens): %s',
                     translate_prompt_tokens, translate_full_prompt)

        # Translation chain invoke
        # query and language are already filled via .partial(), so we invoke without arguments
        translate_response: TranslationOutputScheme = TranslationOutputScheme(translation=answer.text)
        try:
            translate_response = await translate_chain.ainvoke(
                {},
                config=config
            )
            logger.info('translated text: %s', translate_response.translation)

            # Translation completion tokens counting
            translate_completion_tokens = count_tokens(translate_response.translation)
            usage['gpt-4o-mini']['completion_tokens'] += translate_completion_tokens
        except Exception as e:
            logger.error('Error during translation: %s', e, exc_info=True)
            # Use original text if translation fails (already set as default above)

        # Forming questions string
        if answer.is_knowledge_base:
            questions = state.get('kommo_queries', [])
            questions_str = ','.join(f'"{q}"' for q in questions)

        elif answer.is_global:
            questions = state.get('global_queries', [])
            questions_str = ','.join(f'"{q}"' for q in questions)
        else:
            questions_str = ''

        # Prompt tokens counting
        full_prompt = validate_prompt_template.format(
            dialog=dialog,
            answer=answer.text,
            questions_str=questions_str
        )

        print('\n\n')
        print('full_prompt'.upper())
        print(full_prompt)
        print('\n\n')
        prompt_tokens = count_tokens(full_prompt)
        logger.info('usage: %s', usage)
        usage['gpt-4o-mini']['prompt_tokens'] += prompt_tokens
        logger.debug('validate_answer prompt (%d tokens) (validating: %s): %s',
                     prompt_tokens, questions_str, full_prompt)
        
        # Chain invoke
        response: ValidateAnswerOutputScheme = await chain.ainvoke(
            {
                'dialog': dialog,
                'answer': answer.text,
                'questions_str': questions_str
            },
            config=config
        )
        logger.info('response: %s', response)

        # Completion tokens counting
        completion_tokens = count_tokens(str(response.is_support))
        usage['gpt-4o-mini']['completion_tokens'] += completion_tokens

        # Need support
        need_support: bool = response.is_support
        is_plug: bool = state.get('is_plug', False)
        
        print('\n\n')
        print(answer)
        print('\n\n')

        if is_plug:
            logger.warning('is_plug=True, redirect to support')
            valid_answers.append(plugs.unable_to_use())

        elif need_support:
            logger.warning('need_support=True, redirect to support')
            valid_answers.append(plugs.info_not_found())

        else:
            logger.info('response is valid, continue dialog')
            # answer = str(translate_response.translation)            
            valid_answers.append(answer)

    return Command(
        goto='__end__',
        update={
            'answers': valid_answers,
            'usage': usage,
            'small_talk_queries': [],
            'global_queries': [],
            'kommo_queries': []
        }
    )

