import logging
from typing import (
    Annotated,
    Union,
    Sequence,
    TypedDict,
    List,
    Dict,
    Any,
    Set,
    Tuple
)

from src.app.schemes.contexts import Contexts
from src.app.schemes.agent import CopilotAnswersScheme, StateInputData

from src.copilot.utils.schemes import SmallTalk

from langchain_core.messages import BaseMessage, trim_messages
from langgraph.graph import add_messages
from langsmith import traceable

from src.copilot.services.usage.counters import count_tokens
from src.copilot.utils.plugs import PlugsBuilder

from src.utils.config import inited_config as config

logger = logging.getLogger(__name__)




@traceable
def manage_messages(
    messages: Sequence[BaseMessage],
    max_tokens: int = 1200
) -> List[BaseMessage]:
    """Controls the size of the message list."""
    try:
        result = trim_messages(
            messages=messages,
            strategy='last',
            max_tokens=max_tokens,
            token_counter=count_tokens,
            include_system=True,
            allow_partial=False,
            start_on='human'
        )
        logger.info('controlled messages: (len: %d)', len(result))
        return result

    except Exception:
        logger.error("error when removing extra messages", exc_info=True)

        if isinstance(messages, list):
            logger.warning(
                "returns original messages (tokens: %d), (returns %s)",
                count_tokens(messages), messages
            )
            logger.info('length of messages list: %s', len(messages))
            return messages

        logger.warning("returns empty messages list")
        return []


@traceable
def custom_messages_reducer(
    existing: List[BaseMessage],
    updates: Union[List[BaseMessage], dict]
) -> List[BaseMessage]:
    """Custom reducer for messages."""
    if isinstance(updates, dict):
        new_messages = updates.get('messages', [])
    else:
        new_messages = updates

    try:
        combined = add_messages(existing, new_messages)
        result = manage_messages(
            combined,
            max_tokens=config.copilot.history_max_tokens
        )

        logger.info("messages processed, len: %d", len(result))
        logger.debug("graph_state_messages: %s", result)
        return result
    except Exception:
        logger.error("error when adding messages", exc_info=True)
        return existing


def merge_state_updates(acc: Dict, update: Dict) -> Dict:
    """Merge state updates

    Args:
        acc: Dict - accumulator
        update: Dict - update

    Returns:
        Dict - merged state updates"""

    for key, value in update.items():
        if key in acc and isinstance(acc[key], list) and isinstance(value, list):  # noqa
            acc[key].extend(value)
        elif key in acc and isinstance(acc[key], str) and isinstance(value, str):  # noqa
            acc[key] = acc[key] + value
        else:
            acc[key] = value
    return acc


class CopilotState(TypedDict):
    # -| Messages |-
    messages: Annotated[List[BaseMessage], custom_messages_reducer]
    answer: str
    answers: List[CopilotAnswersScheme]
    contexts: Contexts

    loop_count: int

    # -| Subggraph Contexts |-
    retrieve_kommo_context: str
    kommo_articles_links: List[str]
    kommo_queries: List[str]

    kommo_tool_calls: List[str]

    # -| Global Contexts |-
    search_in_web_context: str
    global_queries: List[str]

    # -| Small Talk |-
    small_talk_queries: List[SmallTalk]

    to_run: List[Set[str]]

    usage: Dict[str, Dict[str, Any]]


def ensure_contexts(obj: Any) -> Contexts:
    """Universal factory for getting Contexts object
    from different types of input data."""
    if isinstance(obj, Contexts):
        return obj

    if isinstance(obj, dict):
        if "contexts" in obj:
            return ensure_contexts(obj["contexts"])
        return Contexts(**obj)

    if hasattr(obj, "contexts"):
        return ensure_contexts(getattr(obj, "contexts"))

    raise ValueError(
        "Failed to get Contexts from object of type %s",
        type(obj)
    )


def get_contexts_and_plugs(
    state: CopilotState | StateInputData
) -> Tuple[Contexts, PlugsBuilder]:
    """
    Получить Contexts и PlugsBuilder из состояния.
    """
    contexts = ensure_contexts(state)
    plugs = PlugsBuilder(language=contexts.user.language)
    return contexts, plugs
