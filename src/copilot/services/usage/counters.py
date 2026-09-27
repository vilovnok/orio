import tiktoken
from typing import List, Union
from langchain_core.messages import BaseMessage


encoding = tiktoken.get_encoding('cl100k_base')


def count_tokens_from_str(text: str) -> int:
    """Count tokens from string."""
    return len(encoding.encode(text))


def count_tokens_from_messages(messages: List[BaseMessage]) -> int:
    """Count tokens from messages."""
    return sum(
        count_tokens_from_str(msg.content)
        if isinstance(msg.content, str) else 0
        for msg in messages
        if isinstance(msg, BaseMessage)
    )


def count_tokens(input: Union[str, List[BaseMessage]]) -> int:
    """Count tokens from input."""
    if isinstance(input, str):
        return count_tokens_from_str(input)
    elif isinstance(input, list):
        return count_tokens_from_messages(input)
    else:
        raise ValueError('Invalid input type')
