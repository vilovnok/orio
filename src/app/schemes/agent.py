from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage

from src.app.schemes.contexts import Contexts, Question


class StateInputData(BaseModel):
    messages: List[BaseMessage]
    thread_id: str
    contexts: Contexts

    def __repr__(self):
        return f"""\
        StateInputData(
            messages={self.messages},
            thread_id={self.thread_id},
            contexts={self.contexts})"""


class CopilotAnswersScheme(BaseModel):
    text: str
    is_knowledge_base: bool = Field(default=False)
    is_global: bool = Field(default=False)
    is_plug: bool = Field(default=False)
    is_support: bool = Field(default=False)

    # constants:
    feature: str = Field(default='suggestion_reply')
    action: str = Field(default='ask_question_about_kommo')


class GraphOutput(BaseModel):
    """Graph output Scheme"""
    answers: List[CopilotAnswersScheme]
    usage: Dict[str, Dict[str, Any]]


class CopilotResponse(BaseModel):
    request_id: str
    chat_id: str
    contexts: Contexts
    question: Question
    answers: List[CopilotAnswersScheme]
    sla: Dict[str, int]
    enrich: Dict = Field(default_factory=dict)
