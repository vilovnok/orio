import logging
from pydantic import BaseModel, Field
from typing import Dict, Optional, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.app.schemes.contexts import Contexts, Question
from src.app.schemes.agent import StateInputData, CopilotAnswersScheme


logger = logging.getLogger(__name__)


class CopilotRequest(BaseModel):
    """Copilot request scheme for beanstalkd jobs processing"""
    request_id: str
    chat_id: str
    contexts: Contexts
    question: Question
    enrich: Dict = Field(default_factory=dict)
    sla: Dict[str, int] = Field(default_factory=dict)

    question_id: Optional[str] = None
    feature_id: Optional[str] = None
    time_question_task_start_process_in_backend: Optional[int] = 0
    time_question_task_redirect_in_router: Optional[int] = 0
    time_question_task_redirect_in_internal_gateway: Optional[int] = 0
    time_question_task_redirect_in_feature: Optional[int] = 0

    def prepare_data_for_agent(self) -> StateInputData:
        """Preparing input data for agent."""
        # Convert history messages to LangChain format
        history_messages: List[BaseMessage] = []
        for item in self.contexts.history:
            if item.is_question:
                history_messages.append(HumanMessage(
                    content=item.text,
                    id=item.id
                ))
            else:
                history_messages.append(AIMessage(
                    content=item.text,
                    id=item.id
                ))
        # Add last question to the end of the list
        history_messages.append(HumanMessage(
            content=self.question.text,
            id=self.question.id
        ))

        contexts_dict = self.contexts.model_dump(exclude={'history'})
        contexts_obj = Contexts(**contexts_dict)
        logger.info("contexts_dict: %r", contexts_dict)

        return StateInputData(
            messages=history_messages,
            thread_id=self.chat_id,
            contexts=contexts_obj
        )


class CopilotResponse(BaseModel):
    request_id: str
    chat_id: str
    contexts: Contexts
    question: Question
    answers: List[CopilotAnswersScheme]

    sla: Dict[str, int]
    enrich: Dict = Field(default_factory=dict)
