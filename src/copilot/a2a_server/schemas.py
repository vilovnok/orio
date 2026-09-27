import logging
import uuid

from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.app.schemes.agent import StateInputData, CopilotAnswersScheme
from src.app.schemes.contexts import Contexts, Question

logger = logging.getLogger(__name__)


class AgentMessageScheme(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str = Field(
        description='role of message; Example: "conversation_agent", "human"'
    )
    content: str


class CopilotPayload(BaseModel):
    """Copilot payload scheme for A2A server"""
    version: str = Field(default='0.1.0')
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chat_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    contexts: Contexts
    question: Question
    answers: List[CopilotAnswersScheme] = Field(default_factory=list)
    messages: List[AgentMessageScheme] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)

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
        logger.info("contexts_dict: %r", contexts_dict)
        contexts_obj = Contexts(**contexts_dict)

        return StateInputData(
            messages=history_messages,
            thread_id=self.chat_id,
            contexts=contexts_obj
        )

    def __repr__(self):
        return f"""\
        CopilotRequest(
            request_id={self.request_id},
            chat_id={self.chat_id},
            contexts={self.contexts},
            question={self.question})"""
