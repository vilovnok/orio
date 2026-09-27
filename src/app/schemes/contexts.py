import uuid
import time
from pydantic import BaseModel, Field
from typing import List, Optional


class Question(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str


class HistoryItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    is_question: bool
    is_plug: bool
    is_support: bool
    is_global: bool
    is_knowledge_base: bool

    # Optional fields (for back compatibility)
    score: Optional[str] = ""
    is_question_preprocessed: Optional[bool] = None
    is_answer_helpful: Optional[bool] = None
    created_at: int = Field(default_factory=lambda: int(time.time()))


class AccountContext(BaseModel):
    id: int
    subdomain: str
    tariff: str = Field(
        description="User's tariff Literal['base', 'advanced', 'enterprise']",
        default="base",
    )
    installed_integrations: List[str] = Field(
        description="List of installed integrations",
        default_factory=list
    )


class UserContext(BaseModel):
    id: int
    language: str
    email: str = "user@example.com"
    name: str = "User"
    group: Optional[int] = 0


class WindowArguments(BaseModel):
    entity_id: int
    entity_type: str
    entity_name: str = "Default"


class WindowContext(BaseModel):
    location: str
    arguments: WindowArguments


class Contexts(BaseModel):
    account: AccountContext
    user: UserContext
    window: WindowContext
    history: List[HistoryItem] = Field(default_factory=list)
