from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
import uuid


class Submitter(BaseModel):
    account_id: int
    user_id: int
    group: Optional[int] = 0


class Tokens(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0


class Pricing(BaseModel):
    prompt_tokens: float
    completion_tokens: float
    cached_tokens: float = 0.0


class Cost(BaseModel):
    prompt_tokens: float
    completion_tokens: float
    cached_tokens: float = 0.0


class Metrics(BaseModel):
    request_time: float
    processing_time: float


class UsageItem(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    submitter: Submitter
    model: str
    used_by: str = "copilot_rag"
    tokens: Tokens
    metrics: Optional[Metrics]
    created_at: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


class UsageItemWithPricing(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    submitter: Submitter
    model: str
    used_by: str = "copilot_rag"
    tokens: Tokens
    pricing: Optional[Pricing]
    cost: Optional[Cost]
    metrics: Optional[Metrics] = None
    created_at: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


class UsageTrackingRequest(BaseModel):
    usages: List[UsageItem]
