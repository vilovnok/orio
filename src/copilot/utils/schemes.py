from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal

from src.app.schemes.agent import CopilotAnswersScheme


class ToolResult(BaseModel):
    output: str
    state_updates: Optional[Dict[str, Any]] = {}

    def __str__(self):
        return self.output


class SearchResult(BaseModel):
    title: str
    url: str
    content: str = ""


class WebSearchResult(BaseModel):
    results: list[SearchResult] = Field(default_factory=list)
    state_updates: Dict[str, Any] = Field(default_factory=dict)


class SubgraphResult(BaseModel):
    answers: List[CopilotAnswersScheme]
    state_updates: Optional[Dict[str, Any]] = {}


class Query(BaseModel):
    text: str = Field(..., description="Query text")
    route: Literal["kommo", "global", "small_talk"]


class QueryList(BaseModel):
    q_list: List[Query]


class SmallTalk(BaseModel):
    text: str
    target: Literal["small_talk", "kommo", "global"]
