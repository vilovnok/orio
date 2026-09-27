from pydantic import BaseModel, Field
from typing import Optional, Any, Dict

import logging

logger = logging.getLogger(__name__)


class ElasticsearchArticle(BaseModel):
    """Model for representing an article from Elasticsearch"""
    id: str = Field(..., description="Document ID")
    score: float = Field(..., description="Relevance of the document")

    account_id: int = Field(..., description="Account ID")
    user_id: Optional[int] = Field(None, description="User ID")
    link: Optional[str] = Field(None, description="Source link")
    page_id: Optional[int] = Field(None, description="Page ID")

    # -| params
    lang: Optional[str] = Field(None, description="Document language")
    token_count: Optional[int] = Field(None, description="Token count")

    # -| content
    content: Optional[str] = Field(None, description="Document content")
    title: Optional[str] = Field(None, description="Document title")

    # -| flags
    is_disabled: Optional[bool] = Field(None, description="Disabled flag")

    # -| Custom Fields
    page_content: Optional[str] = Field(
        default=None,
        description="Composite field: title + content"
    )

    page_content_and_url: Optional[str] = Field(
        default=None,
        description="Composite field: page_content + url"
    )

    reranked_context: Optional[str] = Field(
        description='Reranked context of the document content',
        default=None
    )

    @classmethod
    def from_es_hit(
        cls,
        hit: Dict[str, Any]
    ) -> "ElasticsearchArticle":
        """Create an object from the ES search result"""
        source = hit.get("_source", {})

        # Лучший чанк внутри группы
        inner = hit.get("inner_hits", {}) \
            .get("best_chunk", {}) \
            .get("hits", {}) \
            .get("hits", [])

        if inner:
            best = inner[0]["_source"]
            content = best.get("content")
        else:
            content = source.get("content")  # fallback

        title = source.get("title")
        link = source.get("source")

        logger.info('raw_url: %s', link)

        page_content = f'title: {title}\n\n{content}'
        page_content_and_url = f'title: {title}\nurl: {link}\n\n{content}'

        return cls(
            id=hit.get("_id", ""),
            score=hit.get("_score", 0.0),
            account_id=source.get("account_id"),
            user_id=source.get("user_id"),
            link=source.get("source"),
            page_id=source.get("page_id"),
            lang=source.get("lang"),
            content=content,
            title=title,
            page_content=page_content,
            page_content_and_url=page_content_and_url,
            is_disabled=source.get("is_disabled"),
            token_count=source.get("token_count"),
        )
