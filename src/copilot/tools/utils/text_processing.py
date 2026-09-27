from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from datetime import datetime
import asyncio

from pydantic import BaseModel


import logging

from src.copilot.services.usage.counters import count_tokens

from src.copilot.utils.schemes import WebSearchResult, SearchResult

logger = logging.getLogger(__name__)



class SearchResponse(BaseModel):
    content: str
    web_search_used: bool
    urls: List[str]

    def get_text_and_top_url(self) -> str:
        """Get text and the top URL

        Return format:
            text
            top_url"""
        top_url = self.urls[0] if self.urls else None
        if top_url:
            return f"{self.content}\n\n[{top_url}]({top_url})"
        return self.content


def parse_response(response: WebSearchResult) -> SearchResponse:
    """Parse response from OpenAI Responses API"""

    # content = response.output_text

    # results = [
    #     SearchResult(
    #         title=result["title"],
    #         url=result["url"],
    #         content=result.get("content", ""),
    #     )
    #     for result in data.get("results", [])[:5]
    # ]
    sources = []
    for item in response.get("results", [])[:5]:
        source = SearchResult(
            title=item["title"],
            url=item["url"],
            content=item.get("content", ""),
        )

# answer_text = "\n".join([f"{item.title}: [{item.url}]({item.url})" for item in response.results])
        sources.append(f"{source.title}: [{source.url}]({source.url})")


    # web_search_used = any(
    #     getattr(
    #         tool, 'name', ''
    #     ).lower() == 'web search' for tool in response.tools
    # )

    # for output in reversed(response.output):
    #     if getattr(output, 'type', '') == 'web_search_call':
    #         web_search_used = True
    #     if getattr(
    #         output, 'type', ''
    #     ) == 'message' and getattr(output, 'role', '') == 'assistant':
    #         last_message = output
    #         break

    # if last_message and last_message.content:
    #     first_content = last_message.content[0]
    #     annotations = getattr(first_content, 'annotations', [])
    #     urls = {
    #         normalize_url(annotation.url)
    #         for annotation in annotations
    #         if hasattr(annotation, 'url') and annotation.url
    #     }

    logger.info('found %d urls; URLS: %s', len(sources))

    if not sources:
        logger.warning("No URLs found in response")
        urls = {}

    return OpenAISearchResponse(
        content=content,
        web_search_used=web_search_used,
        urls=list(urls)
    )


async def split_article(
    content: str,
    chunk_size: int = 600,
    chunk_overlap: int = 100
) -> List[Document]:
    """Split article into chunks

    Args:
        content (str): Article content
        chunk_size (int, optional): Chunk size. Defaults to 350.
        chunk_overlap (int, optional): Chunk overlap. Defaults to 100.

    Returns:
        List[Document]: List of documents"""
    logger.info("Start split article..")
    start_time = datetime.now()
    content_tokens = count_tokens(content)

    if content_tokens < 800:
        logger.warning("Content is too short for splitting")
        return [Document(page_content=content)]

    if not content:
        logger.error("Content is empty for doc")
        return []
    try:
        logger.info("Creating splitter..")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=count_tokens,

        )

        logger.info("Splitter created")

        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(
            None,
            lambda: splitter.create_documents(
                texts=[content if content else 'No content'],
                metadatas=[{
                        "id": None,
                        "link": None,
                        "title": None
                    }]
            )
        )

        logger.info("Split %d docs", len(docs))
        logger.debug("Docs: %s", docs)
        logger.info("timecount: %s", datetime.now() - start_time)
        return docs
    except Exception as e:
        logger.error(
            "Error in split article: %s", e
        )
        return []
