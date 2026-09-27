import asyncio
import logging
from datetime import datetime
from langsmith import traceable

from langchain_core.tools import tool

from src.copilot.services.elastic.search import hybrid_search_articles

from src.copilot.tools.utils.rerank import get_reranked_context_from_article
from src.copilot.tools.utils.text_processing import normalize_url

from src.copilot.utils.schemes import ToolResult

logger = logging.getLogger(__name__)


@tool
@traceable(run_type="tool", name="retrieve_kommo")
async def retrieve_kommo(query: str) -> ToolResult:
    """Retrieve information in Kommo docs and articles (vector search)
    Args:
        query: (str) string for english language, (very important)
        
    Advice:
        - Use it only if you need more information about Kommo.

    Rules:
    - query (str) MUST be strictly written in ENGLISH language
    - Use all tool_calls in one time. One tool call per one actual question.
        
    """  # noqa
    start_time = datetime.now()
    logger.info('query: %s', query)

    try:
        # 1. -- Getting all articles --
        logger.info('start processing query: %s', query)
        articles = await hybrid_search_articles(
            query=query,
            k=6,
            min_score=0.5,
            text_weight=0.1,
            vector_weight=0.9
        )

        links = set()
        for article in articles:
            try:
                url = normalize_url(article.link)
                links.add(url)
            except Exception:
                logger.warning('Error in normalize_url: %s', article.link)
                continue

    except Exception:
        logger.error('Error in search_articles', exc_info=True)
        raise

    logger.info(
        "Time spent in knowledge base search: %.2f",
        (datetime.now() - start_time).total_seconds()
    )

    if not articles:
        logger.warning('No information found in [search_kommo_docs]')
        raise ValueError('No information found in [search_kommo_docs]')
    logger.info('found %d articles', len(articles))

    # 2. -- Reranking --
    try:
        reranked_contexts = await asyncio.gather(
            *[
                get_reranked_context_from_article(
                    query=query,
                    article=article,
                    k=3
                )
                for article in sorted(
                    articles, key=lambda x: x.score, reverse=True
                )
            ]
        )
        if not reranked_contexts:
            contexts = []
        else:
            contexts = []
            for article, ctx_list in zip(articles, reranked_contexts):
                # ctx_list может быть строкой или списком, зависит от твоей функции
                if isinstance(ctx_list, list):
                    ctx = " ".join(ctx_list[:2])  # берём топ-2 куска
                else:
                    ctx = ctx_list
                contexts.append({
                    "source": normalize_url(article.link),
                    "text": ctx
                })

    except Exception as e:
        logger.error(
            'Error in get_reranked_context: %s',
            e, exc_info=True
        )
        raise

    logger.info('reranked %d articles..', len(articles))
    logger.info('Links: %s;', links)

    return ToolResult(
        output="Success",
        state_updates={
            'retrieve_kommo_context': contexts,
            'kommo_articles_links': list(links),
            'kommo_queries': [query]
        }
    )
