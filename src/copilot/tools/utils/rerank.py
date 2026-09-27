import asyncio
import logging
from datetime import datetime
from typing import List, Literal

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity       # type: ignore

from src.copilot.services.elastic.schemes import ElasticsearchArticle
from src.copilot.tools.utils.text_processing import split_article

logger = logging.getLogger(__name__)


def tfidf_rerank(docs: List[str], query: str, k: int) -> List[str]:
    """Reranking documents by similarity to the query using TF-IDF.

    Args:
        docs: List of strings with documents/excerpts.
        query: Query by which relevance is determined.
        k: Number of returned most relevant documents.

    Returns:
        List of k most relevant documents."""

    start_time = datetime.now()
    vectorizer = TfidfVectorizer()

    doc_vectors = vectorizer.fit_transform(docs)
    after_fit_time = datetime.now()
    logger.debug(
        '[tfidf_rerank] TF-IDF vectorizer initialized in %.2f seconds',
        (after_fit_time - start_time).total_seconds()
    )
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity between the query vector and each document.
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    logger.info(
        '[tfidf_rerank] Cosine similarity calculated in %.2f seconds',
        (datetime.now() - after_fit_time).total_seconds()
    )

    # Getting indices of documents sorted by similarity.
    top_indices = np.argsort(similarities)[::-1][:k]

    # Forming a list of the most relevant documents.
    ranked_docs: List[str] = [docs[i] for i in top_indices]
    logger.info(
        '[tfidf_rerank] Reranking completed in %.2f seconds',
        (datetime.now() - start_time).total_seconds()
    )
    return ranked_docs


async def get_reranked_context_from_article(
    query: str,
    article: ElasticsearchArticle,
    k: int = 4,
    model_name: Literal['tfidf'] = 'tfidf'
) -> str:
    """Return reranked context for query and article

    Args:
        query (str): Query
        article (ElasticsearchArticle): Article
        k (int, optional): Number of documents chunking each article.
        model_name (str, optional): Model name. Defaults to 'BAAI/bge-reranker-base'.

    Returns:
        str: Reranked context"""  # noqa

    logger.info("start reranking for Q: %s", query)
    start_time = datetime.now()

    docs = await split_article(content=article.page_content)

    if model_name == 'tfidf':
        try:
            loop = asyncio.get_event_loop()
            reranked_docs: List[str] = await loop.run_in_executor(
                None,
                lambda: tfidf_rerank(
                    query=query,
                    docs=[doc.page_content for doc in docs],
                    k=k
                )
            )
        except Exception as e:
            logger.error("[tfidf reranking] Error in Reranker: %s", e)
            return 'Reranking failed'

        logger.info("got %d reranked docs", len(reranked_docs))  # noqa: E501
        logger.debug("reranked docs: %s", reranked_docs)
        logger.info("start formatting body..")

        body = '\n'.join(
            f'{i + 1}. {doc}'
            for i, doc in enumerate(
                reranked_docs[:3]
            )
        )

    # Группируем отрывки по принадлежности к статье
    head = f'title: {article.title}\nurl: {article.link}\n\n'

    logger.info("time: %s", datetime.now() - start_time)

    return f'{head}\n\n{body}'
