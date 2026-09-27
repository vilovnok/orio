
import logging
from typing import List

from elasticsearch import AsyncElasticsearch

from langchain_openai.embeddings import OpenAIEmbeddings

from src.copilot.services.elastic.schemes import ElasticsearchArticle
from src.utils.config import inited_config as config


logger = logging.getLogger(__name__)


# async_es = AsyncElasticsearch(
#     hosts=f"{config.elasticsearch.host}:{config.elasticsearch.port}",
#     # Заголовок нужен для маршрутизации на Prod
#     # ID кластера
#     # Заголовок необходим для маршрутизации, т.к. индексы аккаунтов могут
#     # находиться на разных кластерах.
#     headers={"X-C-ID": "c1"}
# )

# Initialize OpenAIEmbeddings with a synchronous client
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-large",
#     dimensions=1024,
#     api_key=config.openai.api_key
# )


async def hybrid_search_articles(
    query: str,
    k: int = 3,
    k_buffer: int = 2,
    min_score: float = 0.7,
    index: str = config.elasticsearch.index,
    text_weight: float = 0.5,
    vector_weight: float = 0.5,
    minTextScore: float = 1.0,
    maxTextScore: float = 100.0,
) -> List[ElasticsearchArticle]:
    """
    Hybrid search articles using BM25 + vector similarity.

    Args:
        query (str): поисковой запрос.
        k (int, optional): Количество возвращаемых результатов. По умолчанию 10.
        min_score (float, optional): Минимальный score для включения документа. По умолчанию 1.55.
        index (str, optional): Название индекса. По умолчанию берется из конфигурации.
        text_weight (float, optional): Вес текстового компонента (BM25).
        vector_weight (float, optional): Вес векторного компонента.
        minTextScore (float, optional): Минимальное значение для нормализации BM25.
        maxTextScore (float, optional): Максимальное значение для нормализации BM25.

    Returns:
        List[ElasticsearchArticle]: Список результатов, преобразованных в модель ElasticsearchArticle.
    """  # Noqa
    logger.debug("[hybrid_search_articles] Start of hybrid search query: %s",
                 query)

    # Получаем векторное представление запроса
    try:
        logger.debug("[hybrid_search_articles] Getting vector query")
        # vector = await embeddings.aembed_query(query)
    except Exception as e:
        logger.error("[hybrid_search_articles] Error getting vector: %s",
                     e, exc_info=True)
        return []

    account_id = config.elasticsearch.account_id

    # Формируем гибридный запрос с использованием script_score:
    query_body = {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "content"],
                    }
                }
            ],
            "filter": {
                "bool": {
                    "must": [
                        {"term": {"account_id": account_id}}  # Noqa
                    ],
                    "must_not": [
                        {"term": {"is_disabled": True}}
                    ]
                }
            }
        }
    }

    # script = {
    #     "source": (
    #         "double text_score = _score; "
    #         "double vector_score = (cosineSimilarity(params.query_vector, 'content_vector') + 1) / 2; "  # noqa
    #         "double normalizedTextScore = (text_score - params.minTextScore) / "  # noqa
    #         "(params.maxTextScore - params.minTextScore); "  # noqa
    #         "return (normalizedTextScore * params.text_weight) + (vector_score * params.vector_weight); "  # noqa
    #     ),
    #     "params": {
    #         "query_vector": vector,
    #         "text_weight": text_weight,
    #         "vector_weight": vector_weight,
    #         "minTextScore": minTextScore,
    #         "maxTextScore": maxTextScore
    #     }
    # }
    # body = {
    #     "query": {
    #         "bool": {
    #             "must": [
    #                 {
    #                     "script_score": {
    #                         "query": query_body,
    #                         "script": script
    #                     }
    #                 }
    #             ],
    #             "filter": {
    #                 "bool": {
    #                     "must": [
    #                         {"term": {"account_id": account_id}}
    #                     ],
    #                     "must_not": [
    #                         {"term": {"is_disabled": True}}
    #                     ]
    #                 }
    #             }
    #         },
    #     },
        # Collapse duplicates by source
        # "collapse": {
        #     "field": "source",
        #     "inner_hits": {
        #         "name": "best_chunk",
        #         "size": 1,
        #         "sort": ["_score"]
        #     }
        # },
    #     "size": k+k_buffer,
    #     "min_score": min_score
    # }

    try:
        # response = await async_es.search(index=index, body=body)
        response = {"hits": {'some': "заглушка"}}
        logger.debug('got response from ES successfully')
    except Exception as e:
        logger.error("[hybrid_search_articles] Error in search: %s",
                     e, exc_info=True)
        return []
    hits = response.get('hits', {}).get('hits', [])
    logger.info('got %d results from ES', len(hits))
    
    if not hits:
        logger.warning("query: '%s' not found documents.", query)
        return []

    try:
        logger.info(f'ELASTIC SCORES: {[hit["_score"] for hit in hits]} ')
        # all_articles = [ElasticsearchArticle.from_es_hit(hit) for hit in hits]
        all_articles = []
        logger.debug('found %d articles; articles: %s',
                     len(all_articles),
                     [article.title for article in all_articles if article])

        articles = [
            ElasticsearchArticle(
                id="mock-article-1",
                score=0.95,
                account_id=account_id,
                user_id=None,
                link="https://example.com/kommo/create-lead",
                page_id=1,
                lang="ru",
                token_count=120,
                content=(
                    "Чтобы создать нового лида в Kommo CRM, "
                    "откройте раздел «Сделки» и нажмите кнопку "
                    "«Добавить сделку». Заполните необходимые поля "
                    "и сохраните сделку."
                ),
                title="Как создать нового лида в Kommo CRM",
                is_disabled=False,
                page_content=(
                    "title: Как создать нового лида в Kommo CRM\n\n"
                    "Чтобы создать нового лида в Kommo CRM, "
                    "откройте раздел «Сделки» и нажмите кнопку "
                    "«Добавить сделку». Заполните необходимые поля "
                    "и сохраните сделку."
                ),
                page_content_and_url=(
                    "title: Как создать нового лида в Kommo CRM\n"
                    "url: https://example.com/kommo/create-lead\n\n"
                    "Чтобы создать нового лида в Kommo CRM, "
                    "откройте раздел «Сделки» и нажмите кнопку "
                    "«Добавить сделку». Заполните необходимые поля "
                    "и сохраните сделку."
                ),
            ),
            ElasticsearchArticle(
                id="mock-article-2",
                score=0.88,
                account_id=account_id,
                user_id=None,
                link="https://example.com/kommo/leads",
                page_id=2,
                lang="ru",
                token_count=90,
                content=(
                    "Для работы с лидами перейдите в раздел сделок. "
                    "Здесь можно создавать новые сделки, изменять "
                    "их данные и переводить их между этапами воронки."
                ),
                title="Работа с лидами",
                is_disabled=False,
                page_content=(
                    "title: Работа с лидами\n\n"
                    "Для работы с лидами перейдите в раздел сделок. "
                    "Здесь можно создавать новые сделки, изменять "
                    "их данные и переводить их между этапами воронки."
                ),
                page_content_and_url=(
                    "title: Работа с лидами\n"
                    "url: https://example.com/kommo/leads\n\n"
                    "Для работы с лидами перейдите в раздел сделок. "
                    "Здесь можно создавать новые сделки, изменять "
                    "их данные и переводить их между этапами воронки."
                ),
            ),
        ]

        # articles = list(
        #     {article.title: article for article in all_articles}.values()
        # )
        # Log, if there are duplicates
        if len(articles) < len(all_articles):
            logger.warning('Found %d duplicates for query: "%s"',
                           len(all_articles) - len(articles), query)
    except Exception as e:
        logger.error('error processing results: %s', e, exc_info=True)
        return []

    return articles[:k]
