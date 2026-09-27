import logging
from langchain_core.tools import tool

from langsmith import traceable

from src.copilot.tools.utils.text_processing import (
    parse_response,
    OpenAISearchResponse
)


from src.copilot.utils.schemes import ToolResult, SearchResult, WebSearchResult
from src.utils.config import inited_config as config

import requests


logger = logging.getLogger(__name__)


# @traceable(type="llm", name="gpt_searching")
# async def gpt_searching(query: str) -> OpenAISearchResponse:
#     """Search information in web using GPT-4o-mini

#     query: str - FULL prompt to gpt-4o-mini-search

#     Returns:
#         OpenAIResponse: Response from gpt-4o-mini-search
#     """

#     try:
#         response = await async_client.responses.create(
#             model=config.copilot.models.gpt_search_preview,
#             temperature=0.3,
#             tools=[{"type": "web_search_preview"}],
#             tool_choice="required",
#             input=query
#         )

#         logger.info("llm_response: %s", response)

#         llm_response: OpenAISearchResponse = parse_response(response)
#         return llm_response

#     except Exception as e:
#         logger.exception("Error in search_in_web:", exc_info=True)
#         return f"Error: {e}"


@tool
@traceable(type="llm", name="searching")
def web_search(query: str) -> str:
    """Search the internet for up-to-date information."""

    url = f'http://{config.web_search.host}:{config.web_search.port}'

    response = requests.get(
        f"{url}/search",
        params={
            "q": query,
            "format": "json",
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    # results = [
    #     SearchResult(
    #         title=result["title"],
    #         url=result["url"],
    #         content=result.get("content", ""),
    #     )
    #     for result in data.get("results", [])[:5]
    # ]

    # print('\n\n\n')
    # print('&'*100)
    # print(results[0])
    # print('&'*100)
    # print('\n\n\n')
    
    return WebSearchResult(
        results=results,
    )




# @tool
# async def search_in_web(query: str) -> ToolResult:

#     return ToolResult(
#         output='Search in web',
#         state_updates={
#             'search_in_web_context': 'Search in web'
#         }
#     )
