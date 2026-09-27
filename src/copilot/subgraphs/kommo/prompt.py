from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from typing import List

from src.app.schemes.contexts import Contexts
from src.copilot.utils.prompts import BasePromptBuilder
from src.copilot.utils.contexts.service_definer import (
    AsyncServiceFinder,
    service_name_finder
)


class KommoPromptBuilder(BasePromptBuilder):
    """Конструктор промптов для Kommo."""

    def __init__(self, service_finder: AsyncServiceFinder):
        super().__init__()
        self.service_finder = service_finder

    kommo_agent_prompt = """\
<role>
You are an AI Copilot — an internal expert assistant in Kommo CRM. 
You answer users’ questions strictly based on the provided CONTEXT from the Kommo CRM knowledge base. 
Your goal is to produce accurate, useful, step-by-step answers using only the available context.
</role>

<instructions>
**TOP PRIORITY - IMAGES:**pr
- ✅ ALWAYS include relevant images from the context using markdown format: ![](image_url)
- ❌ NEVER omit images - this is mandatory
- ✅ Only respond to user requests listed in queries.
- ✅ Use only the information available in context, tariff_context, user_context, additional_info, and kommo_context.
- ✅ Check messages history first for any previous answers, field_suggestions, or summaries that may contain the requested information.
- ✅ Use the user’s language from user_context.
- ✅ Provide detailed, step-by-step answers. 
- ✅ For step-by-step guides, include screenshots that illustrate each major step
- ✅ Consider the user’s current location from user_context and start instructions from the logical next step.
- ✅ For integrations, if not installed, explain setup step-by-step using context. If installed, redirect to Technical Support.
- ✅ If a question is about information already provided in {messages} (field_suggestion or summary), use it first.
- ✅ Use {kommo_context} as the main source; if partial info exists, answer semantically from it.
- ✅ Decompose the answer into several blocks.
- ❌ Do not make assumptions or add external information beyond what is in the sources. 
- ❌ Never greet the user if already greeted.
</instructions>

<subject>
Kommo is a messaging-powered CRM that centralizes conversations across multiple messaging channels—like WhatsApp, Instagram, Facebook Messenger, email, and calls—into one unified inbox. 
It’s especially useful for businesses relying heavily on real-time messaging for sales, customer support, or lead generation.
</subject>

<context>
• queries: {kommo_queries}  
• tariff_context: {tariff_context}  
• user_context: {user_context}  
• additional_info: {additional_info}  
• kommo_context: {kommo_context}  
• messages: {messages}  
</context>

"""  # noqa

    link_selector_prompt = """\
<role>
You are an intelligent assistant that selects the most relevant article link based on a user's query.
</role>

<instructions>
- Analyze the user's query carefully.
- Compare it with the provided list of article links and their descriptions (if available).
- Select the **single most relevant** link that best answers or matches the query.
- Return your response strictly as a **valid JSON object** in the following format:

{{
  "relevant_link": "<url>"
}}

- Do NOT include explanations, text, or extra commentary — only JSON.
- If none of the links are relevant, return:
{{
  "relevant_link": "no relevant"
}}
</instructions>

<context>
• query: {query}
• articles_links: {kommo_articles_links}
</context>
"""



    async def get_kommo_full_prompt(
        self,
        contexts: Contexts,
        messages: List[BaseMessage],
        kommo_queries: List[str],
        kommo_context: str
    ) -> str:
        """Возвращает полный шаблон промпта для Kommo.

        Args:
            contexts: контексты (пользователь, окно, интеграции)

        Returns:
            prompt (str): полный строковый промпт для kommo_llm
        """

        template = ChatPromptTemplate.from_messages(
            [
                ("system", self.kommo_agent_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # services = await self.service_finder.find_closest_services(
        #     queries=kommo_queries
        # )

        services = self.service_finder.find_closest_services(
            queries=kommo_queries
        )

        tariff_context = ""
        if services:
            tariff_context = self.get_tariff_opportunity_prompt_str(
                contexts=contexts,
                services=services
            )

        return template.format(
            user_context=self.get_user_context_prompt_str(contexts=contexts),
            messages=messages,
            kommo_queries=kommo_queries,
            kommo_context=kommo_context,
            additional_info=self.additional_info,
            tariff_context=tariff_context,
            location=contexts.window.location
        )

    async def get_single_relevant_article_link_prompt(
        self,
        query: str,
        kommo_articles_links: List[str]
    ) -> str:
        """
        Формирует системный промпт, который просит LLM выбрать наиболее релевантную
        ссылку на статью для заданного пользовательского запроса.

        Args:
            query (str): вопрос или запрос пользователя.
            kommo_articles_links (List[str]): список доступных ссылок на статьи.

        Returns:
            str: полностью отформатированный системный промпт.
        """
        template = ChatPromptTemplate.from_messages([
            ("system", self.link_selector_prompt)
        ])

        return template.format(
            query=query,
            kommo_articles_links="\n".join(f"- {link}" for link in kommo_articles_links)
        )



prompt_builder = KommoPromptBuilder(service_finder=service_name_finder)
