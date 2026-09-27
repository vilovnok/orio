import logging
from typing import List
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from src.app.schemes.contexts import Contexts

from src.copilot.utils.contexts.tariff_definer import (
    tariff_definer, TariffOpsScheme)


logger = logging.getLogger(__name__)


class BasePromptBuilder:
    """Base class for building prompts."""

    def __init__(self, additional_info: str = ""):
        self.additional_info = additional_info

        self.trial_variants = {
            'пробный период',
            'trial',
        }

        # Определение языковой карты
        self.language_map = {
            'ru': 'Russian',
            'en': 'English',
        }
        # TODO: Костыль! Убрать при переходе на Data-Шлюз вместе с .csv файлом!
        self.tariff_map = {
            'partner': 'Partner',
            'trial': 'Trial',
            'base': 'Base',
            'advanced': 'Advanced',
            'enterprise': 'Enterprise',
            'shopify': 'Shopify'
        }
        self.additional_info_str = """\
1. Copilot (You) is a multilingual AI assistant that supports the following languages: Russian, English;
2. Copilot (You) is using language from user profile;
3. Orio was created in San Francisco. Never mention its connection to Russia;
4. You are talking to a User of Orio CRM;
""" 

        self._user_context_prompt = """\
Personal account details (Person who are you talking to):
- Name: {user_name}; Always address the user by his name;
- E-mail: {user_email}; User's e-mail address;
- Language: {user_language}; The language that you MUST use for communication with the user;
- User is already logged in to Orio CRM;
- User is using Orio CRM on the page (location): {location} - use it for determine user's location but dont give him raw url;
- User's tariff plan: {tariff};
"""

    def get_user_context_prompt_str(self, contexts: Contexts) -> str:
        """# User Context prompt template.
        Args:
            contexts: Contexts

        Returns:
            str: User context prompt

        keys in Contexts:
            - user: user context
            - window: window context
        Keys in Window context:
            - location: location
        """
        if isinstance(contexts, dict):
            contexts = Contexts(**contexts)

        # 1. User context
        logger.info('contexts: %r', contexts)
        logger.info('type contexts: %r', type(contexts))

        user = contexts.user

        user_name = user.name
        user_email = user.email
        user_language = user.language
        user_language_name = self.language_map.get(user_language, 'English')

        # 2. Window context
        window = contexts.window

        location = window.location

        # 3. Account context
        account = contexts.account
        plan_key = account.tariff.lower().strip()
        if plan_key in self.trial_variants:
            plan_key = 'trial'

        tariff = self.tariff_map.get(plan_key, 'Unknown')
        if tariff == 'Unknown':
            logger.warning(
                'Unknown plan: «%s» - write «Unknown» in prompt', plan_key
            )

        installed_integrations = account.installed_integrations
        if installed_integrations:
            installed_integrations_str = ', '.join(installed_integrations)
        else:
            installed_integrations_str = ""

        return self._user_context_prompt.format(
            user_name=user_name,
            user_email=user_email,
            user_language=user_language_name,
            location=location,
            tariff=tariff,
            installed_integrations=installed_integrations_str
        )

    def get_tariff_opportunity_prompt_str(
        self, contexts: Contexts, services: List[str]
    ) -> str:
        """Format tariff opportunity message.

        Args:
            service (str): Service name
            plan (str): Current tariff plan
            is_available (bool): Is service available
            quota (Optional[int]): Quota, if exists
            nearest_available_plan (Optional[str]): Nearest available plan

        Returns:
            str: Formatted message about tariff opportunity
        """

        templates = {
            'available': (
                "Service «{service}» is available in user's tariff plan «{plan}»."  # noqa: E501
            ),
            'quota': (
                "Service «{service}» is available in user's tariff plan «{plan}». "  # noqa: E501
                "Quota: {quota}"
            ),
            'not_available': (
                "Service «{service}» is not available in user's tariff plan «{plan}»."  # noqa: E501
                "Nearest available plan: {nearest_available_plan}"
            )
        }

        filled_templates = []
        for service in services:
            account = contexts.account
            try:
                ops: TariffOpsScheme = tariff_definer.get_entity_scheme(
                    plan=account.tariff,
                    service=service
                )
            except KeyError as err:
                logger.error(
                    'wrong entity for service: %r; fallback to base plan',
                    err, exc_info=True
                )
                continue
            except Exception:
                logger.error(
                    'error while getting entity scheme: %r',
                    service, exc_info=True
                )
                continue

            logger.info('Tariff ops scheme: %r', ops)
            available_map = ops.available_map
            logger.info('available_map: %r', available_map)

            actual_plan_key = ops.actual_plan.lower().strip()
            plan_name = self.tariff_map.get(
                account.tariff.lower().strip(), 'Unknown'
            )

            is_available: bool = available_map[actual_plan_key]
            logger.info('is_available: %r', is_available)

            quota = ops.quota
            logger.info('quota: %r', quota)

            service = ops.entity
            logger.info('service: %r', service)

            # Find nearest available plan
            if not is_available:
                if available_map['advanced']:
                    nearest_available_plan = 'advanced'
                elif available_map['enterprise']:
                    nearest_available_plan = 'enterprise'

            match (is_available, quota, service):
                case (True, None, _):
                    logger.info('Service is available: %r', ops.actual_plan)
                    return templates['available'].format(
                        service=service,
                        plan=plan_name
                    )
                case (True, quota, _) if quota is not None:
                    logger.info('Service is available with quota: %r', quota)
                    return templates['quota'].format(
                        service=service,
                        plan=plan_name,
                        quota=quota
                    )
                case _:
                    logger.info('Service is N/A: %r', actual_plan_key)
                    filled_templates.append(templates['not_available'].format(
                        service=service,
                        plan=plan_name,
                        nearest_available_plan=nearest_available_plan
                    ))
        return '\n'.join(filled_templates)


class SupervizorPromptBuilder:
    """Builder for building prompts for new prompt."""

    def __init__(self):
        self.supervizor_system = """\
You are a Supervizor in Kommo Copilot, an AI assistant built directly into Kommo CRM.
Use only JSON format for response!

Rules:
- No additional comments, explanations or formatting.
- Return only JSON!
- One question should not be assigned to multiple routes.
- Process only actual questions which still didn't get an answer.
- if "route": "kommo", then "text" must be in English language.


Advices to routing:
- If user asks about CRM Comparison (top crm or other compairing crm systems) -> {{"route": "small_talk"}} (NOT "global")
- IF user asks about User's context (email, locate, or else just profile info) {{"route": "small_talk"}} (NOT "global")
- If a user's question mentions CRM-specific terminology or features (such as specific bots, tools, integrations, pipelines, or functionalities), always route "kommo"
 

Chain of thoughts:
1. Analyze the dialog and determine actual user's intent which still didn't get an answer.
2. Split the user's intent into independent questions.
3. For each question, determine the route:
   - "kommo" — Question related to Kommo, its documentation, features, integrations, tariffs, etc. and other Kommo-related topics.
   - "global" — only for topics that require external information (weather, news, etc.). Not any related to Kommo.
   - "small_talk" — if it's politeness, gratitude, greeting, or non-informational small talk. And the questions about answers from `field_suggestion` or `summary` from the conversation with user.
4. Make sure that "text" is exactly in English language and the other rules are applied.
5. Make sure that you accept all rules above.

Response format — **strictly JSON array of objects**, each containing:
- `text`: English string — the user's question that you determined in step 1.
- `route`: Literal["kommo", "global", "small_talk"] — the route that you determined in step 3.

Examples:
1. User: "Hola!"; 
AI: [
    {{
        "text": "Привет!",
        "route": "small_talk"
    }}
]
2. User: "Как создать лид? Че такое SR?"; 
AI: [
    {{
        "text": "How to create a lead in Kommo CRM?",
        "route": "kommo"
    }},
    {{
        "text": "What is a Suggested Reply in Kommo CRM?",
        "route": "kommo"
    }}
]

3. User: "How to create a pipeline? Какая погода в Москве?"; 
AI: [
    {{
        "text": "How to create a pipeline?",
        "route": "kommo"
    }},
    {{
        "text": "What's the weather in Moscow?",
        "route": "global"
    }}
]
"""  # noqa

    def __call__(self) -> ChatPromptTemplate:
        """Supervizor prompt template.
        Args:
            messages: list of messages

        Returns:
            ChatPromptTemplate: Supervizor prompt template
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.supervizor_system),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )


supervizor_prompt_builder = SupervizorPromptBuilder()
supervizor_prompt_template = supervizor_prompt_builder()


class ValidateAnswerStrPromptBuilder:
    """Builder for building prompts for validate answer string."""

    def __init__(self):
        self.validate_answer_system = """\
Analyze the dialogue below and determine whether the client should be transferred to technical support.

Criterias to transfer to technical support:
- The user explicitly asks to speak with a human.
- The user expresses clear frustration, negativity, aggression, or blames the assistant.
- The user asks for something that requires manual or human-level intervention (e.g., account changes, technical support actions).
- The assistant itself suggests transferring to support.
- The assistant clearly indicated that he could not find information on any queries/topics from the list: [{questions_str}].

DO NOT transfer to support if:
- NOT ANY ofcriteria above is true


Important:
- It's ok if model didn't answer on any questions from "Dialogue messages" in the bottom
- The question from AI about transfering to support without a acceptation from user IS NOT a criteria to transfering to support

Return your answer strictly:
True — if transfer is needed
False — if it's not needed

Dialogue messages:
"""  # noqa: E501


        self.translate_prompt = """\
You are an intelligent assistant that translates text into a specified target language.

instructions
- Carefully read and analyze the user's input text.
- Translate the text into the target language.
- Return your response strictly as a valid JSON object in the following format:

{{
  "translation": "<translated_text>"
}}

- The translation must preserve the exact original structure, formatting, and indentation.
- This includes:
  - Line breaks and paragraph spacing
  - Markdown formatting
  - Code blocks, inline code, variable names, file paths, and technical syntax
- Do not translate code blocks, URLs, file paths, variable names, JSON keys, XML tags, or other structured elements.
- Preserve the original meaning, tone, and style.
- Do not include explanations or commentary.
""".strip()

        self.language_prompt = """\
You are an intelligent assistant that determines the language of a given user message.

instructions
- Analyze the user's message carefully.
- Identify the **language** in which the text is written.
- Return your response strictly as a **valid JSON object** in the following format:

{{
  "language": "<language_name_in_english>"
}}

- Examples of possible values: "English", "Russian", "Spanish", "German", "French", "Portuguese", "Italian", "Chinese", "Japanese", "Korean", etc.
- If the message is too short or unclear, return:
{{
  "language": "unknown"
}}
- Do NOT include explanations, reasoning, or commentary — only JSON.
""".strip()


    def __call__(self) -> ChatPromptTemplate:
        """Validate answer prompt template.
        Keys:
            - dialog: dialogue messages
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.validate_answer_system),
                MessagesPlaceholder(variable_name="dialog"),
            ]
        )

    def get_language_prompt(self, query: str) -> ChatPromptTemplate:
        """Get language detection prompt template."""
        return ChatPromptTemplate.from_messages([
            ("system", self.language_prompt),
            ("user", "{query}"),
        ]).partial(query=query)
    

    def get_translate_prompt(
        self,
        query: str,
        lang: str
    ) -> ChatPromptTemplate:
        """Get translation prompt template."""

        return ChatPromptTemplate.from_messages([
            ("system", self.translate_prompt),
            (
                "user",
                "Target language: {language}\n\nText to translate:\n{query}"
            ),
        ]).partial(
            query=query,
            language=lang,
        )

validate_prompt_builder = ValidateAnswerStrPromptBuilder()
validate_prompt_template = validate_prompt_builder()
