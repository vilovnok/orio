from a2a.types import AgentSkill


skills = [
    AgentSkill(
        id='kommo_agent',
        name='Kommo CRM Questions',
        description=(
            'Handles questions related to Kommo CRM, its documentation, '
            'features, integrations, tariffs, and other Kommo-related '
            'topics. Includes questions mentioning CRM-specific terminology '
            'or features.'
        ),
        tags=['kommo', 'crm', 'features', 'documentation', 'support'],
        examples=[
            "How to create a lead in Kommo CRM?",
            "What is a Suggested Reply in Kommo CRM?",
            "How to create a pipeline?",
            "Integration with WhatsApp"
        ],
    ),
    AgentSkill(
        id='global_agent',
        name='General Knowledge Questions',
        description=(
            'Handles questions requiring external information (weather, '
            'news, etc.), not related to Kommo.'
        ),
        tags=['general knowledge', 'external information', 'weather', 'news'],
        examples=[
            "What's the weather in Moscow?",
            "Latest news",
            "Capital of France"
        ],
    ),
    AgentSkill(
        id='small_talk_agent',
        name='Small Talk, Contextual, and Comparative Queries',
        description=(
            'Handles requests routed to "small_talk": politeness, gratitude, '
            'greetings, non-informational small talk. Also handles questions '
            'about CRM comparison and user context (e.g., email, location) '
            'according to routing rules.'
        ),
        tags=[
            'small talk', 'greeting', 'politeness', 'crm comparison',
            'user context', 'non-informational queries'
        ],
        examples=[
            "Hi!",
            "Thanks",
            "How are you?",
            "Which CRM is better, Kommo or X?",
            "What is my email in the system?",
            "Compare Kommo and AmoCRM"
        ],
    )
]
