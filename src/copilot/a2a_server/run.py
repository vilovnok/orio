"""A2A Agent Entrypoint"""
import uvicorn
from starlette.applications import Starlette

# Импорты для аутентификации
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware

# A2A imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
)
from starlette.routing import Mount

from prometheus_client import make_asgi_app

# Local imports
from src.copilot.a2a_server.agent_executor import CopilotAgentExecutor
from src.copilot.a2a_server.auth import JWTAuthBackend, on_auth_error
from src.copilot.a2a_server.skills import skills

# Config imports
from src.utils.log_config import LogConfig
from src.utils.log_config import config as log_config
from src.utils.config import inited_config as config
from src.copilot.metrics.middleware import MetricsMiddleware

logger = LogConfig.setup_logging()



agent_url = f'http://{config.a2a.host}:{config.a2a.port}'
agent_card = AgentCard(
    name='conversation_agent',
    description=(
        'Conversation agent for consulting user about Kommo CRM.'
        'Agent can only write the text, not any other actions in the system.'
    ),
    url=agent_url,
    version='1.0.0',
    defaultInputModes=['text'],
    defaultOutputModes=['text'],
    capabilities=AgentCapabilities(streaming=True),
    skills=skills
)


# 1. Request Handler
request_handler = DefaultRequestHandler(
    agent_executor=CopilotAgentExecutor(),
    task_store=InMemoryTaskStore(),
)

# Настройка Middleware для аутентификации
auth_middleware = Middleware(
    AuthenticationMiddleware,
    backend=JWTAuthBackend(),
    on_error=on_auth_error
)

metrics_middleware = Middleware(MetricsMiddleware)

# 2. A2A Starlette Application
server_app_builder = A2AStarletteApplication(
    agent_card=agent_card, http_handler=request_handler
)


def main():
    logger.info('Starting A2A server')

    # Собираем Starlette приложение с middleware через метод build()
    a2a_app = server_app_builder.build(middleware=[auth_middleware, metrics_middleware])

    app = Starlette(routes=[
        Mount("/metrics", app=make_asgi_app(), name="metrics_app"),
        Mount("/", app=a2a_app, name="a2a_app"),
    ])

    uvicorn.run(
        app,  # Передаем собранное приложение
        host=config.a2a.host,
        port=config.a2a.port,
        log_level=log_config.log_level.lower()
    )


if __name__ == '__main__':
    main()
