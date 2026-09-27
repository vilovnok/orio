import uuid
import asyncio
from pprint import pprint
from httpx import AsyncClient

from a2a.client import A2AClient, A2ACardResolver
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
    SendMessageSuccessResponse,
    Message,
    Part,
    Role,
    DataPart
)
from a2a.client.errors import A2AClientHTTPError
import httpx

from src.copilot.a2a_server.schemas import CopilotPayload
from src.copilot.a2a_server.auth import JWTAuthBackend

from src.app.schemes.contexts import (
    Contexts,
    Question,
    AccountContext,
    UserContext,
    WindowContext,
    WindowArguments,
)


from src.utils.config import inited_config as config
from src.utils.log_config import LogConfig


logger = LogConfig.setup_logging()

contexts = Contexts(
    account=AccountContext(
        id=42,
        subdomain="testorio",
        tariff="enterprise",
        installed_integrations=[
            'WhatsApp Cloud-API',
            'Telegram',
            'Instagram'
        ]
    ),
    user=UserContext(
        id=789,
        language="ru",
        email="test@example.com",
        name="Тестовый Пользователь",
        group=1
    ),
    window=WindowContext(
        location="/clients/",
        arguments=WindowArguments(
            entity_id=456,
            entity_type="lead",
            entity_name="ООО Тестовая Компания"
        ),
    ),
    history=[],
)


request_model = CopilotPayload(
    contexts=contexts,
    question=Question(
        text="Кто является генеральным директором компании Сбера?"
    ),
)

# Преобразуем модель в словарь и выведем для отладки
data_dict = request_model.model_dump()
print("Отправляемые данные:")
pprint(data_dict)

message = Message(
    messageId=str(uuid.uuid4()),
    # taskId=str(uuid.uuid4()),
    role=Role.user,
    parts=[Part(root=DataPart(data=data_dict))]
)

request = SendMessageRequest(
    id=str(uuid.uuid4()),
    params=MessageSendParams(message=message),
)

agent_url = f'http://{config.a2a.host}:{config.a2a.port}'

auth_backend = JWTAuthBackend(
    secret_key=config.a2a.secret.get_secret_value(),
    algorithm="HS256"
)
token = auth_backend.create_jwt_token(
    user_id=789,
    account_id=42,
    service="ai-conversation"
)


async def main():
    async with AsyncClient(
        timeout=30,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
    ) as httpx_client:
        try:
            # client = await A2AClient.get_client_from_agent_card_url(
            #     httpx_client,
            #     agent_url
            # )
            resolver = A2ACardResolver(
                httpx_client,
                agent_url,
            )
            agent_card = await resolver.get_agent_card()
            client = A2AClient(
                httpx_client,
                agent_card=agent_card,
            )

            response = await client.send_message(request)

            if isinstance(response.root, SendMessageSuccessResponse):
                task = response.root.result
                for artifact in task.artifacts:
                    for part in artifact.parts:
                        data = part.root.data
                        print('\n\n')
                        pprint(data)
                        print('\n\n')
                        print('*'*100)
            else:
                raise RuntimeError(
                    f"Got error response: {response.root.model_dump(mode='json')}"
                )
        except A2AClientHTTPError as e:
            if e.__cause__ and isinstance(e.__cause__, httpx.HTTPStatusError):
                http_error = e.__cause__
                logger.error(
                    "Status code: %s; Response body: %s",
                    http_error.response.status_code,
                    http_error.response.text,
                    exc_info=e
                )

                try:
                    response_json = http_error.response.json()
                    logger.error(
                        "Детали из JSON: %s", response_json.get('detail')
                    )
                except ValueError:
                    logger.error(
                        "Тело ответа не является валидным JSON.",
                        exc_info=True
                    )
        except Exception as e:
            logger.error("Exception: %s", e, exc_info=e)


if __name__ == "__main__":
    asyncio.run(main())
