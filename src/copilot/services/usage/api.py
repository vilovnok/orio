from typing import Any, Dict, Optional

import logging
import aiohttp

from datetime import datetime, timedelta

from jose import jwt

from src.copilot.services.usage.schemes import (
    Submitter,
    Tokens,
    UsageItem,
    UsageTrackingRequest,
    Metrics
)
from src.utils.config import inited_config as config

logger = logging.getLogger(__name__)


class JWTGenerator:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def generate_jwt(self, token_payload: Dict[str, Any]):
        expiration_time = (
            datetime.now() + timedelta(minutes=5)
        )
        token_payload = {
            "client": "ai-copilot-multiagent",
            "exp": int(expiration_time.timestamp())
        }

        token = jwt.encode(
            token_payload,
            key=self.secret_key,
            algorithm="HS256"
        )
        return token

    def decode_jwt_token(self, token: str):
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"]
            )
            return payload
        except jwt.JWTError as e:
            raise ValueError("Invalid token") from e


jwt_generator = JWTGenerator(
    secret_key=config.billing.secret_key.get_secret_value()
)


async def send_usage_tracking(
    usage_by_model: Dict[str, Dict[str, int]],
    account_id: int,
    user_id: int,
    group: int = 0,
    request_time: float = 0.0,
    processing_time: Optional[float] = 0.0
) -> Dict[str, Any]:
    """
    Отправляет данные об использовании моделей на API для трекинга.

    Args:
        usage_by_model: Словарь с использованием токенов по моделям
        account_id: ID аккаунта
        user_id: ID пользователя
        group: Группа (бит-флаг)
        used_by: Описание функционала
        api_url: URL API для отправки данных
        request_time: Время запроса в секундах
        processing_time: Время обработки в секундах

    Returns:
        Dictionary с ответом API"""
    usages = []

    submitter = Submitter(
        account_id=account_id,
        user_id=user_id,
        group=group
    )
    if not group:
        logger.warning("Sending usages track with group = 0")

    metrics = Metrics(
        request_time=request_time,
        processing_time=processing_time
    )

    for model_name, tokens_data in usage_by_model.items():
        # Получаем токены из данных
        tokens = Tokens(
            prompt_tokens=tokens_data.get("prompt_tokens", 0),
            completion_tokens=tokens_data.get("completion_tokens", 0),
            cached_tokens=0
        )

        model_name = model_name.replace("_", "-")

        # Создаем запись по конкретной модели
        if not config.billing.copilot_service_name:
            config.billing.copilot_service_name = 'ai-copilot-multiagent-small-talk'

        usage_item = UsageItem(
            submitter=submitter,
            model=model_name,
            used_by=config.billing.copilot_service_name,
            tokens=tokens,
            metrics=metrics
        )

        usages.append(usage_item)

    # Создаем полный запрос
    request_data = UsageTrackingRequest(usages=usages)
    url = f'{config.billing.base_url}/api/v1/billing/usages/track'

    # Логи
    logger.info('Sending usage tracking to %s', url)
    logger.info('Request data: %s', request_data.model_dump())
    logger.debug('Request data keys: %s', request_data.model_dump().keys())

    # Генерируем и логируем JWT для диагностики
    jwt_token = jwt_generator.generate_jwt(
        token_payload={
            "client": config.billing.copilot_service_name,
            "exp": int(datetime.now().timestamp())
        }
    )

    logger.debug(
        'JWT payload used: client=%s',
        config.billing.copilot_service_name,
    )
    logger.debug(
        'JWT token first 10 chars: %s...',
        jwt_token[:10] if jwt_token else 'None'
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=url,
                json=request_data.model_dump(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {jwt_token}"
                }
            ) as response:
                response_data = await response.json()
                if response_data['status'] == 'success':
                    logger.info('Usage tracking sent successfully')
                    return response_data

                elif response.status >= 400:
                    logger.error('Status code: %s', response.status)
                    logger.error("Error tracking usage: %s", response_data)
                    error_message = (
                        response_data.get('message')
                        or response_data.get('detail', 'Unknown error')
                    )
                    return {"error": error_message}

                elif response.status >= 500:
                    logger.error(
                        'Status code: %s; Response: %s',
                        response.status,
                        response_data
                    )
                    error_message = (
                        response_data.get('message')
                        or response_data.get('detail', 'Unknown error')
                    )
                    return {"error": error_message}
                else:
                    logger.warning(
                        'Status code not success: %s', response.status
                    )
                    logger.warning('Response data: %s', response_data)

                return response_data
    except Exception:
        logger.error(
            "Failed to send usage tracking",
            exc_info=True
        )
        return {'error': 'failed to send usage tracking'}
