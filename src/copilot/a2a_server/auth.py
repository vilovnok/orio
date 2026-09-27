import logging

from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from jose.constants import ALGORITHMS
from starlette.authentication import (
    AuthCredentials,
    AuthenticationBackend,
    AuthenticationError,
    SimpleUser,
)

from starlette.requests import Request
from starlette.responses import JSONResponse
from src.utils.config import inited_config as config

logger = logging.getLogger(__name__)


# Кастомный backend
class JWTAuthBackend(AuthenticationBackend):
    """Custom authentication backend for A2A server using JWT."""
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = ALGORITHMS.HS256
    ):

        self.algorithm = algorithm
        resolved_secret_key = secret_key if secret_key is not None \
            else config.a2a.secret.get_secret_value()

        if not resolved_secret_key:
            raise ValueError("JWT secret key is not configured.")
        self.secret_key: str = resolved_secret_key

    async def authenticate(self, conn):
        logger.debug('start auth')
        auth_header = conn.headers.get("Authorization")
        if not self.secret_key:
            logger.error('ENV VAR A2A_SECRET_KEY is not set')
            raise AuthenticationError("ENV VAR A2A_SECRET_KEY is not set")

        if not auth_header:
            raise AuthenticationError("Authorization header is missing")

        try:
            scheme, token = auth_header.split()
            if scheme.lower() != "bearer":
                msg = "Invalid authentication scheme. Use Bearer."
                raise AuthenticationError(msg)
            logger.debug('auth header is valid')
        except ValueError:
            msg_part1 = "Invalid Authorization header format."
            msg_part2 = "Expected 'Bearer <token>'."
            raise AuthenticationError(f"{msg_part1} {msg_part2}")

        if not token:
            raise AuthenticationError("Token is missing after Bearer scheme")

        try:
            payload = jwt.decode(
                token, self.secret_key, algorithms=[self.algorithm]
            )
            service = payload.get("service")
            if not service:
                msg = "Token payload missing 'service'"
                raise AuthenticationError(msg)

            logger.debug('service: %s', service)
            return AuthCredentials(["authenticated"]), SimpleUser(service)
        except JWTError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
        except Exception:
            msg = "Authentication failed due to a server error."
            raise AuthenticationError(msg)

    def create_jwt_token(
        self,
        user_id: int,
        account_id: int,
        service: str,
        exp: Optional[datetime] = None
    ) -> str:
        payload = {
            "user_id": user_id,
            "account_id": account_id,
            "service": service,
            "exp": exp or datetime.utcnow() + timedelta(hours=1)
        }
        return jwt.encode(
            payload,
            self.secret_key,
            algorithm=self.algorithm
        )


# Обработчик ошибок для middleware
def on_auth_error(request: Request, exc: AuthenticationError):
    return JSONResponse(
        {"error": "AuthenticationFailed", "detail": str(exc)}, status_code=401
    )
