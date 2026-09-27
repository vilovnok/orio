import logging
import asyncio

from pystalkd.Beanstalkd import (
    Connection,
    SocketError,
    DeadlineSoon,
    Job
)

INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 60

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Sets up and maintains a connection to beanstalkd."""
    def __init__(self, host: str, port: int, watch_tube: str, use_tube: str):
        self.host = host
        self.port = port
        self.watch_tube = watch_tube
        self.use_tube = use_tube
        self._client: Connection | None = None

    async def get_client(self) -> Connection:
        if self._client is not None:
            try:
                self._client.stats_tube(self.watch_tube)
                return self._client
            except Exception:
                logger.warning(
                    "Stale connection detected, reconnecting...",
                    exc_info=True
                )
                self._client = None

        backoff = INITIAL_BACKOFF
        while True:
            try:
                client = Connection(host=self.host, port=self.port)
                client.watch(self.watch_tube)
                client.use(self.use_tube)
                client.ignore('default')

                logger.info("🟢 Connected to beanstalkd: %s", client)
                try:
                    stats = client.stats_tube(self.watch_tube)
                    logger.info("Tube stats: %s", stats)
                except Exception:
                    logger.error("failed to get tube stats", exc_info=True)

                self._client = client
                return client

            except Exception:
                logger.error(
                    "error connecting to beanstalkd, retrying in %s seconds",
                    backoff, exc_info=True
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.618, MAX_BACKOFF)

    def reset(self) -> None:
        self._client = None


class JobFetcher:
    """Reserves jobs from the queue, handles network errors."""
    def __init__(
        self,
        connection_manager: ConnectionManager,
        reserve_timeout: int = 10
    ):
        self.conn_mgr = connection_manager
        self.reserve_timeout = reserve_timeout

    async def fetch(self) -> Job | None:
        client: Connection = await self.conn_mgr.get_client()
        try:
            job = client.reserve(timeout=self.reserve_timeout)
            return job

        except DeadlineSoon as e:
            logger.warning("DeadlineSoon, skipping: %s", e, exc_info=True)
            return None

        except SocketError as e:
            logger.error("SocketError: %s; reconnecting...", e, exc_info=True)
            self.conn_mgr.reset()
            return None

        except Exception as e:
            logger.error(
                "Error reserving job: %s; reconnecting...",
                e, exc_info=True
            )
            self.conn_mgr.reset()
            return None
