import time
from starlette.types import ASGIApp, Receive, Scope, Send

from src.metrics.metrics import SSE_CONNECTIONS_TOTAL, SSE_CONNECTIONS_ACTIVE, SSE_MESSAGES_SENT_TOTAL, \
    SSE_CONNECTION_DURATION, REQUEST_LATENCY, REQUEST_COUNT


class MetricsMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        path = scope["path"]
        state = {
            "status_code": None,
            "is_sse": False,
            "sse_open": False
        }
        start_time = time.time()

        async def send_wrapper(message: dict) -> None:
            # фиксируем статус
            if message["type"] == "http.response.start":
                state["status_code"] = str(message["status"])

                # Проверим — это SSE?
                headers = {k.decode().lower(): v.decode() for k, v in message.get("headers", [])}
                if headers["content-type"].startswith("text/event-stream"):
                    state["is_sse"] = True
                    state["sse_open"] = True
                    SSE_CONNECTIONS_TOTAL.inc()
                    SSE_CONNECTIONS_ACTIVE.inc()

            elif message["type"] == "http.response.body":
                if state["is_sse"]:
                    # Считаем сообщения (если есть содержимое)
                    if message.get("body"):
                        SSE_MESSAGES_SENT_TOTAL.inc()
                if not message.get("more_body", False):
                    # Завершение ответа
                    duration = time.time() - start_time

                    if state["is_sse"]:
                        SSE_CONNECTION_DURATION.observe(duration)
                        if state["sse_open"]:
                            SSE_CONNECTIONS_ACTIVE.dec()
                            state["sse_open"] = False  # чтобы не декнуть дважды
                    else:
                        # обычный HTTP запрос
                        REQUEST_LATENCY.labels(endpoint=path).observe(duration)
                        REQUEST_COUNT.labels(
                            method=method,
                            endpoint=path,
                            status=state["status_code"] or "unknown"
                        ).inc()

            await send(message)

        await self.app(scope, receive, send_wrapper)
