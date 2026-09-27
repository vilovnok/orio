# Метрики
from prometheus_client import Counter, Histogram, Gauge

from config.config import inited_config


def build_name(name: str) -> str:
    """
    Build a metric name with the configured prefix.
    """
    return f"{inited_config.metrics_config.prefix}_{name}"


REQUEST_COUNT = Counter(
    build_name("http_requests_total"),
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    build_name("http_request_duration_seconds"),
    "Request latency",
    ["endpoint"],
)
SSE_CONNECTIONS_TOTAL = Counter(
    build_name("sse_connections_total"),
    "Total SSE connections opened",
)
SSE_CONNECTIONS_ACTIVE = Gauge(build_name("sse_connections_active"), "Active SSE connections")
SSE_MESSAGES_SENT_TOTAL = Counter(build_name("sse_messages_sent_total"), "Total SSE messages sent")
SSE_CONNECTION_DURATION = Histogram(
    build_name("sse_connection_duration_seconds"),
    "SSE connection duration in seconds"
)

