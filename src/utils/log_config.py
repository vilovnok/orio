import datetime
import json
import logging

from contextlib import contextmanager
from contextvars import ContextVar
from logging import LogRecord
from typing import Literal, cast, Optional

import graypy
from environs import Env
from pydantic import BaseModel, Field


class LogConfigScheme(BaseModel):
    log_format: Literal['json', 'plaintext'] = Field(
        default='json',
        description='Log format; Literal["json", "plaintext"]'
    )
    log_level: str = Field(
        default='INFO',
        description='Level of logging'
    )
    graylog_hostname: Optional[str] = Field(description='Graylog host', default=None)
    graylog_port: Optional[int] = Field(description='Graylog port', default=None)
    graylog_log_level: Optional[str] = Field(default='INFO', description='Graylog log level')
    graylog_stream: Optional[str] = Field(description='Graylog stream', default='ai-copilot-multiagent-small-talk')


def load_log_config() -> LogConfigScheme:
    env = Env()
    env.read_env()

    log_format = cast(
        Literal['json', 'plaintext'],
        env.str('LOG_FORMAT', default='json')
    )

    return LogConfigScheme(
        log_format=log_format,
        log_level=env.str('LOG_LEVEL', default='INFO'),
        graylog_hostname=env.str('GRAYLOG_HOSTNAME', default=None),
        graylog_port=env.int('GRAYLOG_PORT', default=None),
        graylog_log_level=env.str('GRAYLOG_LOG_LEVEL', default='INFO'),
        graylog_stream=env.str('GRAYLOG_STREAM', default='ai-copilot-multiagent-small-talk')
    )


config = load_log_config()

# Глобальная переменная контекста
log_context: ContextVar[dict] = ContextVar("log_context", default={})


@contextmanager
def push_log_context(**fields):
    """Устанавливает значения в `log_context`
    и гарантированно сбрасывает их в finally.
    Использовать только в верхнем уровне обработки задачи"""

    token = log_context.set(fields)
    try:
        yield
    finally:
        log_context.reset(token)


class ContextFilter(logging.Filter):
    """
    Фильтр для добавления контекстных данных из contextvars в лог-запись.
    """
    def __init__(self, custom_fields=None):
        super().__init__()
        self.custom_fields = custom_fields or [
            "job_id",
            "request_id",
            "chat_id",
            "account_id",
            "user_id",
            "thread_id"
        ]

    def filter(self, record: LogRecord):
        context = log_context.get()
        # Для каждого указанного поля пытаемся установить значение из контекста
        for key in self.custom_fields:
            setattr(record, key, context.get(key, ""))
        return True


class GelfCustomFilter(logging.Filter):
    def __init__(self, graylog_stream):
        super().__init__()
        self.graylog_stream = graylog_stream

    def filter(self, record):
        record.stream = self.graylog_stream
        extra = record.__dict__.get("extra", {})
        for key in extra:
            record.__dict__[key] = extra[key]
        return True


# Класс форматирования логов
class CustomFormatter(logging.Formatter):
    """Custom formatter for logs"""

    # ЦВЕТОКОДЫ (ANSI-escape codes)
    GREEN = "\x1b[32;20m"
    GREY = "\x1b[38;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    # СТРУКТУРА СООБЩЕНИЯ
    FORMAT = "[%(asctime)s] %(levelname)-7s - %(module)10s:%(lineno)3d|%(funcName)-24s - %(message)s" # noqa
    INFO_FORMAT = "\x1b[34;20m[%(asctime)s] %(levelname)-7s - %(module)10s:%(lineno)3d|%(funcName)-24s -\x1b[0m %(message)s"  # noqa

    # СОБРАННЫЙ ФОРМАТ (ЦВЕТОКОДЫ + СТРУКТУРА СООБЩЕНИЯ)
    FORMATS = {
        logging.DEBUG: GREEN + FORMAT + RESET,
        logging.INFO: INFO_FORMAT,
        logging.WARNING: YELLOW + FORMAT + RESET,
        logging.ERROR: RED + FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + FORMAT + RESET
    }

    def format(self, record: LogRecord):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Гибридный форматтер, совмещающий читаемые логи и JSON формат
class HybridFormatter(logging.Formatter):
    """Hybrid formatter"""

    # ЦВЕТОКОДЫ (ANSI-escape codes)
    GREEN = "\x1b[32;20m"
    GREY = "\x1b[38;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    BLUE = "\x1b[34;20m"
    RESET = "\x1b[0m"

    # СТРУКТУРА СООБЩЕНИЯ
    FORMAT = "[%(asctime)s] %(levelname)-7s - %(module)10s:%(lineno)3d|%(funcName)-24s - %(message)s" # noqa
    INFO_FORMAT = BLUE + "[%(asctime)s] %(levelname)-7s - %(module)10s:%(lineno)3d|%(funcName)-24s -" + RESET + " %(message)s"  # noqa

    # СОБРАННЫЙ ФОРМАТ (ЦВЕТОКОДЫ + СТРУКТУРА СООБЩЕНИЯ)
    FORMATS = {
        logging.DEBUG: GREEN + FORMAT + RESET,
        logging.INFO: INFO_FORMAT,
        logging.WARNING: YELLOW + FORMAT + RESET,
        logging.ERROR: RED + FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + FORMAT + RESET
    }

    def __init__(
        self,
        log_format: str = config.log_format,
        custom_fields=None,
        **kwargs
    ):
        """Initializes HybridFormatter

        Args:
            use_json (bool): Use JSON format
            custom_fields (list): List of custom fields in JSON
            **kwargs: Additional arguments to pass to the parent class"""

        super().__init__(**kwargs)
        self.log_format = log_format
        # Задаем полный список обязательных полей
        default_fields = [
            "job_id",
            "request_id",
            "chat_id",
            "account_id",
            "user_id",
            "thread_id"
        ]
        self.custom_fields = custom_fields or default_fields

    def formatTime(self, record, datefmt=None):
        ct = datetime.datetime.utcfromtimestamp(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
            s = s.replace('%f', f"{ct.microsecond:06d}")
        else:
            s = ct.isoformat(timespec='microseconds') + 'Z'
        return s

    def format(self, record: LogRecord):
        if self.log_format == 'json':
            formatted = {
                "timestamp": self.formatTime(record, self.datefmt),
                "level": record.levelname.lower(),
                "message": record.getMessage(),
                "module": record.module,
                "line": record.lineno,
                "function": record.funcName,
                "thread": record.thread,
                "stream": getattr(record, "stream", config.graylog_stream),
            }

            # Добавляем все пользовательские поля контекста в JSON
            for key in self.custom_fields:
                if hasattr(record, key):
                    formatted[key] = getattr(record, key)
                else:
                    formatted[key] = ""

            return json.dumps(formatted, ensure_ascii=False)

        else:
            # Человекочитаемый формат с цветами
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)


class LogConfig:
    """Class for logging configuration
    setup_logging() - method for setting up logging"""

    # Словарь с уровнями логирования по умолчанию для каждого логгера
    LOG_LEVEL = config.log_level.upper()

    DEFAULT_LOGGER_LEVELS = {
        "langchain": LOG_LEVEL,
        "langgraph": LOG_LEVEL,
        "langsmith": LOG_LEVEL,
        "openai": 'DEBUG',
        "httpx": 'DEBUG',
        "urllib3": 'DEBUG',
        "workers": LOG_LEVEL,
        "bot": LOG_LEVEL,
        "app": LOG_LEVEL,
        "root": LOG_LEVEL,
    }

    @staticmethod
    def setup_logging(
        log_format: Literal['json', 'plaintext'] = config.log_format,
        custom_fields=None,
        logger_levels=None
    ):
        """Setup logging.

        Args:
            use_json (bool): If True, logs are output in JSON format.
            custom_fields (list): List of additional fields for logs
            logger_levels (dict): Logging level settings for specific loggers.
        """
        default_level = config.log_level

        levels = LogConfig.DEFAULT_LOGGER_LEVELS.copy()
        if logger_levels:
            levels.update(logger_levels)

        root_level = levels.get("root", default_level)
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, root_level))

        sh = logging.StreamHandler()
        sh.setLevel(getattr(logging, root_level))

        # Набор полей по умолчанию расширен
        default_custom_fields = [
            "job_id",
            "request_id",
            "chat_id",
            "account_id",
            "user_id",
            "thread_id",
        ]

        formatter = HybridFormatter(
            log_format=log_format,
            custom_fields=custom_fields or default_custom_fields,
            datefmt='%Y-%m-%dT%H:%M:%S.%fZ'
        )
        sh.setFormatter(formatter)

        # Добавляем в обработчик фильтр, который инжектит кастомные поля
        # из contextvars.
        sh.addFilter(
            ContextFilter(
                custom_fields=custom_fields or default_custom_fields
            )
        )

        # Добавляем обработчик в корневой логгер.
        root_logger.addHandler(sh)

        if config.graylog_hostname and config.graylog_port:
            # Добавляем грейлог обработчик
            gelf_handler = graypy.GELFUDPHandler(config.graylog_hostname, config.graylog_port)
            gelf_handler.setLevel(config.graylog_log_level)
            gelf_handler.addFilter(GelfCustomFilter(config.graylog_stream))
            gelf_handler.addFilter(
                ContextFilter(
                    custom_fields=custom_fields or default_custom_fields
                )
            )
            root_logger.addHandler(gelf_handler)

        loggers = [
            "langchain", "langgraph", "langsmith",
            "openai", "httpx", "urllib3",
            "workers", "bot", "app"
        ]
        configured_loggers = []
        for logger_name in loggers:
            logger_level = levels.get(logger_name, default_level)
            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, logger_level))
            logger.propagate = True

            if not logger.handlers:
                logger.addHandler(sh)

            configured_loggers.append(f"{logger_name}:{logger_level}")

        # Получаем логгер текущего модуля для логирования информации.
        logger = logging.getLogger(__name__)
        format_type = log_format.upper()
        logger.info(
            "Logging setup completed. Root level: %s, format: %s",
            root_level, format_type
        )
        logger.debug("Configured loggers: %s", ", ".join(configured_loggers))
        return logger


# Пример использования с пользовательскими полями и JSON форматом
def example_usage():
    """Пример использования логгера с пользовательскими полями и уровнями"""
    # Пример настройки уровней логирования для разных компонентов
    custom_levels = {
        "openai": "DEBUG",       # Подробные логи для OpenAI
        "langchain": "ERROR",    # Только ошибки от LangChain
        "httpx": "WARNING",      # Предупреждения от HTTP клиента
        "app": "INFO",           # Инфо от приложения
        "root": "INFO",          # Базовый уровень
        "langsmith": "ERROR",
    }

    # Инициализируем логгер с PLAINTEXT форматом, доп. полями и уровнями
    logger = LogConfig.setup_logging(
        log_format='plaintext',
        custom_fields=["request_id", "user_id", "operation"],
        logger_levels=custom_levels
    )

    # Пример прямого использования с extra
    logger.info(
        "Событие произошло",
        extra={
            "request_id": "req-67890",
            "user_id": "user-12345",
            "operation": "authorization"
        }
    )

    # Пример логов с разных компонентов (они будут иметь разные уровни)
    openai_logger = logging.getLogger("openai")    # LEVEL: DEBUG
    openai_logger.debug("Отправка запроса к API")  # Будет залогировано

    langchain_logger = logging.getLogger("langchain")  # LEVEL: ERROR
    langchain_logger.warning("Предупреждение")         # НЕ будет залогировано
    langchain_logger.error("Критическая ошибка")       # Будет залогировано

    return logger
