from typing import Any, Optional

from langchain_gigachat import GigaChat
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

from src.utils.config import Config
from src.utils.config_schemes import LLMConfig
from src.copilot.provider.schemes import Message


class LLMProvider:
    SUPPORTED = ("gigachat", "openai")


    def __init__(
        self,
        model: str,
        api_key: str,
        provider: str = "gigachat",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        enable_thinking: bool = False,

        scope: str = "GIGACHAT_API_PERS",
        verify_ssl_certs: bool = False,
        **kwargs: Any,
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking

        self._llm = self._build_llm(scope, verify_ssl_certs, kwargs)


    def _build_llm(self, scope: str, verify_ssl_certs: bool, extra: dict[str, Any]):
        
        if self.provider == "gigachat":

            llm_kwargs: dict[str, Any] = {
                "credentials": self.api_key,
                "scope": scope,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "verify_ssl_certs": verify_ssl_certs,
                **extra,
            }
            if self.base_url:
                llm_kwargs["base_url"] = self.base_url
            return GigaChat(**llm_kwargs)

        if self.provider == "openai":
            
            llm_kwargs = {
                "api_key": self.api_key,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **extra,
            }

            if self.base_url:
                llm_kwargs["base_url"] = self.base_url

            llm_kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking,
                }
            }
            return ChatOpenAI(**llm_kwargs)

        raise ValueError(
            f"Unsupported provider: {self.provider!r}. Supported: {', '.join(self.SUPPORTED)}"
        )


    @classmethod
    def from_config(cls, config: "LLMConfig") -> "LLMProvider":
        """Create provider from LLMConfig."""
        return cls(
            provider=config.provider, 
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            enable_thinking=config.enable_thinking,
        )


    async def complete(self, messages, tools=None, **kwargs) -> AIMessage:
        llm = self._llm.bind_tools(tools) if tools else self._llm
        return await llm.ainvoke(messages, **kwargs)


    @staticmethod
    def text_of(message: AIMessage) -> str:
        content = message.content
        if isinstance(content, list):
            content = "".join(
                p if isinstance(p, str) else p.get("text", "") for p in content
            )
        return content or ""


    async def chat(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> str:        
        response = await self._llm.ainvoke(messages, **kwargs)
        content = response.content
        if isinstance(content, list): # на случай мультимодальных частей
            content = "".join(
                part if isinstance(part, str) else part.get("text", "")
                for part in content
            )

        return content or ""