from pydantic import BaseModel, Field, field_validator, SecretStr



class LLMConfig(BaseModel):
    """LLM provider configuration."""

    model: str
    api_key: str
    provider: str
    base_url: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    enable_thinking: bool = False


    @field_validator("base_url")
    @classmethod
    def api_base_must_be_url(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith(("http://", "https://")):
            raise ValueError("base_url must be a valid URL")
        return v


class LLMProvidersConfig(BaseModel):
    openai: LLMConfig
    gigachat: LLMConfig


class LLMConfigScheme(BaseModel):
    providers: LLMProvidersConfig


class A2AScheme(BaseModel):
    host: str = Field(description='Host for A2A agent')
    port: int = Field(description='Port for A2A agent')
    secret: SecretStr = Field(description='Secret key for A2A agent')


class ModelsConfigScheme(BaseModel):
    orio_chain: str = Field(description='orio openai model')
    gpt_search_preview: str = Field(
        description='gpt_search_preview openai model'
    )
    validate_answers: str = Field(
        description='validate_answers openai model'
    )
    small_talk: str = Field(
        description='small_talk openai model'
    )


class CopilotConfigScheme(BaseModel):
    max_loop_count: int = Field(
        description='Maximum number of loops',
        default=12
    )
    history_max_tokens: int = Field(
        description='Maximum number of tokens in history',
        default=1500
    )
    models: ModelsConfigScheme = Field(
        description='Models config'
    )


class ElasticsearchConfigScheme(BaseModel):
    host: str = Field(description='Elasticsearch host')
    port: int = Field(description='Elasticsearch port', default=9200)
    account_id: int = Field(
        description='Knowledge base account ID'
    )
    index: str = Field(
        description='Elasticsearch index',
        default='articles_with_embeddings_en'
    )


class BeanstalkdScheme(BaseModel):
    host: str = Field(
        description='host for Beanstalkd',
        default='beanstalkd'
    )
    port: int = Field(description='port for Beanstalkd', default=11300)
    jobs_queue: str = Field(description='queue for jobs')
    results_queue: str = Field(description='queue for results')


class BillingScheme(BaseModel):
    base_url: str = Field(description='Base URL for billing')
    secret_key: SecretStr = Field(description='Secret key for billing')
    copilot_service_name: str = Field(description='Service name for copilot')


class WebSearchScheme(BaseModel):
    host: str = Field(description='Web search host')
    port: int = Field(description='Port for web search', default=8080)


class LocalesConfig(BaseModel):
    locales_dir: str = Field(
        description='directory for locales files',
        default='./src/langs/locales'
    )


class MetricsConfig(BaseModel):
    prefix: str = Field(description="Metrics prefix")
