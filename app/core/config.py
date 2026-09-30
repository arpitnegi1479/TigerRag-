from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="agentic-graphrag-backend")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    postgres_dsn: str = Field(default="")
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = None
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="")
    graph_backend: str = Field(default="memory")

    max_file_size_mb: int = Field(default=25)
    allowed_file_types: str = Field(default=".pdf,.txt,.md,.docx,.html")
    rate_limit_requests_per_minute: int = Field(default=120, ge=1)
    rate_limit_window_seconds: int = Field(default=60, ge=1)

    gemini_api_key: str | None = None
    gemini_model: str = Field(default="gemini-2.5-flash")
    embedding_provider: str = Field(default="auto")
    embedding_model: str = Field(default="gemini-embedding-001")
    embedding_dimension: int = Field(default=768)

    max_agent_steps: int = Field(default=8)
    max_tool_calls: int = Field(default=20)
    max_graph_depth: int = Field(default=3)
    default_chunk_size: int = Field(default=800)
    default_chunk_overlap: int = Field(default=120)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
