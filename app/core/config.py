"""
Centralized configuration — all values come from environment / .env file.
Never hardcode secrets.
"""

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "Production RAG API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ── HuggingFace ───────────────────────────────────────────────────────────
    HUGGINGFACEHUB_API_TOKEN: str = Field(..., description="HuggingFace API token")

    # Embedding model (runs locally via HF sentence-transformers)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM via HuggingFace Inference API
    LLM_MODEL_ID: str = "HuggingFaceH4/zephyr-7b-beta"
    LLM_MAX_NEW_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.3
    LLM_TOP_P: float = 0.95

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "rag_documents"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    RETRIEVAL_K: int = 4                  # number of docs to retrieve
    RETRIEVAL_FETCH_K: int = 10           # fetch before MMR reranking
    RETRIEVAL_SEARCH_TYPE: str = "mmr"   # "similarity" | "mmr"

    # ── Text Splitting ────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS: int = 60        # requests per window
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v.upper()

    @field_validator("RETRIEVAL_SEARCH_TYPE")
    @classmethod
    def validate_search_type(cls, v: str) -> str:
        allowed = {"similarity", "mmr"}
        if v not in allowed:
            raise ValueError(f"RETRIEVAL_SEARCH_TYPE must be one of {allowed}")
        return v


@lru_cache
def get_settings() -> Settings:
    """Cached settings — instantiated once per process."""
    return Settings()


settings = get_settings()
