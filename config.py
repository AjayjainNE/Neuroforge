"""
NeuroForge Configuration
Loads and validates all environment settings.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class LLMProvider(str, Enum):
    NVIDIA = "nvidia"
    MISTRAL = "mistral"
    ANTHROPIC = "anthropic"


class ExpertiseLevel(str, Enum):
    SKETCH = "sketch"        # Level 1 — Quick architectural sketch
    DETAILED = "detailed"    # Level 2 — Full design + rationale
    COMPLETE = "complete"    # Level 3 — Implementation + optimizations
    RESEARCH = "research"    # Level 4 — Novel combinations + citations


class Settings(BaseSettings):
    # ── Provider ────────────────────────────────────────────────
    LLM_PROVIDER: LLMProvider = LLMProvider.NVIDIA

    # ── NVIDIA NIM ──────────────────────────────────────────────
    NVIDIA_API_KEY: str = ""
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    NVIDIA_PRIMARY_MODEL: str = "meta/llama-3.1-70b-instruct"
    NVIDIA_CODE_MODEL: str = "meta/llama-3.1-70b-instruct"

    # ── Mistral ─────────────────────────────────────────────────
    MISTRAL_API_KEY: str = ""
    MISTRAL_BASE_URL: str = "https://api.mistral.ai/v1"
    MISTRAL_PRIMARY_MODEL: str = "mistral-large-latest"
    MISTRAL_CODE_MODEL: str = "codestral-latest"

    # ── App ─────────────────────────────────────────────────────
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Agent Memory ────────────────────────────────────────────
    MAX_CONTEXT_TOKENS: int = 12000
    SESSION_TTL_HOURS: int = 24
    ENABLE_SELF_CRITIQUE: bool = True
    ENABLE_ANTI_PATTERN_CHECK: bool = True

    # ── Architecture DNA ────────────────────────────────────────
    ENABLE_NOVELTY_DETECTION: bool = True
    ENABLE_COMPUTE_ESTIMATION: bool = True
    MAX_CANDIDATE_ARCHITECTURES: int = 3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def active_api_key(self) -> str:
        if self.LLM_PROVIDER == LLMProvider.NVIDIA:
            return self.NVIDIA_API_KEY
        return self.MISTRAL_API_KEY

    @property
    def active_base_url(self) -> str:
        if self.LLM_PROVIDER == LLMProvider.NVIDIA:
            return self.NVIDIA_BASE_URL
        return self.MISTRAL_BASE_URL

    @property
    def primary_model(self) -> str:
        if self.LLM_PROVIDER == LLMProvider.NVIDIA:
            return self.NVIDIA_PRIMARY_MODEL
        return self.MISTRAL_PRIMARY_MODEL

    @property
    def code_model(self) -> str:
        if self.LLM_PROVIDER == LLMProvider.NVIDIA:
            return self.NVIDIA_CODE_MODEL
        return self.MISTRAL_CODE_MODEL

    def validate_provider(self) -> None:
        if self.LLM_PROVIDER == LLMProvider.NVIDIA and not self.NVIDIA_API_KEY:
            raise ValueError(
                "NVIDIA_API_KEY is required. Get a free key at https://build.nvidia.com/"
            )
        if self.LLM_PROVIDER == LLMProvider.MISTRAL and not self.MISTRAL_API_KEY:
            raise ValueError(
                "MISTRAL_API_KEY is required. Get a free key at https://console.mistral.ai/"
            )
        # anthropic provider uses ambient credentials — no explicit key needed


settings = Settings()
