"""Configurações da aplicação, lidas de variáveis de ambiente."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    # Modelo rápido e barato; troque via variável de ambiente se necessário.
    model_name: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # Tradução deve ser o mais determinística possível.
    temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    # Timeout (s) e tentativas extras em caso de erro transitório (429/5xx).
    timeout: float = float(os.getenv("OPENAI_TIMEOUT", "30"))
    max_retries: int = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
    max_input_chars: int = int(os.getenv("MAX_INPUT_CHARS", "5000"))
    # Cache em memória: textos idênticos não geram nova chamada à OpenAI.
    cache_enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"


settings = Settings()
