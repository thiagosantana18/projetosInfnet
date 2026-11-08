"""Configurações da aplicação, lidas de variáveis de ambiente."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("MODEL_NAME", "Helsinki-NLP/opus-mt-en-de")
    # "cpu", "cuda" ou "cuda:0".
    device: str = os.getenv("DEVICE", "cpu")
    max_input_chars: int = int(os.getenv("MAX_INPUT_CHARS", "10000"))
    # O modelo Marian suporta até 512 tokens; deixamos margem de segurança.
    max_tokens_per_chunk: int = int(os.getenv("MAX_TOKENS_PER_CHUNK", "400"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "8"))


settings = Settings()
