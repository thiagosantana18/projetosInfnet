"""Modelos de entrada e saída da API (validação via Pydantic)."""
from pydantic import BaseModel, Field, field_validator

from app.config import settings


class TranslationRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=settings.max_input_chars,
        description="Texto em inglês a ser traduzido.",
        examples=["Hello, how are you today?"],
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("O texto não pode conter apenas espaços em branco.")
        return value


class TranslationResponse(BaseModel):
    source_text: str
    translated_text: str = Field(..., description="Texto traduzido para o alemão.")
    model: str = Field(..., description="Modelo da HuggingFace utilizado.")
