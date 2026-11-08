"""Modelos de entrada e saída da API (validação via Pydantic)."""
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Mensagem enviada ao chatbot.",
        examples=["O que é LangChain?"],
    )

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A mensagem não pode conter apenas espaços em branco.")
        return value


class ChatResponse(BaseModel):
    message: str
    response: str = Field(..., description="Resposta simulada pelo Fake LLM.")


class QuestionsResponse(BaseModel):
    questions: list[str]
