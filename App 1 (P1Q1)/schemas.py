"""Modelos de entrada e saída da API (validação via Pydantic)."""
from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Frase inicial que será continuada pelo modelo.",
        examples=["Once upon a time, in a distant galaxy"],
    )
    max_new_tokens: int = Field(
        50, ge=1, le=200, description="Quantidade máxima de tokens gerados."
    )
    temperature: float = Field(
        0.8, gt=0.0, le=2.0, description="Maior valor = texto mais aleatório."
    )
    top_k: int = Field(50, ge=1, le=200, description="Amostragem top-k.")
    top_p: float = Field(0.95, gt=0.0, le=1.0, description="Amostragem nucleus (top-p).")


class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str = Field(..., description="Continuação gerada (sem o prompt).")
    full_text: str = Field(..., description="Prompt + continuação.")
