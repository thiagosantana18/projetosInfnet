"""Aplicação FastAPI para geração de texto com GPT-2."""
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from app.config import settings
from app.schemas import GenerationRequest, GenerationResponse
from app.service import TextGenerator

logger = logging.getLogger("gpt2-api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo na inicialização (e não a cada requisição)."""
    logger.info("Carregando modelo '%s'...", settings.model_name)
    app.state.generator = TextGenerator(settings.model_name)
    logger.info("Modelo carregado.")
    yield


app = FastAPI(
    title="GPT-2 Text Generation API",
    description="Gera continuações de texto com GPT-2 (HuggingFace Transformers).",
    version="1.0.0",
    lifespan=lifespan,
)


def get_generator(request: Request) -> TextGenerator:
    """Dependência que fornece o gerador carregado (fácil de mockar em testes)."""
    return request.app.state.generator


@app.get("/health", tags=["infra"])
def health() -> dict:
    return {"status": "ok", "model": settings.model_name}


@app.post("/generate", response_model=GenerationResponse, tags=["generation"])
def generate(
    payload: GenerationRequest,
    generator: TextGenerator = Depends(get_generator),
) -> GenerationResponse:
    """Recebe uma frase e retorna a continuação gerada pelo GPT-2.

    Declarado como `def` (síncrono) de propósito: o FastAPI o executa em uma
    thread separada, evitando bloquear o event loop durante a inferência.
    """
    try:
        generated = generator.generate(
            prompt=payload.prompt,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_k=payload.top_k,
            top_p=payload.top_p,
        )
    except Exception:
        logger.exception("Falha ao gerar texto")
        raise HTTPException(status_code=500, detail="Erro ao gerar o texto.")

    return GenerationResponse(
        prompt=payload.prompt,
        generated_text=generated,
        full_text=payload.prompt + generated,
    )
