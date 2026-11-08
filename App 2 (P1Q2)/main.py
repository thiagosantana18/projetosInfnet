"""Aplicação FastAPI para tradução de inglês para francês."""
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from app.config import settings
from app.schemas import TranslationRequest, TranslationResponse
from app.service import Translator

logger = logging.getLogger("translator-api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo na inicialização (e não a cada requisição)."""
    logger.info("Carregando modelo '%s'...", settings.model_name)
    app.state.translator = Translator(
        settings.model_name,
        max_tokens_per_chunk=settings.max_tokens_per_chunk,
        batch_size=settings.batch_size,
    )
    logger.info("Modelo carregado.")
    yield


app = FastAPI(
    title="EN→FR Translation API",
    description="Traduz textos do inglês para o francês com Helsinki-NLP/opus-mt-en-fr.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_translator(request: Request) -> Translator:
    """Dependência que fornece o tradutor carregado (fácil de mockar em testes)."""
    return request.app.state.translator


@app.get("/health", tags=["infra"])
def health() -> dict:
    return {"status": "ok", "model": settings.model_name}


@app.post("/translate", response_model=TranslationResponse, tags=["translation"])
def translate(
    payload: TranslationRequest,
    translator: Translator = Depends(get_translator),
) -> TranslationResponse:
    """Recebe um texto em inglês e retorna a tradução em francês.

    Declarado como `def` (síncrono) de propósito: o FastAPI o executa em uma
    thread separada, evitando bloquear o event loop durante a inferência.
    """
    try:
        translated = translator.translate(payload.text)
    except Exception:
        logger.exception("Falha ao traduzir o texto")
        raise HTTPException(status_code=500, detail="Erro ao traduzir o texto.")

    return TranslationResponse(source_text=payload.text, translated_text=translated)
