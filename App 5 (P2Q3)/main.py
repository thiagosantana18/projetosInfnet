"""Aplicação FastAPI: tradução EN→DE com LangChain + HuggingFace."""
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from langchain_core.runnables import Runnable

from app.config import settings
from app.schemas import TranslationRequest, TranslationResponse
from app.translator import build_translation_chain

logger = logging.getLogger("translator-api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo e monta a chain uma única vez, na inicialização."""
    logger.info("Carregando modelo '%s'...", settings.model_name)
    app.state.chain = build_translation_chain(settings)
    logger.info("Modelo carregado.")
    yield


app = FastAPI(
    title="EN→DE Translation API (LangChain + HuggingFace)",
    description="Traduz textos do inglês para o alemão com Helsinki-NLP/opus-mt-en-de via LangChain.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_chain(request: Request) -> Runnable:
    """Dependência que fornece a chain (fácil de substituir em testes)."""
    return request.app.state.chain


@app.get("/health", tags=["infra"])
def health() -> dict:
    return {"status": "ok", "model": settings.model_name}


@app.post("/translate", response_model=TranslationResponse, tags=["translation"])
def translate(
    payload: TranslationRequest,
    chain: Runnable = Depends(get_chain),
) -> TranslationResponse:
    """Recebe um texto em inglês e retorna a tradução em alemão (JSON).

    Declarado como `def` (síncrono) de propósito: o FastAPI o executa em uma
    thread separada, evitando bloquear o event loop durante a inferência.
    """
    try:
        translated = chain.invoke(payload.text)
    except Exception:
        logger.exception("Falha ao traduzir o texto")
        raise HTTPException(status_code=500, detail="Erro ao traduzir o texto.")

    return TranslationResponse(
        source_text=payload.text,
        translated_text=translated,
        model=settings.model_name,
    )
