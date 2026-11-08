"""Aplicação FastAPI: tradução EN→FR com LangChain + OpenAI."""
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.runnables import Runnable

from app.config import settings
from app.schemas import TranslationRequest, TranslationResponse
from app.translator import TranslationError, build_translation_chain, translate

logger = logging.getLogger("translator-api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Valida a configuração e cria a chain uma única vez (cliente HTTP reutilizado)."""
    if not settings.openai_api_key:
        raise RuntimeError("Defina a variável de ambiente OPENAI_API_KEY.")
    app.state.chain = build_translation_chain(settings)
    logger.info("Chain pronta (modelo: %s)", settings.model_name)
    yield


app = FastAPI(
    title="EN→FR Translation API (LangChain + OpenAI)",
    description="Traduz textos do inglês para o francês usando um modelo da OpenAI via LangChain.",
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
async def translate_text(
    payload: TranslationRequest,
    chain: Runnable = Depends(get_chain),
) -> TranslationResponse:
    """Recebe um texto em inglês e retorna a tradução em francês (JSON)."""
    try:
        translated = await translate(chain, payload.text)
    except TranslationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    return TranslationResponse(
        source_text=payload.text,
        translated_text=translated,
        model=settings.model_name,
    )


@app.post("/translate/stream", tags=["translation"])
async def translate_stream(
    payload: TranslationRequest,
    chain: Runnable = Depends(get_chain),
) -> StreamingResponse:
    """Versão com streaming: devolve a tradução em texto puro conforme é gerada.

    Reduz a latência percebida em textos longos (o primeiro trecho chega rápido).
    """

    async def token_stream():
        try:
            async for chunk in chain.astream({"text": payload.text}):
                yield chunk
        except Exception:
            logger.exception("Falha durante o streaming da tradução")

    return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8")
