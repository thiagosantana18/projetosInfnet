"""Aplicação FastAPI: chatbot simples com LangChain + Fake LLM."""
import logging

from fastapi import Depends, FastAPI, HTTPException
from langchain_core.runnables import Runnable

from app.chatbot import build_chain
from app.knowledge_base import KNOWLEDGE_BASE
from app.schemas import ChatRequest, ChatResponse, QuestionsResponse

logger = logging.getLogger("chatbot-api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Fake LLM Chatbot",
    description="Chatbot de demonstração com LangChain e um Fake LLM.",
    version="1.0.0",
)

# Criada uma única vez; não há modelo pesado para carregar.
_chain = build_chain()


def get_chain() -> Runnable:
    """Dependência que fornece a chain (fácil de substituir em testes)."""
    return _chain


@app.get("/health", tags=["infra"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/questions", response_model=QuestionsResponse, tags=["chat"])
def list_questions() -> QuestionsResponse:
    """Lista as perguntas pré-definidas que o chatbot entende."""
    return QuestionsResponse(questions=list(KNOWLEDGE_BASE))


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    payload: ChatRequest,
    chain: Runnable = Depends(get_chain),
) -> ChatResponse:
    """Recebe uma mensagem e retorna a resposta simulada pelo Fake LLM."""
    try:
        answer = await chain.ainvoke({"question": payload.message})
    except Exception:
        logger.exception("Falha ao executar a chain")
        raise HTTPException(status_code=500, detail="Erro ao processar a mensagem.")

    return ChatResponse(message=payload.message, response=answer)
