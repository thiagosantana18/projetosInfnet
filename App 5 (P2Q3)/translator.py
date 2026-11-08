"""Integração LangChain + HuggingFace (modelo opus-mt-en-de rodando localmente)."""
from langchain_core.runnables import Runnable
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from app.chain import compose_translation_chain
from app.config import Settings


def build_translation_chain(settings: Settings) -> Runnable:
    """Carrega o modelo e monta a chain de tradução."""
    hf_pipe = pipeline("translation", model=settings.model_name, device=settings.device)

    # Adaptador do LangChain: expõe o pipeline da HuggingFace como um Runnable.
    llm = HuggingFacePipeline(
        pipeline=hf_pipe,
        batch_size=settings.batch_size,
        pipeline_kwargs={"max_length": 512},
    )

    tokenizer = hf_pipe.tokenizer
    return compose_translation_chain(
        translator=llm,
        count_tokens=lambda text: len(tokenizer.tokenize(text)),
        max_tokens=settings.max_tokens_per_chunk,
    )
