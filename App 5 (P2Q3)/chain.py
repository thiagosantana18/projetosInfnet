"""Composição da chain (LCEL) em torno de qualquer "tradutor" Runnable.

Fluxo:  texto -> prepare (divide em trechos) -> translator.map() (em lote)
        -> assemble (remonta os parágrafos) -> texto traduzido

Como `translator` é um Runnable genérico (str -> str), a mesma chain funciona
com o HuggingFacePipeline em produção e com um Runnable falso nos testes.
"""
from typing import Callable

from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from app.text_utils import split_into_chunks


def compose_translation_chain(
    translator: Runnable,
    count_tokens: Callable[[str], int],
    max_tokens: int,
) -> Runnable:
    def prepare(text: str) -> dict:
        """Divide o texto em parágrafos e cada parágrafo em trechos."""
        per_paragraph = [
            split_into_chunks(paragraph, count_tokens, max_tokens)
            for paragraph in text.split("\n")
        ]
        return {
            "layout": [len(chunks) for chunks in per_paragraph],
            "chunks": [chunk for chunks in per_paragraph for chunk in chunks],
        }

    def assemble(data: dict) -> str:
        """Reagrupa as traduções na estrutura original de parágrafos."""
        translations = iter(data["translations"])
        paragraphs = [
            " ".join(next(translations).strip() for _ in range(n))
            for n in data["layout"]
        ]
        return "\n".join(paragraphs)

    return (
        RunnableLambda(prepare)
        # `.map()` aplica o tradutor a cada trecho, em uma única chamada em lote.
        | RunnablePassthrough.assign(
            translations=RunnableLambda(lambda d: d["chunks"]) | translator.map()
        )
        | RunnableLambda(assemble)
    )
