"""Funções puras para dividir textos longos em trechos traduzíveis.

Modelos de tradução têm limite de tokens por entrada. Para suportar textos
longos, dividimos o texto em sentenças e as agrupamos em trechos que respeitam
esse limite. As funções recebem um contador de tokens, o que as mantém
independentes do modelo (e fáceis de testar).
"""
import re
from typing import Callable

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Divide um parágrafo em sentenças usando pontuação final."""
    return [s for s in _SENTENCE_BOUNDARY.split(text.strip()) if s]


def _split_long_sentence(
    sentence: str, count_tokens: Callable[[str], int], max_tokens: int
) -> list[str]:
    """Divide por palavras uma sentença que sozinha excede o limite."""
    pieces: list[str] = []
    current: list[str] = []
    for word in sentence.split():
        candidate = " ".join(current + [word])
        if current and count_tokens(candidate) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def split_into_chunks(
    paragraph: str, count_tokens: Callable[[str], int], max_tokens: int
) -> list[str]:
    """Agrupa sentenças em trechos com no máximo `max_tokens` tokens cada."""
    chunks: list[str] = []
    current = ""

    for sentence in split_sentences(paragraph):
        if count_tokens(sentence) > max_tokens:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_sentence(sentence, count_tokens, max_tokens))
            continue

        candidate = f"{current} {sentence}".strip()
        if current and count_tokens(candidate) > max_tokens:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks
