"""Camada de serviço: encapsula o modelo de tradução da HuggingFace."""
from transformers import pipeline

from app.text_utils import split_into_chunks


class Translator:
    """Carrega o modelo uma única vez e traduz textos curtos ou longos."""

    def __init__(
        self,
        model_name: str,
        max_tokens_per_chunk: int = 400,
        batch_size: int = 8,
    ) -> None:
        self._pipe = pipeline("translation", model=model_name)
        self._tokenizer = self._pipe.tokenizer
        self._max_tokens = max_tokens_per_chunk
        self._batch_size = batch_size

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.tokenize(text))

    def translate(self, text: str) -> str:
        """Traduz `text` (EN -> FR) preservando a divisão em parágrafos."""
        paragraphs = text.split("\n")

        # Divide cada parágrafo em trechos; parágrafos vazios são mantidos.
        chunks_per_paragraph = [
            split_into_chunks(p, self._count_tokens, self._max_tokens)
            for p in paragraphs
        ]
        flat_chunks = [c for chunks in chunks_per_paragraph for c in chunks]
        if not flat_chunks:
            return text

        # Uma única chamada em lote para todos os trechos.
        outputs = self._pipe(flat_chunks, batch_size=self._batch_size, max_length=512)
        translations = iter(o["translation_text"] for o in outputs)

        # Reagrupa as traduções na estrutura original de parágrafos.
        translated_paragraphs = [
            " ".join(next(translations) for _ in chunks)
            for chunks in chunks_per_paragraph
        ]
        return "\n".join(translated_paragraphs)
