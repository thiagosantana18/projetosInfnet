"""Camada de serviço: encapsula o modelo GPT-2 da HuggingFace."""
from transformers import pipeline


class TextGenerator:
    """Carrega o modelo uma única vez e expõe um método de geração."""

    def __init__(self, model_name: str = "gpt2") -> None:
        self._pipe = pipeline("text-generation", model=model_name)
        # GPT-2 não possui pad token; reutilizamos o EOS para evitar warnings.
        self._pad_token_id = self._pipe.tokenizer.eos_token_id

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """Retorna apenas a continuação do texto (sem repetir o prompt)."""
        outputs = self._pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=self._pad_token_id,
            return_full_text=False,
            num_return_sequences=1,
        )
        return outputs[0]["generated_text"]
