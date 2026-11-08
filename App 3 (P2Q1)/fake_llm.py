"""Fake LLM customizado: responde a perguntas pré-definidas.

O `FakeListLLM` do LangChain devolve respostas em sequência, sem olhar para a
pergunta. Para simular um chatbot que responde a perguntas específicas, criamos
um LLM próprio que herda de `langchain_core.language_models.llms.LLM` e
implementa apenas `_call`. Por ser um `Runnable` padrão do LangChain, ele pode
ser usado em chains (LCEL) exatamente como um modelo real.
"""
import re
import unicodedata
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

# Extrai a pergunta do usuário do prompt formatado pelo PromptTemplate.
_QUESTION_PATTERN = re.compile(r"Usuário:\s*(.*?)\s*\nAssistente:", re.DOTALL)


def normalize(text: str) -> str:
    """Minúsculas, sem acentos, sem pontuação e com espaços colapsados."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class KnowledgeBaseFakeLLM(LLM):
    """LLM simulado que consulta um dicionário de perguntas e respostas."""

    knowledge_base: Dict[str, str]
    fallback_response: str = "Desculpe, não sei responder a isso."

    @property
    def _llm_type(self) -> str:
        return "knowledge-base-fake-llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        match = _QUESTION_PATTERN.search(prompt)
        question = match.group(1) if match else prompt
        key = normalize(question)

        for known_question, answer in self.knowledge_base.items():
            if normalize(known_question) == key:
                return answer
        return self.fallback_response
