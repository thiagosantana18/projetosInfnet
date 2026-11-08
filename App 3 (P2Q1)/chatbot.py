"""Montagem da chain do chatbot (LCEL): prompt -> LLM -> parser."""
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from app.fake_llm import KnowledgeBaseFakeLLM
from app.knowledge_base import FALLBACK_RESPONSE, KNOWLEDGE_BASE

PROMPT = PromptTemplate.from_template("Usuário: {question}\nAssistente:")


def build_chain(llm: LLM | None = None) -> Runnable:
    """Cria a chain. Para usar um LLM real, basta passar outro `llm`.

    A chain não conhece o tipo de modelo: trocar o Fake LLM por OpenAI,
    Anthropic, etc. não exige alterar o resto da aplicação.
    """
    llm = llm or KnowledgeBaseFakeLLM(
        knowledge_base=KNOWLEDGE_BASE,
        fallback_response=FALLBACK_RESPONSE,
    )
    return PROMPT | llm | StrOutputParser()
