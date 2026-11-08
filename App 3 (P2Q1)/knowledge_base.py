"""Perguntas pré-definidas e suas respostas simuladas.

As chaves devem ser escritas de forma "normalizada": minúsculas, sem acentos
e sem pontuação (a normalização é aplicada também à pergunta do usuário).
"""

KNOWLEDGE_BASE: dict[str, str] = {
    "ola": "Olá! Sou um chatbot de demonstração. Como posso ajudar?",
    "qual e o seu nome": "Eu me chamo FakeBot, um assistente simulado com LangChain.",
    "o que e langchain": (
        "LangChain é um framework para construir aplicações com modelos de "
        "linguagem, usando componentes como prompts, modelos e parsers."
    ),
    "o que e um fake llm": (
        "Um Fake LLM é um modelo simulado que devolve respostas pré-definidas, "
        "útil para testar aplicações sem custo, sem rede e de forma determinística."
    ),
    "como voce pode me ajudar": (
        "Posso responder a algumas perguntas pré-definidas. "
        "Consulte GET /questions para ver quais."
    ),
    "tchau": "Até logo! Foi um prazer conversar com você.",
}

FALLBACK_RESPONSE = (
    "Desculpe, não sei responder a isso. "
    "Consulte GET /questions para ver as perguntas que eu entendo."
)
