# APIs de geração de texto e tradução com FastAPI, LangChain e HuggingFace

Coleção de cinco aplicações em **FastAPI** que exploram geração de texto e tradução com
modelos da HuggingFace, um chatbot simulado com LangChain e a API da OpenAI.

| # | Pasta | O que faz | Tecnologias | Precisa de internet para rodar? |
|---|-------|-----------|-------------|---------------------------------|
| 1 | [`gpt2-fastapi`](./gpt2-fastapi) | Continua uma frase com GPT-2 | FastAPI + transformers | Só na 1ª execução (download do modelo) |
| 2 | [`translator-fastapi`](./translator-fastapi) | Traduz inglês → francês | FastAPI + transformers (`opus-mt-en-fr`) | Só na 1ª execução (download do modelo) |
| 3 | [`fake-llm-chatbot`](./fake-llm-chatbot) | Chatbot com respostas pré-definidas | FastAPI + LangChain + Fake LLM | Não |
| 4 | [`openai-translator`](./openai-translator) | Traduz inglês → francês | FastAPI + LangChain + OpenAI | Sim (chamadas à API) |
| 5 | [`hf-translator-de`](./hf-translator-de) | Traduz inglês → alemão | FastAPI + LangChain + HuggingFace (`opus-mt-en-de`) | Só na 1ª execução (download do modelo) |

## Sumário

- [Estrutura do repositório](#estrutura-do-repositório)
- [Pré-requisitos](#pré-requisitos)
- [Como executar (passo comum a todas as apps)](#como-executar-passo-comum-a-todas-as-apps)
- [App 1 – Geração de texto com GPT-2](#app-1--geração-de-texto-com-gpt-2)
- [App 2 – Tradução EN→FR com HuggingFace](#app-2--tradução-enfr-com-huggingface)
- [App 3 – Chatbot com LangChain e Fake LLM](#app-3--chatbot-com-langchain-e-fake-llm)
- [App 4 – Tradução EN→FR com LangChain e OpenAI](#app-4--tradução-enfr-com-langchain-e-openai)
- [App 5 – Tradução EN→DE com LangChain e HuggingFace](#app-5--tradução-ende-com-langchain-e-huggingface)
- [Executando os testes](#executando-os-testes)
- [Solução de problemas](#solução-de-problemas)

Cada pasta é um projeto independente, com seu próprio `requirements.txt`, e segue a mesma organização:

```
<app>/
├── app/
│   ├── main.py        # rotas FastAPI
│   ├── config.py      # configurações via variáveis de ambiente (quando aplicável)
│   ├── schemas.py     # validação de entrada/saída (Pydantic)
│   └── ...            # módulos específicos de cada app
├── requirements.txt
```

## Pré-requisitos

- **Python 3.10 ou superior** (`python --version`).
- **pip** e o módulo **venv** (já vêm com o Python na maioria das instalações).
- **Espaço em disco e RAM** para as apps com modelos locais (1, 2 e 5): o `torch` e os modelos ocupam
  alguns GB no total, e os modelos são baixados do Hugging Face Hub na primeira execução e ficam em cache.
- **Chave da API da OpenAI**, somente para a App 4 ([criar uma chave](https://platform.openai.com/api-keys)).
- **Git**, para clonar o repositório.

> Recomendação: use um ambiente virtual **separado para cada app**. As dependências são diferentes
> (por exemplo, só as apps 3, 4 e 5 usam LangChain) e isso evita conflitos de versão.

## Como executar (passo comum a todas as apps)

Todas as apps seguem o mesmo roteiro. Substitua `<pasta-da-app>` pelo nome da pasta.

**1. Clone o repositório e entre na pasta da app**

```bash
git clone <URL-DO-SEU-REPOSITORIO>
cd <nome-do-repositorio>/<pasta-da-app>
```

**2. Crie e ative um ambiente virtual**

```bash
python -m venv .venv
```

| Sistema | Comando de ativação |
|---------|---------------------|
| Linux / macOS | `source .venv/bin/activate` |
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |

**3. Instale as dependências**

```bash
pip install -r requirements.txt
```

**4. Inicie o servidor** (sempre **de dentro da pasta da app**, pois o pacote se chama `app`)

```bash
uvicorn app.main:app --reload
```

O servidor sobe em `http://localhost:8000`. A documentação interativa (Swagger) fica em
**http://localhost:8000/docs**, onde é possível testar os endpoints direto do navegador.

**5. Para rodar outra app ao mesmo tempo**, use outra porta:

```bash
uvicorn app.main:app --reload --port 8001
```

**6. Para encerrar**, pressione `Ctrl + C` no terminal. Para sair do ambiente virtual, use `deactivate`.

> Nos exemplos abaixo, os comandos `curl` usam aspas simples no JSON (Linux/macOS/Git Bash).
> No PowerShell do Windows, é mais simples usar o Swagger em `/docs` ou `Invoke-RestMethod`.

---

## App 1 – Geração de texto com GPT-2

Recebe uma frase em JSON e devolve uma continuação gerada pelo modelo `gpt2` da HuggingFace.

**Executar**

```bash
cd gpt2-fastapi
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Na primeira execução, o modelo `gpt2` (~500 MB) é baixado automaticamente.
Aguarde a mensagem `Modelo carregado.` no terminal antes de fazer requisições.

**Endpoints**

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/generate` | Gera a continuação de um texto |
| `GET` | `/health` | Verifica se a API está no ar |

**Exemplo**

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time, in a distant galaxy", "max_new_tokens": 40}'
```

**Resposta**

```json
{
  "prompt": "Once upon a time, in a distant galaxy",
  "generated_text": " there lived a young explorer who...",
  "full_text": "Once upon a time, in a distant galaxy there lived a young explorer who..."
}
```

**Parâmetros do corpo (JSON)**

| Campo | Obrigatório | Padrão | Faixa | Descrição |
|-------|-------------|--------|-------|-----------|
| `prompt` | sim | – | 1 a 500 caracteres | Frase inicial |
| `max_new_tokens` | não | 50 | 1 a 200 | Máximo de tokens gerados |
| `temperature` | não | 0.8 | 0 (exclusivo) a 2 | Maior = mais aleatório |
| `top_k` | não | 50 | 1 a 200 | Amostragem top-k |
| `top_p` | não | 0.95 | 0 (exclusivo) a 1 | Amostragem nucleus |

**Variáveis de ambiente (opcionais)**

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `MODEL_NAME` | `gpt2` | Modelo da HuggingFace |

**Observações:** o GPT-2 foi treinado majoritariamente em inglês, então prompts em inglês dão resultados melhores.
Como a geração usa amostragem, o mesmo prompt produz textos diferentes a cada chamada.

---

## App 2 – Tradução EN→FR com HuggingFace

Traduz textos do inglês para o francês com o modelo `Helsinki-NLP/opus-mt-en-fr`. Textos longos são divididos
em trechos (por parágrafos e sentenças), traduzidos em lote e remontados, preservando as quebras de linha.

**Executar**

```bash
cd translator-fastapi
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Na primeira execução, o modelo (~300 MB) é baixado automaticamente.

**Endpoints**

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/translate` | Traduz um texto do inglês para o francês |
| `GET` | `/health` | Verifica se a API está no ar |

**Exemplo**

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you today?"}'
```

**Resposta**

```json
{
  "source_text": "Hello, how are you today?",
  "translated_text": "Bonjour, comment allez-vous aujourd'hui ?"
}
```

O `translated_text` acima é ilustrativo; o texto real depende do modelo.

**Variáveis de ambiente (opcionais)**

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `MODEL_NAME` | `Helsinki-NLP/opus-mt-en-fr` | Modelo da HuggingFace |
| `MAX_INPUT_CHARS` | `10000` | Tamanho máximo do texto de entrada |
| `MAX_TOKENS_PER_CHUNK` | `400` | Tokens por trecho enviado ao modelo (limite do modelo: 512) |
| `BATCH_SIZE` | `8` | Trechos traduzidos por lote |

---

## App 3 – Chatbot com LangChain e Fake LLM

Chatbot simples que responde a perguntas pré-definidas usando uma chain do LangChain com um **Fake LLM**.
Não baixa modelo, não usa internet e não exige chave de API, então é a app mais rápida de rodar.

**Executar**

```bash
cd fake-llm-chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Endpoints**

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/chat` | Envia uma mensagem e recebe a resposta simulada |
| `GET` | `/questions` | Lista as perguntas que o chatbot entende |
| `GET` | `/health` | Verifica se a API está no ar |

**Exemplo**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "O que é um Fake LLM?"}'
```

**Resposta**

```json
{
  "message": "O que é um Fake LLM?",
  "response": "Um Fake LLM é um modelo simulado que devolve respostas pré-definidas, ..."
}
```

**Perguntas reconhecidas:** `Olá`, `Qual é o seu nome?`, `O que é LangChain?`, `O que é um Fake LLM?`,
`Como você pode me ajudar?` e `Tchau`. A comparação ignora maiúsculas, acentos e pontuação.
Qualquer outra pergunta recebe uma resposta padrão (fallback). Para adicionar perguntas, edite
`app/knowledge_base.py`.

---

## App 4 – Tradução EN→FR com LangChain e OpenAI

Traduz inglês para francês usando um modelo da OpenAI via LangChain (`ChatOpenAI`).
**Requer uma chave de API da OpenAI**, e as chamadas à API são cobradas conforme o uso.

**Executar**

```bash
cd openai-translator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Defina a chave de API **antes** de iniciar o servidor:

| Sistema | Comando |
|---------|---------|
| Linux / macOS | `export OPENAI_API_KEY="sk-..."` |
| Windows (PowerShell) | `$env:OPENAI_API_KEY="sk-..."` |
| Windows (cmd) | `set OPENAI_API_KEY=sk-...` |

Depois, inicie o servidor:

```bash
uvicorn app.main:app --reload
```

Sem a variável `OPENAI_API_KEY`, a aplicação encerra na inicialização com uma mensagem de erro.
Nunca versione o arquivo `.env` nem a sua chave.

**Endpoints**

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/translate` | Traduz um texto e devolve JSON |
| `POST` | `/translate/stream` | Devolve a tradução em texto puro, conforme é gerada |
| `GET` | `/health` | Verifica se a API está no ar |

**Exemplos**

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you today?"}'
```

```json
{
  "source_text": "Hello, how are you today?",
  "translated_text": "Bonjour, comment allez-vous aujourd'hui ?",
  "model": "gpt-4o-mini"
}
```

Streaming (o `-N` desativa o buffer do curl):

```bash
curl -N -X POST http://localhost:8000/translate/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

**Variáveis de ambiente**

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `OPENAI_API_KEY` | (obrigatória) | Chave da API da OpenAI |
| `OPENAI_MODEL` | `gpt-4o-mini` | Modelo utilizado (troque se quiser outro) |
| `OPENAI_TEMPERATURE` | `0` | Aleatoriedade da resposta |
| `OPENAI_TIMEOUT` | `30` | Timeout da chamada, em segundos |
| `OPENAI_MAX_RETRIES` | `2` | Tentativas extras em erros transitórios |
| `MAX_INPUT_CHARS` | `5000` | Tamanho máximo do texto de entrada |
| `CACHE_ENABLED` | `true` | Cache em memória para textos idênticos |

**Códigos de erro da API**

| Código | Causa |
|--------|-------|
| `422` | Entrada inválida (texto vazio, só espaços ou acima do limite) |
| `429` | Limite de requisições da OpenAI excedido |
| `503` | Falha de conexão com a OpenAI |
| `504` | Timeout na chamada à OpenAI |
| `502` | Erro retornado pela OpenAI |
| `500` | Erro de configuração (por exemplo, chave inválida) ou erro interno |

---

## App 5 – Tradução EN→DE com LangChain e HuggingFace

Traduz inglês para alemão com o modelo `Helsinki-NLP/opus-mt-en-de`, executado **localmente** e orquestrado pelo
LangChain (`HuggingFacePipeline`). A chain divide o texto em trechos, traduz em lote e remonta o resultado.

**Executar**

```bash
cd hf-translator-de
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Na primeira execução, o modelo (~300 MB) é baixado automaticamente.

**Endpoints**

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/translate` | Traduz um texto do inglês para o alemão |
| `GET` | `/health` | Verifica se a API está no ar |

**Exemplo**

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you today?"}'
```

**Resposta**

```json
{
  "source_text": "Hello, how are you today?",
  "translated_text": "Hallo, wie geht es dir heute?",
  "model": "Helsinki-NLP/opus-mt-en-de"
}
```

O `translated_text` acima é ilustrativo; o texto real depende do modelo.

**Variáveis de ambiente (opcionais)**

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `MODEL_NAME` | `Helsinki-NLP/opus-mt-en-de` | Modelo da HuggingFace |
| `DEVICE` | `cpu` | `cpu`, `cuda` ou `cuda:0` (GPU exige `torch` com suporte a CUDA) |
| `MAX_INPUT_CHARS` | `10000` | Tamanho máximo do texto de entrada |
| `MAX_TOKENS_PER_CHUNK` | `400` | Tokens por trecho enviado ao modelo |
| `BATCH_SIZE` | `8` | Trechos traduzidos por lote |

Exemplo usando GPU:

```bash
DEVICE=cuda uvicorn app.main:app --reload
```

---

## Solução de problemas

| Problema | Causa provável | Solução |
|----------|----------------|---------|
| `ModuleNotFoundError: No module named 'app'` | Comando executado fora da pasta da app | Entre na pasta da app (`cd <pasta-da-app>`) e rode de novo |
| `ModuleNotFoundError` para `fastapi`, `transformers`, `langchain...` | Ambiente virtual não ativado ou dependências não instaladas | Ative o `.venv` e rode `pip install -r requirements.txt` |
| `Address already in use` | Porta 8000 ocupada (outra app rodando) | Use `--port 8001` ou encerre a outra app |
| Servidor demora a iniciar na primeira vez | Download do modelo no Hugging Face Hub | Aguarde; nas próximas execuções o modelo vem do cache |
| Erro de download do modelo | Sem internet ou Hub bloqueado | Verifique a conexão ou use um proxy/mirror configurado |
| `RuntimeError: Defina a variável de ambiente OPENAI_API_KEY` (App 4) | Chave não definida | Defina `OPENAI_API_KEY` antes de iniciar o servidor |
| Resposta `500` na App 4 | Chave inválida ou sem permissão | Confira a chave e o acesso ao modelo configurado |
| Resposta `429` na App 4 | Cota ou limite de requisições da OpenAI | Aguarde, reduza o ritmo de chamadas ou revise o plano |
| `422 Unprocessable Entity` | Corpo JSON inválido (campo ausente, vazio ou fora dos limites) | Confira os campos e os limites descritos em cada app |
| Respostas lentas nas apps com modelo local | Inferência em CPU | Use GPU (`DEVICE=cuda` na App 5) ou reduza `max_new_tokens` / tamanho do texto |
| Erro de `torch` ou CUDA | Versão do `torch` sem suporte à sua GPU | Instale o `torch` conforme as instruções em [pytorch.org](https://pytorch.org/get-started/locally/) |

## Sugestão de `.gitignore`

Para não versionar ambientes virtuais, caches e segredos, crie um `.gitignore` na raiz do repositório:

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.env
```
