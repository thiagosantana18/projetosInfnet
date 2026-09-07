"""
Turismo no Rio de Janeiro — Chegada de turistas (Data.Rio)
===========================================================
Aplicação Streamlit que lê os datasets do portal Data.Rio (seção
Turismo) sobre chegada mensal de turistas ao Rio de Janeiro por via
Aérea (tabela 2675) ou por via Marítima (tabela 2676), e oferece uma
interface para explorar, filtrar, visualizar e exportar esses dados.
"""

import io
import re
import time

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------------------------------
# LEITURA E "ARRUMAÇÃO" DOS DADOS DO DATA.RIO
# ---------------------------------------------------------------------------
# As tabelas 2675 (via Aérea) e 2676 (via Marítima) do Data.Rio compartilham
# exatamente o mesmo layout hierárquico "de impressão":
#
#   linha 0  -> "Data.Rio"
#   linha 2  -> título da tabela (contém o ano e a via de acesso)
#   linha 4  -> cabeçalho agrupado ("Continentes e países...", "Número de turistas")
#   linha 5  -> cabeçalho das colunas: (vazio), Total, Janeiro, ..., Dezembro
#   linha 7+ -> dados. Linhas SEM recuo = continente (subtotal).
#               Linhas COM recuo (espaços à esquerda) = país, pertencente
#               ao último continente lido. A primeira linha de dados é
#               sempre "Total" (o total geral).
#   ...      -> bloco de notas/fonte ao final (ignorado)
#
# As funções abaixo transformam esse layout em um DataFrame "arrumado"
# (tidy), com uma linha por continente e uma linha por país.

MESES_PADRAO = [
    "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
    "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro",
]


def _limpar_numero(valor):
    """Converte células como '1.234', '-', NaN em float utilizável."""
    if pd.isna(valor):
        return 0.0
    if isinstance(valor, (int, float)):
        return float(valor)
    texto = str(valor).strip()
    if texto in ("-", "", "nan", "None"):
        return 0.0
    texto = texto.replace(".", "").replace(",", ".")
    try:
        return float(texto)
    except ValueError:
        return 0.0


def _detectar_via(titulo: str) -> str:
    t = titulo.lower()
    if "aérea" in t or "aerea" in t:
        return "Aérea"
    if "marítima" in t or "maritima" in t:
        return "Marítima"
    return "Não identificada"


def parse_datario_xls(file_like):
    """
    Lê um arquivo XLS do Data.Rio (turismo) e devolve um dicionário com:
      - titulo:        texto original da tabela
      - ano:            ano de referência (string)
      - via:            "Aérea" / "Marítima" / "Não identificada"
      - meses:          lista com os nomes das 12 colunas mensais
      - df:             DataFrame "arrumado" (uma linha por continente e por país)
      - total_geral:    dict com Total/meses do total geral da tabela
      - ok:             True se a estrutura esperada foi reconhecida
      - aviso:          mensagem de aviso quando ok=False
    """
    resultado = {
        "titulo": None, "ano": None, "via": None, "meses": MESES_PADRAO,
        "df": pd.DataFrame(), "total_geral": {}, "ok": False, "aviso": None,
    }

    try:
        if hasattr(file_like, "seek"):
            file_like.seek(0)
        raw = pd.read_excel(file_like, header=None, engine="xlrd")
    except Exception as e:
        resultado["aviso"] = f"Não foi possível ler o arquivo XLS ({e})."
        return resultado

    try:
        titulo = str(raw.iloc[2, 0]).strip()
        # O título traz "Tabela NNNN - ... - Via X - AAAA": o ano de
        # referência é sempre o ÚLTIMO grupo de 4 dígitos (o primeiro é o
        # número da tabela, não o ano).
        anos_encontrados = re.findall(r"(\d{4})", titulo)
        ano = anos_encontrados[-1] if anos_encontrados else "N/D"
        via = _detectar_via(titulo)

        # cabeçalho das colunas mensais está na linha 5, colunas 2..13
        meses = [str(raw.iloc[5, c]).strip() for c in range(2, 14)]
        if len(meses) != 12 or not meses[0].lower().startswith("jan"):
            raise ValueError("Layout de colunas mensais inesperado.")

        registros = []
        total_geral = {}
        continente_atual = None
        r = 7
        n_linhas = len(raw)

        while r < n_linhas:
            rotulo = raw.iloc[r, 0]
            if pd.isna(rotulo) or str(rotulo).strip() == "":
                break  # fim do bloco de dados

            rotulo_txt = str(rotulo)
            rotulo_limpo = rotulo_txt.strip()
            recuado = rotulo_txt.startswith(("   ", "\xa0", " \xa0"))

            total_val = _limpar_numero(raw.iloc[r, 1])
            valores_meses = [_limpar_numero(raw.iloc[r, c]) for c in range(2, 14)]

            if rotulo_limpo.lower() == "total" and total_geral == {}:
                total_geral = {"Total": total_val, **dict(zip(meses, valores_meses))}
                r += 1
                continue

            if not recuado:
                continente_atual = rotulo_limpo
                registros.append({
                    "Nivel": "Continente",
                    "Continente": rotulo_limpo,
                    "Pais": "(subtotal do continente)",
                    "Total": total_val,
                    **dict(zip(meses, valores_meses)),
                })
            else:
                registros.append({
                    "Nivel": "País",
                    "Continente": continente_atual,
                    "Pais": rotulo_limpo,
                    "Total": total_val,
                    **dict(zip(meses, valores_meses)),
                })
            r += 1

        if not registros:
            raise ValueError("Nenhuma linha de dados foi reconhecida.")

        df = pd.DataFrame(registros)

        resultado.update({
            "titulo": titulo, "ano": ano, "via": via, "meses": meses,
            "df": df, "total_geral": total_geral, "ok": True,
        })
        return resultado

    except Exception as e:
        resultado["aviso"] = (
            "A estrutura do arquivo não é a esperada para as tabelas "
            f"2675/2676 do Data.Rio ({e}). Exibindo dados brutos, sem "
            "interpretação hierárquica."
        )
        try:
            if hasattr(file_like, "seek"):
                file_like.seek(0)
            resultado["df"] = pd.read_excel(file_like, engine="xlrd")
        except Exception:
            pass
        return resultado

# ---------------------------------------------------------------------------
# 1) OBJETIVO, MOTIVAÇÃO E CONFIGURAÇÃO GERAL DA PÁGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Turismo no Rio de Janeiro — Data.Rio",
    page_icon="🌎",
    layout="wide",
)

# --- Session State: valores padrão (item 9 - persistência de preferências) --
DEFAULTS = {
    "cor_fundo": "#0E1117",
    "cor_fonte": "#FAFAFA",
    "nivel_sel": "Ambos",
    "continentes_sel": None,  # definido após carregar os dados (todos por padrão)
    "ocultar_zero": False,
    "apenas_total": False,
    "colunas_sel": None,   # definido após carregar os dados
    "ordenar_por": "Total",
    "ordem_desc": True,
    "df_carregado": None,
    "meta_carregada": None,
}
for chave, valor in DEFAULTS.items():
    if chave not in st.session_state:
        st.session_state[chave] = valor


def aplicar_tema(cor_fundo: str, cor_fonte: str):
    """Injeta CSS para customizar o fundo e a cor das fontes (item 7)."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {cor_fundo};
            color: {cor_fonte};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {cor_fundo};
            color: {cor_fonte};
        }}
        .stApp, .stApp p, .stApp span, .stApp label, .stApp h1,
        .stApp h2, .stApp h3, .stApp h4, .stMarkdown, .stMetric label {{
            color: {cor_fonte} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


aplicar_tema(st.session_state["cor_fundo"], st.session_state["cor_fonte"])

st.title("🌎 Chegada de Turistas no Rio de Janeiro")
st.caption("Fonte dos dados: Data.Rio — seção Turismo (Departamento de Polícia Federal / EMBRATUR)")

with st.expander("📌 Objetivo, motivação e funcionalidades do projeto (item 1)", expanded=True):
    st.markdown(
        """
**Dataset escolhido:** *Chegada mensal de turistas pelo Rio de Janeiro*, disponível em duas
variantes no portal Data.Rio — **via Aérea** (tabela 2675) e **via Marítima** (tabela 2676),
com a chegada de turistas discriminada por mês, continente e país de residência permanente.

**Objetivo e motivação:** o turismo é um dos pilares econômicos da cidade do Rio de Janeiro, e
entender de onde vêm os visitantes internacionais — e em quais meses — ajuda a compreender a
sazonalidade do setor (por exemplo, o efeito do verão, do Carnaval ou de eventos internacionais)
e a relevância relativa de cada continente/país de origem. Esses dados públicos permitem
comparar o perfil dos turistas que chegam de avião com os que chegam de navio (cruzeiros),
fluxos que têm dinâmicas bem diferentes ao longo do ano.

**Funcionalidades implementadas nesta aplicação:**
- Upload de arquivo `.xls` do Data.Rio, com leitura em cache;
- Filtros por nível de detalhe (continente/país), continentes de interesse e colunas/meses;
- Tabela interativa ordenável e exportável para Excel;
- Barra de progresso e spinner durante o carregamento/processamento;
- Personalização de cores (color picker) para fundo e fonte;
- Persistência das preferências do usuário via `session_state`;
- Gráficos simples (barras, linhas, pizza) e avançados (histograma, dispersão);
- Métricas-resumo (total de registros, soma, média, mês de pico etc.).
        """
    )

# ---------------------------------------------------------------------------
# 2) UPLOAD DE ARQUIVO XLS
# ---------------------------------------------------------------------------
st.sidebar.header("1️⃣ Arquivo de dados")
arquivo = st.sidebar.file_uploader(
    "Envie um arquivo .xls do Data.Rio (tabela 2675 ou 2676)",
    type=["xls"],
    key="uploader",
)

usar_exemplo = None
if arquivo is None:
    usar_exemplo = st.sidebar.selectbox(
        "…ou use um exemplo já incluído no projeto:",
        ["(nenhum)", "2675 — Via Aérea", "2676 — Via Marítima"],
        key="exemplo_sel",
    )

# ---------------------------------------------------------------------------
# 8) CACHE — evita reprocessar o mesmo arquivo a cada interação
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def carregar_dados_cache(conteudo_bytes: bytes, nome_arquivo: str):
    buffer = io.BytesIO(conteudo_bytes)
    return parse_datario_xls(buffer)


def carregar_com_feedback(conteudo_bytes: bytes, nome_arquivo: str):
    """Mostra spinner + barra de progresso (item 6) enquanto processa os dados."""
    progresso = st.sidebar.progress(0, text="Iniciando leitura do arquivo…")
    with st.spinner(f"Lendo e organizando os dados de **{nome_arquivo}**…"):
        for pct, msg in [
            (20, "Abrindo arquivo XLS…"),
            (45, "Identificando cabeçalhos e período…"),
            (70, "Estruturando continentes e países…"),
        ]:
            progresso.progress(pct, text=msg)
            time.sleep(0.15)
        meta = carregar_dados_cache(conteudo_bytes, nome_arquivo)
        progresso.progress(100, text="Concluído!")
        time.sleep(0.2)
    progresso.empty()
    return meta


meta = None
nome_fonte = None

if arquivo is not None:
    conteudo = arquivo.getvalue()
    nome_fonte = arquivo.name
    meta = carregar_com_feedback(conteudo, nome_fonte)
elif usar_exemplo and usar_exemplo != "(nenhum)":
    caminho = "2675.xls" if usar_exemplo.startswith("2675") else "2676.xls"
    with open(caminho, "rb") as f:
        conteudo = f.read()
    nome_fonte = caminho.split("/")[-1]
    meta = carregar_com_feedback(conteudo, nome_fonte)

if meta is None:
    st.info("⬅️ Envie um arquivo `.xls` na barra lateral (ou escolha um exemplo) para começar.")
    st.stop()

if not meta["ok"]:
    st.warning(meta["aviso"])
    if meta["df"] is not None and not meta["df"].empty:
        st.dataframe(meta["df"], use_container_width=True)
    st.stop()

df = meta["df"]
meses = meta["meses"]
st.session_state["df_carregado"] = df
st.session_state["meta_carregada"] = {k: v for k, v in meta.items() if k != "df"}

st.success(
    f"**{meta['titulo']}** — Ano de referência: **{meta['ano']}** • "
    f"Via de acesso: **{meta['via']}** • Fonte: `{nome_fonte}`"
)

# ---------------------------------------------------------------------------
# EXIBIÇÃO INICIAL DO DATASET (parte do item 3)
# ---------------------------------------------------------------------------
with st.expander("🔎 Ver dataset completo (sem filtros)"):
    st.dataframe(df, use_container_width=True, height=300)

# ---------------------------------------------------------------------------
# 3) FILTROS E SELEÇÃO — radio, checkbox e dropdown (multiselect/selectbox)
# ---------------------------------------------------------------------------
st.sidebar.header("2️⃣ Filtros")

# --- seletor tipo RADIO: nível de detalhamento das linhas -------------------
nivel = st.sidebar.radio(
    "Nível de detalhamento (linhas):",
    ["Continentes", "Países", "Ambos"],
    key="nivel_sel",
)

# --- seletor tipo DROPDOWN (multiselect): continentes de interesse ---------
continentes_disponiveis = sorted(df["Continente"].dropna().unique().tolist())
if st.session_state["continentes_sel"] is None:
    st.session_state["continentes_sel"] = continentes_disponiveis
continentes_escolhidos = st.sidebar.multiselect(
    "Continentes a incluir:",
    options=continentes_disponiveis,
    key="continentes_sel",
)

# --- seletores tipo CHECKBOX -------------------------------------------------
ocultar_zero = st.sidebar.checkbox(
    "Ocultar linhas com Total = 0", key="ocultar_zero"
)
apenas_total = st.sidebar.checkbox(
    "Mostrar apenas a coluna Total (ocultar meses)", key="apenas_total"
)

# --- seletor tipo DROPDOWN (multiselect): colunas/meses a exibir -----------
colunas_valores = ["Total"] + meses
if st.session_state["colunas_sel"] is None:
    st.session_state["colunas_sel"] = colunas_valores
colunas_escolhidas = st.sidebar.multiselect(
    "Colunas (meses) a exibir na tabela:",
    options=colunas_valores,
    default=st.session_state["colunas_sel"],
    key="colunas_sel",
    disabled=apenas_total,
)

# --- seletor tipo DROPDOWN (selectbox): ordenação da tabela -----------------
ordenar_por = st.sidebar.selectbox(
    "Ordenar tabela por:",
    options=["Total"] + meses + ["Continente", "Pais"],
    key="ordenar_por",
)
ordem_desc = st.sidebar.checkbox("Ordem decrescente", key="ordem_desc")

st.sidebar.header("3️⃣ Aparência")
cor_fundo = st.sidebar.color_picker("Cor de fundo do painel", key="cor_fundo")
cor_fonte = st.sidebar.color_picker("Cor das fontes", key="cor_fonte")
aplicar_tema(cor_fundo, cor_fonte)  # reaplica caso o usuário tenha mudado agora

if st.sidebar.button("↩️ Resetar preferências"):
    for chave, valor in DEFAULTS.items():
        st.session_state[chave] = valor
    st.rerun()

# ---------------------------------------------------------------------------
# APLICA OS FILTROS
# ---------------------------------------------------------------------------
df_filtrado = df.copy()

if nivel != "Ambos":
    alvo = "Continente" if nivel == "Continentes" else "País"
    df_filtrado = df_filtrado[df_filtrado["Nivel"] == alvo]

if continentes_escolhidos:
    df_filtrado = df_filtrado[df_filtrado["Continente"].isin(continentes_escolhidos)]

if ocultar_zero:
    df_filtrado = df_filtrado[df_filtrado["Total"] > 0]

colunas_base = ["Nivel", "Continente", "Pais"]
if apenas_total:
    colunas_dado = ["Total"]
else:
    colunas_dado = [c for c in colunas_valores if c in colunas_escolhidas] or ["Total"]

df_exibir = df_filtrado[colunas_base + colunas_dado].copy()

if ordenar_por in df_exibir.columns:
    df_exibir = df_exibir.sort_values(by=ordenar_por, ascending=not ordem_desc)

# ---------------------------------------------------------------------------
# 4) TABELA INTERATIVA (ordenação/filtragem adicional pela própria interface)
# ---------------------------------------------------------------------------
st.subheader("📋 Dados filtrados")
st.caption(
    "A tabela abaixo já reflete os filtros da barra lateral. Você também pode "
    "clicar no cabeçalho de qualquer coluna para reordenar, ou usar a lupa "
    "(ícone no canto superior da tabela) para buscar/filtrar diretamente."
)
st.dataframe(
    df_exibir,
    use_container_width=True,
    height=380,
    column_config={
        col: st.column_config.NumberColumn(format="%.0f")
        for col in colunas_dado
    },
)
st.caption(f"{len(df_exibir):,} linha(s) exibida(s) de {len(df):,} no total.".replace(",", "."))

# ---------------------------------------------------------------------------
# 5) DOWNLOAD DOS DADOS FILTRADOS
# ---------------------------------------------------------------------------
def gerar_excel_bytes(dataframe: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="dados_filtrados")
    return buffer.getvalue()


st.download_button(
    label="⬇️ Baixar dados filtrados (Excel)",
    data=gerar_excel_bytes(df_exibir),
    file_name="turismo_rio_filtrado.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
st.caption(
    "Observação: o arquivo é gerado no formato `.xlsx` (padrão atual do Excel), "
    "totalmente compatível com o `.xls` original, mas com suporte nativo em "
    "bibliotecas atuais de geração de planilhas."
)

st.divider()

# ---------------------------------------------------------------------------
# 12) MÉTRICAS BÁSICAS
# ---------------------------------------------------------------------------
st.subheader("📊 Métricas-resumo")

col_pais = df_filtrado[df_filtrado["Nivel"] == "País"]
soma_total = df_filtrado.loc[df_filtrado["Nivel"] == "Continente", "Total"].sum()
media_mensal = df_filtrado.loc[df_filtrado["Nivel"] == "Continente", meses].values.mean() \
    if not df_filtrado.empty else 0
mes_totais = {m: df_filtrado.loc[df_filtrado["Nivel"] == "Continente", m].sum() for m in meses}
mes_pico = max(mes_totais, key=mes_totais.get) if mes_totais else "-"
maior_pais = (
    col_pais.sort_values("Total", ascending=False).iloc[0]
    if not col_pais.empty else None
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Registros exibidos", f"{len(df_exibir):,}".replace(",", "."))
c2.metric("Soma (continentes)", f"{soma_total:,.0f}".replace(",", "."))
c3.metric("Média mensal (continentes)", f"{media_mensal:,.0f}".replace(",", "."))
c4.metric("Mês de pico", mes_pico, f"{mes_totais.get(mes_pico, 0):,.0f}".replace(",", "."))
c5.metric(
    "País com mais turistas",
    maior_pais["Pais"] if maior_pais is not None else "-",
    f"{maior_pais['Total']:,.0f}".replace(",", ".") if maior_pais is not None else None,
)

st.divider()

# ---------------------------------------------------------------------------
# 10) GRÁFICOS SIMPLES — barras, linhas e pizza
# ---------------------------------------------------------------------------
st.subheader("📈 Gráficos simples")

df_continentes = df_filtrado[df_filtrado["Nivel"] == "Continente"].copy()
df_continentes = df_continentes.sort_values("Total", ascending=False)

aba_bar, aba_linha, aba_pizza = st.tabs(["Barras", "Linhas", "Pizza"])


def novo_eixo(figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(cor_fundo)
    ax.set_facecolor(cor_fundo)
    ax.tick_params(colors=cor_fonte)
    ax.xaxis.label.set_color(cor_fonte)
    ax.yaxis.label.set_color(cor_fonte)
    ax.title.set_color(cor_fonte)
    for spine in ax.spines.values():
        spine.set_color(cor_fonte)
    return fig, ax


with aba_bar:
    if df_continentes.empty:
        st.info("Nenhum continente disponível com os filtros atuais.")
    else:
        fig, ax = novo_eixo()
        ax.bar(df_continentes["Continente"], df_continentes["Total"], color="#3AA6FF")
        ax.set_ylabel("Total de turistas")
        ax.set_title("Total de turistas por continente")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

with aba_linha:
    if not meta["total_geral"]:
        st.info("Total geral mensal não disponível.")
    else:
        valores = [meta["total_geral"].get(m, 0) for m in meses]
        fig, ax = novo_eixo()
        ax.plot(meses, valores, marker="o", color="#FF7A3A")
        ax.set_ylabel("Total de turistas")
        ax.set_title("Evolução mensal do total geral de turistas")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

with aba_pizza:
    if df_continentes.empty:
        st.info("Nenhum continente disponível com os filtros atuais.")
    else:
        top = df_continentes.head(6)
        outros_total = df_continentes["Total"].iloc[6:].sum()
        rotulos = top["Continente"].tolist()
        valores = top["Total"].tolist()
        if outros_total > 0:
            rotulos.append("Outros")
            valores.append(outros_total)
        fig, ax = novo_eixo(figsize=(6, 6))
        wedges, texts, autotexts = ax.pie(
            valores, labels=rotulos, autopct="%1.1f%%", textprops={"color": cor_fonte}
        )
        ax.set_title("Participação por continente")
        st.pyplot(fig)

# ---------------------------------------------------------------------------
# 11) GRÁFICOS AVANÇADOS — histograma e dispersão
# ---------------------------------------------------------------------------
st.subheader("🔬 Gráficos avançados")

aba_hist, aba_disp = st.tabs(["Histograma", "Dispersão (scatter)"])

with aba_hist:
    dados_hist = df_filtrado[(df_filtrado["Nivel"] == "País") & (df_filtrado["Total"] > 0)]
    if dados_hist.empty:
        st.info("Sem dados de países (Total > 0) para o histograma com os filtros atuais.")
    else:
        fig, ax = novo_eixo()
        ax.hist(dados_hist["Total"], bins=20, color="#3AA6FF", edgecolor=cor_fundo)
        ax.set_xlabel("Total de turistas (por país)")
        ax.set_ylabel("Frequência (nº de países)")
        ax.set_title("Distribuição do total de turistas entre países")
        st.pyplot(fig)

with aba_disp:
    mes_x = st.selectbox("Mês para o eixo X:", options=meses, index=0, key="mes_scatter")
    dados_disp = df_filtrado[df_filtrado["Nivel"] == "País"]
    if dados_disp.empty:
        st.info("Sem dados de países para o gráfico de dispersão com os filtros atuais.")
    else:
        continentes_unicos = dados_disp["Continente"].unique().tolist()
        cores = plt.cm.tab10.colors
        fig, ax = novo_eixo()
        for i, cont in enumerate(continentes_unicos):
            sub = dados_disp[dados_disp["Continente"] == cont]
            ax.scatter(sub[mes_x], sub["Total"], label=cont, color=cores[i % len(cores)], alpha=0.8)
        ax.set_xlabel(f"Turistas em {mes_x}")
        ax.set_ylabel("Total anual de turistas")
        ax.set_title(f"Relação entre {mes_x} e o Total anual (por país)")
        legenda = ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
        for texto in legenda.get_texts():
            texto.set_color(cor_fonte)
        st.pyplot(fig, bbox_inches="tight")

st.divider()
st.caption(
    "Projeto acadêmico desenvolvido com Streamlit a partir de dados abertos do "
    "portal Data.Rio (https://www.data.rio)."
)
