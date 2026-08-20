import os

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots

# -------------------------------------------------------------------------------------
# Configuração Geral da Página
# -------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Painel COVID-19 Brasil 2025",
    page_icon="🦠",
    layout="wide",
)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ARQ_PARTE1 = os.path.join(DATA_DIR, "HIST_PAINEL_COVIDBR_2025_Parte1_05set2025.csv")
ARQ_PARTE2 = os.path.join(DATA_DIR, "HIST_PAINEL_COVIDBR_2025_Parte2_05set2025.csv")

REGIOES_ORDENADAS = ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]

# -------------------------------------------------------------------------------------
# Coordenadas Geográficas Curadas (para os exercícios 5 e 12)
# -------------------------------------------------------------------------------------
# O painel oficial do Ministério da Saúde NÃO traz latitude/longitude por município. 
# Para viabilizar as visualizações em mapa (st.map e PyDeck) sem depender de internet, 
# construí um dicionário curado com as coordenadas (aproximadas, de domínio público) das 
# capitais estaduais e de outros grandes municípios brasileiros, cobrindo as 5 regiões do país.
# Isso é suficiente para demonstrar a técnica de visualização geográfica, ainda que não cubra 
# os ~5.570 municípios do Brasil.

COORD_MUNICIPIOS = {
    # Região Norte
    ("AC", "Rio Branco"): (-9.9750, -67.8243),
    ("AP", "Macapá"): (0.0349, -51.0694),
    ("AM", "Manaus"): (-3.1190, -60.0217),
    ("AM", "Parintins"): (-2.6283, -56.7358),
    ("PA", "Belém"): (-1.4558, -48.4902),
    ("PA", "Ananindeua"): (-1.3656, -48.3722),
    ("RO", "Porto Velho"): (-8.7619, -63.9039),
    ("RR", "Boa Vista"): (2.8235, -60.6758),
    ("TO", "Palmas"): (-10.2491, -48.3243),
    # Região Nordeste
    ("MA", "São Luís"): (-2.5297, -44.3028),
    ("PI", "Teresina"): (-5.0892, -42.8019),
    ("CE", "Fortaleza"): (-3.7172, -38.5433),
    ("CE", "Juazeiro do Norte"): (-7.2130, -39.3155),
    ("RN", "Natal"): (-5.7945, -35.2110),
    ("PB", "João Pessoa"): (-7.1195, -34.8450),
    ("PE", "Recife"): (-8.0476, -34.8770),
    ("PE", "Jaboatão dos Guararapes"): (-8.1130, -35.0148),
    ("PE", "Olinda"): (-8.0089, -34.8553),
    ("AL", "Maceió"): (-9.6498, -35.7089),
    ("SE", "Aracaju"): (-10.9472, -37.0731),
    ("BA", "Salvador"): (-12.9777, -38.5016),
    ("BA", "Feira de Santana"): (-12.2667, -38.9667),
    # Região Sudeste
    ("MG", "Belo Horizonte"): (-19.9167, -43.9345),
    ("MG", "Uberlândia"): (-18.9186, -48.2772),
    ("MG", "Contagem"): (-19.9317, -44.0536),
    ("MG", "Juiz de Fora"): (-21.7642, -43.3503),
    ("ES", "Vitória"): (-20.3155, -40.3128),
    ("RJ", "Rio de Janeiro"): (-22.9068, -43.1729),
    ("RJ", "Niterói"): (-22.8833, -43.1036),
    ("RJ", "Duque de Caxias"): (-22.7856, -43.3117),
    ("RJ", "Nova Iguaçu"): (-22.7592, -43.4511),
    ("SP", "São Paulo"): (-23.5505, -46.6333),
    ("SP", "Campinas"): (-22.9099, -47.0626),
    ("SP", "Guarulhos"): (-23.4538, -46.5333),
    ("SP", "Santo André"): (-23.6639, -46.5383),
    ("SP", "São Bernardo do Campo"): (-23.6944, -46.5654),
    ("SP", "Osasco"): (-23.5329, -46.7918),
    ("SP", "Sorocaba"): (-23.5015, -47.4526),
    # Região Sul
    ("PR", "Curitiba"): (-25.4284, -49.2733),
    ("PR", "Londrina"): (-23.3103, -51.1628),
    ("PR", "Maringá"): (-23.4205, -51.9331),
    ("SC", "Florianópolis"): (-27.5954, -48.5480),
    ("RS", "Porto Alegre"): (-30.0346, -51.2177),
    ("RS", "Caxias do Sul"): (-29.1634, -51.1797),
    ("RS", "Pelotas"): (-31.7654, -52.3376),
    # Região Centro-Oeste
    ("DF", "Brasília"): (-15.7939, -47.8828),
    ("GO", "Goiânia"): (-16.6869, -49.2648),
    ("MT", "Cuiabá"): (-15.6014, -56.0979),
    ("MS", "Campo Grande"): (-20.4697, -54.6201),
}

# -------------------------------------------------------------------------------------
# Carga e Preparação dos Dados
# -------------------------------------------------------------------------------------
COLUNAS_USADAS = [
    "regiao", "estado", "municipio", "data", "semanaEpi",
    "populacaoTCU2019", "casosAcumulado", "casosNovos",
    "obitosAcumulado", "obitosNovos", "Recuperadosnovos", "emAcompanhamentoNovos",
]


@st.cache_data(show_spinner="Carregando base de dados da COVID-19 (arquivos grandes, aguarde)...")
def carregar_dados():
    """Lê os dois arquivos CSV oficiais e devolve um único DataFrame consolidado."""
    dtypes = {
        "regiao": "category",
        "estado": "category",
        "municipio": "string",
        "semanaEpi": "int16",
        "populacaoTCU2019": "float64",
        "casosAcumulado": "int64",
        "casosNovos": "int64",
        "obitosAcumulado": "int64",
        "obitosNovos": "int64",
        "Recuperadosnovos": "float64",
        "emAcompanhamentoNovos": "float64",
    }
    partes = []
    for caminho in (ARQ_PARTE1, ARQ_PARTE2):
        parte = pd.read_csv(
            caminho, sep=";", usecols=COLUNAS_USADAS, dtype=dtypes, parse_dates=["data"]
        )
        partes.append(parte)
    df = pd.concat(partes, ignore_index=True)
    return df


@st.cache_data
def separar_niveis(df):
    """Separa a tabela única nos 3 níveis geográficos empilhados na fonte original."""
    df_brasil = df[df["regiao"] == "Brasil"].sort_values("data").reset_index(drop=True)
    df_estados = (
        df[df["estado"].notna() & df["municipio"].isna()]
        .sort_values(["estado", "data"])
        .reset_index(drop=True)
    )
    df_municipios = (
        df[df["municipio"].notna()].sort_values(["estado", "municipio", "data"]).reset_index(drop=True)
    )
    return df_brasil, df_estados, df_municipios


@st.cache_data
def agregados_semanais_estado(df_estados):
    """Agrega os dados diários por (estado, semana epidemiológica).
    casosNovos/obitosNovos: somados na semana (são contagens diárias de novos casos).
    casosAcumulado/obitosAcumulado: valor máximo da semana (são contadores cumulativos)."""
    return (
        df_estados.groupby(["regiao", "estado", "semanaEpi"], as_index=False, observed=True)
        .agg(
            casosNovos=("casosNovos", "sum"),
            obitosNovos=("obitosNovos", "sum"),
            casosAcumulado=("casosAcumulado", "max"),
            obitosAcumulado=("obitosAcumulado", "max"),
        )
    )


@st.cache_data
def agregados_semanais_brasil(df_brasil):
    return (
        df_brasil.groupby("semanaEpi", as_index=False)
        .agg(
            casosNovos=("casosNovos", "sum"),
            obitosNovos=("obitosNovos", "sum"),
            casosAcumulado=("casosAcumulado", "max"),
            obitosAcumulado=("obitosAcumulado", "max"),
        )
    )


@st.cache_data
def agregados_semanais_regiao(df_estados):
    """Como a fonte não traz um nível 'região' pronto, somei os estados de cada região."""
    return (
        df_estados.groupby(["regiao", "semanaEpi"], as_index=False, observed=True)
        .agg(
            casosNovos=("casosNovos", "sum"),
            obitosNovos=("obitosNovos", "sum"),
            casosAcumulado=("casosAcumulado", "sum"),
        )
    )


df = carregar_dados()
df_brasil, df_estados, df_municipios = separar_niveis(df)

sem_estado = agregados_semanais_estado(df_estados)
sem_brasil = agregados_semanais_brasil(df_brasil)
sem_regiao = agregados_semanais_regiao(df_estados)

LISTA_ESTADOS = sorted(df_estados["estado"].dropna().unique().tolist())
DATA_MAX_ESTADOS = df_estados["data"].max()
DATA_MAX_MUNICIPIOS = df_municipios["data"].max()
SEMANA_MAIS_RECENTE = int(sem_estado["semanaEpi"].max())

# -------------------------------------------------------------------------------------
# Cabeçalho / Visão Geral
# -------------------------------------------------------------------------------------
st.title("🦠 Painel Interativo COVID-19 Brasil — 2025")
st.caption(
    "Fonte: Ministério da Saúde / Portal Coronavírus Brasil · "
    f"Período disponível: {df['data'].min().strftime('%d/%m/%Y')} a {df['data'].max().strftime('%d/%m/%Y')}"
)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Casos acumulados (Brasil)", f"{int(sem_brasil['casosAcumulado'].max()):,}".replace(",", "."))
col_b.metric("Óbitos acumulados (Brasil)", f"{int(sem_brasil['obitosAcumulado'].max()):,}".replace(",", "."))
col_c.metric("Semana epidemiológica mais recente", SEMANA_MAIS_RECENTE)
col_d.metric("UFs monitoradas", len(LISTA_ESTADOS))

st.divider()

# =====================================================================================
# Exercício 1 — Importância da Visualização de Dados
# =====================================================================================

st.header("1. Importância da Visualização de Dados na Pandemia")
st.markdown(
    """
A visualização de dados é o elo entre a enorme quantidade de registros brutos coletados
diariamente (neste conjunto de dados, cerca de 1,4 milhão de linhas, uma por
município/estado/Brasil a cada dia) e as decisões que gestores de saúde pública e a
população precisam tomar rapidamente.

- **Para gestores:** gráficos de tendência semanal (casos/óbitos) funcionam como um
  sistema de alerta precoce para reforçar leitos, insumos e campanhas de vacinação
  antes que uma onda se agrave. Mapas geográficos apontam onde concentrar recursos.

- **Para a população:** painéis simples tornam o risco local compreensível sem exigir
  leitura de tabelas, favorecendo adesão a medidas de proteção.

- **Para comparações:** boxplots e mapas de calor revelam desigualdades regionais que
  números isolados escondem, ajudando a formular políticas mais equitativas.

A visualização não substitui a análise estatística, mas é o que viabiliza uma
resposta rápida e baseada em evidências em um cenário de emergência sanitária.
"""
)

st.divider()

# =====================================================================================
# Exercício 2 — Gráfico de Barras com Streamlit
# =====================================================================================
# Comentário (Exercício 2): usei st.bar_chart sobre os dados agregados por semana
# epidemiológica (sem_estado). O estado padrão é SP, escolhido por ser o mais populoso
# do Brasil e o que concentra o maior volume absoluto de notificações, produzindo uma
# série mais rica para identificar picos e vales semanais. O usuário pode trocar o
# estado no seletor para comparar outros perfis.

st.header("2. Gráfico de Barras com Streamlit")

indice_sp = LISTA_ESTADOS.index("SP") if "SP" in LISTA_ESTADOS else 0
estado_barras = st.selectbox(
    "Escolha o estado:", LISTA_ESTADOS, index=indice_sp, key="sel_estado_barras"
)

dados_barras = (
    sem_estado[sem_estado["estado"] == estado_barras]
    .sort_values("semanaEpi")
    .set_index("semanaEpi")["casosNovos"]
)
st.bar_chart(dados_barras)

# Análise - Exercício 2:
st.markdown(
    f"""
**Análise:** o gráfico acima mostra a evolução semanal de casos novos notificados em
**{estado_barras}** ao longo de 2025 (semanas 1 a {SEMANA_MAIS_RECENTE}). Barras mais
altas indicam semanas de maior transmissão ou de maior volume de notificações
retroativas, quedas bruscas seguidas de picos costumam refletir atrasos de digitação
de exames (represamento) mais do que uma queda real na circulação do vírus, por isso
é sempre recomendável observar a tendência de várias semanas, e não uma semana isolada.
Escolhi **SP** como padrão por ser o estado mais populoso do país, o que garante uma
série com maior volume de dados e, portanto, mais informativa para este tipo de análise.
"""
)

st.divider()

# =====================================================================================
# Exercício 3 — Gráfico de Linha com Streamlit
# =====================================================================================
# Comentário (Exercício 3): usei st.line_chart sobre a série nacional (sem_brasil), com
# a coluna obitosAcumulado, que por definição é monotonicamente não decrescente.

st.header("3. Gráfico de Linha com Streamlit")

dados_linha = sem_brasil.sort_values("semanaEpi").set_index("semanaEpi")["obitosAcumulado"]
st.line_chart(dados_linha)

# Análise - Exercício 3:
st.markdown(
    """
**Análise:** por se tratar de um contador cumulativo, a curva nunca é
decrescente, ela apenas sobe ou permanece estável. O que importa observar é a
**inclinação** da curva em cada trecho:

- trechos **mais inclinados** (íngremes) indicam semanas com mais óbitos novos, ou
  seja, uma fase mais grave da pandemia;

- trechos **quase horizontais (patamares)** indicam semanas de baixa letalidade
  registrada, sugerindo estabilização ou controle da transmissão.

Assim, mesmo sem olhar diretamente para óbitos *novos*, a curva acumulada permite
identificar visualmente os períodos mais críticos apenas pela mudança de inclinação.
"""
)

st.divider()

# =====================================================================================
# Exercício 4 — Gráfico de Área com Streamlit
# =====================================================================================
# Comentário (Exercício 4): comparei três estados de regiões diferentes, SP (Sudeste),
# BA (Nordeste) e RS (Sul), para evidenciar diferenças de escala e de trajetória entre
# regiões distintas do país. Usei stack=False para que as áreas sejam sobrepostas
# (e não empilhadas), permitindo comparar diretamente os níveis de cada estado, já que
# empilhar somaria os valores e distorceria a leitura comparativa.

st.header("4. Gráfico de Área com Streamlit")

opcoes_padrao = [e for e in ["SP", "BA", "RS"] if e in LISTA_ESTADOS]
estados_area = st.multiselect(
    "Escolha até 3 estados para comparar:",
    LISTA_ESTADOS,
    default=opcoes_padrao,
    max_selections=3,
    key="sel_estados_area",
)

if len(estados_area) == 0:
    st.info("Selecione ao menos um estado para visualizar o gráfico de área.")
else:
    dados_area = (
        sem_estado[sem_estado["estado"].isin(estados_area)]
        .pivot(index="semanaEpi", columns="estado", values="casosAcumulado")
        .sort_index()
    )
    st.area_chart(dados_area, stack=False)

# Análise - Exercício 4:
    st.markdown(
        f"""
**Análise:** os estados selecionados ({", ".join(estados_area)}) pertencem a regiões
diferentes do país, o que evidencia diferenças de escala (estados mais populosos
concentram, em geral, mais casos em números absolutos) e de trajetória (a rapidez com
que os casos acumulados crescem ao longo das semanas). Diferenças no ritmo de
crescimento entre estados podem refletir densidade populacional, mobilidade urbana,
cobertura vacinal e capacidade de testagem/notificação de cada região, por isso
comparações em números absolutos devem ser complementadas, sempre que possível, por
indicadores relativos à população (ex.: casos por 100 mil habitantes).
"""
    )

st.divider()

# =====================================================================================
# Exercício 5 — Mapa com Streamlit
# =====================================================================================
# Comentário (Exercício 5): como a base do MS não traz coordenadas geográficas, usei o
# dicionário curado COORD_MUNICIPIOS (ver topo do arquivo) para localizar 5 municípios
# do estado de São Paulo, os cinco escolhidos estão entre os mais populosos do estado,
# o que garante bom volume de casos para tornar a comparação visual mais informativa.

st.header("5. Mapa com Streamlit")

municipios_sp_mapa = ["São Paulo", "Campinas", "Guarulhos", "Santo André", "São Bernardo do Campo"]

mapa_df = df_municipios[
    (df_municipios["estado"] == "SP")
    & (df_municipios["municipio"].isin(municipios_sp_mapa))
    & (df_municipios["data"] == DATA_MAX_MUNICIPIOS)
][["municipio", "casosAcumulado", "populacaoTCU2019"]].copy()

mapa_df["lat"] = mapa_df["municipio"].apply(lambda m: COORD_MUNICIPIOS[("SP", m)][0])
mapa_df["lon"] = mapa_df["municipio"].apply(lambda m: COORD_MUNICIPIOS[("SP", m)][1])
# Escala visual do marcador (o parâmetro "size" do st.map espera um raio em metros,
# por isso dividimos o número absoluto de casos por um fator para um resultado legível)
mapa_df["tamanho_marcador"] = mapa_df["casosAcumulado"] / 6

st.map(mapa_df, latitude="lat", longitude="lon", size="tamanho_marcador")
st.dataframe(
    mapa_df[["municipio", "casosAcumulado", "populacaoTCU2019"]]
    .sort_values("casosAcumulado", ascending=False)
    .rename(columns={"populacaoTCU2019": "populacao"}),
    width='stretch',
    hide_index=True,
)

# Análise - Exercício 5:
st.markdown(
    """
**Análise:** a visualização geográfica permite identificar rapidamente, em um único
olhar, quais municípios concentram o maior número absoluto de casos, normalmente os
mais populosos e/ou com maior mobilidade urbana (como São Paulo e Guarulhos). Esse
tipo de mapa é útil para gestores planejarem a alocação geográfica de recursos
(equipes móveis de vacinação, testagem, leitos), pois localiza fisicamente onde a
demanda é maior, algo que uma tabela numérica não comunica de forma tão imediata.
"""
)

st.divider()

# =====================================================================================
# Exercíco 6 — Visualização com Matplotlib
# =====================================================================================
# Comentário (Exercício 6): como casos novos e óbitos novos têm escalas muito diferentes
# (milhares vs. dezenas/centenas), usei duas escalas no mesmo gráfico (eixo Y duplo - twinx): 
# Barras para casos novos (eixo esquerdo) e uma linha com marcadores para óbitos novos (eixo direito), 
# ordenados pelo volume de casos.

st.header("6. Visualização com Matplotlib")

dados_recentes = (
    sem_estado[sem_estado["semanaEpi"] == SEMANA_MAIS_RECENTE]
    .sort_values("casosNovos", ascending=False)
    .reset_index(drop=True)
)

fig_mpl, ax1 = plt.subplots(figsize=(14, 6))
cor_casos = "tab:blue"
cor_obitos = "tab:red"

ax1.bar(dados_recentes["estado"], dados_recentes["casosNovos"], color=cor_casos, label="Casos novos")
ax1.set_xlabel("Estado (UF)")
ax1.set_ylabel("Casos novos", color=cor_casos)
ax1.tick_params(axis="y", labelcolor=cor_casos)
ax1.tick_params(axis="x", rotation=90)

ax2 = ax1.twinx()
ax2.plot(
    dados_recentes["estado"],
    dados_recentes["obitosNovos"],
    color=cor_obitos,
    marker="o",
    linewidth=2,
    label="Óbitos novos",
)
ax2.set_ylabel("Óbitos novos", color=cor_obitos)
ax2.tick_params(axis="y", labelcolor=cor_obitos)

fig_mpl.suptitle(f"Casos novos vs. óbitos novos por estado — Semana Epidemiológica {SEMANA_MAIS_RECENTE}")
fig_mpl.tight_layout()
st.pyplot(fig_mpl)

# Análise - Exercício 6:
st.markdown(
    """
**Análise:** de modo geral, os estados com maior número de casos novos também tendem a
registrar mais óbitos novos em termos absolutos, o que é esperado (mais infecções
tendem a gerar mais óbitos). Porém, a proporção entre as duas curvas não é constante:
alguns estados com muitos casos apresentam relativamente poucos óbitos na mesma semana
(o que pode refletir maior cobertura vacinal, população mais jovem ou perfil de
notificação diferente), enquanto outros mostram óbitos desproporcionalmente altos em
relação aos casos novos, um sinal de alerta que merece investigação (subnotificação
de casos leves, sobrecarga hospitalar ou atraso na confirmação de óbitos).
"""
)

st.divider()

# =====================================================================================
# Exercício 7 — Boxplot com Seaborn
# =====================================================================================
# Comentário (Exercício 7): usei os dados agregados por região/semana (sem_regiao) e
# comparei a distribuição semanal de casos novos entre Norte, Nordeste e Sudeste,
# conforme pedido no enunciado.

st.header("7. Boxplot com Seaborn)")

regioes_boxplot = ["Norte", "Nordeste", "Sudeste"]
dados_boxplot = sem_regiao[sem_regiao["regiao"].isin(regioes_boxplot)]

fig_box, ax_box = plt.subplots(figsize=(9, 5.5))
sns.boxplot(
    data=dados_boxplot,
    x="regiao",
    y="casosNovos",
    order=regioes_boxplot,
    hue="regiao",
    palette="Set2",
    legend=False,
    ax=ax_box,
)
ax_box.set_xlabel("Região")
ax_box.set_ylabel("Casos novos (por semana epidemiológica)")
ax_box.set_title("Distribuição semanal de casos novos por região — 2025")
st.pyplot(fig_box)

# Análise - Exercício 7:
st.markdown(
    """
**Análise:** o boxplot evidencia diferenças tanto de **nível** quanto de
**variabilidade** entre as regiões. O Sudeste, por concentrar os estados mais
populosos do país (SP, RJ, MG, ES), tende a apresentar mediana e amplitude
(caixa maior, mais outliers) bem superiores às demais regiões. Norte e Nordeste,
embora tenham geografias e contextos socioeconômicos distintos, costumam apresentar
medianas mais baixas e menor dispersão, o que pode refletir tanto uma transmissão
menos intensa quanto uma cobertura de testagem/notificação mais limitada. Outliers
indicam semanas atípicas, picos pontuais de notificação que fogem do padrão observado 
no restante do período.
"""
)

st.divider()

# =====================================================================================
# Exercício 8 — Gráfico de Área com Altair
# =====================================================================================
# Comentário (Exercício 8): escolhi a região Nordeste, por ser a segunda mais populosa
# do país e apresentar uma dinâmica epidemiológica distinta do eixo Sul/Sudeste,
# oferecendo um contraste interessante em relação aos exercícios anteriores.

st.header("8. Gráfico de Área com Altair")

regiao_area_altair = st.selectbox(
    "Escolha a região:",
    REGIOES_ORDENADAS,
    index=REGIOES_ORDENADAS.index("Nordeste"),
    key="sel_regiao_area_altair",
)

dados_area_altair = sem_regiao[sem_regiao["regiao"] == regiao_area_altair].sort_values("semanaEpi")

grafico_area_altair = (
    alt.Chart(dados_area_altair)
    .mark_area(opacity=0.65, interpolate="monotone", color="#2E86AB")
    .encode(
        x=alt.X("semanaEpi:Q", title="Semana Epidemiológica"),
        y=alt.Y("casosNovos:Q", title="Casos Novos"),
        tooltip=[
            alt.Tooltip("semanaEpi:Q", title="Semana"),
            alt.Tooltip("casosNovos:Q", title="Casos novos", format=","),
        ],
    )
    .properties(height=400)
    .interactive()
)
st.altair_chart(grafico_area_altair, width='stretch')

# Análise textual (item 8):
st.markdown(
    f"""
**Análise:** o gráfico de área mostra a evolução semanal de casos novos na região
**{regiao_area_altair}** ao longo de 2025. Picos indicam semanas de maior transmissão
(ou de maior volume de notificações retroativas), enquanto vales sugerem períodos de
menor circulação viral ou de subnotificação (por exemplo, em torno de feriados
prolongados, quando o sistema de notificação tende a operar com atraso).
"""
)

st.divider()

# =====================================================================================
# Exercício 9 — Heatmap com Altair
# =====================================================================================
# Comentário (Exercício 9): a base de dados do Ministério da Saúde utilizada aqui NÃO
# contém informação sobre leitos hospitalares ocupados. As colunas "Recuperadosnovos" e
# "emAcompanhamentoNovos", que existem na planilha, praticamente não são mais preenchidas
# em 2025 (eram usadas sobretudo em 2020/2021 e aqui estão quase 100% vazias), então também
# não servem como proxy. Por isso, seguindo a ressalva do próprio enunciado ("caso os dados 
# estejam disponíveis"), usei como indicadores de carga assistencial as quatro variáveis
# numéricas que de fato estão consistentemente preenchidas: casos novos, óbitos novos, 
# casos acumulados e óbitos acumulados.

st.header("9. Heatmap com Altair")
st.caption(
    "⚠️ A base de dados não contém informação de leitos hospitalares ocupados, e as "
    "colunas de 'recuperados'/'em acompanhamento' estão praticamente vazias em 2025. "
    "Por isso, utilizei como indicadores disponíveis: casos novos, óbitos novos, "
    "casos acumulados e óbitos acumulados."
)

estado_heatmap = st.selectbox(
    "Escolha o estado:", LISTA_ESTADOS, index=indice_sp, key="sel_estado_heatmap"
)
 
colunas_corr = ["casosNovos", "obitosNovos", "casosAcumulado", "obitosAcumulado"]
nomes_amigaveis = {
    "casosNovos": "Casos novos",
    "obitosNovos": "Óbitos novos",
    "casosAcumulado": "Casos acumulados",
    "obitosAcumulado": "Óbitos acumulados",
}
 
dados_corr = df_estados[df_estados["estado"] == estado_heatmap][colunas_corr].rename(columns=nomes_amigaveis)
matriz_corr = dados_corr.corr().reset_index().melt(id_vars="index")
matriz_corr.columns = ["Variável 1", "Variável 2", "Correlação"]
 
base_heatmap = alt.Chart(matriz_corr).encode(
    x=alt.X("Variável 1:N", title=None),
    y=alt.Y("Variável 2:N", title=None),
)
retangulos = base_heatmap.mark_rect().encode(
    color=alt.Color("Correlação:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]))
)
rotulos = base_heatmap.mark_text(baseline="middle").encode(
    text=alt.Text("Correlação:Q", format=".2f"),
    color=alt.condition(
        "abs(datum.Correlação) > 0.5", alt.value("white"), alt.value("black")
    ),
)
st.altair_chart((retangulos + rotulos).properties(height=380), width='stretch')

# Análise - Exercício 9:
st.markdown(
    f"""
**Análise:** em **{estado_heatmap}**, espera-se uma correlação positiva forte entre
*casos novos* e *óbitos novos* (mais infecções tendem a anteceder mais óbitos, com
alguma defasagem de dias/semanas não capturada aqui), e uma correlação positiva quase
perfeita entre as variáveis *acumuladas* entre si (já que ambas só crescem ao longo do
tempo). A correlação entre uma variável "nova" (casosNovos/obitosNovos) e sua
contraparte "acumulada" tende a ser mais moderada, pois a acumulada carrega o efeito
de todo o histórico, enquanto a nova reflete apenas o dia/semana em questão. Vale
reforçar que correlação não implica causalidade. Mudanças na testagem, feriados e
revisões de metodologia também afetam essas séries.
"""
)

st.divider()

# =====================================================================================
# Exercício 10 — Gráfico de Pizza com Plotly
# =====================================================================================
# Comentário (item 10): somamos os casos acumulados de todos os estados de cada região
# na data mais recente disponível, e calculamos a participação percentual de cada
# região no total nacional.

st.header("10. Gráfico de Pizza com Plotly")

dados_pizza = (
    df_estados[df_estados["data"] == DATA_MAX_ESTADOS]
    .groupby("regiao", as_index=False, observed=True)["casosAcumulado"]
    .sum()
)
dados_pizza["regiao"] = dados_pizza["regiao"].astype(str)

fig_pizza = px.pie(
    dados_pizza,
    names="regiao",
    values="casosAcumulado",
    category_orders={"regiao": REGIOES_ORDENADAS},
    color_discrete_sequence=px.colors.qualitative.Set2,
    hole=0.35,
    title=f"Participação de cada região nos casos acumulados — {DATA_MAX_ESTADOS.strftime('%d/%m/%Y')}",
)
fig_pizza.update_traces(textposition="inside", textinfo="percent+label")
st.plotly_chart(fig_pizza, width='stretch')

# Análise - Exercício 10:
st.markdown(
    """
**Análise:** a região **Sudeste** concentra a maior fatia dos casos acumulados do
país, refletindo, sobretudo, sua enorme densidade populacional (é a região mais
populosa do Brasil, reunindo os estados de SP, RJ, MG e ES) e sua alta capacidade de
testagem e notificação. Isso não significa necessariamente que a transmissão per
capita seja mais alta ali, apenas que, em números absolutos, mais pessoas testadas
e mais casos confirmados são esperados onde há mais habitantes. Para uma comparação
justa entre regiões, o ideal seria complementar esta visão com indicadores relativos
(casos por 100 mil habitantes), o que reduz o viés populacional.
"""
)

st.divider()

# =====================================================================================
# Exercício 11 — Subplots com Plotly
# =====================================================================================
# Comentário (Exercício 11): construí 2 subplots lado a lado (um por região), cada um
# com eixo Y duplo (secondary_y) para acomodar as diferentes escalas de casos novos e
# óbitos novos na mesma área do gráfico.

st.header("11. Subplots com Plotly")

col_r1, col_r2 = st.columns(2)
with col_r1:
    regiao_1 = st.selectbox(
        "Região 1:", REGIOES_ORDENADAS, index=REGIOES_ORDENADAS.index("Sudeste"), key="sel_regiao1"
    )
with col_r2:
    regiao_2 = st.selectbox(
        "Região 2:", REGIOES_ORDENADAS, index=REGIOES_ORDENADAS.index("Nordeste"), key="sel_regiao2"
    )

fig_subplots = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[regiao_1, regiao_2],
    specs=[[{"secondary_y": True}, {"secondary_y": True}]],
)

for coluna_idx, regiao_escolhida in enumerate((regiao_1, regiao_2), start=1):
    dados_reg = sem_regiao[sem_regiao["regiao"] == regiao_escolhida].sort_values("semanaEpi")
    fig_subplots.add_trace(
        go.Bar(
            x=dados_reg["semanaEpi"],
            y=dados_reg["casosNovos"],
            name=f"Casos novos",
            marker_color="steelblue",
            showlegend=(coluna_idx == 1),
        ),
        row=1,
        col=coluna_idx,
        secondary_y=False,
    )
    fig_subplots.add_trace(
        go.Scatter(
            x=dados_reg["semanaEpi"],
            y=dados_reg["obitosNovos"],
            name=f"Óbitos novos",
            mode="lines+markers",
            marker_color="indianred",
            showlegend=(coluna_idx == 1),
        ),
        row=1,
        col=coluna_idx,
        secondary_y=True,
    )

fig_subplots.update_xaxes(title_text="Semana Epidemiológica")
fig_subplots.update_yaxes(title_text="Casos novos", secondary_y=False)
fig_subplots.update_yaxes(title_text="Óbitos novos", secondary_y=True)
fig_subplots.update_layout(height=500, legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig_subplots, width='stretch')

# Análise - Exercício 11:
st.markdown(
    f"""
**Análise:** comparando **{regiao_1}** e **{regiao_2}** lado a lado, é possível notar
diferenças tanto na magnitude (número absoluto de casos e óbitos, refletindo o
tamanho populacional de cada região) quanto no formato da curva ao longo do ano
(semanas em que uma região apresenta pico enquanto a outra está estável, o que pode
indicar diferentes calendários de eventos de aglomeração, campanhas de vacinação ou
condições climáticas locais). O uso de eixo Y duplo em cada subplot permite enxergar,
na mesma região, se os óbitos novos acompanham de perto os picos de casos novos ou se
apresentam uma defasagem, um sinal indireto do tempo entre infecção e óbito.
"""
)

st.divider()

# =====================================================================================
# Exercício 12 — Mapa Interativo com PyDeck
# =====================================================================================
# Comentário (item 12): Calculei a "incidência por 100 mil habitantes", 
# que é justamente uma medida de casos AJUSTADA pela população. 
# Ao contrário do número absoluto de casos, ela permite comparar municípios de tamanhos 
# muito diferentes em pé de igualdade. Usei essa incidência para definir o tamanho e a cor 
# dos círculos no mapa PyDeck. Assim como no item 5, como a base não traz coordenadas, 
# usei o dicionário curado COORD_MUNICIPIOS, aqui restrito aos municípios da região Sudeste.

st.header("12. Mapa Interativo com PyDeck")
st.caption("Região Sudeste")

municipios_sudeste = [m for (uf, m) in COORD_MUNICIPIOS.keys() if uf in ("SP", "RJ", "MG", "ES")]
estados_sudeste = ["SP", "RJ", "MG", "ES"]

dados_pydeck = df_municipios[
    (df_municipios["estado"].isin(estados_sudeste))
    & (df_municipios["municipio"].isin(municipios_sudeste))
    & (df_municipios["data"] == DATA_MAX_MUNICIPIOS)
][["estado", "municipio", "casosAcumulado", "populacaoTCU2019"]].copy()

dados_pydeck["lat"] = dados_pydeck.apply(
    lambda linha: COORD_MUNICIPIOS[(linha["estado"], linha["municipio"])][0], axis=1
)
dados_pydeck["lon"] = dados_pydeck.apply(
    lambda linha: COORD_MUNICIPIOS[(linha["estado"], linha["municipio"])][1], axis=1
)
# Incidência ajustada pela população (casos por 100 mil habitantes)
dados_pydeck["incidencia_100k"] = (
    dados_pydeck["casosAcumulado"] / dados_pydeck["populacaoTCU2019"] * 100_000
).round(1)
# Raio do círculo em metros, escalado a partir da incidência (ajuste visual)
dados_pydeck["raio_m"] = dados_pydeck["incidencia_100k"] * 40
# Intensidade de cor (canal vermelho) proporcional à incidência, normalizada 0-255
max_incidencia = dados_pydeck["incidencia_100k"].max()
dados_pydeck["cor_r"] = (dados_pydeck["incidencia_100k"] / max_incidencia * 200 + 55).astype(int)

camada_pydeck = pdk.Layer(
    "ScatterplotLayer",
    data=dados_pydeck,
    get_position=["lon", "lat"],
    get_radius="raio_m",
    get_fill_color="[cor_r, 60, 120, 160]",
    pickable=True,
    stroked=True,
    get_line_color=[255, 255, 255],
    line_width_min_pixels=1,
)

vista_inicial = pdk.ViewState(latitude=-21.5, longitude=-44.5, zoom=5.3, pitch=30)

st.pydeck_chart(
    pdk.Deck(
        layers=[camada_pydeck],
        initial_view_state=vista_inicial,
        tooltip={
            "text": "{municipio} ({estado})\nIncidência: {incidencia_100k} casos/100mil hab.\nCasos acumulados: {casosAcumulado}"
        },
    )
)

st.dataframe(
    dados_pydeck[["estado", "municipio", "casosAcumulado", "populacaoTCU2019", "incidencia_100k"]]
    .sort_values("incidencia_100k", ascending=False)
    .rename(columns={"populacaoTCU2019": "populacao"}),
    width='stretch',
    hide_index=True,
)

# Análise - Exercício 12:
st.markdown(
    """
**Análise:** ao ajustar os casos acumulados pela população de cada município
(incidência por 100 mil habitantes), municípios menores porém com alta circulação
relativa do vírus aparecem em destaque no mapa mesmo tendo poucos casos em números
absolutos, o que não aconteceria em um mapa baseado apenas na contagem bruta. A
densidade populacional tende a favorecer a disseminação da COVID-19 por facilitar o
contato próximo e prolongado entre pessoas, por isso é comum observarmos incidência elevada
tanto nas grandes metrópoles quanto em municípios menores e mais adensados que fazem
parte de suas regiões metropolitanas.
"""
)

st.divider()
st.caption(
    "Painel construído com Streamlit, Matplotlib, Seaborn, Altair, Plotly e PyDeck · "
    "Dados: Ministério da Saúde / Portal Coronavírus Brasil."
)
