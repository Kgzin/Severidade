"""Ferramenta de análise de severidade de doenças foliares — PlantCV + Streamlit.

Executar:  .venv\\Scripts\\streamlit run app.py
"""

from dataclasses import asdict
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from analise import diagramatica, foto
from analise.escala import ESCALAS, escala_de_texto

EXEMPLOS = Path(__file__).parent / "exemplos"
MODO_DIAG = "Escala diagramática / desenho P&B"
MODO_FOTO = "Foto de folha (qualquer doença)"
SEM_ESCALA = "Nenhuma (apenas % de severidade)"

st.set_page_config(page_title="Severidade foliar · PlantCV", page_icon="🌿", layout="wide")


# ---------------------------------------------------------------- utilidades
def decodificar(conteudo: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(conteudo, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Formato de imagem não reconhecido.")
    return img


@st.cache_data(show_spinner=False, max_entries=16)  # limita a memória no servidor
def rodar_diagramatica(conteudo: bytes, params: dict):
    return diagramatica.analisar(decodificar(conteudo), diagramatica.ParamsDiagramatica(**params))


@st.cache_data(show_spinner=False, max_entries=16)  # limita a memória no servidor
def rodar_foto(conteudo: bytes, params: dict):
    return foto.analisar(decodificar(conteudo), foto.ParamsFoto(**params))


def fmt(v: float, casas: int = 2) -> str:
    return f"{v:.{casas}f}".replace(".", ",")


def png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def csv_br(df: pd.DataFrame) -> bytes:
    # ';' e vírgula decimal abrem direto no Excel em português
    return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def grafico_medido_vs_ref(df: pd.DataFrame) -> alt.LayerChart:
    lim = [min(df["Referência (%)"].min(), df["Medida (%)"].min()) * 0.7,
           max(df["Referência (%)"].max(), df["Medida (%)"].max()) * 1.4]
    escala = alt.Scale(type="log", domain=lim, nice=False)
    diagonal = alt.Chart(pd.DataFrame({"x": lim, "y": lim})).mark_line(
        color="#9aa0a6", strokeDash=[4, 4], strokeWidth=1
    ).encode(x=alt.X("x:Q", scale=escala), y=alt.Y("y:Q", scale=escala))
    base = alt.Chart(df).encode(
        x=alt.X("Referência (%):Q", scale=escala, title="Severidade de referência (%) — log"),
        y=alt.Y("Medida (%):Q", scale=escala, title="Severidade medida (%) — log"),
        tooltip=["Folha", alt.Tooltip("Referência (%):Q", format=".2f"),
                 alt.Tooltip("Medida (%):Q", format=".2f"), "Nota atribuída"],
    )
    pontos = base.mark_circle(size=90, color="#2a6fdb", opacity=0.9, stroke="white", strokeWidth=2)
    rotulos = base.mark_text(dx=9, dy=-9, fontSize=11, color="#5f6368").encode(text="Folha:N")
    return (diagonal + pontos + rotulos).properties(height=340).configure_axis(
        gridColor="#e8eaed", domainColor="#bdc1c6", labelColor="#5f6368", titleColor="#5f6368"
    )


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Configuração")
    modo = st.radio("Tipo de imagem", [MODO_FOTO, MODO_DIAG],
                    help="Fotos: qualquer folha sobre fundo uniforme; os sintomas são destacados "
                         "em preto e branco. Desenhos/escalas: lesões pretas sobre folha branca.")

    st.subheader("Escala de referência")
    culturas = ["Todas"] + sorted({n.split(" – ")[0] for n in ESCALAS})
    cultura = st.selectbox("Cultura", culturas,
                           index=0 if modo == MODO_FOTO else culturas.index("Cana"))
    opcoes = [n for n in ESCALAS if cultura == "Todas" or n.startswith(cultura + " –")]
    padrao = "Cana – Ferrugem alaranjada (Puccinia kuehnii)"
    lista = [SEM_ESCALA] + opcoes + ["Personalizada"]
    nome_escala = st.selectbox("Escala", lista,
                               index=lista.index(padrao) if modo == MODO_DIAG and padrao in lista else 0)
    if nome_escala == SEM_ESCALA:
        escala = None
    elif nome_escala == "Personalizada":
        texto = st.text_input("Severidades das notas (%; separadas por ';')", "0,5; 1; 5; 10; 25; 50")
        try:
            escala = escala_de_texto("Personalizada", texto)
        except ValueError as e:
            st.error(f"Escala inválida: {e}")
            st.stop()
    else:
        escala = ESCALAS[nome_escala]
    if escala:
        st.caption("Notas: " + escala.descricao_notas())
        if escala.referencia:
            with st.expander("Referência da escala"):
                st.caption(escala.referencia)

    st.subheader("Parâmetros")
    if modo == MODO_DIAG:
        medida = st.radio("Medida principal", ["Binária (limiar)", "Ponderada (intensidade)"],
                          help="Binária: pixel é lesão se mais escuro que o limiar. "
                               "Ponderada: cada pixel conta proporcionalmente à sua escuridão "
                               "(útil para figuras digitalizadas com bordas suavizadas).")
        pd_ = diagramatica.ParamsDiagramatica()
        with st.expander("Avançado"):
            pd_.limiar_lesao = st.slider("Limiar de lesão (cinza)", 50, 230, pd_.limiar_lesao)
            pd_.limiar_tinta = st.slider("Limiar de tinta/fundo (cinza)", 150, 250, pd_.limiar_tinta)
            pd_.excluir_nervura = st.checkbox("Excluir nervura central", pd_.excluir_nervura)
            pd_.frac_altura_folha = st.slider("Altura mínima da folha (fração da imagem)",
                                              0.05, 0.9, pd_.frac_altura_folha)
            pd_.frac_altura_linha = st.slider("Comprimento mínimo de linha vertical (fração)",
                                              0.05, 0.9, pd_.frac_altura_linha)
            pd_.margem_linha = st.slider("Margem das linhas (px)", 0, 6, pd_.margem_linha)
        params = asdict(pd_)
    else:
        pf = foto.ParamsFoto()
        with st.expander("Detecção de sintomas", expanded=True):
            if st.checkbox("Limiar automático (relativo ao verde da folha)", True,
                           help="O verde do tecido sadio muda com câmera, luz e cultivar. "
                                "O limiar é o verde típico da folha (percentil 25 do canal 'a') "
                                "mais a margem abaixo."):
                pf.margem_verde = st.slider("Margem acima do verde sadio", 6, 35, pf.margem_verde,
                                            help="Menor = mais sensível (capta halos e cloroses leves). "
                                                 "Maior = só lesões nítidas; aumente se nervuras ou "
                                                 "tecido sadio estiverem sendo marcados.")
            else:
                pf.limiar_a = st.slider("Limiar fixo (canal 'a' do LAB)", 100, 140, 118,
                                        help="Pixel com 'a' ≥ limiar deixou de ser verde e é sintoma. "
                                             "128 = sem cor; tecido sadio fica tipicamente em 95–112.")
            pf.refinar = st.checkbox("Detalhe fino das manchas", pf.refinar,
                                     help="Usa a luminância (resolução cheia) para recortar cada "
                                          "pústula/mancha; a cor da foto tem só metade da resolução "
                                          "e, sozinha, gera manchas 'inchadas'.")
            if pf.refinar:
                pf.contraste_escuro = st.slider("Contraste mínimo da mancha", 0.0, 10.0,
                                                pf.contraste_escuro, 0.5,
                                                help="Quanto a mancha precisa ser mais escura que o "
                                                     "tecido em volta. Menor = mais manchas pequenas.")
            if not st.checkbox("Ampliar imagens pequenas", True,
                               help=f"Imagens com lado menor que {pf.resolucao_min} px são ampliadas "
                                    "antes da análise: contornos mais precisos. Áreas continuam em "
                                    "px da imagem original."):
                pf.resolucao_min = 0
            pf.area_min_lesao = st.number_input("Área mínima de lesão (px)", 0, 5000, pf.area_min_lesao)
            pf.borda_px = st.slider("Margem da folha ignorada (px)", 0, 10, pf.borda_px,
                                    help="Pixels da borda misturam folha e fundo e podem parecer lesão.")
        with st.expander("Segmentação da folha"):
            fundos = {"Automático": "auto", "Uniforme (escaneada / papel)": "uniforme",
                      "Complexo (foto de campo)": "complexo"}
            pf.modo_fundo = fundos[st.selectbox(
                "Tipo de fundo", list(fundos),
                help="Uniforme: folha separada pela diferença de cor para o fundo (pega lesões na "
                     "margem). Complexo: folha encontrada pela cor verde, para fotos com vasos, solo, "
                     "outras plantas ou folha encostando na borda. Automático decide pela borda da imagem.")]
            maior = st.selectbox("Folhas analisadas", ["Automático", "Só a maior", "Todas"],
                                 help="Automático: todas no fundo uniforme; só a maior no complexo.")
            pf.so_maior_folha = {"Automático": None, "Só a maior": True, "Todas": False}[maior]
            pf.remover_peciolo = st.checkbox("Remover pecíolo", pf.remover_peciolo,
                                             help="Exclui estruturas finas presas à folha (pecíolo, "
                                                  "pedaço de caule) da área foliar.")
            auto = st.checkbox("Limiar automático (Otsu)", True)
            if not auto:
                pf.limiar_folha = st.slider("Limiar da folha", 0, 150, 40,
                                            help="Fundo uniforme: diferença mínima de cor para o fundo. "
                                                 "Fundo complexo: valor máximo de 'a' do tecido verde.")
            pf.frac_area_folha = st.slider("Área mínima de uma folha (% da imagem)", 0.1, 20.0,
                                           pf.frac_area_folha * 100, 0.1) / 100
            pf.area_min_ruido = st.number_input("Área mínima de objeto (px)", 0, 100000,
                                                pf.area_min_ruido)
        pf.max_numeros = st.slider("Manchas numeradas na imagem", 5, 300, pf.max_numeros,
                                   help="Só as N maiores recebem número, para a imagem continuar "
                                        "legível; a tabela lista todas.")
        params = asdict(pf)


# ---------------------------------------------------------------- entrada
st.title("🌿 Análise de severidade de doenças foliares")
st.caption("Segmentação com PlantCV · sintomas destacados em preto e branco · "
           "% de área foliar lesionada e nota na escala diagramática")

arquivos = st.file_uploader("Imagens (PNG, JPG, TIF)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
                            accept_multiple_files=True)
entradas: list[tuple[str, bytes]] = [(a.name, a.getvalue()) for a in arquivos or []]
if modo == MODO_DIAG:
    exemplo = "escala_ferrugem_alaranjada.png"
    if st.checkbox(f"Usar imagem de exemplo ({exemplo})", value=not arquivos) and (EXEMPLOS / exemplo).exists():
        entradas.insert(0, (exemplo, (EXEMPLOS / exemplo).read_bytes()))
if not entradas:
    st.info("Envie uma ou mais imagens para começar.")
    st.stop()


# ---------------------------------------------------------------- análise
resumo: list[dict] = []

for nome, conteudo in entradas:
    st.divider()
    st.subheader(nome)
    try:
        with st.spinner("Analisando…"):
            res = rodar_diagramatica(conteudo, params) if modo == MODO_DIAG else rodar_foto(conteudo, params)
    except Exception as e:  # imagem corrompida ou parâmetros que não segmentam nada
        st.error(f"Falha ao analisar: {e}")
        continue

    for aviso in res.avisos:
        st.warning(aviso)

    if modo == MODO_DIAG:
        col_a, col_b = st.columns(2)
        col_a.image(conteudo, caption="Original", width="stretch")
        col_b.image(res.sobreposicao, channels="BGR", width="stretch",
                    caption="Verde: limbo · vermelho: lesão · azul: contorno/nervura")
    else:
        col_a, col_b, col_c = st.columns(3)
        col_a.image(conteudo, caption="Original", width="stretch")
        col_b.image(res.preto_branco, width="stretch", caption="Preto e branco: sintomas em preto")
        col_c.image(res.sobreposicao, channels="BGR", width="stretch",
                    caption="Contorno laranja: folha · magenta: sintoma")
        col_b.download_button("Baixar P&B (PNG)", png(res.preto_branco),
                              f"{Path(nome).stem}_pb.png", "image/png", key=f"pb_{nome}")

    if modo == MODO_DIAG:
        ponderada = medida.startswith("Ponderada")
        linhas = []
        for f in res.folhas:
            sev = f.severidade_ponderada if ponderada else f.severidade_binaria
            nota, ref_nota = escala.classificar(sev) if escala else (None, None)
            linhas.append({
                "Imagem": nome, "Folha": f.indice,
                "Medida (%)": round(sev, 3),
                "Binária (%)": round(f.severidade_binaria, 3),
                "Ponderada (%)": round(f.severidade_ponderada, 3),
                "Nota atribuída": nota, "Nota (% ref.)": ref_nota,
                "Nº lesões": f.n_lesoes, "Área limbo (px)": f.area_limbo_px,
                "Área lesão (px)": f.area_lesao_px,
            })
        df = pd.DataFrame(linhas)

        # Se o nº de folhas bate com o nº de notas, é a própria escala: compara posição a posição.
        if escala and len(df) == len(escala.valores):
            df["Referência (%)"] = escala.valores
            df["Acerto"] = df["Nota atribuída"] == df["Folha"]
            acertos = int(df["Acerto"].sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Folhas detectadas", len(df))
            c2.metric("Notas corretas", f"{acertos}/{len(df)}")
            erro = np.mean(np.abs(np.log10(df["Medida (%)"].clip(lower=1e-3) / df["Referência (%)"])))
            c3.metric("Erro médio (fator)", f"×{fmt(10 ** erro)}",
                      help="Média geométrica da razão medida/referência (1,00 = perfeito).")
            st.altair_chart(grafico_medido_vs_ref(df), width="stretch")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Folhas detectadas", len(df))
            c2.metric("Severidade média (%)", fmt(df["Medida (%)"].mean()))

        if not escala:
            df = df.drop(columns=["Nota atribuída", "Nota (% ref.)"])
        st.dataframe(df, hide_index=True, width="stretch")
        resumo.extend(df.to_dict("records"))

    else:
        cols = st.columns(4 if escala else 3)
        cols[0].metric("Severidade" + (" (todas as folhas)" if len(res.folhas) > 1 else ""),
                       f"{fmt(res.severidade)}%")
        cols[1].metric("Nº de lesões", res.n_lesoes)
        cols[2].metric("Área média da lesão", f"{fmt(res.area_media_lesao_px, 1)} px")
        if escala:
            nota, ref_nota = escala.classificar(res.severidade)
            cols[3].metric("Nota na escala", f"{nota}", help=f"Valor de referência: {fmt(ref_nota)}%")

        linhas = []
        for f in res.folhas:
            linha = {"Imagem": nome, "Folha": f.indice, "Severidade (%)": round(f.severidade, 3)}
            if escala:
                linha["Nota atribuída"], linha["Nota (% ref.)"] = escala.classificar(f.severidade)
            linha |= {"Nº lesões": f.n_lesoes, "Área média lesão (px)": round(f.area_media_lesao_px, 1),
                      "Área folha (px)": f.area_folha_px, "Área lesão (px)": f.area_lesao_px,
                      "Limiar 'a'": f.limiar_a}
            linhas.append(linha)
        st.caption(f"Fundo detectado: **{res.modo_fundo}** · limiar de verde ('a'): "
                   + ", ".join(str(f.limiar_a) for f in res.folhas))
        if len(linhas) > 1:
            st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")
        if res.lesoes:
            with st.expander("Manchas numeradas (da maior para a menor)", expanded=True):
                n1, n2 = st.columns([3, 2])
                n1.image(res.numerada, channels="BGR", width="stretch",
                         caption="Nº 1 = maior mancha da imagem"
                         + (f" · numeradas as {params['max_numeros']} maiores de {len(res.lesoes)}"
                            if len(res.lesoes) > params["max_numeros"] else ""))
                df_les = pd.DataFrame([{
                    "Mancha": l.numero, "Folha": l.folha, "Área (px)": l.area_px,
                    "% da folha": round(l.pct_folha, 3),
                    "% do total lesionado": round(100 * l.area_px / res.area_lesao_px, 2),
                } for l in res.lesoes])
                n2.dataframe(df_les, hide_index=True, width="stretch", height=380)
                d1, d2 = n1.columns(2)
                d1.download_button("Baixar imagem numerada (PNG)", png(res.numerada),
                                   f"{Path(nome).stem}_numerada.png", "image/png", key=f"num_{nome}")
                d2.download_button("Baixar tabela de manchas (CSV)", csv_br(df_les.assign(Imagem=nome)),
                                   f"{Path(nome).stem}_manchas.csv", "text/csv", key=f"les_{nome}")
        with st.expander("Máscaras intermediárias"):
            m1, m2, m3 = st.columns(3)
            m1.image(res.mascara_folha, caption="Folha (diferença de cor para o fundo)", width="stretch")
            m2.image(res.canal_a, width="stretch",
                     caption="Canal 'a' do LAB (escuro = verde; claro = sem verde)")
            m3.image(res.mascara_lesao, caption="Sintomas", width="stretch")
        resumo.extend(linhas)


# ---------------------------------------------------------------- resumo
if resumo:
    st.divider()
    st.subheader("Resumo")
    df_resumo = pd.DataFrame(resumo)
    st.dataframe(df_resumo, hide_index=True, width="stretch")
    st.download_button("Baixar CSV", csv_br(df_resumo), "severidade.csv", "text/csv")
