"""Ferramenta de análise de severidade de doenças foliares — PlantCV + Streamlit.

Executar:  .venv\\Scripts\\streamlit run app.py
"""

import copy
import hashlib
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import pillow_heif
import streamlit as st
from PIL import Image, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates

from analise import diagramatica, foto, graficos
from analise.escala import ESCALAS, escala_de_texto

EXEMPLOS = Path(__file__).parent / "exemplos"
MODO_DIAG = "Escala diagramática / desenho P&B"
MODO_FOTO = "Foto de folha (qualquer doença)"
SEM_ESCALA = "Nenhuma (apenas % de severidade)"

TIPOS = ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "heic", "heif"]

st.set_page_config(page_title="Severidade foliar · PlantCV", page_icon="🌿", layout="wide")
pillow_heif.register_heif_opener()  # Pillow passa a abrir HEIC/HEIF (fotos de iPhone)


# ---------------------------------------------------------------- utilidades
@st.cache_data(show_spinner=False, max_entries=16)
def decodificar(conteudo: bytes) -> np.ndarray:
    """Bytes do arquivo -> imagem BGR (uint8).

    OpenCV lê PNG/JPG/TIF/BMP (já aplicando a orientação EXIF). HEIC/HEIF, que o OpenCV
    não lê, vai pelo Pillow + pillow-heif, com a orientação EXIF aplicada à parte.
    """
    img = cv2.imdecode(np.frombuffer(conteudo, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        with Image.open(BytesIO(conteudo)) as pil:
            rgb = np.asarray(ImageOps.exif_transpose(pil).convert("RGB"))
    except Exception as e:
        raise ValueError("Formato de imagem não reconhecido.") from e
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@st.cache_data(show_spinner=False, max_entries=16)
def previa(conteudo: bytes, lado_max: int = 1600) -> np.ndarray:
    """Original reduzido para exibir: navegadores não mostram HEIC e fotos grandes pesam."""
    img = decodificar(conteudo)
    escala = lado_max / max(img.shape[:2])
    return cv2.resize(img, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA) if escala < 1 else img


@st.cache_data(show_spinner=False, max_entries=16)  # limita a memória no servidor
def rodar_diagramatica(conteudo: bytes, params: dict):
    return diagramatica.analisar(decodificar(conteudo), diagramatica.ParamsDiagramatica(**params))


@st.cache_data(show_spinner=False, max_entries=16)  # limita a memória no servidor
def rodar_foto(conteudo: bytes, params: dict):
    return foto.analisar(decodificar(conteudo), foto.ParamsFoto(**params))


def tema_escuro() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # versões antigas / execução sem navegador
        return False


def mostrar_graficos(linhas: pd.DataFrame, escala, chave: str) -> None:
    """Gráficos de interpretação: posição na escala, folhas por nota e severidade por folha."""
    escuro = tema_escuro()
    df = linhas.rename(columns={"Nota atribuída": "Nota"})
    if escala:
        st.altair_chart(graficos.posicao_na_escala(df, escala, escuro), width="stretch",
                        key=f"pos_{chave}")
    else:
        st.caption("Escolha uma escala na barra lateral para ver as faixas de nota nos gráficos.")
    if len(df) > 1:
        g1, g2 = st.columns(2) if escala else (st.container(), None)
        g1.altair_chart(graficos.severidade_por_folha(df, escala, escuro), width="stretch",
                        key=f"sev_{chave}")
        if escala:
            g2.altair_chart(graficos.folhas_por_nota(df, escala, escuro), width="stretch",
                            key=f"nota_{chave}")


LARGURA_EDITOR = 1100  # px da imagem exibida no editor de contorno
RAIO_ALCA = 10         # px (na imagem exibida) para "pegar" um ponto
MODOS_EDICAO = ["Mover pontos", "Remover ponto", "Nova folha", "Excluir folha"]
# Cores das folhas no editor (BGR): ciclo só para separar folhas vizinhas; o número identifica.
_CORES_EDITOR = [(0, 140, 255), (255, 160, 0), (60, 200, 60), (200, 80, 230), (0, 220, 255), (255, 90, 90)]


def poligonos_da_mascara(mascara: np.ndarray, largura_original: int) -> list[list[list[float]]]:
    """Contornos externos da máscara automática -> polígonos simplificados (px da imagem original).

    Simplificação de Douglas-Peucker com tolerância de 1,5 px da imagem original: nas fotos de
    teste, 9–46 pontos por folha, área ±0,1% e severidade igual à da detecção automática.
    """
    esc = largura_original / mascara.shape[1]
    contornos, _ = cv2.findContours((mascara > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    pols = []
    for c in contornos:
        if cv2.contourArea(c) < 50:
            continue
        s = cv2.approxPolyDP(c, 1.5 / esc, True)[:, 0, :]
        if len(s) >= 3:
            pols.append((s * esc).tolist())
    # Mesma ordem da análise automática: da esquerda para a direita, de cima para baixo.
    pols.sort(key=lambda p: (min(x for x, _ in p), min(y for _, y in p)))
    return pols


def _desenhar_contornos(img: np.ndarray, estado: dict, esc: float, alcas: bool) -> np.ndarray:
    """Prévia do editor: folhas numeradas com seus pontos e o contorno em andamento (magenta)."""
    vis = cv2.resize(img, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA) if esc < 1 else img.copy()
    camada = vis.copy()
    for i, pol in enumerate(estado["pol"]):
        pts = np.round(np.asarray(pol) * esc).astype(np.int32)
        cv2.fillPoly(camada, [pts], _CORES_EDITOR[i % len(_CORES_EDITOR)])
    vis = cv2.addWeighted(camada, 0.22, vis, 0.78, 0)
    for i, pol in enumerate(estado["pol"]):
        cor = _CORES_EDITOR[i % len(_CORES_EDITOR)]
        pts = np.round(np.asarray(pol) * esc).astype(np.int32)
        cv2.polylines(vis, [pts], True, cor, 2, cv2.LINE_AA)
        if alcas:
            for x, y in pts:
                cv2.circle(vis, (int(x), int(y)), 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(vis, (int(x), int(y)), 4, cor, 1, cv2.LINE_AA)
        x0, y0 = pts[np.argmin(pts[:, 1])]
        rot = str(i + 1)
        (tw, th), _ = cv2.getTextSize(rot, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        x0 = int(np.clip(x0 - tw // 2, 0, vis.shape[1] - tw - 4))
        y0 = int(max(th + 4, y0 - 6))
        cv2.rectangle(vis, (x0 - 2, y0 - th - 3), (x0 + tw + 2, y0 + 3), (255, 255, 255), -1)
        cv2.putText(vis, rot, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    if estado["atual"]:
        pts = np.round(np.asarray(estado["atual"]) * esc).astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(vis, [pts], False, (255, 0, 255), 2, cv2.LINE_AA)
        for j, (x, y) in enumerate(pts):
            cv2.circle(vis, (int(x), int(y)), 7 if j == 0 else 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, (int(x), int(y)), 7 if j == 0 else 4, (255, 0, 255), 2, cv2.LINE_AA)
    return vis


def _ponto_mais_proximo(pols: list, p: np.ndarray, raio: float) -> tuple[int, int] | None:
    melhor, dist = None, raio
    for i, pol in enumerate(pols):
        d = np.hypot(*(np.asarray(pol) - p).T)
        j = int(np.argmin(d))
        if d[j] <= dist:
            melhor, dist = (i, j), d[j]
    return melhor


def _aresta_mais_proxima(pols: list, p: np.ndarray, raio: float) -> tuple[int, int] | None:
    """(folha, índice onde inserir) da aresta mais próxima de p, se estiver a até `raio`."""
    melhor, dist = None, raio
    for i, pol in enumerate(pols):
        a = np.asarray(pol, float)
        b = np.roll(a, -1, axis=0)
        ab = b - a
        t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
        d = np.hypot(*(a + ab * t[:, None] - p).T)
        j = int(np.argmin(d))
        if d[j] <= dist:
            melhor, dist = (i, j + 1), d[j]
    return melhor


def editor_contorno(conteudo: bytes, params: dict) -> dict:
    """Edição da seleção: parte dos contornos automáticos e permite arrastar, incluir e remover
    pontos, criar e excluir folhas. Devolve o estado {pol, atual, recortar} desta imagem."""
    chave = hashlib.md5(conteudo).hexdigest()[:12]
    todos = st.session_state.setdefault("contornos", {})
    estado = todos.setdefault(chave, {"pol": [], "atual": [], "t": None, "recortar": False,
                                      "hist": []})

    def guardar():  # histórico para "Desfazer"
        estado["hist"].append((copy.deepcopy(estado["pol"]), list(estado["atual"])))
        del estado["hist"][:-50]

    def carregar_automatica():
        guardar()
        res = rodar_foto(conteudo, params)
        estado["pol"] = poligonos_da_mascara(res.mascara_folha, decodificar(conteudo).shape[1])
        estado["atual"] = []
        estado["recortar"] = False  # o usuário define a borda; o ajuste desfaria a correção

    def comecar_do_zero():
        guardar()
        estado.update(pol=[], atual=[], recortar=True)
        st.session_state[f"modo_{chave}"] = "Nova folha"
        estado["editando"] = True

    def fechar():
        if len(estado["atual"]) >= 3:
            guardar()
            estado["pol"].append(estado["atual"])
        estado["atual"] = []

    def desfazer():
        if estado["hist"]:
            estado["pol"], estado["atual"] = estado["hist"].pop()

    def descartar():
        estado.update(pol=[], atual=[], hist=[], editando=False)

    editando = bool(estado["pol"] or estado["atual"] or estado.get("editando"))
    rotulo = "✏️ Editar a seleção das folhas"
    if estado["pol"]:
        rotulo += f" — {len(estado['pol'])} folha(s) editada(s), substituindo a detecção automática"
    with st.expander(rotulo, expanded=editando):
        if not editando:
            st.caption("Se a detecção automática errou o contorno de alguma folha, carregue a "
                       "seleção dela como pontos editáveis e arraste os pontos para o lugar certo.")
            c1, c2 = st.columns(2)
            if c1.button("✏️ Editar a seleção automática", key=f"carregar_{chave}", type="primary",
                         width="stretch"):
                carregar_automatica()
                estado["editando"] = True
                st.rerun()
            c2.button("Desenhar do zero", key=f"zero_{chave}", on_click=comecar_do_zero, width="stretch")
            return estado

        modo = st.radio("Ferramenta", MODOS_EDICAO, horizontal=True, key=f"modo_{chave}",
                        help="Mover pontos: arraste um ponto até a nova posição; clique na borda, "
                             "fora de um ponto, para criar um ponto novo ali. Remover ponto: clique "
                             "no ponto. Nova folha: clique em volta dela, ponto a ponto, e feche "
                             "clicando perto do primeiro ponto. Excluir folha: clique dentro dela.")
        img = decodificar(conteudo)
        H, W = img.shape[:2]
        esc = min(1.0, LARGURA_EDITOR / W)
        vis = _desenhar_contornos(img, estado, esc, alcas=modo in ("Mover pontos", "Remover ponto"))
        ev = streamlit_image_coordinates(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), width="content",
                                         key=f"clique_{chave}", click_and_drag=True,
                                         cursor="crosshair")
        if ev and ev.get("unix_time") != estado["t"] and "x1" in ev:
            estado["t"] = ev["unix_time"]
            # Coordenadas do evento estão na imagem exibida; converte para a imagem original.
            fx, fy = W / ev["width"], H / ev["height"]
            p1 = np.array([ev["x1"] * fx, ev["y1"] * fy])
            p2 = np.array([min(max(ev["x2"], 0), ev["width"]) * fx,
                           min(max(ev["y2"], 0), ev["height"]) * fy])
            raio = RAIO_ALCA * fx
            arrastou = np.hypot(*(p2 - p1)) > 3 * fx
            if modo == "Mover pontos":
                alvo = _ponto_mais_proximo(estado["pol"], p1, raio)
                if alvo and arrastou:
                    guardar()
                    estado["pol"][alvo[0]][alvo[1]] = p2.tolist()
                elif not alvo and not arrastou:
                    aresta = _aresta_mais_proxima(estado["pol"], p1, raio)
                    if aresta:
                        guardar()
                        estado["pol"][aresta[0]].insert(aresta[1], p1.tolist())
            elif modo == "Remover ponto":
                alvo = _ponto_mais_proximo(estado["pol"], p2, raio)
                if alvo and len(estado["pol"][alvo[0]]) > 3:
                    guardar()
                    estado["pol"][alvo[0]].pop(alvo[1])
            elif modo == "Nova folha":
                if len(estado["atual"]) >= 3 and np.hypot(*(p2 - estado["atual"][0])) < raio * 1.5:
                    fechar()
                else:
                    guardar()
                    estado["atual"].append(p2.tolist())
            elif modo == "Excluir folha":
                for i, pol in enumerate(estado["pol"]):
                    if cv2.pointPolygonTest(np.asarray(pol, np.float32), tuple(map(float, p2)), False) >= 0:
                        guardar()
                        estado["pol"].pop(i)
                        break
            st.rerun()

        b1, b2, b3, b4 = st.columns(4)
        b1.button("↶ Desfazer", on_click=desfazer, disabled=not estado["hist"], key=f"desf_{chave}",
                  width="stretch")
        if modo == "Nova folha":
            b2.button("✔ Fechar folha", on_click=fechar, disabled=len(estado["atual"]) < 3,
                      key=f"fechar_{chave}", width="stretch")
        if b3.button("↺ Restaurar a automática", key=f"rest_{chave}", width="stretch"):
            carregar_automatica()
            st.rerun()
        b4.button("✖ Descartar edição", on_click=descartar, key=f"desc_{chave}", width="stretch",
                  help="Volta a usar a detecção automática nesta imagem.")
        estado["recortar"] = st.checkbox(
            "Ajustar o contorno à borda real da folha", value=estado["recortar"], key=f"rec_{chave}",
            help="Tira de dentro do contorno o fundo que ficou incluído (preto, papel, grade). Útil "
                 "para folhas desenhadas do zero; desligado na edição da seleção automática, para "
                 "respeitar a borda que você ajustou.")
        if estado["atual"]:
            st.info(f"Folha nova em andamento: {len(estado['atual'])} ponto(s). Feche a folha para "
                    "ela entrar na análise.")
        elif not estado["pol"]:
            st.info("Nenhuma folha na seleção: a imagem continua com a detecção automática.")
    return estado


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
            fundos = {"Automático": "auto", "Fundo removido (preto)": "preto",
                      "Gabarito (várias folhas sobre quadro impresso)": "gabarito",
                      "Uniforme (escaneada / papel)": "uniforme", "Complexo (foto de campo)": "complexo"}
            pf.modo_fundo = fundos[st.selectbox(
                "Tipo de fundo", list(fundos),
                help="Fundo removido (preto): padrão do projeto; folha = tudo que não é o preto. "
                     "Gabarito: várias folhas/faixas sobre quadro com grade, números ou bordas "
                     "impressas; cada folha é achada pelo excesso de verde. Uniforme: folha separada "
                     "pela diferença de cor para um fundo liso (pega lesões na margem). Complexo: foto "
                     "de campo com vasos, solo ou outras plantas; analisa a maior folha. "
                     "Automático escolhe sozinho.")]
            maior = st.selectbox("Folhas analisadas", ["Automático", "Só a maior", "Todas"],
                                 help="Automático: todas no fundo preto, no gabarito e no fundo "
                                      "uniforme; só a maior no complexo.")
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

arquivos = st.file_uploader("Imagens (PNG, JPG, TIF, HEIC)", type=TIPOS, accept_multiple_files=True)
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
    params_img = params
    if modo == MODO_FOTO:
        # Contorno manual (se houver) substitui a detecção automática só desta imagem.
        estado = editor_contorno(conteudo, params)
        if estado["pol"]:
            # O limiar de sintoma sai do verde de todas as folhas; na edição ele fica o da
            # detecção automática, para que mexer numa folha não mude a avaliação das outras.
            limiar = params["limiar_a"]
            if limiar is None:
                auto = rodar_foto(conteudo, params)
                limiar = auto.folhas[0].limiar_a if auto.folhas else None
            params_img = dict(params, poligonos=tuple(tuple(map(tuple, p)) for p in estado["pol"]),
                              recortar_manual=estado["recortar"], limiar_a=limiar)
    try:
        with st.spinner("Analisando…"):
            res = (rodar_diagramatica(conteudo, params_img) if modo == MODO_DIAG
                   else rodar_foto(conteudo, params_img))
    except Exception as e:  # imagem corrompida ou parâmetros que não segmentam nada
        st.error(f"Falha ao analisar: {e}")
        continue

    for aviso in res.avisos:
        st.warning(aviso)

    if modo == MODO_DIAG:
        col_a, col_b = st.columns(2)
        col_a.image(previa(conteudo), channels="BGR", caption="Original", width="stretch")
        col_b.image(res.sobreposicao, channels="BGR", width="stretch",
                    caption="Verde: limbo · vermelho: lesão · azul: contorno/nervura")
    else:
        col_a, col_b, col_c = st.columns(3)
        col_a.image(previa(conteudo), channels="BGR", caption="Original", width="stretch")
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
                   + ", ".join(str(v) for v in sorted({f.limiar_a for f in res.folhas})))
        if len(linhas) > 1:
            st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")
        if linhas:
            with st.expander("Gráficos", expanded=True):
                mostrar_graficos(pd.DataFrame(linhas), escala, chave=nome)
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
    if modo == MODO_FOTO and df_resumo["Imagem"].nunique() > 1:
        st.markdown("**Gráficos — todas as imagens**")
        geral = df_resumo.assign(Folha=df_resumo["Imagem"].map(lambda s: Path(s).stem) + " · "
                                 + df_resumo["Folha"].astype(str))
        mostrar_graficos(geral, escala, chave="__resumo__")
