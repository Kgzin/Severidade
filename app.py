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


@st.cache_data(show_spinner=False, max_entries=8)
def sem_fundo(conteudo: bytes, mascara: np.ndarray, fundo_preto: bool) -> bytes:
    """Imagem original (resolução cheia) com o fundo retirado pela máscara das folhas."""
    img = decodificar(conteudo)
    H, W = img.shape[:2]
    m = cv2.resize(mascara, (W, H), interpolation=cv2.INTER_NEAREST) > 0
    if fundo_preto:
        out = img.copy()
        out[~m] = 0
    else:
        out = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        out[..., 3] = np.where(m, 255, 0).astype(np.uint8)
    return cv2.imencode(".png", out)[1].tobytes()


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
MODOS_EDICAO = ["Navegar", "Mover pontos", "Remover ponto", "Nova folha", "Excluir folha",
                "Marcar mancha", "Apagar sintoma"]
MODOS_DESENHO = {"Nova folha": "pol", "Marcar mancha": "add", "Apagar sintoma": "rem"}
_COR_DESENHO = {"pol": (255, 0, 255), "add": (0, 0, 255), "rem": (255, 200, 0)}  # BGR
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


ALTURA_EDITOR = 760    # altura máxima (px) da vista do editor: fotos em pé não ficam enormes
ZOOMS = [1, 2, 4, 8, 16]


def vista_editor(estado: dict, W: int, H: int) -> tuple[int, int, int, int, float]:
    """Janela da imagem original mostrada no editor: (x0, y0, largura, altura, escala).

    Zoom 1 = imagem inteira cabendo em LARGURA_EDITOR × ALTURA_EDITOR; zoom z mostra 1/z dela,
    centrada em estado["centro"] (ajustado para não sair da imagem), ampliada z vezes.
    """
    base = min(LARGURA_EDITOR / W, ALTURA_EDITOR / H)
    z = estado.get("zoom", 1)
    ww, wh = max(8, round(W / z)), max(8, round(H / z))
    cx, cy = estado.get("centro") or (W / 2, H / 2)
    x0 = int(np.clip(round(cx - ww / 2), 0, W - ww))
    y0 = int(np.clip(round(cy - wh / 2), 0, H - wh))
    # O ponto escolhido fica guardado como está (aproximar perto da borda não o desloca);
    # só a janela é ajustada para caber na imagem.
    return x0, y0, ww, wh, base * z


def _desenhar_contornos(img: np.ndarray, estado: dict, vista: tuple, alcas: bool,
                        sintoma: np.ndarray | None = None, alvo: str = "pol") -> np.ndarray:
    """Prévia do editor na janela `vista`: folhas numeradas com seus pontos, sintomas
    detectados (opcional), manchas marcadas (vermelho), áreas apagadas (azul-claro) e o
    desenho em andamento. Linhas e pontos têm espessura fixa na tela, qualquer que seja o zoom."""
    x0, y0, ww, wh, s = vista
    recorte = img[y0:y0 + wh, x0:x0 + ww]
    tam = (max(1, round(ww * s)), max(1, round(wh * s)))
    # Ampliado: pixels nítidos (vizinho mais próximo) para ver a borda real das manchas.
    vis = cv2.resize(recorte, tam, interpolation=cv2.INTER_NEAREST if s >= 2 else cv2.INTER_AREA)
    T = lambda pol: np.round((np.asarray(pol, float) - [x0, y0]) * s).astype(np.int32)

    if sintoma is not None:  # sintomas detectados pela análise atual, em vermelho translúcido
        H, W = img.shape[:2]
        fy, fx = sintoma.shape[0] / H, sintoma.shape[1] / W
        sub = sintoma[int(y0 * fy):int(np.ceil((y0 + wh) * fy)), int(x0 * fx):int(np.ceil((x0 + ww) * fx))]
        if sub.size:
            m = cv2.resize(sub, tam, interpolation=cv2.INTER_NEAREST) > 0
            vis[m] = (vis[m] * 0.35 + np.array([40, 40, 255]) * 0.65).astype(np.uint8)
    camada = vis.copy()
    for i, pol in enumerate(estado["pol"]):
        cv2.fillPoly(camada, [T(pol)], _CORES_EDITOR[i % len(_CORES_EDITOR)])
    for lista in ("add", "rem"):
        for pol in estado.get(lista, []):
            cv2.fillPoly(camada, [T(pol)], _COR_DESENHO[lista])
    vis = cv2.addWeighted(camada, 0.22, vis, 0.78, 0)
    for lista in ("add", "rem"):
        for pol in estado.get(lista, []):
            cv2.polylines(vis, [T(pol)], True, _COR_DESENHO[lista], 2, cv2.LINE_AA)
    for i, pol in enumerate(estado["pol"]):
        cor = _CORES_EDITOR[i % len(_CORES_EDITOR)]
        pts = T(pol)
        cv2.polylines(vis, [pts], True, cor, 2, cv2.LINE_AA)
        if alcas:
            for x, y in pts:
                if 0 <= x < tam[0] and 0 <= y < tam[1]:
                    cv2.circle(vis, (int(x), int(y)), 4, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(vis, (int(x), int(y)), 4, cor, 1, cv2.LINE_AA)
        # número da folha no ponto mais alto visível do contorno
        dentro = pts[(pts[:, 0] >= 0) & (pts[:, 0] < tam[0]) & (pts[:, 1] >= 0) & (pts[:, 1] < tam[1])]
        if len(dentro):
            xr, yr = dentro[np.argmin(dentro[:, 1])]
            rot = str(i + 1)
            (tw, th), _ = cv2.getTextSize(rot, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            xr = int(np.clip(xr - tw // 2, 0, max(0, tam[0] - tw - 4)))
            yr = int(max(th + 4, yr - 6))
            cv2.rectangle(vis, (xr - 2, yr - th - 3), (xr + tw + 2, yr + 3), (255, 255, 255), -1)
            cv2.putText(vis, rot, (xr, yr), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    if estado["atual"]:
        cor = _COR_DESENHO[alvo]
        pts = T(estado["atual"])
        if len(pts) > 1:
            cv2.polylines(vis, [pts], False, cor, 2, cv2.LINE_AA)
        for j, (x, y) in enumerate(pts):
            cv2.circle(vis, (int(x), int(y)), 7 if j == 0 else 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, (int(x), int(y)), 7 if j == 0 else 4, cor, 2, cv2.LINE_AA)
    return vis


def _minimapa(img: np.ndarray, vista: tuple, largura: int = 220) -> np.ndarray:
    """Imagem inteira pequena com o retângulo da área mostrada no editor."""
    x0, y0, ww, wh, _ = vista
    k = largura / img.shape[1]
    mini = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    cv2.rectangle(mini, (int(x0 * k), int(y0 * k)), (int((x0 + ww) * k) - 1, int((y0 + wh) * k) - 1),
                  (0, 220, 255), 2)
    return mini


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


def params_da_imagem(params: dict, estado: dict, conteudo: bytes) -> dict:
    """Parâmetros da análise desta imagem, com as edições do avaliador (folhas e manchas)."""
    extra = {}
    if estado["pol"]:
        # O limiar de sintoma sai do verde de todas as folhas; na edição ele fica o da
        # detecção automática, para que mexer numa folha não mude a avaliação das outras.
        limiar = params["limiar_a"]
        if limiar is None:
            auto = rodar_foto(conteudo, params)
            limiar = auto.folhas[0].limiar_a if auto.folhas else None
        extra |= dict(poligonos=tuple(tuple(map(tuple, p)) for p in estado["pol"]),
                      recortar_manual=estado["recortar"], limiar_a=limiar)
    if estado["add"]:
        extra["lesoes_add"] = tuple(tuple(map(tuple, p)) for p in estado["add"])
    if estado["rem"]:
        extra["lesoes_rem"] = tuple(tuple(map(tuple, p)) for p in estado["rem"])
    return dict(params, **extra) if extra else params


def editor_contorno(conteudo: bytes, params: dict) -> dict:
    """Edição da seleção de folhas (arrastar/incluir/remover pontos, criar/excluir folhas) e
    das manchas (marcar à mão as que a análise não pegou, apagar sintomas falsos).
    Devolve o estado {pol, add, rem, atual, recortar} desta imagem."""
    chave = hashlib.md5(conteudo).hexdigest()[:12]
    todos = st.session_state.setdefault("contornos", {})
    estado = todos.setdefault(chave, {"pol": [], "atual": [], "t": None, "recortar": False,
                                      "hist": []})
    estado.setdefault("add", [])
    estado.setdefault("rem", [])
    chave_modo = f"modo_{chave}"

    def guardar():  # histórico para "Desfazer"
        estado["hist"].append(copy.deepcopy({k: estado[k] for k in ("pol", "atual", "add", "rem")}))
        del estado["hist"][:-50]

    def carregar_automatica():
        guardar()
        res = rodar_foto(conteudo, params)
        estado["pol"] = poligonos_da_mascara(res.mascara_folha, decodificar(conteudo).shape[1])
        estado["atual"] = []
        estado["recortar"] = False  # o usuário define a borda; o ajuste desfaria a correção

    def comecar(modo_inicial: str, do_zero: bool = False):
        if do_zero:
            guardar()
            estado.update(pol=[], atual=[], recortar=True)
        st.session_state[chave_modo] = modo_inicial
        estado["editando"] = True

    def fechar(modo_atual: str):
        if len(estado["atual"]) >= 3:
            guardar()
            estado[MODOS_DESENHO[modo_atual]].append(estado["atual"])
        estado["atual"] = []

    def desfazer():
        if estado["hist"]:
            estado.update(estado["hist"].pop())

    def descartar():
        estado.update(pol=[], atual=[], add=[], rem=[], hist=[], editando=False)

    editando = bool(estado["pol"] or estado["atual"] or estado["add"] or estado["rem"]
                    or estado.get("editando"))
    partes = []
    if estado["pol"]:
        partes.append(f"{len(estado['pol'])} folha(s) editada(s)")
    if estado["add"]:
        partes.append(f"{len(estado['add'])} mancha(s) marcada(s)")
    if estado["rem"]:
        partes.append(f"{len(estado['rem'])} área(s) de sintoma apagada(s)")
    rotulo = "✏️ Editar folhas e manchas" + (" — " + ", ".join(partes) if partes else "")
    with st.expander(rotulo, expanded=editando):
        if not editando:
            st.caption("Corrija o que a análise automática errou: o contorno das folhas (arrastando "
                       "os pontos da seleção automática) ou as manchas (marcando as que não foram "
                       "detectadas e apagando sintomas falsos).")
            c1, c2, c3 = st.columns(3)
            if c1.button("✏️ Editar a seleção automática", key=f"carregar_{chave}", type="primary",
                         width="stretch"):
                carregar_automatica()
                comecar("Mover pontos")
                st.rerun()
            c2.button("🔴 Marcar / apagar manchas", key=f"manchas_{chave}", width="stretch",
                      on_click=comecar, args=("Marcar mancha",))
            c3.button("Desenhar folhas do zero", key=f"zero_{chave}", width="stretch",
                      on_click=comecar, args=("Nova folha", True))
            return estado

        modo = st.radio("Ferramenta", MODOS_EDICAO, horizontal=True, key=chave_modo,
                        help="Navegar: arraste para mover a vista ou clique para centralizar nesse "
                             "ponto; na vista inteira, o clique já aproxima 4× ali. Mover pontos: arraste um ponto até a nova posição; clique na borda, "
                             "fora de um ponto, para criar um ponto novo ali. Remover ponto: clique "
                             "no ponto. Nova folha: clique em volta dela, ponto a ponto, e feche "
                             "clicando perto do primeiro ponto. Excluir folha: clique dentro dela. "
                             "Marcar mancha: contorne a mancha que a análise não pegou. Apagar "
                             "sintoma: contorne uma área marcada como sintoma por engano.")
        img = decodificar(conteudo)
        H, W = img.shape[:2]

        def mudar_zoom(passo: int | None):
            i = ZOOMS.index(estado.get("zoom", 1))
            estado["zoom"] = 1 if passo is None else ZOOMS[int(np.clip(i + passo, 0, len(ZOOMS) - 1))]

        z1, z2, z3, z4 = st.columns([1, 1, 1, 3])
        z1.button("＋ Aproximar", key=f"zmais_{chave}", on_click=mudar_zoom, args=(1,),
                  disabled=estado.get("zoom", 1) == ZOOMS[-1], width="stretch")
        z2.button("－ Afastar", key=f"zmenos_{chave}", on_click=mudar_zoom, args=(-1,),
                  disabled=estado.get("zoom", 1) == 1, width="stretch")
        z3.button("Vista inteira", key=f"zinteira_{chave}", on_click=mudar_zoom, args=(None,),
                  disabled=estado.get("zoom", 1) == 1, width="stretch")
        z4.caption(f"Zoom **{estado.get('zoom', 1)}×** — com **Navegar**: clique para aproximar ou "
                   "centralizar, arraste para mover a vista.")
        vista = vista_editor(estado, W, H)
        sintoma = None
        if modo in ("Marcar mancha", "Apagar sintoma"):
            sintoma = rodar_foto(conteudo, params_da_imagem(params, estado, conteudo)).mascara_lesao
            st.caption("Em vermelho: o que a análise está contando como sintoma agora. "
                       "Contorno vermelho: manchas marcadas à mão · azul-claro: áreas apagadas.")
        vis = _desenhar_contornos(img, estado, vista, alcas=modo in ("Mover pontos", "Remover ponto"),
                                  sintoma=sintoma, alvo=MODOS_DESENHO.get(modo, "pol"))
        c_img, c_mini = st.columns([5, 1])
        with c_img:
            ev = streamlit_image_coordinates(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), width="content",
                                             key=f"clique_{chave}", click_and_drag=True,
                                             cursor="grab" if modo == "Navegar" else "crosshair")
        c_mini.image(_minimapa(img, vista), channels="BGR", caption="Área mostrada", width="stretch")
        if ev and ev.get("unix_time") != estado["t"] and "x1" in ev:
            estado["t"] = ev["unix_time"]
            # Evento em px da imagem exibida (o navegador pode reduzi-la) -> vista -> original.
            x0, y0, _, _, s_v = vista
            k = vis.shape[1] / ev["width"] / s_v           # px exibido -> px da imagem original
            p1 = np.array([x0 + ev["x1"] * k, y0 + ev["y1"] * k])
            p2 = np.array([x0 + min(max(ev["x2"], 0), ev["width"]) * k,
                           y0 + min(max(ev["y2"], 0), ev["height"]) * k])
            raio = RAIO_ALCA * k
            arrastou = np.hypot(*(p2 - p1)) > 3 * k
            if modo == "Navegar":
                if arrastou:  # parte do centro da janela visível (arrastar além da borda não acumula)
                    x0, y0, ww, wh, _ = vista
                    estado["centro"] = [x0 + ww / 2 - (p2 - p1)[0], y0 + wh / 2 - (p2 - p1)[1]]
                else:  # clique: centraliza ali; na vista inteira, já aproxima 4×
                    estado["centro"] = p2.tolist()
                    if estado.get("zoom", 1) == 1:
                        estado["zoom"] = 4
            elif modo == "Mover pontos":
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
            elif modo in MODOS_DESENHO:
                if len(estado["atual"]) >= 3 and np.hypot(*(p2 - estado["atual"][0])) < raio * 1.5:
                    fechar(modo)
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
        if modo in MODOS_DESENHO:
            texto = {"Nova folha": "✔ Fechar folha", "Marcar mancha": "✔ Fechar mancha",
                     "Apagar sintoma": "✔ Fechar área"}[modo]
            b2.button(texto, on_click=fechar, args=(modo,), disabled=len(estado["atual"]) < 3,
                      key=f"fechar_{chave}", width="stretch")
        if b3.button("↺ Restaurar a automática", key=f"rest_{chave}", width="stretch",
                     help="Recarrega o contorno automático das folhas (mantém as manchas)."):
            carregar_automatica()
            st.rerun()
        b4.button("✖ Descartar edição", on_click=descartar, key=f"desc_{chave}", width="stretch",
                  help="Apaga todas as edições (folhas e manchas) e volta à análise automática.")
        if estado["pol"] or modo in ("Mover pontos", "Remover ponto", "Nova folha", "Excluir folha"):
            estado["recortar"] = st.checkbox(
                "Ajustar o contorno à borda real da folha", value=estado["recortar"],
                key=f"rec_{chave}",
                help="Tira de dentro do contorno o fundo que ficou incluído (preto, papel, grade). "
                     "Útil para folhas desenhadas do zero; desligado na edição da seleção "
                     "automática, para respeitar a borda que você ajustou.")
        if estado["atual"]:
            st.info(f"Contorno em andamento: {len(estado['atual'])} ponto(s). Feche clicando perto "
                    "do primeiro ponto ou no botão ✔.")
        elif not estado["pol"] and modo in ("Mover pontos", "Remover ponto", "Excluir folha"):
            st.info("Nenhuma folha editada: a imagem usa a seleção automática de folhas.")
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
            pf.excluir_furos = st.checkbox(
                "Excluir furos e rasgos", pf.excluir_furos,
                help="Onde o fundo aparece dentro da folha (furo, rasgo), o pixel sai da área da "
                     "folha e não conta como lesão. Desligue se a doença deixa o tecido da mesma "
                     "cor do fundo (ex.: oídio muito branco sobre papel branco).")
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
        params_img = params_da_imagem(params, estado, conteudo)
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
        if res.folhas:
            with col_a.popover("🖼 Remover fundo", width="stretch"):
                st.caption("Imagem original com o fundo retirado pela seleção de folhas do app "
                           "(automática ou editada), já com furos e rasgos vazados. Corrija a "
                           "seleção no editor antes, se precisar.")
                st.download_button("PNG transparente", sem_fundo(conteudo, res.mascara_folha, False),
                                   f"{Path(nome).stem}-nobg.png", "image/png",
                                   key=f"nobg_{nome}", width="stretch")
                st.download_button("Fundo preto", sem_fundo(conteudo, res.mascara_folha, True),
                                   f"{Path(nome).stem}-preto.png", "image/png",
                                   key=f"preto_{nome}", width="stretch")

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
                      "Furos/rasgos (px)": f.area_furos_px, "Limiar 'a'": f.limiar_a}
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
