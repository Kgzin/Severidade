"""Análise de escalas diagramáticas / desenhos de folhas em preto e branco.

Cada folha é um retângulo desenhado com contorno fino e nervura central representada
por duas linhas verticais paralelas. As lesões são manchas escuras no limbo.

Pipeline:
1. Tinta = pixels mais escuros que o fundo (PlantCV threshold).
2. Linhas do desenho = estruturas verticais/horizontais longas (abertura morfológica
   com elemento estruturante linear). Assim contorno e nervura são separados das lesões
   por geometria, não por intensidade (as lesões pequenas têm o mesmo cinza das linhas).
3. Cada componente conexo do "esqueleto" de linhas alto o suficiente é uma folha.
4. Limbo = interior da folha - linhas - faixa da nervura central.
5. Severidade = área de lesão / área do limbo. Duas medidas:
   - ponderada: cada pixel contribui com sua escuridão (trata anti-aliasing/digitalização);
   - binária: pixel é lesão se mais escuro que o limiar.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from plantcv import plantcv as pcv

pcv.params.debug = None


@dataclass
class ParamsDiagramatica:
    limiar_tinta: int = 215          # abaixo disso o pixel é "tinta" (linha ou lesão)
    limiar_lesao: int = 170          # limiar da medida binária
    frac_altura_linha: float = 0.3   # comprimento mínimo de linha vertical (fração da altura da imagem)
    frac_largura_linha: float = 0.04  # comprimento mínimo de linha horizontal (fração da largura)
    frac_altura_folha: float = 0.3   # altura mínima de uma folha (fração da altura da imagem)
    margem_linha: int = 2            # dilatação (px) das linhas para descartar o anti-aliasing
    largura_borda: int = 8           # faixa (px) junto ao contorno onde traços lineares são contorno
    excluir_nervura: bool = True


@dataclass
class Folha:
    indice: int
    caixa: tuple[int, int, int, int]  # x, y, w, h
    area_limbo_px: int
    area_lesao_px: int
    severidade_ponderada: float
    severidade_binaria: float
    n_lesoes: int


@dataclass
class ResultadoDiagramatica:
    folhas: list[Folha]
    sobreposicao: np.ndarray  # BGR
    mascara_lesao: np.ndarray
    mascara_linhas: np.ndarray
    avisos: list[str] = field(default_factory=list)


def _linhas(tinta: np.ndarray, comp_v: int, comp_h: int) -> tuple[np.ndarray, np.ndarray]:
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(comp_v, 3)))
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(comp_h, 3), 1))
    vert = cv2.morphologyEx(tinta, cv2.MORPH_OPEN, k_v)
    horiz = cv2.morphologyEx(tinta, cv2.MORPH_OPEN, k_h)
    return vert, horiz


def _erodir(mask: np.ndarray, kernel: np.ndarray, iteracoes: int) -> np.ndarray:
    # Borda constante 0: por padrão o OpenCV não erode junto à borda da imagem/recorte.
    return cv2.erode(mask, kernel, iterations=iteracoes,
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)


def _grupos_colunas(mask_vert: np.ndarray, frac: float = 0.5) -> list[np.ndarray]:
    """Agrupa colunas contíguas onde há linha vertical ocupando > frac da altura."""
    h = mask_vert.shape[0]
    cols = np.where((mask_vert > 0).sum(axis=0) > frac * h)[0]
    if cols.size == 0:
        return []
    return np.split(cols, np.where(np.diff(cols) > 1)[0] + 1)


def _faixa_nervura(grupos: list[np.ndarray], largura: int) -> tuple[int, int] | None:
    """Nervura central = par de linhas internas mais próximo do centro da folha.

    Grupos colados às bordas são o contorno; lesões densas podem gerar falsas "linhas"
    verticais (ex.: nota 9), por isso só o par mais central é considerado.
    """
    centro = largura / 2
    internos = [g for g in grupos if 0.15 * largura < g.mean() < 0.85 * largura]
    if len(internos) < 2:
        return None
    internos.sort(key=lambda g: abs(g.mean() - centro))
    a, b = sorted(internos[:2], key=lambda g: g[0])
    if b[0] - a[-1] > 0.3 * largura:  # par distante demais para ser uma nervura
        return None
    return int(a[0]), int(b[-1])


def analisar(img_bgr: np.ndarray, params: ParamsDiagramatica | None = None) -> ResultadoDiagramatica:
    p = params or ParamsDiagramatica()
    cinza = pcv.rgb2gray(img_bgr)
    H, W = cinza.shape
    avisos: list[str] = []

    tinta = pcv.threshold.binary(cinza, p.limiar_tinta, object_type="dark")
    vert, horiz = _linhas(tinta, int(H * p.frac_altura_linha), int(W * p.frac_largura_linha))
    esqueleto = pcv.logical_or(vert, horiz)

    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(esqueleto, connectivity=8)
    caixas = sorted(
        (tuple(int(v) for v in stats[r, :4]) + (r,) for r in range(1, n)
         if stats[r, 3] > p.frac_altura_folha * H),
        key=lambda c: (c[0], c[1]),
    )
    if not caixas:
        avisos.append("Nenhum contorno de folha encontrado; a imagem inteira foi tratada como uma folha.")
        caixas = [(0, 0, W, H, None)]

    kernel_margem = np.ones((2 * p.margem_linha + 1, 2 * p.margem_linha + 1), np.uint8)
    mascara_lesao = np.zeros_like(cinza)
    mascara_limbo = np.zeros_like(cinza)
    folhas: list[Folha] = []

    for i, (x, y, w, h, rotulo) in enumerate(caixas, start=1):
        sub_cinza = cinza[y:y + h, x:x + w].astype(np.float32)

        # Limbo = envoltória convexa do contorno, erodida para descartar a própria linha.
        # A envoltória tolera falhas no traço do contorno (digitalização) e não desconta
        # estrias longas de lesão que a abertura linear também detecta nas notas altas.
        if rotulo is None:
            limbo = np.ones((h, w), bool)
        else:
            contorno = (rotulos[y:y + h, x:x + w] == rotulo).astype(np.uint8)
            pontos = cv2.findNonZero(contorno)
            envoltoria = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(envoltoria, cv2.convexHull(pontos), 255)
            # Traço do contorno junto à borda (pode ser inclinado/espesso) não é limbo.
            faixa_borda = envoltoria & ~_erodir(envoltoria, np.ones((3, 3), np.uint8), p.largura_borda)
            traco = cv2.dilate(esqueleto[y:y + h, x:x + w] & faixa_borda, kernel_margem)
            limbo = (_erodir(envoltoria, kernel_margem, 2) > 0) & (traco == 0)

        if p.excluir_nervura:
            faixa = _faixa_nervura(_grupos_colunas(vert[y:y + h, x:x + w]), w)
            if faixa:
                a, b = faixa[0] - p.margem_linha, faixa[1] + p.margem_linha
                limbo[:, max(a, 0):b + 1] = False
            elif rotulo is not None:
                avisos.append(f"Folha {i}: nervura central não identificada; considerada como limbo.")

        # Fundo pode não ser 255 (papel/digitalização): normaliza pelo percentil 95 do limbo.
        fundo = float(np.percentile(sub_cinza[limbo], 95)) if limbo.any() else 255.0
        escuridao = np.clip((fundo - sub_cinza) / max(fundo, 1.0), 0, 1)
        escuridao[sub_cinza >= p.limiar_tinta] = 0  # ruído de papel não conta

        lesao_bin = limbo & (sub_cinza < p.limiar_lesao)
        area_limbo = int(limbo.sum())
        area_lesao = int(lesao_bin.sum())
        sev_pond = 100 * float(escuridao[limbo].sum()) / max(area_limbo, 1)
        sev_bin = 100 * area_lesao / max(area_limbo, 1)

        lesao_u8 = lesao_bin.astype(np.uint8) * 255
        n_les = cv2.connectedComponents(lesao_u8, connectivity=8)[0] - 1

        mascara_lesao[y:y + h, x:x + w] |= lesao_u8
        mascara_limbo[y:y + h, x:x + w] |= limbo.astype(np.uint8) * 255
        folhas.append(Folha(i, (x, y, w, h), area_limbo, area_lesao, sev_pond, sev_bin, n_les))

    sobreposicao = _desenhar(img_bgr, mascara_limbo, mascara_lesao, esqueleto, folhas)
    return ResultadoDiagramatica(folhas, sobreposicao, mascara_lesao, esqueleto, avisos)


def _desenhar(img, limbo, lesao, linhas, folhas):
    base = img.copy()
    # limbo levemente esverdeado, lesões em vermelho, linhas em azul
    verde = np.zeros_like(base)
    verde[:] = (170, 230, 170)
    base = np.where(limbo[..., None] > 0, cv2.addWeighted(base, 0.6, verde, 0.4, 0), base)
    base[(linhas > 0) & (limbo == 0)] = (200, 120, 40)
    base[lesao > 0] = (40, 40, 220)
    for f in folhas:
        x, y, w, h = f.caixa
        cv2.rectangle(base, (x, y), (x + w, y + h), (0, 140, 255), 1)
        rotulo = str(f.indice)
        (tw, th), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(base, (x + 2, y + 2), (x + 6 + tw, y + 6 + th), (255, 255, 255), -1)
        cv2.putText(base, rotulo, (x + 4, y + 4 + th), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return base
