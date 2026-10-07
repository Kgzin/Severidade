"""Análise de fotos coloridas de folhas — qualquer espécie, qualquer doença.

Pipeline:
1. Segmentação da(s) folha(s), com o tipo de fundo detectado pela moldura da imagem:
   - uniforme (escaneada, papel, cartolina): folha = pixels cuja cor difere da cor da
     moldura (Otsu na distância LAB) -> fill (ruído) -> fill_holes. Pega lesões na margem.
   - complexo (foto de campo, folha encostando na borda, vasos/solo ao fundo): folha =
     pixels verdes (canal 'a' baixo) -> fechamento morfológico + fill_holes para incluir
     as manchas; só a maior folha é analisada.
   - gabarito (várias folhas/faixas sobre quadro com grade, números ou bordas impressas):
     folha = excesso de verde (ExG, Otsu) + nervura central esbranquiçada (canal 'a') ->
     fill_holes; todas as folhas são analisadas.
   O pecíolo é removido por abertura morfológica (estrutura fina), exceto no gabarito.
2. Tecido sadio x sintoma pelo canal 'a' do LAB (eixo verde <-> vermelho, 128 = neutro).
   Todo sintoma deixa de ser verde: necrose bege/marrom, clorose amarela, pústulas
   alaranjadas, manchas pretas e oídio branco (cores neutras) ficam acima do limiar.
   O limiar é relativo ao verde sadio da imagem (percentil 10 de 'a' nas folhas + margem), pois
   câmera, luz e cultivar mudam o verde do tecido sadio.
3. Severidade = área sintomática / área da folha; contagem e tamanho médio de lesões.
4. Saída em preto e branco no padrão das escalas diagramáticas: fundo e tecido sadio
   brancos, contorno da folha e lesões em preto.
"""

from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from plantcv import plantcv as pcv

pcv.params.debug = None


@dataclass
class ParamsFoto:
    modo_fundo: str = "auto"          # "auto", "uniforme" (escaneada/papel) ou "complexo" (campo)
    limiar_folha: int | None = None   # None = Otsu (distância ao fundo ou canal 'a', conforme o modo)
    so_maior_folha: bool | None = None  # None = só no fundo complexo
    frac_area_folha: float = 0.01     # objeto menor que essa fração da imagem não é folha
    area_min_ruido: int = 200         # objetos menores que isso (px) são descartados na máscara
    remover_peciolo: bool = True
    frac_peciolo: float = 0.10       # largura máxima do pecíolo (fração do tamanho da folha)
    limiar_a: int | None = None       # 'a' >= limiar = sintoma; None = verde da folha + margem
    margem_verde: int = 17            # limiar automático = percentil 10 de 'a' nas folhas + margem
    max_numeros: int = 50             # imagem numerada: só as N maiores manchas recebem número
    area_min_lesao: int = 5           # lesões menores que isso (px) são descartadas
    borda_px: int = 2                 # faixa da margem ignorada na busca de lesões (pixels mistos)
    resolucao_min: int = 1200         # imagens menores são ampliadas até isso (lado menor); 0 = não
    resolucao_max: int = 2400         # imagens maiores são reduzidas até isso (lado maior); 0 = não
    refinar: bool = True              # recorta as manchas pela luminância (detalhe fino)
    contraste_escuro: float = 3.0     # quanto (L, 0-255) a mancha é mais escura que a vizinhança
    forte: int = 8                    # 'a' >= limiar + forte é sintoma mesmo sem ser escuro


@dataclass
class FolhaFoto:
    indice: int
    caixa: tuple[int, int, int, int]  # x, y, w, h
    area_folha_px: int
    area_lesao_px: int
    severidade: float
    n_lesoes: int
    area_media_lesao_px: float
    limiar_a: int                     # limiar de 'a' usado nesta folha


@dataclass
class Lesao:
    numero: int                # 1 = maior lesão da imagem
    folha: int
    area_px: int
    pct_folha: float           # % da área da folha ocupada por esta lesão
    ancora: tuple[int, int]    # ponto mais interno da lesão (onde vai o número)


@dataclass
class ResultadoFoto:
    folhas: list[FolhaFoto]
    area_folha_px: int
    area_lesao_px: int
    severidade: float
    n_lesoes: int
    area_media_lesao_px: float
    sobreposicao: np.ndarray   # BGR
    preto_branco: np.ndarray   # cinza: lesões e contorno pretos, resto branco
    numerada: np.ndarray       # BGR: P&B com as lesões numeradas da maior para a menor
    lesoes: list[Lesao]
    mascara_folha: np.ndarray
    mascara_lesao: np.ndarray
    canal_a: np.ndarray
    modo_fundo: str
    avisos: list[str] = field(default_factory=list)


def _mascara_exg(img_bgr: np.ndarray, p: ParamsFoto) -> np.ndarray:
    """Folhas pelo excesso de verde (ExG = 2g - r - b, cromaticidades normalizadas).

    Índice clássico de fenotipagem: separa tecido verde de papel branco, tinta vermelha/preta
    (grade e números de gabarito) e solo, independentemente do brilho. Lesões internas viram
    buracos e são preenchidas (fill_holes); um fechamento curto reúne pequenas falhas da borda
    sem unir folhas vizinhas.
    """
    H, W = img_bgr.shape[:2]
    bgr = cv2.medianBlur(img_bgr, 5).astype(np.float32)
    soma = bgr.sum(axis=2) + 1e-6
    exg = (2 * bgr[..., 1] - bgr[..., 2] - bgr[..., 0]) / soma          # -1..2, verde > 0
    exg_u8 = np.clip((exg + 0.2) / 0.8 * 255, 0, 255).astype(np.uint8)
    if p.limiar_folha is None:
        verde = pcv.threshold.otsu(exg_u8, object_type="light")
    else:
        verde = pcv.threshold.binary(exg_u8, p.limiar_folha, object_type="light")

    # Nervura central esbranquiçada (cana, milho, sorgo) tem ExG baixo e dividiria a faixa em
    # duas metades. Ela ainda é levemente verde no canal 'a' (≈115–119) e o papel não (≈126):
    # entra o que estiver abaixo do meio-termo entre o 'a' da folha e o do fundo, desde que
    # ligado a tecido verde.
    canal_a = cv2.cvtColor(cv2.medianBlur(img_bgr, 5), cv2.COLOR_BGR2LAB)[..., 1]
    if verde.any() and (~(verde > 0)).any():
        corte = (np.median(canal_a[verde > 0]) + np.median(canal_a[verde == 0])) / 2
        uniao = ((verde > 0) | (canal_a < corte)).astype(np.uint8)
        n, rot = cv2.connectedComponents(uniao, connectivity=8)
        ligados = np.zeros(n, bool)
        ligados[np.unique(rot[verde > 0])] = True
        ligados[0] = False
        verde = ligados[rot].astype(np.uint8) * 255

    k = max(3, int(0.006 * min(H, W)) | 1)
    verde = cv2.morphologyEx(verde, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return pcv.fill_holes(pcv.fill(verde, size=p.area_min_ruido))


def _varias_folhas(mask: np.ndarray, frac_min: float = 0.002) -> bool:
    """Gabarito: 3 ou mais objetos verdes de tamanho parecido (>= 25% do maior)."""
    n, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA]
    areas = areas[areas >= frac_min * mask.size]
    return len(areas) >= 3 and int((areas >= 0.25 * areas.max()).sum()) >= 3


def _engoliu_fundo(mask: np.ndarray) -> bool:
    """A máscara 'folha' do fundo uniforme pegou o gabarito impresso (grade, bordas, números)."""
    n, rot, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return False
    r = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    maior = rot == r
    bordas = sum(bool(x.any()) for x in (maior[0], maior[-1], maior[:, 0], maior[:, -1]))
    return st[r, cv2.CC_STAT_AREA] > 0.4 * mask.size or bordas >= 3


def segmentar_folha(img_bgr: np.ndarray, p: ParamsFoto) -> tuple[np.ndarray, str]:
    """Retorna (máscara das folhas, modo de fundo usado: "uniforme", "gabarito" ou "complexo")."""
    modo = p.modo_fundo
    if modo == "gabarito":
        return _mascara_exg(img_bgr, p), modo
    if modo == "auto":
        exg = _mascara_exg(img_bgr, p)
        if _varias_folhas(exg):
            # Várias folhas lado a lado (gabarito, bancada): ainda tenta o fundo uniforme,
            # que preserva lesões na margem, e só fica com ele se não pegou a impressão.
            uni, modo_uni = _segmentar_fundo(img_bgr, replace(p, modo_fundo="auto"))
            if modo_uni == "uniforme" and not _engoliu_fundo(uni):
                return uni, "uniforme"
            return exg, "gabarito"
    mask, modo = _segmentar_fundo(img_bgr, p)
    if p.modo_fundo == "auto" and modo == "uniforme" and _engoliu_fundo(mask):
        return _mascara_exg(img_bgr, p), "gabarito"
    return mask, modo


def _segmentar_fundo(img_bgr: np.ndarray, p: ParamsFoto) -> tuple[np.ndarray, str]:
    """Segmentação pela cor do fundo (uniforme) ou pelo canal 'a' (complexo, foto de campo)."""
    lab_u8 = cv2.cvtColor(cv2.medianBlur(img_bgr, 5), cv2.COLOR_BGR2LAB)
    lab = lab_u8.astype(np.float32)
    H, W = lab.shape[:2]
    b = max(2, min(H, W) // 50)
    moldura = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                              lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    # Pixels neutros da moldura (branco/cinza/preto: papel, scanner). Quando boa parte da
    # moldura é neutra, o fundo é essa cor, mesmo que as folhas encostem na borda
    # (ex.: faixas de folha de gramínea cortadas no topo e na base).
    neutros = moldura[np.hypot(moldura[:, 1] - 128, moldura[:, 2] - 128) < 15]
    if len(neutros) >= 0.3 * len(moldura):
        fundo, uniforme = np.median(neutros, axis=0), True
    else:
        # Fundo colorido também é uniforme se quase toda a moldura tem a mesma cor. Em foto
        # de campo o fundo tem vasos, solo, caules -> moldura heterogênea.
        fundo = np.median(moldura, axis=0)
        uniforme = np.mean(np.linalg.norm(moldura - fundo, axis=1) < 20) >= 0.85

    modo = p.modo_fundo
    if modo == "auto":
        modo = "uniforme" if uniforme else "complexo"

    if modo == "uniforme":
        dist = np.clip(np.linalg.norm(lab - fundo, axis=2), 0, 255).astype(np.uint8)
        if p.limiar_folha is None:
            mask = pcv.threshold.otsu(dist, object_type="light")
        else:
            mask = pcv.threshold.binary(dist, p.limiar_folha, object_type="light")
        mask = pcv.fill(mask, size=p.area_min_ruido)
        return pcv.fill_holes(mask), modo

    # Fundo complexo: a folha é achada pela cor verde (canal 'a' baixo). As manchas não são
    # verdes e ficam como buracos/recortes: um fechamento morfológico une as que tocam a
    # margem e fill_holes preenche as internas.
    canal_a = np.ascontiguousarray(lab_u8[..., 1])
    if p.limiar_folha is None:
        corte = min(cv2.threshold(canal_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0], 125)
    else:
        corte = p.limiar_folha
    verde = pcv.threshold.binary(canal_a, int(corte), object_type="dark")
    verde = pcv.fill(verde, size=p.area_min_ruido)
    k = max(5, int(0.03 * min(H, W)) | 1)
    verde = cv2.morphologyEx(verde, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return pcv.fill_holes(verde), modo


def _sem_peciolo(folha: np.ndarray, frac: float) -> np.ndarray:
    """Abertura com disco maior que a largura do pecíolo; mantém o maior pedaço (o limbo).

    A abertura também arredonda a ponta e os dentes da margem; esses pixels são devolvidos
    se estiverem a menos de d/4 do limbo (o pecíolo, mais comprido, continua de fora).
    """
    d = max(3, int(frac * np.sqrt(folha.sum())) | 1)
    aberta = cv2.morphologyEx(folha.astype(np.uint8), cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    n, rot, stats, _ = cv2.connectedComponentsWithStats(aberta, connectivity=8)
    if n < 2:
        return folha
    limbo = (rot == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    r = max(3, d // 4 | 1)
    return folha & (cv2.dilate(limbo, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))) > 0)


def _filtro_guiado(guia: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Filtro guiado (He et al., 2010): suaviza `src` preservando as bordas de `guia`.

    Câmeras e JPEG guardam a cor em meia resolução (subamostragem de croma); a luminância
    tem resolução cheia. Guiar o canal 'a' pela luminância devolve à cor os contornos
    finos das pústulas e manchas.
    """
    m = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mI, mp = m(guia), m(src)
    a = (m(guia * src) - mI * mp) / (m(guia * guia) - mI * mI + eps)
    b = mp - a * mI
    return m(a) * guia + m(b)


def analisar(img_bgr: np.ndarray, params: ParamsFoto | None = None) -> ResultadoFoto:
    p0 = params or ParamsFoto()
    avisos: list[str] = []

    # Imagens pequenas são ampliadas (bicúbica) antes da análise: os contornos das manchas
    # ficam com precisão de fração de pixel original. Imagens muito grandes (foto de celular)
    # são reduzidas para caber na memória do servidor. Áreas são devolvidas em px originais.
    H0, W0 = img_bgr.shape[:2]
    f = 1.0
    if p0.resolucao_min and min(H0, W0) < p0.resolucao_min:
        f = float(min(4, int(np.ceil(p0.resolucao_min / min(H0, W0)))))
        f = min(f, p0.resolucao_max / max(H0, W0)) if p0.resolucao_max else f
        f = max(f, 1.0)
    elif p0.resolucao_max and max(H0, W0) > p0.resolucao_max:
        f = p0.resolucao_max / max(H0, W0)
    if f != 1.0:
        img_bgr = cv2.resize(img_bgr, None, fx=f, fy=f,
                             interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
    p = replace(p0, area_min_ruido=round(p0.area_min_ruido * f * f),
                area_min_lesao=round(p0.area_min_lesao * f * f), borda_px=max(1, round(p0.borda_px * f)))
    H, W = img_bgr.shape[:2]

    mascara, modo_fundo = segmentar_folha(img_bgr, p)
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    # No gabarito cada folha/faixa ocupa pouco da foto (~0,5–2%); só fiapos ficam de fora.
    frac_min = min(p.frac_area_folha, 0.002) if modo_fundo == "gabarito" else p.frac_area_folha
    objs = [r for r in range(1, n) if stats[r, cv2.CC_STAT_AREA] >= frac_min * H * W]
    so_maior = p.so_maior_folha if p.so_maior_folha is not None else modo_fundo == "complexo"
    if so_maior and objs:
        objs = [max(objs, key=lambda r: stats[r, cv2.CC_STAT_AREA])]
    objs.sort(key=lambda r: (stats[r, cv2.CC_STAT_LEFT], stats[r, cv2.CC_STAT_TOP]))
    if not objs:
        avisos.append("Nenhuma folha segmentada — ajuste o limiar de segmentação.")

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lum = lab[..., 0]
    a_fino = _filtro_guiado(lum / 255, lab[..., 1], r=max(2, round(2 * f)), eps=1e-4)
    canal_a = np.clip(a_fino, 0, 255).astype(np.uint8)
    k_borda = np.ones((3, 3), np.uint8)
    folha_total = np.zeros((H, W), np.uint8)
    lesao_total = np.zeros((H, W), np.uint8)
    folhas: list[FolhaFoto] = []
    lesoes: list[Lesao] = []

    regioes = []
    for r in objs:
        folha = rotulos == r
        if p.remover_peciolo and modo_fundo != "gabarito":  # faixas de gabarito não têm pecíolo
            folha = _sem_peciolo(folha, p.frac_peciolo)
        nucleo = cv2.erode(folha.astype(np.uint8), k_borda, iterations=p.borda_px,
                           borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        regioes.append((folha, nucleo))

    # Cada câmera/luz/cultivar dá um verde diferente (escaneada: a ~ 100; campo: a ~ 106;
    # print de baixa resolução: a ~ 111), por isso o limiar parte do verde sadio da imagem:
    # percentil 10 de 'a' em todas as folhas juntas. É comum a todas as folhas (mesma luz)
    # e continua sendo tecido sadio mesmo com folhas quase totalmente doentes.
    todos = np.concatenate([canal_a[n_] for _, n_ in regioes]) if regioes else np.array([])
    if p.limiar_a is not None:
        limiar = p.limiar_a
    else:
        limiar = int(np.percentile(todos, 10)) + p.margem_verde if todos.size else 128
    if p.refinar:
        # A cor (mesmo guiada) "vaza" alguns pixels para o tecido em volta das pústulas.
        # Na faixa duvidosa só conta o pixel mais escuro que a vizinhança (pústulas,
        # manchas); cor claramente não-verde (necrose bege, oídio) conta sempre.
        escuro = lum < cv2.GaussianBlur(lum, (0, 0), max(4.0, 4 * f)) - p.contraste_escuro
        sintoma = (a_fino >= limiar + p.forte) | ((a_fino >= limiar - 5) & escuro)
    else:
        sintoma = a_fino >= limiar

    for i, (folha, nucleo) in enumerate(regioes, start=1):
        lesao_u8 = (nucleo & sintoma).astype(np.uint8) * 255
        if p.area_min_lesao > 0 and lesao_u8.any():
            lesao_u8 = pcv.fill(lesao_u8, size=p.area_min_lesao)

        f2 = f * f  # áreas em px da imagem original
        area_folha = round(int(folha.sum()) / f2)
        area_lesao = round(int((lesao_u8 > 0).sum()) / f2)
        n_comp, rot_les, st_les, _ = cv2.connectedComponentsWithStats(lesao_u8, connectivity=8)
        n_les = n_comp - 1
        for c in range(1, n_comp):
            a_px = st_les[c, cv2.CC_STAT_AREA] / f2
            lesoes.append(Lesao(0, i, max(1, round(a_px)), 100 * a_px / area_folha,
                                _ponto_interno(rot_les == c, st_les[c])))
        x, y, w, h = cv2.boundingRect(folha.astype(np.uint8))
        folhas.append(FolhaFoto(i, (x, y, w, h), area_folha, area_lesao,
                                100 * int((lesao_u8 > 0).sum()) / int(folha.sum()), n_les,
                                area_lesao / n_les if n_les else 0.0, limiar))
        folha_total[folha] = 255
        lesao_total |= lesao_u8

    if folha_total.sum() / 255 > 0.97 * H * W:
        avisos.append("A 'folha' ocupa quase toda a imagem — o fundo pode não ter sido removido. "
                      "Fotografe sobre fundo uniforme com margem livre em volta da folha.")

    area_folha = sum(f.area_folha_px for f in folhas)
    area_lesao = sum(f.area_lesao_px for f in folhas)
    n_les = sum(f.n_lesoes for f in folhas)
    # Numeração única na imagem, da maior para a menor lesão.
    lesoes.sort(key=lambda l: -l.area_px)
    for n_ord, l in enumerate(lesoes, start=1):
        l.numero = n_ord
    pb = _preto_branco(folha_total, lesao_total)
    return ResultadoFoto(
        folhas, area_folha, area_lesao,
        100 * area_lesao / area_folha if area_folha else 0.0,
        n_les, area_lesao / n_les if n_les else 0.0,
        _desenhar(img_bgr, folha_total, lesao_total, folhas),
        pb, _numerar(pb, lesoes[:p.max_numeros], f), lesoes,
        folha_total, lesao_total, canal_a, modo_fundo, avisos,
    )


def _ponto_interno(mask: np.ndarray, stats: np.ndarray) -> tuple[int, int]:
    """Ponto mais distante da borda da lesão (centro "visual"; o centroide pode cair fora)."""
    x, y, w, h = (int(v) for v in stats[:4])
    sub = np.pad(mask[y:y + h, x:x + w].astype(np.uint8), 1)
    dist = cv2.distanceTransform(sub, cv2.DIST_L2, 3)
    py, px = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return x + int(px) - 1, y + int(py) - 1


def _numerar(pb: np.ndarray, lesoes: list[Lesao], fator: float = 1.0) -> np.ndarray:
    """Imagem P&B com o número de cada lesão em vermelho.

    Lesões grandes recebem o número dentro (fonte proporcional ao tamanho); as pequenas,
    ao lado, ligado por uma linha fina. Desenha da menor para a maior para que os
    números das lesões principais fiquem por cima.
    """
    img = cv2.cvtColor(pb, cv2.COLOR_GRAY2BGR)
    H, W = pb.shape
    base = min(H, W) / 900
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    vermelho, branco = (0, 0, 220), (255, 255, 255)
    for l in reversed(lesoes):
        rotulo = str(l.numero)
        raio = np.sqrt(l.area_px / np.pi) * fator  # em px da imagem desenhada (ampliada)
        escala = float(np.clip(raio / 25, 0.55, 1.4)) * max(base, 0.7)
        esp = max(1, int(round(escala * 2)))
        (tw, th), _ = cv2.getTextSize(rotulo, fonte, escala, esp)
        cx, cy = l.ancora
        if raio >= max(tw, th) * 1.25:          # cabe com folga dentro da lesão
            org = (cx - tw // 2, cy + th // 2)
        else:                                   # fora, à direita e acima
            dx = int(raio + 4 + 6 * base)
            org = (min(cx + dx, W - tw - 2), max(cy - dx // 2, th + 2))
            cv2.line(img, (cx, cy), (org[0], org[1] - th // 2), vermelho, 1, cv2.LINE_AA)
        cv2.putText(img, rotulo, org, fonte, escala, branco, esp + 2, cv2.LINE_AA)
        cv2.putText(img, rotulo, org, fonte, escala, vermelho, esp, cv2.LINE_AA)
    return img


def _espessura(shape) -> int:
    return max(1, round(min(shape[:2]) / 300))


def _preto_branco(folha: np.ndarray, lesao: np.ndarray) -> np.ndarray:
    pb = np.full(folha.shape, 255, np.uint8)
    pb[lesao > 0] = 0
    contornos, _ = cv2.findContours(folha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(pb, contornos, -1, 0, _espessura(folha.shape))
    return pb


def _desenhar(img, folha, lesao, folhas):
    fora = (img * 0.35 + 255 * 0.65).astype(np.uint8)  # esmaece o fundo
    base = np.where(folha[..., None] > 0, img, fora)
    contornos, _ = cv2.findContours(folha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(base, contornos, -1, (0, 140, 255), _espessura(img.shape) + 1)
    base[lesao > 0] = (255, 0, 255)
    if len(folhas) > 1:
        escala = max(0.5, min(img.shape[:2]) / 800)
        for f in folhas:
            x, y, _, _ = f.caixa
            rotulo = str(f.indice)
            (tw, th), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)
            cv2.rectangle(base, (x, y), (x + tw + 6, y + th + 6), (255, 255, 255), -1)
            cv2.putText(base, rotulo, (x + 3, y + th + 3), cv2.FONT_HERSHEY_SIMPLEX,
                        escala, (0, 0, 0), max(1, int(escala * 2)), cv2.LINE_AA)
    return base
