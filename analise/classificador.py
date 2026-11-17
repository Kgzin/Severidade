"""Classificador de sintomas treinado com exemplos marcados pelo avaliador.

Regras fixas de cor ("o que deixou de ser verde") não separam sintomas sutis (pontuações
esbranquiçadas, manchas pequenas) de nervuras, bordas e reflexos. Aqui o app aprende com
exemplos: o avaliador marca pixels de lesão e de tecido sadio em algumas fotos e uma
floresta aleatória (Random Forest) aprende a separar os dois a partir de atributos de cor,
brilho e contraste local em várias escalas — a mesma ideia dos classificadores de pixel do
PlantCV (naive Bayes), ilastik e Trainable Weka Segmentation.

Os atributos são relativos ao tecido da própria folha (mediana da imagem) e ao entorno
(médias locais calculadas só sobre a folha, sem "puxar" o fundo para dentro da borda), para
o modelo valer entre fotos com luz e câmera diferentes.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

ESCALAS = (1.0, 3.0, 8.0)          # sigmas (px da imagem processada) do contexto local
MAX_POR_CLASSE = 60_000            # amostras de treino por classe (limita memória e tempo)
VERSAO = 2  # mudou o conjunto de atributos (forma pela Hessiana)


def _media_na_folha(x: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """Média gaussiana só sobre pixels de folha (convolução normalizada)."""
    num = cv2.GaussianBlur(x * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return num / np.maximum(den, 1e-3)


def nomes_atributos() -> list[str]:
    nomes = ["L_rel", "a_rel", "b_rel", "croma"]
    for s in ESCALAS:
        nomes += [f"dL_{s:g}", f"da_{s:g}", f"db_{s:g}", f"Lmed_{s:g}"]
    nomes += ["grad_1", "grad_2", "log_1.5", "log_3", "dist_borda"]
    for s in FORMA:
        nomes += [f"hess_l1_{s:g}", f"hess_l2_{s:g}", f"anisot_{s:g}"]
    return nomes


FORMA = (1.5, 3.0)  # sigmas da Hessiana: separa ponto/mancha (redondo) de nervura/risco (linha)


def _hessiana(L: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Autovalores (l1 >= l2) da Hessiana normalizada pela escala e a anisotropia.

    Ponto claro: l1 e l2 negativos e parecidos (anisotropia baixa). Nervura/risco claro:
    um autovalor forte e o outro ~0 (anisotropia alta).
    """
    g = cv2.GaussianBlur(L, (0, 0), s)
    dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3) * s * s
    dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3) * s * s
    dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3) * s * s
    meio, raiz = (dxx + dyy) / 2, np.sqrt(((dxx - dyy) / 2) ** 2 + dxy ** 2)
    l1, l2 = meio + raiz, meio - raiz
    anis = (np.abs(l1) - np.abs(l2)) / (np.abs(l1) + np.abs(l2) + 1e-3)
    return l1, l2, anis


def atributos(img_bgr: np.ndarray, folha: np.ndarray) -> np.ndarray:
    """Matriz (n_pixels_de_folha, n_atributos) na ordem de np.flatnonzero(folha)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    w = folha.astype(np.float32)
    idx = folha.ravel()
    mL, ma, mb = (float(np.median(c[folha])) for c in (L, a, b))
    cols = [L - mL, a - ma, b - mb, np.hypot(a, b)]
    for s in ESCALAS:
        Ls, as_, bs = (_media_na_folha(c, w, s) for c in (L, a, b))
        cols += [L - Ls, a - as_, b - bs, Ls - mL]
    for s in (1.0, 2.0):
        g = cv2.GaussianBlur(L, (0, 0), s)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        cols.append(np.hypot(gx, gy))
    for s in (1.5, 3.0):
        cols.append(cv2.Laplacian(cv2.GaussianBlur(L, (0, 0), s), cv2.CV_32F) * s * s)
    dist = cv2.distanceTransform(folha.astype(np.uint8), cv2.DIST_L2, 3)
    cols.append(np.minimum(dist, 15))
    for s in FORMA:
        cols += list(_hessiana(L, s))
    return np.stack([c.ravel()[idx] for c in cols], axis=1).astype(np.float32)


def amostras(exemplos: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
             semente: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Atributos (X) e classes (y: 1 lesão, 0 sadio) dos exemplos [(img, folha, lesão, sadio)].

    Sadio = o marcado como sadio + uma amostra do restante da folha longe das lesões marcadas
    (o tecido comum também precisa estar representado).
    """
    rng = np.random.default_rng(semente)
    X_pos, X_neg = [], []
    for img, folha, pos, neg in exemplos:
        F = atributos(img, folha)
        lin = np.flatnonzero(folha.ravel())
        pos_f, neg_f = pos.ravel()[lin], neg.ravel()[lin]
        perto = cv2.dilate(pos.astype(np.uint8), np.ones((9, 9), np.uint8)).ravel()[lin] > 0
        resto = ~perto & ~neg_f
        X_pos.append(F[pos_f])
        X_neg.append(F[neg_f])
        n_resto = min(int(resto.sum()), max(4 * int(pos_f.sum()), 20_000))
        if n_resto:
            X_neg.append(F[rng.choice(np.flatnonzero(resto), n_resto, replace=False)])
    Xp = np.concatenate(X_pos) if X_pos else np.zeros((0, len(nomes_atributos())), np.float32)
    Xn = np.concatenate(X_neg) if X_neg else np.zeros((0, len(nomes_atributos())), np.float32)
    return np.concatenate([Xp, Xn]), np.r_[np.ones(len(Xp), np.uint8), np.zeros(len(Xn), np.uint8)]


def treinar_amostras(X: np.ndarray, y: np.ndarray, semente: int = 0) -> dict:
    """Floresta aleatória sobre as amostras (limitadas a MAX_POR_CLASSE por classe)."""
    rng = np.random.default_rng(semente)
    partes = []
    for c in (1, 0):
        i = np.flatnonzero(y == c)
        if len(i) > MAX_POR_CLASSE:
            i = rng.choice(i, MAX_POR_CLASSE, replace=False)
        partes.append(i)
    if len(partes[0]) == 0:
        raise ValueError("Nenhum pixel de lesão marcado para treinar.")
    if len(partes[1]) == 0:
        raise ValueError("Nenhum pixel de tecido sadio disponível para treinar.")
    sel = np.concatenate(partes)
    X, y = X[sel].astype(np.float32), y[sel]
    rf = RandomForestClassifier(n_estimators=80, max_depth=18, min_samples_leaf=2,
                                class_weight="balanced_subsample", n_jobs=-1, random_state=semente)
    rf.fit(X, y)
    return {"versao": VERSAO, "modelo": rf, "X": X, "y": y,
            "n_lesao": int((y == 1).sum()), "n_sadio": int((y == 0).sum())}


def treinar(exemplos: list, base: dict | None = None, semente: int = 0) -> dict:
    """Treina com os exemplos novos somados às amostras de um modelo anterior (`base`)."""
    X, y = amostras(exemplos, semente)
    if base is not None:
        X, y = np.concatenate([base["X"], X]), np.r_[base["y"], y]
    return treinar_amostras(X, y, semente)


def probabilidade(pacote: dict, img_bgr: np.ndarray, folha: np.ndarray,
                  bloco: int = 400_000) -> np.ndarray:
    """Mapa de probabilidade de lesão (0–1) na forma da imagem; 0 fora da folha."""
    prob = np.zeros(folha.shape, np.float32)
    if not folha.any():
        return prob
    F = atributos(img_bgr, folha)
    p = np.empty(len(F), np.float32)
    for i in range(0, len(F), bloco):
        p[i:i + bloco] = pacote["modelo"].predict_proba(F[i:i + bloco])[:, 1]
    prob.ravel()[np.flatnonzero(folha.ravel())] = p
    return prob


def salvar(pacote: dict) -> bytes:
    """Arquivo do modelo = só as amostras de treino (.npz, sem pickle).

    Carregar um modelo nunca executa código vindo do arquivo (pickle/joblib permitiriam):
    o modelo é retreinado a partir das amostras, em segundos. E os exemplos acumulam:
    treinar com um modelo carregado como base soma as amostras antigas às novas.
    """
    buf = io.BytesIO()
    np.savez_compressed(buf, X=pacote["X"].astype(np.float16), y=pacote["y"],
                        versao=np.array(VERSAO), atributos=np.array(nomes_atributos()))
    return buf.getvalue()


def carregar(dados: bytes) -> dict:
    try:
        with np.load(io.BytesIO(dados), allow_pickle=False) as z:
            versao, nomes = int(z["versao"]), list(z["atributos"])
            X, y = z["X"].astype(np.float32), z["y"].astype(np.uint8)
    except Exception as e:
        raise ValueError("Arquivo de modelo inválido.") from e
    if versao != VERSAO or nomes != nomes_atributos():
        raise ValueError("Arquivo de modelo de outra versão do app; treine novamente.")
    return treinar_amostras(X, y)
