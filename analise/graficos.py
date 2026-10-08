"""Gráficos (Altair) para interpretar a severidade nas faixas de nota de uma escala.

- posicao_na_escala: régua com as faixas de nota (eixo log, como as escalas
  diagramáticas) e cada folha marcada na faixa em que caiu.
- folhas_por_nota: quantas folhas caíram em cada nota (distribuição do lote).
- severidade_por_folha: % de cada folha, colorida pela nota.

Cor das notas: rampa sequencial de um só tom (azul), mais escuro = nota maior no tema
claro e mais claro = nota maior no tema escuro. Uma rampa de um tom só distingue
com segurança até ~8 passos (ΔL OKLCH >= 0,06), por isso cada faixa traz o número
da nota escrito e a posição no eixo carrega a severidade: a cor nunca é a única pista.
"""

import math

import altair as alt
import numpy as np
import pandas as pd

from .escala import Escala

# Rampa azul da paleta de referência. Claro: passos 250 -> 700 (o mais claro ainda
# tem 2,1:1 sobre o fundo branco). Escuro: 600 -> 100 (o mais escuro tem 2,3:1).
_RAMPA = {
    False: ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
            "#1c5cab", "#184f95", "#104281", "#0d366b"],
    True: ["#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7",
           "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"],
}
# Tinta e fundo de cada tema do Streamlit (o anel de 2px entre marcas usa a cor do fundo).
_TEMA = {
    False: {"fundo": "#ffffff", "tinta": "#0b0b0b", "secundaria": "#52514e", "serie": "#2a78d6"},
    True: {"fundo": "#0e1117", "tinta": "#ffffff", "secundaria": "#c3c2b7", "serie": "#3987e5"},
}
_EIXO_PCT = "replace(format(datum.value, '~g'), '.', ',') + '%'"


def _hex_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


def _lum(rgb: np.ndarray) -> float:
    c = rgb / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def cores_notas(n: int, escuro: bool) -> list[str]:
    """n cores da rampa, distribuídas por toda a extensão (interpolação entre passos)."""
    rampa = [_hex_rgb(h) for h in _RAMPA[escuro]]
    if n == 1:
        return [_RAMPA[escuro][len(rampa) // 2]]
    cores = []
    for t in np.linspace(0, len(rampa) - 1, n):
        i = min(int(t), len(rampa) - 2)
        c = rampa[i] + (rampa[i + 1] - rampa[i]) * (t - i)
        cores.append("#" + "".join(f"{int(round(v)):02x}" for v in c))
    return cores


def _tinta_sobre(cor: str) -> str:
    """Preto ou branco, o que tiver mais contraste sobre a cor da faixa."""
    l = _lum(_hex_rgb(cor))
    return "#0b0b0b" if (l + 0.05) / 0.05 >= 1.05 / (l + 0.05) else "#ffffff"


def _br(v: float) -> str:
    return f"{v:.3g}".replace(".", ",")


def _cortes(escala: Escala) -> list[float]:
    """Severidade onde muda a nota (mesma regra de Escala.classificar)."""
    if escala.limites is not None:
        return [float(x) for x in escala.limites[:-1]]
    v = escala.valores
    positivos = [x for x in v if x > 0]
    return [min(positivos) / 2 if a <= 0 else math.sqrt(a * b) for a, b in zip(v, v[1:])]


def faixas(escala: Escala, sev_min: float | None = None) -> tuple[pd.DataFrame, float]:
    """Tabela de faixas (nota, ini, fim, texto) no domínio de exibição [lo, 100]."""
    cortes = _cortes(escala)
    candidatos = [c for c in cortes if c > 0] + ([sev_min] if sev_min and sev_min > 0 else [])
    lo = 10 ** math.floor(math.log10(min(candidatos) / 3)) if candidatos else 0.01
    lo = max(lo, 1e-3)
    bordas = [lo] + [min(max(c, lo), 100.0) for c in cortes] + [100.0]
    linhas = []
    for i in range(len(escala.valores)):
        ini, fim = bordas[i], max(bordas[i + 1], bordas[i])
        if fim <= ini * 1.0001:          # faixa "0%" ou "100%": largura mínima para aparecer
            if i == 0:
                fim = ini * 1.6
                bordas[1] = max(bordas[1], fim)
            else:
                continue
        real_ini = 0.0 if i == 0 else cortes[i - 1]
        real_fim = cortes[i] if i < len(cortes) else 100.0
        if escala.limites is not None:
            if real_fim == real_ini:
                texto = f"{_br(real_fim)}%"
            elif np.isinf(escala.limites[i]):
                texto = f">{_br(real_ini)}%"
            else:
                texto = f"{_br(real_ini)}–{_br(real_fim)}%"
        else:
            texto = f"ref. {_br(escala.valores[i])}% (faixa {_br(real_ini)}–{_br(min(real_fim, 100))}%)"
        linhas.append({"nota": i + escala.nota_inicial, "ini": ini, "fim": fim,
                       "centro": math.sqrt(ini * fim), "faixa": texto,
                       "decadas": math.log10(fim / ini)})
    return pd.DataFrame(linhas), lo


def _cor_scale(escala: Escala, escuro: bool) -> alt.Scale:
    notas = [i + escala.nota_inicial for i in range(len(escala.valores))]
    return alt.Scale(domain=notas, range=cores_notas(len(notas), escuro))


def posicao_na_escala(folhas: pd.DataFrame, escala: Escala, escuro: bool) -> alt.LayerChart:
    """folhas: colunas Folha, Severidade (%), Nota (e opcionalmente Imagem)."""
    t = _TEMA[escuro]
    positivos = folhas.loc[folhas["Severidade (%)"] > 0, "Severidade (%)"]
    df_f, lo = faixas(escala, float(positivos.min()) if len(positivos) else None)
    cores = cores_notas(len(escala.valores), escuro)
    df_f["tinta"] = [_tinta_sobre(cores[n - escala.nota_inicial]) for n in df_f["nota"]]
    df_f["rotulo"] = df_f["nota"].astype(str)

    x = alt.X("ini:Q", scale=alt.Scale(type="log", domain=[lo, 100], nice=False),
              axis=alt.Axis(title="Severidade (% da área foliar) — escala log", labelExpr=_EIXO_PCT,
                            values=[10 ** k for k in range(int(math.log10(lo)), 3)], grid=False))
    base = alt.Chart(df_f)
    tooltip_faixa = [alt.Tooltip("nota:O", title="Nota"), alt.Tooltip("faixa:N", title="Faixa")]
    bandas = base.mark_rect(stroke=t["fundo"], strokeWidth=2, cornerRadius=4).encode(
        x=x, x2="fim:Q", y=alt.value(52), y2=alt.value(92),
        color=alt.Color("nota:O", scale=_cor_scale(escala, escuro), legend=None),
        tooltip=tooltip_faixa,
    )
    rotulos = base.transform_filter("datum.decadas >= 0.12").mark_text(
        fontSize=12, fontWeight="bold").encode(
        x="centro:Q", y=alt.value(72), text="rotulo:N", color=alt.Color("tinta:N", scale=None),
        tooltip=tooltip_faixa,
    )
    # Folhas: pontos acima da régua, espalhados na vertical para não se sobreporem.
    pts = folhas.copy()
    pts["x"] = pts["Severidade (%)"].clip(lower=lo * 1.05, upper=100)
    # Folhas sem sintoma (0%) não têm posição no eixo log: viram um marcador só, com a contagem.
    zeros = pts[pts["Severidade (%)"] <= 0]
    pts = pts[pts["Severidade (%)"] > 0].sort_values("x").reset_index(drop=True)
    # Empilhamento sem sorteio: cada ponto vai para a 1ª linha livre (distância mínima no eixo log).
    linhas_y, ultimo = [38, 22, 6], [-np.inf] * 3
    folga = 0.05 * math.log10(100 / lo)
    ys = []
    for x in np.log10(pts["x"]):
        i = next((k for k in range(3) if x - ultimo[k] >= folga), int(np.argmin(ultimo)))
        ultimo[i] = x
        ys.append(linhas_y[i])
    pts["y"] = ys
    pts["rotulo"] = pts["Folha"].astype(str)
    tooltip_pts = [c for c in ("Imagem", "Folha") if c in pts.columns]
    tooltip = [*tooltip_pts, alt.Tooltip("Severidade (%):Q", format=".3f"), alt.Tooltip("Nota:O")]
    pontos = alt.Chart(pts).mark_circle(size=90, opacity=1, stroke=t["fundo"], strokeWidth=2,
                                        color=t["tinta"]).encode(
        x="x:Q", y=alt.Y("y:Q", scale=None, axis=None), tooltip=tooltip)
    if len(pts) <= 10:  # rótulo direto só enquanto cabe; acima disso, tooltip e tabela
        pontos = pontos + alt.Chart(pts).mark_text(align="left", dx=8, fontSize=11,
                                                   color=t["secundaria"]).encode(
            x="x:Q", y=alt.Y("y:Q", scale=None, axis=None), text="rotulo:N", tooltip=tooltip)
    if len(zeros):
        n0 = len(zeros)
        z = pd.DataFrame({"x": [lo * 1.05], "y": [38],
                          "t": [f"{n0} folha{'s' if n0 > 1 else ''} com 0% (sem sintoma)"],
                          "folhas": [", ".join(zeros["Folha"].astype(str))]})
        tip0 = [alt.Tooltip("t:N", title="Sem sintoma"), alt.Tooltip("folhas:N", title="Folhas")]
        pontos = pontos + alt.Chart(z).mark_point(shape="diamond", size=110, filled=True,
                                                  color=t["tinta"], stroke=t["fundo"], strokeWidth=2).encode(
            x="x:Q", y=alt.Y("y:Q", scale=None, axis=None), tooltip=tip0) \
            + alt.Chart(z).mark_text(align="left", dx=9, fontSize=11, color=t["secundaria"]).encode(
            x="x:Q", y=alt.Y("y:Q", scale=None, axis=None), text="t:N", tooltip=tip0)
    sentido = "mais claro" if escuro else "mais escuro"
    return (bandas + rotulos + pontos).properties(
        height=120, title=alt.TitleParams(f"Posição na escala — {escala.nome}",
                                          subtitle=f"Cada ponto é uma folha; faixas numeradas = notas da "
                                                   f"escala ({sentido} = nota maior)",
                                          anchor="start"))


def folhas_por_nota(folhas: pd.DataFrame, escala: Escala, escuro: bool) -> alt.LayerChart:
    t = _TEMA[escuro]
    df_f, _ = faixas(escala)
    notas = [i + escala.nota_inicial for i in range(len(escala.valores))]
    cont = folhas.groupby("Nota").agg(n=("Folha", "size"),
                                      lista=("Folha", lambda s: ", ".join(map(str, s)))).reindex(notas)
    cont = cont.fillna({"n": 0, "lista": "—"}).reset_index().rename(columns={"index": "Nota"})
    cont["Nota"] = notas
    cont["faixa"] = cont["Nota"].map(dict(zip(df_f["nota"], df_f["faixa"]))).fillna("")
    cont["pct"] = 100 * cont["n"] / max(1, int(cont["n"].sum()))
    base = alt.Chart(cont).encode(
        x=alt.X("Nota:O", sort=notas, axis=alt.Axis(labelAngle=0, title="Nota da escala")),
        tooltip=[alt.Tooltip("Nota:O"), alt.Tooltip("faixa:N", title="Faixa"),
                 alt.Tooltip("n:Q", title="Nº de folhas"), alt.Tooltip("pct:Q", title="% das folhas", format=".1f"),
                 alt.Tooltip("lista:N", title="Folhas")],
    )
    n_max = int(cont["n"].max()) if len(cont) else 1
    passo = max(1, math.ceil(n_max / 6))  # marcas inteiras, no máximo ~6
    barras = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, stroke=t["fundo"],
                           strokeWidth=2).encode(
        y=alt.Y("n:Q", title="Nº de folhas", scale=alt.Scale(domain=[0, n_max + passo], nice=False),
                axis=alt.Axis(values=list(range(0, n_max + passo + 1, passo)), format="d")),
        color=alt.Color("Nota:O", scale=_cor_scale(escala, escuro), legend=None),
    )
    rotulos = base.transform_filter("datum.n > 0").mark_text(dy=-8, fontSize=12,
                                                             color=t["secundaria"]).encode(
        y="n:Q", text=alt.Text("n:Q", format="d"))
    return (barras + rotulos).properties(
        height=260, title=alt.TitleParams("Folhas por nota", subtitle="Distribuição do lote nas faixas da escala",
                                          anchor="start"))


def severidade_por_folha(folhas: pd.DataFrame, escala: Escala | None, escuro: bool) -> alt.LayerChart:
    t = _TEMA[escuro]
    df = folhas.copy()
    df["Folha"] = df["Folha"].astype(str)
    tooltip = [c for c in ("Imagem", "Folha") if c in df.columns] + \
        [alt.Tooltip("Severidade (%):Q", format=".3f")] + (["Nota:O"] if escala else [])
    ordem = list(df["Folha"])
    base = alt.Chart(df).encode(x=alt.X("Folha:N", sort=ordem, axis=alt.Axis(labelAngle=0, title="Folha")),
                                tooltip=tooltip)
    if escala:
        cor = alt.Color("Nota:O", scale=_cor_scale(escala, escuro),
                        legend=alt.Legend(title="Nota", orient="right"))
    else:
        cor = alt.value(t["serie"])
    barras = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, stroke=t["fundo"],
                           strokeWidth=2).encode(
        y=alt.Y("Severidade (%):Q", title="Severidade (%)", axis=alt.Axis(labelExpr=_EIXO_PCT)),
        color=cor)
    media = alt.Chart(pd.DataFrame({"m": [df["Severidade (%)"].mean()]})).mark_rule(
        strokeDash=[4, 4], strokeWidth=1.5, color=t["secundaria"]).encode(
        y="m:Q", tooltip=[alt.Tooltip("m:Q", title="Média (%)", format=".3f")])
    return (barras + media).properties(
        height=260, title=alt.TitleParams("Severidade por folha",
                                          subtitle=f"Linha tracejada = média das folhas "
                                                   f"({_br(df['Severidade (%)'].mean())}%)", anchor="start"))
