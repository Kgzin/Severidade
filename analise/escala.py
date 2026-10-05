"""Escalas diagramáticas de referência e classificação de severidade.

Dois tipos de escala:
- por valores (a maioria das escalas diagramáticas): cada nota é uma severidade de
  referência; a nota atribuída é a mais próxima em escala logarítmica (lei de Weber-Fechner).
- por faixas (ex.: Horsfall & Barratt, diagramas com intervalos): cada nota cobre um
  intervalo de severidade; a nota atribuída é a da faixa que contém o valor medido.

Todas as escalas abaixo foram conferidas na publicação original (resumo, texto ou figura).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Escala:
    nome: str
    valores: tuple[float, ...]   # severidade (% de área foliar lesionada) de referência de cada nota
    referencia: str = ""
    limites: tuple[float, ...] | None = None  # escala por faixas: limite superior de cada nota (%)
    nota_inicial: int = 1        # número da primeira nota (Horsfall & Barratt começa em 0)

    def classificar(self, severidade: float) -> tuple[int, float]:
        """Retorna (nota, severidade de referência da nota)."""
        if self.limites is not None:
            idx = int(np.searchsorted(np.array(self.limites), severidade, side="left"))
            idx = min(idx, len(self.valores) - 1)
            return idx + self.nota_inicial, self.valores[idx]

        # Escalas diagramáticas seguem a lei de Weber-Fechner (progressão logarítmica),
        # por isso a distância é medida em log e não em valor absoluto. Nota de 0% (sem
        # sintoma) só é atribuída abaixo da metade da menor severidade positiva da escala.
        valores = np.array(self.valores, dtype=float)
        positivos = valores > 0
        if not positivos.all() and severidade < valores[positivos].min() / 2:
            idx = int(np.flatnonzero(~positivos)[0])
            return idx + self.nota_inicial, self.valores[idx]
        ref = np.where(positivos, np.log10(np.where(positivos, valores, 1)), np.inf)
        alvo = np.log10(max(severidade, 1e-3))
        idx = int(np.argmin(np.abs(ref - alvo)))
        return idx + self.nota_inicial, self.valores[idx]

    def descricao_notas(self) -> str:
        if self.limites is not None:
            partes, inf = [], 0.0
            for i, sup in enumerate(self.limites):
                n = i + self.nota_inicial
                if np.isinf(sup) and inf >= 99.9:   # nota de 100% (Horsfall & Barratt)
                    partes.append(f"{n}: 100%")
                elif sup == inf:
                    partes.append(f"{n}: {_br(sup)}%")
                elif sup >= 99.9:
                    partes.append(f"{n}: {_br(inf)}–<100%")
                elif np.isinf(sup):
                    partes.append(f"{n}: >{_br(inf)}%")
                else:
                    partes.append(f"{n}: {_br(inf)}–{_br(sup)}%")
                inf = sup
            return " · ".join(partes)
        return " · ".join(f"{i + self.nota_inicial}: {_br(v)}%" for i, v in enumerate(self.valores))


def _br(v: float) -> str:
    return f"{v:g}".replace(".", ",")


_INF = float("inf")

_LISTA = [
    # ------------------------------------------------------------------ SOJA
    Escala("Soja – Ferrugem asiática (Phakopsora pachyrhizi)",
           (0.6, 2, 7, 18, 42, 78.5),
           "Godoy, C.V.; Koga, L.J.; Canteri, M.G. Diagrammatic scale for assessment of soybean rust "
           "severity. Fitopatologia Brasileira, v.31, n.1, p.63-68, 2006."),
    Escala("Soja – Mancha-alvo (Corynespora cassiicola)",
           (1, 2, 5, 9, 19, 33, 52),
           "Soares, R.M.; Godoy, C.V.; Oliveira, M.C.N. Escala diagramática para avaliação da "
           "severidade da mancha alvo da soja. Tropical Plant Pathology, v.34, n.5, p.333-338, 2009."),
    Escala("Soja – Doenças de final de ciclo (septoriose + crestamento de cercospora)",
           (2.4, 15.2, 25.9, 40.5, 66.6),
           "Martins, M.C.; Guerzoni, R.A.; Câmara, G.M.S.; Mattiazzi, P.; Lourenço, S.A.; Amorim, L. "
           "Escala diagramática para a quantificação do complexo de doenças foliares de final de "
           "ciclo em soja. Fitopatologia Brasileira, v.29, n.2, p.179-184, 2004."),
    Escala("Soja – Crestamento foliar de cercospora (Cercospora kikuchii)",
           (1, 4.5, 17.5, 50, 82.2, 95, 99),
           "Lavilla, M.; Ivancovich, A.; Díaz-Paleo, A. Diagrammatic scale for assessment the "
           "severity of Cercospora leaf blight on soybean (Glycine max) leaflets. Agronomía "
           "Mesoamericana, 2021."),
    Escala("Soja – Oídio (Microsphaera diffusa)",
           (0.62, 1.47, 3.29, 7.7, 20.14, 27.05, 43.6, 60),
           "Mattiazzi, P. Efeito do oídio (Microsphaera diffusa) na produção e duração da área "
           "foliar sadia da soja. Dissertação (Mestrado), ESALQ/USP, Piracicaba, 2003. "
           "(Última nota: >60%.)"),
    Escala("Soja – Míldio (Peronospora manshurica)",
           (0.08, 0.30, 1.10, 3.39, 12.85, 34.92, 66.13, 87.65),
           "Kowata, L.S.; May-De-Mio, L.L.; Dalla-Pria, M.; Santos, H.A.A. Escala diagramática "
           "para avaliar severidade de míldio na soja. Scientia Agraria, v.9, n.1, p.105-110, 2008."),
    # ------------------------------------------------------------------ MILHO
    Escala("Milho – Helmintosporiose comum (Exserohilum turcicum)",
           (0.5, 1.0, 2.5, 6.5, 15.5, 30.0, 54.0),
           "Lazaroto, A.; Santos, I.; Konflanz, V.A.; Malagi, G.; Camochena, R.C. Escala "
           "diagramática para avaliação de severidade da helmintosporiose comum em milho. "
           "Ciência Rural, v.42, n.12, p.2131-2137, 2012."),
    Escala("Milho – Mancha branca (Phaeosphaeria maydis / Pantoea ananatis)",
           (1.1, 2.1, 4.2, 7.9, 14.4, 25.0, 39.7),
           "Malagi, G.; Santos, I.; Camochena, R.C.; Moccellin, R. Elaboração e validação da escala "
           "diagramática para avaliação da mancha branca do milho. Revista Ciência Agronômica, "
           "v.42, n.3, 2011."),
    Escala("Milho – Antracnose foliar (Colletotrichum graminicola)",
           (0, 0.5, 2, 10, 20, 30, 50, 60, 70, 80, 95, 100),
           "Trojan, D.G.; Dalla Pria, M.; Castro, A. Validação de escala diagramática para "
           "quantificação da severidade da antracnose da folha do milho. Summa Phytopathologica, "
           "v.44, n.1, p.56-64, 2018."),
    Escala("Milho – Doenças foliares, notas 1–9 (Guia Agroceres)",
           (0, 1, 10, 20, 30, 40, 60, 80, 90),
           "Agroceres. Guia Agroceres de sanidade. 1996. Notas 1 a 9 = 0, 1, 10, 20, 30, 40, 60, "
           "80 e >80% (última nota representada por 90%). Usada para cercosporiose, "
           "helmintosporiose, ferrugens e mancha branca (ex.: Santos et al., J. Biotechnol. "
           "Biodivers., v.2, n.3, p.67-71, 2011)."),
    # ------------------------------------------------------------------ CANA
    Escala("Cana – Ferrugem alaranjada (Puccinia kuehnii)",
           (0.06, 0.14, 0.36, 0.89, 2.17, 5.18, 11.87, 24.92, 45.0),
           "Klosowski, A.C.; Ruaro, L.; Bespalhok Filho, J.C.; May De Mio, L.L. Proposta e "
           "validação de escala para a ferrugem alaranjada da cana-de-açúcar. Tropical Plant "
           "Pathology, v.38, n.2, 2013. doi:10.1590/S1982-56762013000200012."),
    Escala("Cana – Ferrugem marrom (Puccinia melanocephala), 9 níveis",
           (0, 0.5, 2, 4, 6.5, 9.5, 13, 20, 30),
           "Bonadiman, P.A. Cana-de-açúcar: flutuação populacional de insetos-praga, fitopatometria "
           "e dinâmica temporal da ferrugem marrom sob diferentes adubações. Dissertação "
           "(Mestrado), UFES, Alegre, 2021. Diagrama de área padrão com fotografias coloridas; "
           "níveis por faixa, desprezando o halo amarelo.",
           limites=(0, 1, 3, 5, 8, 11, 15, 25, _INF)),
    # ------------------------------------------------------------------ GENÉRICA
    Escala("Genérica – Horsfall & Barratt (qualquer doença)",
           (0, 1.5, 4.5, 9, 18.5, 37.5, 62.5, 81.5, 91, 95.5, 98.5, 100),
           "Horsfall, J.G.; Barratt, R.W. An improved grading system for measuring plant disease. "
           "Phytopathology, v.35, p.655, 1945. Notas 0 a 11.",
           limites=(0, 3, 6, 12, 25, 50, 75, 88, 94, 97, 99.999, _INF), nota_inicial=0),
]

ESCALAS = {e.nome: e for e in _LISTA}


def escala_de_texto(nome: str, texto: str) -> Escala:
    """Cria uma escala a partir de valores separados por ';' (aceita vírgula decimal)."""
    valores = tuple(float(v.strip().replace(",", ".")) for v in texto.split(";") if v.strip())
    if not valores:
        raise ValueError("Informe ao menos um valor de severidade.")
    return Escala(nome, valores)
