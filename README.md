# Análise de severidade de doenças foliares (PlantCV + Streamlit)

Mede a % de área foliar lesionada e atribui a nota de uma escala diagramática
(13 escalas publicadas para soja, milho e cana, além de Horsfall & Barratt; ver *Escalas*).

## Como rodar

Duplo clique em `executar.bat` (cria o ambiente na primeira vez), ou:

```powershell
py install 3.12                       # se ainda não tiver o Python 3.12
py -V:3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\streamlit.exe run app.py
```

## Deploy (Streamlit Community Cloud — gratuito)

1. Crie um repositório no GitHub (pode ser privado) e envie o projeto:
   ```powershell
   git remote add origin https://github.com/<usuario>/<repositorio>.git
   git push -u origin main
   ```
2. Em <https://share.streamlit.io>, entre com a conta do GitHub → **Create app** →
   escolha o repositório, branch `main` e arquivo `app.py`.
3. Em **Advanced settings**, selecione **Python 3.12** (o PlantCV não instala no 3.14).
4. **Deploy**. A primeira instalação leva alguns minutos (PlantCV, SciPy, OpenCV).

Arquivos usados pelo deploy:

| Arquivo | Função |
|---|---|
| `requirements.txt` | pacotes Python (versões fixas, as mesmas testadas localmente) |
| `packages.txt` | bibliotecas do sistema exigidas pelo OpenCV no Linux (`libgl1`, `libglib2.0-0`) |
| `.streamlit/config.toml` | upload de até 50 MB por arquivo; sem telemetria |
| `.gitignore` | mantém `.venv/` e `__pycache__/` fora do repositório |

Limites do plano gratuito: ~1 GB de RAM e o app "dorme" após alguns dias sem acesso (acorda
no primeiro clique). Por isso fotos com lado maior que 2400 px são reduzidas antes da análise
(pico de ~360 MB numa foto de 4000 px; áreas continuam em px da imagem original) e o cache
guarda no máximo 16 resultados. Repositório privado: o app pode ficar restrito a e-mails
convidados em **Settings → Sharing**.

## Modos de análise

Formatos aceitos: PNG, JPG/JPEG, TIF/TIFF, BMP e **HEIC/HEIF** (fotos de iPhone, lidas com
`pillow-heif`, com a orientação EXIF aplicada — foto tirada na vertical aparece em pé).

**Escala diagramática / desenho P&B** — figuras como a de `exemplos/escala_ferrugem_alaranjada.png`.
1. Linhas do desenho (contorno e nervura) são separadas das lesões por geometria:
   abertura morfológica com elementos lineares longos.
2. Cada contorno vira uma folha; o limbo é a envoltória convexa erodida, sem a faixa da nervura central.
3. Severidade binária (pixel mais escuro que o limiar) e ponderada (proporcional à escuridão do pixel).
4. Quando o nº de folhas é igual ao nº de notas, compara cada folha com a nota da escala (validação).

Na figura de exemplo: 9/9 notas corretas (medida binária: 0,06 · 0,16 · 0,36 · 1,01 · 2,23 · 7,12 · 13,39 · 31,27 · 49,90%).

**Foto de folha (qualquer doença)** — modo padrão. Gera a imagem em preto e branco no
padrão das escalas diagramáticas (folha branca com contorno preto, sintomas em preto) e a severidade.
1. Folha(s) — o tipo de fundo é detectado pela borda da imagem (ou escolhido na barra lateral):
   - **Uniforme** (escaneada, papel, cartolina — reconhecido quando ≥ 30% da borda é branco,
     cinza ou preto, mesmo com folhas encostando na borda): folha = pixels com cor diferente do fundo
     (Otsu na distância LAB), `fill` e `fill_holes` do PlantCV. Pega lesões que chegam à margem.
     Várias folhas na mesma imagem são analisadas separadamente.
   - **Complexo** (foto de campo: folha encostando na borda, vasos, solo, outras plantas):
     folha = pixels verdes (canal `a` baixo) + fechamento morfológico e `fill_holes` para
     incluir as manchas. Só a maior folha é analisada.
2. Sintoma: canal `a` do LAB (eixo verde ↔ vermelho; 128 = sem cor). Todo sintoma deixa de
   ser verde — necrose bege/marrom, clorose e halos amarelos, pústulas alaranjadas, manchas
   pretas, oídio branco. Limiar automático relativo ao verde sadio da imagem: percentil 10
   de `a` em todas as folhas + margem (padrão 17), pois câmera, luz e cultivar mudam o verde
   do tecido sadio (escaneada: a ≈ 99; campo: a ≈ 104; print de baixa resolução: a ≈ 111).
   A referência é comum às folhas da imagem e não se contamina com folhas muito doentes.
3. Detalhe fino: a cor de fotos/JPEG tem metade da resolução da luminância, e sozinha gera
   manchas "inchadas". Por isso (a) imagens com lado menor que 1200 px são ampliadas
   (bicúbica) antes da análise; (b) o canal `a` passa por um filtro guiado pela luminância,
   que devolve à cor os contornos finos; (c) na faixa duvidosa de cor só conta o pixel mais
   escuro que a vizinhança (pústula/mancha), enquanto cor claramente não-verde (necrose bege,
   oídio) conta sempre. Áreas são sempre informadas em px da imagem original.
4. Pecíolo removido por abertura morfológica (estrutura fina presa ao limbo).
5. Severidade = área sintomática / área da folha; nº de lesões e área média.
6. A escala é opcional ("Nenhuma" mostra só a %).
7. Imagem com as manchas numeradas da maior (nº 1) para a menor, numeração única na imagem
   (só as N maiores recebem número, padrão 50, para continuar legível), com tabela por mancha (folha, área, % da folha, % do total lesionado) para baixar em CSV.

A nota é a da escala mais próxima em escala logarítmica (escalas diagramáticas seguem a lei de Weber-Fechner).

## Escalas

Conferidas na publicação original (resumo, texto ou figura). Referência completa no app
(barra lateral → "Referência da escala") e em `analise/escala.py`.

| Cultura | Doença | Notas (% de área foliar lesionada) | Fonte |
|---|---|---|---|
| Soja | Ferrugem asiática (*Phakopsora pachyrhizi*) | 0,6 · 2 · 7 · 18 · 42 · 78,5 | Godoy et al., Fitopatol. Bras. 31:63, 2006 |
| Soja | Mancha-alvo (*Corynespora cassiicola*) | 1 · 2 · 5 · 9 · 19 · 33 · 52 | Soares et al., Trop. Plant Pathol. 34:333, 2009 |
| Soja | Doenças de final de ciclo (septoriose + cercospora) | 2,4 · 15,2 · 25,9 · 40,5 · 66,6 | Martins et al., Fitopatol. Bras. 29:179, 2004 |
| Soja | Crestamento foliar (*Cercospora kikuchii*) | 1 · 4,5 · 17,5 · 50 · 82,2 · 95 · 99 | Lavilla et al., Agron. Mesoam., 2021 |
| Soja | Oídio (*Microsphaera diffusa*) | 0,62 · 1,47 · 3,29 · 7,7 · 20,14 · 27,05 · 43,6 · >60 | Mattiazzi, dissertação ESALQ/USP, 2003 |
| Soja | Míldio (*Peronospora manshurica*) | 0,08 · 0,30 · 1,10 · 3,39 · 12,85 · 34,92 · 66,13 · 87,65 | Kowata et al., Sci. Agrar. 9:105, 2008 |
| Milho | Helmintosporiose comum (*Exserohilum turcicum*) | 0,5 · 1 · 2,5 · 6,5 · 15,5 · 30 · 54 | Lazaroto et al., Ciênc. Rural 42:2131, 2012 |
| Milho | Mancha branca (*Phaeosphaeria maydis*) | 1,1 · 2,1 · 4,2 · 7,9 · 14,4 · 25 · 39,7 | Malagi et al., Rev. Ciênc. Agron. 42(3), 2011 |
| Milho | Antracnose foliar (*Colletotrichum graminicola*) | 0 · 0,5 · 2 · 10 · 20 · 30 · 50 · 60 · 70 · 80 · 95 · 100 | Trojan et al., Summa Phytopathol. 44:56, 2018 |
| Milho | Doenças foliares, notas 1–9 | 0 · 1 · 10 · 20 · 30 · 40 · 60 · 80 · >80 | Guia Agroceres de Sanidade, 1996 |
| Cana | Ferrugem alaranjada (*Puccinia kuehnii*) | 0,06 · 0,14 · 0,36 · 0,89 · 2,17 · 5,18 · 11,87 · 24,92 · 45 | Klosowski et al., Trop. Plant Pathol. 38(2), 2013 |
| Cana | Ferrugem marrom (*Puccinia melanocephala*) — faixas | 0 · 0–1 · 1–3 · 3–5 · 5–8 · 8–11 · 11–15 · 15–25 · >25 | Bonadiman, dissertação UFES, 2021 |
| Qualquer | Horsfall & Barratt — faixas, notas 0–11 | 0 · 0–3 · 3–6 · 6–12 · 12–25 · 25–50 · 50–75 · 75–88 · 88–94 · 94–97 · 97–100 · 100 | Horsfall & Barratt, Phytopathology 35:655, 1945 |

Escalas por valores: nota = valor de referência mais próximo em escala log. Escalas por
faixas: nota = faixa que contém a severidade medida.

## Estrutura

```
app.py                  interface Streamlit
analise/escala.py       escalas de referência e classificação
analise/diagramatica.py análise de desenhos/escalas P&B
analise/foto.py         análise de fotos coloridas (qualquer folha/doença) e saída P&B
exemplos/               escala diagramática de exemplo
requirements.txt        pacotes Python
packages.txt            pacotes do sistema (deploy Linux)
.streamlit/config.toml  configuração do servidor
```

## Dicas para fotos de campo

- Melhor resultado: folha destacada sobre fundo uniforme (papel branco ou cartolina) com folga
  em volta, luz difusa, sem sombras nem reflexos. Sombras fortes e reflexos podem virar sintoma.
- Em foto de campo, lesões que chegam à margem da folha podem ficar fora do contorno.
- Calibre a margem de verde com algumas folhas avaliadas visualmente e mantenha-a fixa no lote.
- Halos/cloroses leves não aparecem → diminua a margem; nervuras ou tecido sadio marcados → aumente.
- Folhas naturalmente amareladas ou avermelhadas (senescência, cultivares roxas) exigem
  limiar ajustado, pois o método mede perda de verde.
