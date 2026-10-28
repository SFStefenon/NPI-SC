"""Soma as unidades interrompidas e calcula as médias de DEC/FEC por mês.

As médias usam os registros originais do mês, agrupados por DATA INICIO.
Meses sem registros entre o primeiro e o último mês recebem zero.
DATA identifica cada mês pelo seu primeiro dia (AAAA-MM-01).
"""
import csv
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

ENTRADA = Path(globals().get("ENTRADA", Path(__file__).with_name("DEC_DOC.csv")))
SAIDA = Path(globals().get("SAIDA", Path(__file__).with_name("DEC_DOC_mensal.csv")))

# Por mês: [soma de unidades interrompidas, soma DEC, soma FEC, registros].
mensal = defaultdict(lambda: [0, Decimal("0"), Decimal("0"), 0])
with ENTRADA.open("r", encoding="utf-8-sig", newline="") as f:
    for linha in csv.DictReader(f, delimiter=";"):
        data = datetime.strptime(linha["DATA INICIO"].strip(), "%d/%m/%Y %H:%M")
        mes = data.date().replace(day=1)
        valores = mensal[mes]
        quantidade = linha["QTDE UC EQPTO INTERROMPIDA"].strip()
        if quantidade:
            valores[0] += int(quantidade)
        valores[1] += Decimal(linha["DEC"].strip().replace(",", "."))
        valores[2] += Decimal(linha["FEC"].strip().replace(",", "."))
        valores[3] += 1

if not mensal:
    raise ValueError("O arquivo de entrada não contém registros.")

mes, ultimo = min(mensal), max(mensal)
with SAIDA.open("w", encoding="utf-8", newline="") as f:
    escritor = csv.writer(f)
    escritor.writerow(["DATA", "QTDE UC EQPTO INTERROMPIDA", "DEC", "FEC"])
    while mes <= ultimo:
        quantidade, dec, fec, n = mensal[mes]
        escritor.writerow([
            mes.isoformat(), quantidade,
            f"{dec / n:.8f}" if n else "0",
            f"{fec / n:.8f}" if n else "0",
        ])
        mes = mes.replace(year=mes.year + 1, month=1) if mes.month == 12 else mes.replace(month=mes.month + 1)

print(f"Arquivo gerado: {SAIDA}")


###############################################################################

"""Carrega DEC_DOC_diario.csv e produz um gráfico PDF com três séries temporais."""
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Configuração
CSV_PATH = Path(__file__).resolve().with_name("DEC_DOC_mensal.csv")
OUTPUT_PDF = Path(__file__).resolve().with_name("series_temporais_DEC_FEC_mensal.pdf")
START_DATE = None          # Ex.: "2020-01-01" ou None
END_DATE = None            # Ex.: "2025-12-31" ou None
SMOOTH_DAYS = 30           # Média móvel causal, apenas dias anteriores e atual
SHOW_PLOT = False          # True para abrir a figura ao executar

with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
    linhas = list(csv.DictReader(f))

campos = ["DATA", "QTDE UC EQPTO INTERROMPIDA", "DEC", "FEC"]
if not linhas or any(c not in linhas[0] for c in campos):
    raise ValueError(f"O CSV deve conter as colunas: {', '.join(campos)}")

registros = []
for linha in linhas:
    dia = datetime.strptime(linha["DATA"].strip(), "%Y-%m-%d")
    if START_DATE and dia < datetime.fromisoformat(START_DATE):
        continue
    if END_DATE and dia > datetime.fromisoformat(END_DATE):
        continue
    valores = [float(linha[c].replace(",", ".")) for c in campos[1:]]
    registros.append((dia, *valores))

if not registros:
    raise ValueError("Nenhuma data encontrada no intervalo solicitado.")
registros.sort(key=lambda x: x[0])
datas = [item[0] for item in registros]
serie = np.asarray([item[1:] for item in registros], dtype=float)
if not np.isfinite(serie).all():
    raise ValueError("O CSV contém valores DEC/FEC ou interrupções não numéricos.")

# Média móvel: sem utilizar informações de dias futuros.
def media_movel(x, janela):
    acumulada = np.r_[0.0, np.cumsum(x)]
    fim = np.arange(1, len(x) + 1)
    inicio = np.maximum(0, fim - janela)
    return (acumulada[fim] - acumulada[inicio]) / (fim - inicio)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": True,
    "axes.edgecolor": "#CDD2D8", "axes.linewidth": 0.8,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.grid": False, "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})

fig, eixos = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
fig.subplots_adjust(left=0.092, right=0.955, top=0.785, bottom=0.105, hspace=0.43)
fig.text(0.092, 0.957, "INDICADORES DE CONTINUIDADE", fontsize=10, weight="bold", color="#738091")
fig.text(0.092, 0.922, "Evolução temporal  |  Interrupções, DEC e FEC",
         fontsize=20, weight="medium", color="#182331")
fig.text(0.092, 0.892,
         f"{datas[0]:%d/%m/%Y}  —  {datas[-1]:%d/%m/%Y}     ·     "
         f"{len(datas):,} dias     ·     tendência de {SMOOTH_DAYS} dias".replace(',', '.'),
         fontsize=9.5, color="#748091")

config = [
    ("UNIDADES CONSUMIDORAS INTERROMPIDAS", "Soma diária de QTDE UC EQPTO INTERROMPIDA", "#246B85"),
    ("DEC", "Média diária do indicador DEC", "#246C63"),
    ("FEC", "Média diária do indicador FEC", "#9B6342"),
]

def numero(x, indice):
    if indice == 0:
        return f"{x:,.0f}".replace(",", ".")
    return f"{x:.4f}".replace(".", ",") if indice == 2 else f"{x:.3f}".replace(".", ",")

for i, (ax, (rotulo, descricao, cor)) in enumerate(zip(eixos, config)):
    y = serie[:, i]
    suavizada = media_movel(y, SMOOTH_DAYS)
    ax.plot(datas, y, color=cor, lw=0.65, alpha=0.24, label="Valor diário", zorder=2)
    ax.fill_between(datas, 0, suavizada, color=cor, alpha=0.08, zorder=1)
    ax.plot(datas, suavizada, color=cor, lw=2.0, label=f"Média móvel ({SMOOTH_DAYS} dias)", zorder=3)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos, i=i: numero(val, i)))
    ax.tick_params(axis="y", length=0, pad=11, labelsize=9, colors="#617080")
    ax.tick_params(axis="x", length=4, pad=9, labelsize=9, colors="#617080")
    ax.text(0, 1.21, rotulo, transform=ax.transAxes, fontsize=10.5,
            color="#263748", weight="bold", va="top")
    ax.text(0, 1.085, descricao, transform=ax.transAxes,
            fontsize=8.8, color="#85909B", va="top")
    ax.text(1, 1.16, f"MÉDIA / DIA   {numero(y.mean(), i)}",
            transform=ax.transAxes, ha="right", fontsize=9, color=cor, weight="bold", va="top")
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=8.7,
              handlelength=2.6, labelcolor="#546171", borderaxespad=0.5)
    ax.margins(x=0)

# Mesma escala temporal, com anos identificados de forma clara.
anos = (datas[-1] - datas[0]).days / 365.25
intervalo = 1 if anos <= 12 else 2
eixos[-1].xaxis.set_major_locator(mdates.YearLocator(intervalo))
eixos[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
eixos[-1].tick_params(labelbottom=True)
fig.text(0.092, 0.043, "Fonte: DEC_DOC_diario.csv   ·   Séries diárias; sem suavização dos valores originais",
         fontsize=8.5, color="#8993A0")
fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.26)
if SHOW_PLOT:
    plt.show()
plt.close(fig)
print(f"Gráfico gerado: {OUTPUT_PDF}")
