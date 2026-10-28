"""Agrega o DEC_DOC.csv por dia de DATA INICIO e preenche dias sem registros com zero."""
import csv
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

ENTRADA = Path(__file__).with_name("DEC_DOC.csv")
SAIDA = Path(__file__).with_name("DEC_DOC_diario.csv")

# Em cada data: [soma de unidades interrompidas, soma DEC, soma FEC, quantidade de registros].
diario = defaultdict(lambda: [0, Decimal("0"), Decimal("0"), 0])
with ENTRADA.open("r", encoding="utf-8-sig", newline="") as f:
    for linha in csv.DictReader(f, delimiter=";"):
        data = datetime.strptime(linha["DATA INICIO"].strip(), "%d/%m/%Y %H:%M").date()
        valores = diario[data]
        quantidade = linha["QTDE UC EQPTO INTERROMPIDA"].strip()
        if quantidade:
            valores[0] += int(quantidade)
        valores[1] += Decimal(linha["DEC"].strip().replace(",", "."))
        valores[2] += Decimal(linha["FEC"].strip().replace(",", "."))
        valores[3] += 1

with SAIDA.open("w", encoding="utf-8", newline="") as f:
    escritor = csv.writer(f)
    escritor.writerow(["DATA", "QTDE UC EQPTO INTERROMPIDA", "DEC", "FEC"])
    data, ultima = min(diario), max(diario)
    while data <= ultima:
        quantidade, dec, fec, n = diario[data]
        escritor.writerow([
            data.isoformat(), quantidade,
            f"{dec / n:.8f}" if n else "0",
            f"{fec / n:.8f}" if n else "0",
        ])
        data += timedelta(days=1)

print(f"Arquivo gerado: {SAIDA}")
