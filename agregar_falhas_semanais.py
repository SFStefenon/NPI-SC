"""
Aggregate DEC_DOC.csv by day, week, and month.
Generate a CSV and PDF plot for each aggregation.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Configuration
INPUT = Path(__file__).with_name("DEC_DOC.csv")
OUTPUT = INPUT.parent
SHOW_PLOTS = False

COLUMNS = ["QTDE UC EQPTO INTERROMPIDA", "DEC", "FEC"]
PERIODS = {
    "day": "D",
    "week": "W-MON",
    "month": "MS",
}

# Load data
df = pd.read_csv(INPUT, sep=";", dtype=str, encoding="utf-8-sig")
df["DATA"] = pd.to_datetime(
    df["DATA INICIO"].str.strip(),
    format="%d/%m/%Y %H:%M"
)

for col in COLUMNS:
    df[col] = pd.to_numeric(
        df[col].str.strip().str.replace(",", ".", regex=False),
        errors="coerce"
    )

df = df.set_index("DATA").sort_index()

if df.empty:
    raise ValueError("No records found in the input CSV.")

# Plot configuration
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

# Aggregate and plot
for name, freq in PERIODS.items():

    data = df.resample(
        freq, label="left", closed="left"
    ).agg({
        COLUMNS[0]: "sum",
        "DEC": "mean",
        "FEC": "mean",
    }).fillna(0)

    # Save CSV
    csv_path = OUTPUT / f"DEC_DOC_{name}.csv"
    data.to_csv(
        csv_path,
        index_label="DATA",
        date_format="%Y-%m-%d",
        float_format="%.8f"
    )

    # Generate plot
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 7), sharex=True
    )

    labels = ["Interrupted units", "DEC", "FEC"]

    for ax, col, label in zip(axes, COLUMNS, labels):
        ax.plot(
            data.index,
            data[col],
            color="#245C78",
            linewidth=1.0
        )
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.margins(x=0)
        ax.tick_params(axis="both", labelsize=9)

    locator = mdates.AutoDateLocator()
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator)
    )
    axes[-1].set_xlabel("Date")

    fig.tight_layout()

    # Save PDF
    pdf_path = OUTPUT / f"DEC_FEC_{name}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)

    print(f"Generated: {csv_path.name} | {pdf_path.name}")