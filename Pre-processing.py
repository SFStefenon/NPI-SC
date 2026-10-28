"""Combine DEC_DOC.csv with INMET Lages precipitation (daily, weekly, monthly)."""
from pathlib import Path
import unicodedata

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEC_FILE = ROOT / "DEC_DOC/DEC_DOC.csv"
INMET_DIR = ROOT / "INMET 16-25 LGA-SC"
OUTPUT_DIR = ROOT / "Pre-processed"
SHOW_PLOTS = False

INTERRUPTIONS = "QTDE UC EQPTO INTERROMPIDA"
RAIN = "PRECIPITACAO_TOTAL_MM"
PERIODS = {"day": "D", "week": "W-MON", "month": "MS"}


def numeric(values):
    return pd.to_numeric(
        values.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def normalize(text):
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().upper()


# Load interruption indicators.
dec = pd.read_csv(DEC_FILE, sep=";", encoding="utf-8-sig", dtype=str)
dec["DATA"] = pd.to_datetime(
    dec["DATA INICIO"].str.strip(), format="%d/%m/%Y %H:%M", errors="coerce"
).dt.normalize()
for column in (INTERRUPTIONS, "DEC", "FEC"):
    dec[column] = numeric(dec[column]).fillna(0)
dec = dec.dropna(subset=["DATA"]).set_index("DATA").sort_index()

# Load all yearly INMET files (metadata occupies the first 8 lines).
files = sorted(p for p in INMET_DIR.iterdir() if p.suffix.lower() == ".csv")
if not files:
    raise FileNotFoundError(f"No INMET CSV files found in {INMET_DIR}")

weather_parts = []
for file in files:
    raw = pd.read_csv(file, sep=";", encoding="latin-1", skiprows=8, dtype=str)
    date_col = next(c for c in raw.columns if normalize(c).startswith("DATA"))
    time_col = next(c for c in raw.columns if normalize(c).startswith("HORA"))
    rain_col = next(c for c in raw.columns if "PRECIPITACAO TOTAL" in normalize(c))
    dates = pd.to_datetime(raw[date_col].str.strip(), format="%Y-%m-%d", errors="coerce")
    if dates.notna().sum() == 0:
        dates = pd.to_datetime(raw[date_col].str.strip(), dayfirst=True, errors="coerce")
    # INMET timestamps are UTC; DEC/FEC event dates are treated as local Lages time.
    hours = (raw[time_col].str.strip().str.upper()
             .str.replace(" UTC", "", regex=False)
             .str.replace(":", "", regex=False).str.zfill(4))
    utc = pd.to_datetime(
        dates.dt.strftime("%Y-%m-%d") + " " + hours,
        format="%Y-%m-%d %H%M", errors="coerce", utc=True
    )
    local_dates = utc.dt.tz_convert("America/Sao_Paulo").dt.tz_localize(None).dt.normalize()
    precipitation = numeric(raw[rain_col])
    precipitation = precipitation.where(precipitation >= 0)
    valid_hours = precipitation.notna().astype(int)
    precipitation = precipitation.fillna(0)
    weather_parts.append(pd.DataFrame({"DATA": local_dates, RAIN: precipitation, "HORAS_INMET_VALIDAS": valid_hours}))

weather = pd.concat(weather_parts, ignore_index=True)
weather = weather.dropna(subset=["DATA"]).set_index("DATA").sort_index()
if dec.empty or weather.empty:
    raise ValueError("No valid dates found in DEC_DOC.csv or the INMET files.")

# Restrict to the dates covered by BOTH datasets.
start = max(dec.index.min(), weather.index.min())
end = min(dec.index.max(), weather.index.max())
if start > end:
    raise ValueError("The DEC/FEC and INMET files do not have overlapping dates.")
dec = dec.loc[start:end]
weather = weather.loc[start:end]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": False, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42,
})

# One output CSV and PDF per resolution.
for name, frequency in PERIODS.items():
    settings = {"label": "left", "closed": "left"}
    period_index = pd.Series(1, index=pd.date_range(start, end, freq="D")).resample(
        frequency, **settings
    ).size().index

    dec_period = dec.resample(frequency, **settings).agg({
        INTERRUPTIONS: "sum", "DEC": "mean", "FEC": "mean"
    }).reindex(period_index).fillna(0)

    # Rainfall is a sum of valid hourly observations, not a mean.
    rain_period = weather[RAIN].resample(frequency, **settings).sum()
    hours_period = weather["HORAS_INMET_VALIDAS"].resample(frequency, **settings).sum()

    combined = dec_period.copy()
    combined[RAIN] = rain_period.reindex(period_index)
    combined["HORAS_INMET_VALIDAS"] = hours_period.reindex(period_index, fill_value=0)
    combined.index.name = "DATA"
    combined = combined.fillna(0)
    csv_file = OUTPUT_DIR / f"DEC_INMET_{name}.csv"
    combined.to_csv(csv_file, date_format="%Y-%m-%d", float_format="%.8f")

    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    labels = [
        (INTERRUPTIONS, "Interrupted units"),
        ("DEC", "DEC"),
        ("FEC", "FEC"),
        (RAIN, "Precipitation (mm)"),
    ]
    for ax, (column, label) in zip(axes, labels):
        ax.plot(combined.index, combined[column], color="#245C78", linewidth=0.9)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.margins(x=0)
        ax.tick_params(axis="both", labelsize=9)

    locator = mdates.AutoDateLocator()
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1].set_xlabel("Date")
    fig.tight_layout()

    pdf_file = OUTPUT_DIR / f"DEC_INMET_{name}.pdf"
    fig.savefig(pdf_file, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"Generated: {csv_file.name} | {pdf_file.name}")
