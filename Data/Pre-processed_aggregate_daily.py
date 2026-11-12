"""Aggregate DEC_DOC and all INMET Lages hourly CSVs into one DAILY dataset."""
from pathlib import Path
import re
import unicodedata

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent  # Change if the script is outside the data folder.
DEC_FILE = ROOT / "Original/DEC_DOC/DEC_DOC.csv"
INMET_DIR = ROOT / "Original/INMET 16-25 LGA-SC"
OUTPUT_DIR = ROOT / "Pre-processed"
SHOW_PLOTS = False

INTERRUPTIONS = "QTDE UC EQPTO INTERROMPIDA"
RAIN = "PRECIPITACAO_TOTAL_MM"

# INMET column prefixes after removing accents/punctuation. If a sensor is
# unavailable in a particular year, its daily output is NaN (not zero).
SENSORS = {
    "rain": "PRECIPITACAO TOTAL",
    "pressure": "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO",
    "pressure_max": "PRESSAO ATMOSFERICA MAX",
    "pressure_min": "PRESSAO ATMOSFERICA MIN",
    "solar": "RADIACAO GLOBAL",
    "temp": "TEMPERATURA DO AR BULBO SECO",
    "dew": "TEMPERATURA DO PONTO DE ORVALHO",
    "temp_max": "TEMPERATURA MAXIMA",
    "temp_min": "TEMPERATURA MINIMA",
    "dew_max": "TEMPERATURA ORVALHO MAX",
    "dew_min": "TEMPERATURA ORVALHO MIN",
    "humidity_max": "UMIDADE REL MAX",
    "humidity_min": "UMIDADE REL MIN",
    "humidity": "UMIDADE RELATIVA DO AR",
    "wind_dir": "VENTO DIRECAO HORARIA",
    "gust": "VENTO RAJADA MAXIMA",
    "wind": "VENTO VELOCIDADE HORARIA",
}


def normalize(text):
    ascii_text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z0-9]+", " ", ascii_text.upper()).strip()


def numeric(series):
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    ).replace(-9999, np.nan)


def find_column(columns, prefix, required=False):
    matching = [c for c in columns if normalize(c).startswith(prefix)]
    if not matching and required:
        raise ValueError(f"Missing required INMET column starting with: {prefix}")
    return matching[0] if matching else None


def circular_mean(degrees):
    """Correctly average headings such as 350° and 10° -> 0°, not 180°."""
    degrees = degrees.dropna()
    if degrees.empty:
        return np.nan
    radians = np.deg2rad(degrees.to_numpy(dtype=float))
    sine, cosine = np.sin(radians).mean(), np.cos(radians).mean()
    if np.hypot(sine, cosine) < 1e-10:
        return np.nan  # Undefined direction when vectors cancel.
    return float(np.rad2deg(np.arctan2(sine, cosine)) % 360)


def sum_valid(series):
    return series.sum(min_count=1)  # No measurements => NaN, not 0.


# Load DEC/FEC events.
dec = pd.read_csv(DEC_FILE, sep=";", encoding="utf-8-sig", dtype=str)
dec["DATA"] = pd.to_datetime(
    dec["DATA INICIO"].str.strip(), format="%d/%m/%Y %H:%M", errors="coerce"
).dt.normalize()
for col in (INTERRUPTIONS, "DEC", "FEC"):
    dec[col] = numeric(dec[col]).fillna(0)
dec = dec.dropna(subset=["DATA"]).set_index("DATA").sort_index()

# Read every year; timestamps are in UTC, output dates are local to Lages.
files = sorted(p for p in INMET_DIR.iterdir() if p.suffix.lower() == ".csv")
if not files:
    raise FileNotFoundError(f"No INMET CSV files found in {INMET_DIR}")

parts = []
for path in files:
    raw = pd.read_csv(path, sep=";", encoding="latin-1", skiprows=8, dtype=str)
    date_col = find_column(raw.columns, "DATA", required=True)
    hour_col = find_column(raw.columns, "HORA", required=True)

    dates = pd.to_datetime(raw[date_col].str.strip(), format="%Y-%m-%d", errors="coerce")
    if dates.notna().sum() == 0:
        dates = pd.to_datetime(raw[date_col].str.strip(), dayfirst=True, errors="coerce")

    hours = (raw[hour_col].str.strip().str.upper()
             .str.replace(" UTC", "", regex=False)
             .str.replace(":", "", regex=False).str.zfill(4))
    utc = pd.to_datetime(
        dates.dt.strftime("%Y-%m-%d") + " " + hours,
        format="%Y-%m-%d %H%M", errors="coerce", utc=True,
    )
    local_date = utc.dt.tz_convert("America/Sao_Paulo").dt.tz_localize(None).dt.normalize()

    block = pd.DataFrame({"UTC": utc, "DATA": local_date})
    for name, prefix in SENSORS.items():
        source = find_column(raw.columns, prefix, required=(name == "rain"))
        block[name] = numeric(raw[source]) if source else np.nan

    # Exclude physically impossible / flagged values, without filling gaps.
    for name in ("rain", "solar", "wind", "gust"):
        block.loc[block[name] < 0, name] = np.nan
    for name in ("humidity", "humidity_min", "humidity_max"):
        block.loc[~block[name].between(0, 100), name] = np.nan
    block.loc[~block["wind_dir"].between(0, 360), "wind_dir"] = np.nan
    parts.append(block)

weather = pd.concat(parts, ignore_index=True).dropna(subset=["UTC", "DATA"])
# Avoid counting an hourly observation twice if yearly files overlap.
weather = weather.sort_values("UTC").drop_duplicates(subset="UTC", keep="last")
if dec.empty or weather.empty:
    raise ValueError("No valid dates found in DEC_DOC or INMET data.")

start = max(dec.index.min(), weather["DATA"].min())
end = min(dec.index.max(), weather["DATA"].max())
if start > end:
    raise ValueError("DEC/FEC and INMET files have no overlapping dates.")

index = pd.date_range(start, end, freq="D", name="DATA")
dec_daily = dec.resample("D").agg({
    INTERRUPTIONS: "sum", "DEC": "mean", "FEC": "mean",
}).reindex(index).fillna(0)

weather_daily = weather.groupby("DATA").agg(
    PRECIPITACAO_TOTAL_MM=("rain", sum_valid),
    HORAS_INMET_VALIDAS=("rain", "count"),
    HORAS_INMET_REGISTADAS=("UTC", "count"),
    PRESSAO_ATMOSFERICA_MEDIA_MB=("pressure", "mean"),
    PRESSAO_ATMOSFERICA_MAX_MB=("pressure_max", "max"),
    PRESSAO_ATMOSFERICA_MIN_MB=("pressure_min", "min"),
    RADIACAO_GLOBAL_TOTAL_KJ_M2=("solar", sum_valid),
    HORAS_RADIACAO_VALIDAS=("solar", "count"),
    TEMPERATURA_AR_MEDIA_C=("temp", "mean"),
    TEMPERATURA_AR_MAX_C=("temp_max", "max"),
    TEMPERATURA_AR_MIN_C=("temp_min", "min"),
    PONTO_ORVALHO_MEDIA_C=("dew", "mean"),
    PONTO_ORVALHO_MAX_C=("dew_max", "max"),
    PONTO_ORVALHO_MIN_C=("dew_min", "min"),
    UMIDADE_RELATIVA_MEDIA_PCT=("humidity", "mean"),
    UMIDADE_RELATIVA_MAX_PCT=("humidity_max", "max"),
    UMIDADE_RELATIVA_MIN_PCT=("humidity_min", "min"),
    VENTO_VELOCIDADE_MEDIA_M_S=("wind", "mean"),
    VENTO_VELOCIDADE_MAX_M_S=("wind", "max"),
    VENTO_RAJADA_MAX_M_S=("gust", "max"),
    VENTO_DIRECAO_MEDIA_GRAUS=("wind_dir", circular_mean),
).reindex(index)

# Observation counts are zero if a date has no weather records.
for col in ("HORAS_INMET_VALIDAS", "HORAS_INMET_REGISTADAS", "HORAS_RADIACAO_VALIDAS"):
    weather_daily[col] = weather_daily[col].fillna(0).astype(int)

combined = dec_daily.join(weather_daily)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_file = OUTPUT_DIR / "DEC_INMET_day.csv"
combined.to_csv(csv_file, date_format="%Y-%m-%d", float_format="%.8f")

# A compact overview PDF; all environmental columns are available in the CSV.
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": False, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42,
})
plots = [
    (INTERRUPTIONS, "Interrupted units"),
    ("DEC", "DEC"),
    ("FEC", "FEC"),
    (RAIN, "Precipitation (mm)"),
    ("TEMPERATURA_AR_MEDIA_C", "Temperature (°C)"),
    ("UMIDADE_RELATIVA_MEDIA_PCT", "Humidity (%)"),
    ("VENTO_VELOCIDADE_MEDIA_M_S", "Wind speed (m/s)"),
    ("PRESSAO_ATMOSFERICA_MEDIA_MB", "Pressure (mB)"),
    ("RADIACAO_GLOBAL_TOTAL_KJ_M2", "Solar radiation (kJ/m²)"),
]
fig, axes = plt.subplots(len(plots), 1, figsize=(13, 16), sharex=True)
for ax, (col, label) in zip(axes, plots):
    ax.plot(combined.index, combined[col], color="#245C78", linewidth=0.8)
    ax.set_ylabel(label, fontsize=10)
    if col not in ("TEMPERATURA_AR_MEDIA_C", "PRESSAO_ATMOSFERICA_MEDIA_MB"):
        ax.set_ylim(bottom=0)
    ax.margins(x=0)

locator = mdates.AutoDateLocator()
axes[-1].xaxis.set_major_locator(locator)
axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
axes[-1].set_xlabel("Date")
fig.tight_layout()
pdf_file = OUTPUT_DIR / "DEC_INMET_day.pdf"
fig.savefig(pdf_file, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
plt.close(fig)
print(f"Generated: {csv_file} ({len(combined)} days, {len(combined.columns)} columns)")
print(f"Generated: {pdf_file}")
