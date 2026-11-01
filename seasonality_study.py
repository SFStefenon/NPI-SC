"""Descriptive seasonality study of weekly interrupted consumer units.

Input: DEC_INMET_week.csv (in the same folder as this script).
Output: seasonality_study/ (PDF/PNG plots, CSV statistics, interpretation).
No forecasts or training/test splits.

Requires: pandas, numpy, matplotlib, scipy, statsmodels.
"""
from pathlib import Path
from calendar import month_abbr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, LogLocator
from scipy.stats import friedmanchisquare
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "DEC_INMET_week.csv"
OUT = ROOT / "seasonality_study"
OUT.mkdir(parents=True, exist_ok=True)
TARGET = "QTDE UC EQPTO INTERROMPIDA"
MONTHS = [month_abbr[m] for m in range(1, 13)]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "medium",
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 130, "savefig.dpi": 190,
})

raw = pd.read_csv(INPUT)
needed = {"DATA", TARGET}
if not needed.issubset(raw.columns):
    raise ValueError(f"Missing columns: {sorted(needed - set(raw.columns))}")
raw["DATA"] = pd.to_datetime(raw["DATA"], errors="raise")
raw[TARGET] = pd.to_numeric(raw[TARGET], errors="raise")
df = raw.sort_values("DATA").drop_duplicates(subset="DATA").set_index("DATA")
if not df.index.to_series().diff().dropna().eq(pd.Timedelta(days=7)).all():
    raise ValueError("Expected continuous weekly data at 7-day intervals")
if df[TARGET].isna().any() or (df[TARGET] < 0).any():
    raise ValueError("The target contains missing or negative observations")

y = df[TARGET].astype(float)
month = df.index.month
calendar_year = df.index.year
iso = df.index.isocalendar()
weekly = pd.DataFrame({
    "year": iso.year.astype(int),
    "week": iso.week.astype(int),
    "value": y.values,
}, index=df.index)
full_years = pd.DataFrame({"year": calendar_year, "month": month, "value": y.values})
full_years = full_years[full_years.year.between(2016, 2024)].copy()
monthly_table = full_years.groupby(["year", "month"])["value"].median().unstack()
if monthly_table.isna().any().any():
    raise ValueError("One or more complete years have missing months")
normalized_month = monthly_table.div(monthly_table.median(axis=1), axis=0)
monthly_median = normalized_month.median(axis=0)
month_q25 = normalized_month.quantile(0.25, axis=0)
month_q75 = normalized_month.quantile(0.75, axis=0)

weekly = weekly[weekly.year.between(2016, 2024) & weekly.week.le(52)].copy()
weekly["normalized"] = weekly.value / weekly.groupby("year").value.transform("median")
week_summary = weekly.groupby("week")["normalized"].agg(
    median="median", q25=lambda v: v.quantile(.25), q75=lambda v: v.quantile(.75)
).reindex(range(1, 53))

log_y = np.log1p(y)
stl = STL(log_y, period=52, seasonal=13, robust=True).fit()
season_strength = float(np.clip(1 - np.var(stl.resid) / np.var(stl.resid + stl.seasonal), 0, 1))
trend_strength = float(np.clip(1 - np.var(stl.resid) / np.var(stl.resid + stl.trend), 0, 1))
acf_values = acf(y, nlags=104, fft=True)
acf_bound = 1.96 / np.sqrt(len(y))
fr = friedmanchisquare(*[monthly_table[m].to_numpy() for m in range(1, 13)])
kendall_w = float(fr.statistic / (len(monthly_table) * (12 - 1)))

# Statistics use full time series; blocked monthly comparison uses only 2016-2024.
metrics = pd.DataFrame([{
    "observations": len(y), "start": str(y.index.min().date()),
    "end": str(y.index.max().date()), "median": float(y.median()),
    "mean": float(y.mean()), "max": float(y.max()),
    "stl_strength_log": season_strength, "stl_trend_strength_log": trend_strength,
    "acf_52_raw": float(acf_values[52]), "acf_52_threshold_approx": float(acf_bound),
    "friedman_statistic": float(fr.statistic), "friedman_p": float(fr.pvalue),
    "kendall_W": kendall_w, "years_tested": len(monthly_table),
}])
metrics.to_csv(OUT / "seasonality_statistics.csv", index=False)
normalized_month.rename(columns=lambda m: MONTHS[m - 1]).round(4).to_csv(OUT / "monthly_pattern_by_year.csv")
week_summary.round(4).to_csv(OUT / "weekly_seasonal_profile.csv")

pdf = PdfPages(OUT / "seasonality_study_all_figures.pdf")
figures = []


def save(fig, filename):
    fig.savefig(OUT / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{filename}.png", bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    figures.append(fig)


def short(v, _):
    return f"{v / 1000:g}k" if abs(v) >= 1000 else f"{v:g}"


def cosmetics(ax, xlabel=None, ylabel=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", length=3)


monthly_values = [y[month == m].to_numpy() for m in range(1, 13)]
xx = np.arange(1, 13)
map_values = normalized_month.to_numpy()
lags = np.arange(1, 105)

# 0. Composite preview focused on the principal interpretive views.
fig = plt.figure(figsize=(15.3, 11.3), layout="constrained")
grid = fig.add_gridspec(3, 2)
a = fig.add_subplot(grid[0, :]); a.plot(y.index, y, linewidth=.7, alpha=.6)
a.plot(y.index, y.rolling(13, center=True, min_periods=7).median(), linewidth=2)
a.set_title("A  |  Weekly series and 13-week rolling median", loc="left")
a.set_ylim(bottom=0);a.yaxis.set_major_formatter(FuncFormatter(short))
b = fig.add_subplot(grid[1, 0]); b.boxplot(monthly_values, tick_labels=MONTHS,
    flierprops={"marker": ".", "markersize": 3, "alpha": .3});b.set_yscale("log")
b.set_title("B  |  Monthly distributions (log scale)", loc="left")
c = fig.add_subplot(grid[1, 1]); c.fill_between(xx, month_q25, month_q75, alpha=.2)
c.plot(xx, monthly_median, linewidth=2, marker="o", markersize=3)
c.axhline(1., linestyle="--", linewidth=1);c.set_xticks(xx, MONTHS)
c.set_ylim(bottom=0);c.set_title("C  |  Normalized monthly pattern", loc="left")
d = fig.add_subplot(grid[2, 0]); d.imshow(map_values, origin="lower", aspect="auto", vmin=0, vmax=2.5)
d.set_xticks(np.arange(12), MONTHS);d.set_yticks(np.arange(len(normalized_month)),
    normalized_month.index.astype(str));d.set_title("D  |  Year-month seasonal consistency", loc="left")
e = fig.add_subplot(grid[2, 1]); e.vlines(lags, 0, acf_values[1:105], linewidth=1)
e.axhline(0, linewidth=.8);e.axhline(acf_bound, linestyle="--", linewidth=1)
e.axhline(-acf_bound, linestyle="--", linewidth=1);e.axvline(52, linestyle=":")
e.set_ylim(-.25, .36);e.set_xlim(0, 105);e.set_title("E  |  ACF up to 104 weeks", loc="left")
fig.suptitle("Seasonality of weekly interrupted consumer units | 2015-2025", fontsize=17, weight="medium")
fig.savefig(OUT / "00_seasonality_dashboard.png", bbox_inches="tight", dpi=180)
fig.savefig(OUT / "00_seasonality_dashboard.pdf", bbox_inches="tight")
pdf.savefig(fig, bbox_inches="tight")
plt.close(fig)

# 1. Raw weekly series, with a robust rolling seasonal reference.
fig, ax = plt.subplots(figsize=(11.8, 4.8), layout="constrained")
ax.plot(y.index, y.values, linewidth=.9, alpha=.55, label="Weekly interruptions")
ax.plot(y.index, y.rolling(13, center=True, min_periods=7).median(),
        linewidth=2.2, label="13-week rolling median")
ax.axhline(y.median(), linewidth=1.1, linestyle="--", alpha=.75, label="Historical median")
for date, value in y.nlargest(2).items():
    ax.annotate(f"{int(value):,}", xy=(date, value), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=9)
ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(FuncFormatter(short))
ax.set_title("Weekly interrupted consumer units | 2015-2025", loc="left", pad=12)
cosmetics(ax, ylabel="Interrupted consumer units")
ax.legend(loc="upper left", frameon=False, ncol=3)
save(fig, "01_weekly_series")

# 2. Monthly distributions, with a log axis so outliers don't hide the typical weeks.
fig, ax = plt.subplots(figsize=(11.8, 5.0), layout="constrained")
monthly_values = [y[month == m].to_numpy() for m in range(1, 13)]
ax.boxplot(monthly_values, tick_labels=MONTHS, showfliers=True,
           medianprops={"linewidth": 2},
           flierprops={"marker": ".", "markersize": 4, "alpha": .4})
ax.set_yscale("log")
ax.set_ylim(500, 100000)
ax.yaxis.set_major_formatter(FuncFormatter(short))
ax.set_title("Month-of-year distributions | weekly observations", loc="left", pad=12)
ax.text(.99, .98, "Logarithmic y-axis: the boxes show the median and quartiles",
        transform=ax.transAxes, ha="right", va="top", fontsize=9)
cosmetics(ax, xlabel="Calendar month", ylabel="Interrupted consumer units (log scale)")
save(fig, "02_monthly_distribution")

# 3. Robust month-of-year pattern normalized within each complete year.
fig, ax = plt.subplots(figsize=(11.8, 4.8), layout="constrained")
xx = np.arange(1, 13)
ax.fill_between(xx, month_q25.values, month_q75.values, alpha=.2, label="Interquartile range across years")
ax.plot(xx, monthly_median.values, marker="o", linewidth=2.1, markersize=5,
        label="Median across years")
ax.axhline(1.0, linestyle="--", linewidth=1.1, alpha=.8, label="Within-year reference (1.0)")
ax.set_xticks(xx, MONTHS)
ax.set_xlim(.6, 12.4)
ax.set_ylim(bottom=0)
ax.set_title("Month-of-year pattern | normalized within each year (2016-2024)", loc="left", pad=12)
cosmetics(ax, xlabel="Calendar month", ylabel="Ratio to each year's monthly median")
ax.legend(loc="upper left", frameon=False)
save(fig, "03_normalized_monthly_seasonality")

# 4. Heatmap of month-year medians, eliminating between-year scale shifts.
fig, ax = plt.subplots(figsize=(11.8, 5.0), layout="constrained")
map_values = normalized_month.to_numpy()
img = ax.imshow(map_values, origin="lower", aspect="auto", vmin=0, vmax=2.5)
ax.set_xticks(np.arange(12), MONTHS)
ax.set_yticks(np.arange(len(normalized_month)), normalized_month.index.astype(str))
ax.set_title("How consistent is the monthly pattern across years?", loc="left", pad=12)
cosmetics(ax, xlabel="Calendar month", ylabel="Complete calendar year")
bar = fig.colorbar(img, ax=ax, shrink=.87, pad=.025)
bar.set_label("Monthly median / within-year median (values capped at 2.5)")
save(fig, "04_year_month_heatmap")

# 5. Seasonal weekly profile (ISO week 53 removed since it occurs rarely).
fig, ax = plt.subplots(figsize=(11.8, 4.8), layout="constrained")
wx = np.arange(1, 53)
ax.fill_between(wx, week_summary.q25, week_summary.q75, alpha=.2,
                label="Interquartile range across years")
ax.plot(wx, week_summary["median"], linewidth=1.6, label="Cross-year weekly median")
ax.plot(wx, week_summary["median"].rolling(5, center=True, min_periods=1).median(),
        linewidth=2.3, label="5-week smoothed median")
ax.axhline(1.0, linestyle="--", linewidth=1, alpha=.75, label="Year-normalized baseline")
ax.set_xlim(1, 52)
ax.set_ylim(bottom=0)
ax.set_xticks([1, 9, 17, 26, 35, 44, 52])
ax.set_title("Week-of-year profile | normalized within each year (2016-2024)", loc="left", pad=12)
cosmetics(ax, xlabel="ISO week (1-52)", ylabel="Ratio to each year's weekly median")
ax.legend(loc="upper left", frameon=False, ncol=2)
save(fig, "05_week_of_year_profile")

# 6. Decomposition on log scale. Report seasonality strength on the same scale.
fig, axs = plt.subplots(4, 1, figsize=(11.8, 8.0), sharex=True, layout="constrained")
components = [log_y, stl.trend, stl.seasonal, stl.resid]
labels = ["log(1 + target)", "Trend", "Seasonal", "Remainder"]
for ax, values, ylabel in zip(axs, components, labels):
    ax.plot(y.index, np.asarray(values), linewidth=1.1)
    cosmetics(ax, ylabel=ylabel)
axs[0].set_title(f"Robust STL decomposition | 52-week period | seasonal strength = {season_strength:.3f}",
                 loc="left", pad=12)
axs[-1].set_xlabel("Date")
save(fig, "06_stl_decomposition")

# 7. Annual ACF diagnostic: annual peaks should repeatedly stand out if seasonality is stable.
fig, ax = plt.subplots(figsize=(11.8, 4.8), layout="constrained")
lags = np.arange(1, 105)
ax.vlines(lags, 0, acf_values[1:105], linewidth=1.2, alpha=.8)
ax.axhline(0, linewidth=1)
ax.axhline(acf_bound, linestyle="--", linewidth=1, alpha=.75)
ax.axhline(-acf_bound, linestyle="--", linewidth=1, alpha=.75,
           label="Approx. 95% white-noise reference")
ax.axvline(52, linestyle=":", linewidth=1.2, label="Lag 52 (annual)")
ax.axvline(104, linestyle=":", linewidth=1.2, label="Lag 104 (two years)")
ax.annotate(f"ACF(52) = {acf_values[52]:.3f}", xy=(52, acf_values[52]),
            xytext=(60, .27), fontsize=10, arrowprops={"arrowstyle": "->"})
ax.set_xlim(0, 105)
ax.set_ylim(-.24, .36)
ax.set_title("Autocorrelation | annual recurrence diagnostic", loc="left", pad=12)
cosmetics(ax, xlabel="Lag (weeks)", ylabel="Autocorrelation")
ax.legend(loc="upper right", frameon=False)
save(fig, "07_autocorrelation")

summary = (
    "SEASONALITY STUDY | QTDE UC EQPTO INTERROMPIDA\n"
    f"Weeks: {len(y)}, from {y.index.min().date()} to {y.index.max().date()}\n"
    f"Weekly median: {y.median():,.0f}; maximum: {y.max():,.0f}\n"
    f"52-week autocorrelation (raw): {acf_values[52]:.3f}\n"
    f"Robust STL seasonal strength (log1p, period 52): {season_strength:.3f}\n"
    f"Friedman test, blocked by complete years 2016-2024: chi-square={fr.statistic:.3f}, "
    f"p={fr.pvalue:.3f}; Kendall W={kendall_w:.3f}\n"
    "Interpretation: weak and inconsistent annual seasonality. Differences between calendar months "
    "are visible, but a repeatable month-of-year pattern is not supported by the year-blocked test. "
    "Large episodic spikes and changing annual patterns dominate. "
    "This is descriptive analysis, not a forecast.\n"
    "Rainfall should not be interpreted without checking HORAS_INMET_VALIDAS: "
    "a recorded zero may reflect an absence of measurements rather than no rain.\n"
)
(OUT / "interpretation.txt").write_text(summary, encoding="utf-8")
for fig in figures:
    plt.close(fig)
pdf.close()
print(summary)
print(f"Files saved to {OUT}")
