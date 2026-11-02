"""Module ablation study for weekly interrupted customer-unit counts.

Only QTDE UC EQPTO INTERROMPIDA is used as a model variable. DATA is
used to sort/validate weekly observations and is never passed to the model.

The reference architecture is fixed in advance: RevIN, non-overlapping patches,
learned position embeddings, a two-layer pre-norm Transformer encoder, final
LayerNorm, a flattened direct multi-step head, and inverse RevIN. Experiments
remove one module or replace/add one design choice at a time. Every variant
uses the same chronological splits, seeds, training loss, and stopping rule.
Reference and variants are ranked using validation RMSE; test results must not
be used to choose which configuration to report as the winner.

Run modes can be selected without argparse:
  FORECAST_MODE=smoke    python weekly_interruptions_ablation.py
  FORECAST_MODE=quick    python weekly_interruptions_ablation.py
  FORECAST_MODE=complete python weekly_interruptions_ablation.py

Optional environment variables:
  FORECAST_DATA_FILE=/path/to/DEC_INMET_week.csv
  FORECAST_SHEET=Plan1
  FORECAST_OUTPUT_DIR=/path/to/results
  FORECAST_USE_GPU=0
  FORECAST_GPU=2
  FORECAST_RUNS=50

Primary methodological references:
  Transformer encoder:        https://arxiv.org/abs/1706.03762
  PatchTST (patches, overlap): https://arxiv.org/abs/2211.14730
  RevIN:                     https://openreview.net/forum?id=cGDAkQo1C0p
  DLinear (decomposition):   https://arxiv.org/abs/2205.13504

This is a module study of one architecture, not a comparison with the complete
implementations of those published models. Results require running on the
actual weekly interruption dataset; historical scores are not copied into this file.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


# CUDA visibility must be configured before importing torch.
GPU_USE = os.getenv("FORECAST_USE_GPU", "1").lower() not in {
    "0", "false", "no"
}
GPU_NUMBER = os.getenv("FORECAST_GPU", "0")
if GPU_USE:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUMBER

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------

MODE = os.getenv("FORECAST_MODE", "complete").lower()
if MODE == "complete":
    RUNS, MAX_EPOCHS, MIN_EPOCHS, PATIENCE = 50, 100, 30, 30
elif MODE == "quick":
    RUNS, MAX_EPOCHS, MIN_EPOCHS, PATIENCE = 2, 10, 2, 3
elif MODE == "smoke":
    RUNS, MAX_EPOCHS, MIN_EPOCHS, PATIENCE = 1, 1, 1, 1
else:
    raise ValueError("FORECAST_MODE must be 'smoke', 'quick', or 'complete'.")
RUNS = int(os.getenv("FORECAST_RUNS", RUNS))
if RUNS < 1:
    raise ValueError("FORECAST_RUNS must be a positive integer.")

DATA_FILE = Path(os.getenv("FORECAST_DATA_FILE", str(Path(__file__).with_name("Pre-processed/DEC_INMET_week.csv"))))
OUTPUT_DIR = Path(os.getenv("FORECAST_OUTPUT_DIR", "50runs_Results_weekly_interruptions_module_ablation"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "QTDE UC EQPTO INTERROMPIDA"
TIME_COLUMN: Optional[str] = "DATA"
EXCEL_SHEET_NAME = os.getenv("FORECAST_SHEET", "Plan1")
CSV_SEPARATOR = ","
CSV_DECIMAL = "."
MISSING_POLICY = "raise"  # "raise" or "forward_fill"

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.20
if not math.isclose(TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION, 1.0):
    raise ValueError("Train, validation and test fractions must sum to one.")

LOOKBACK = 104  # Two annual cycles of weekly observations (52 weeks/year).
HORIZON = 7
PATCH_SIZE = 8
PATCH_STRIDE = 8
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.10
DECOMPOSITION_WINDOW = 25

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
NUM_WORKERS = 0
USE_AMP = True

SEEDS = list(range(1, RUNS + 1))
EPSILON = 1e-8


@dataclass(frozen=True)
class ExperimentSettings:
    mode: str
    data_file: str
    excel_sheet_name: str
    target_column: str
    time_column: Optional[str]
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    lookback: int
    horizon: int
    patch_size: int
    patch_stride: int
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float
    decomposition_window: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    runs: int
    max_epochs: int
    min_epochs: int
    patience: int


SETTINGS = ExperimentSettings(
    mode=MODE,
    data_file=str(DATA_FILE),
    excel_sheet_name=EXCEL_SHEET_NAME,
    target_column=TARGET_COLUMN,
    time_column=TIME_COLUMN,
    train_fraction=TRAIN_FRACTION,
    validation_fraction=VALIDATION_FRACTION,
    test_fraction=TEST_FRACTION,
    lookback=LOOKBACK,
    horizon=HORIZON,
    patch_size=PATCH_SIZE,
    patch_stride=PATCH_STRIDE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    decomposition_window=DECOMPOSITION_WINDOW,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    gradient_clip=GRADIENT_CLIP,
    runs=RUNS,
    max_epochs=MAX_EPOCHS,
    min_epochs=MIN_EPOCHS,
    patience=PATIENCE,
)

# -----------------------------------------------------------------------------
# Reproducibility and data preparation
# -----------------------------------------------------------------------------

device = torch.device(
    "cuda:0" if GPU_USE and torch.cuda.is_available() else "cpu"
)
AMP_ENABLED = USE_AMP and device.type == "cuda"

print(f"Device: {device}")
if device.type == "cuda":
    print(f"Physical GPU requested: {GPU_NUMBER}")
    print(f"Visible GPU: {torch.cuda.get_device_name(0)}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_univariate_series(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(path, sheet_name=EXCEL_SHEET_NAME)
    elif suffix in {".csv", ".txt"}:
        dataframe = pd.read_csv(
            path, sep=CSV_SEPARATOR, decimal=CSV_DECIMAL
        )
    else:
        raise ValueError(
            "DATA_FILE must be an Excel (.xlsx/.xls) or CSV/text file."
        )
    if TARGET_COLUMN not in dataframe.columns:
        raise KeyError(
            f"Column '{TARGET_COLUMN}' was not found. Available columns: "
            + ", ".join(map(str, dataframe.columns))
        )

    if TIME_COLUMN is not None:
        if TIME_COLUMN not in dataframe.columns:
            raise KeyError(f"Time column '{TIME_COLUMN}' was not found.")
        dataframe[TIME_COLUMN] = pd.to_datetime(
            dataframe[TIME_COLUMN], errors="raise"
        )
        dataframe = dataframe.sort_values(TIME_COLUMN, kind="stable")
        if dataframe[TIME_COLUMN].duplicated().any():
            raise ValueError("Duplicate timestamps were found in TIME_COLUMN.")
        # Gaps would misalign the weekly forecast horizon and its annual cycles.
        if not dataframe[TIME_COLUMN].diff().dropna().eq(pd.Timedelta(days=7)).all():
            raise ValueError("Weekly DATA timestamps must be consecutive (7 days apart).")

    series = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan)
    missing_count = int(series.isna().sum())

    if missing_count and MISSING_POLICY == "forward_fill":
        series = series.ffill()
        first_valid_positions = np.flatnonzero(series.notna().to_numpy())
        if len(first_valid_positions) == 0:
            raise ValueError(f"'{TARGET_COLUMN}' contains no valid observations.")
        series = series.iloc[int(first_valid_positions[0]):]
    elif missing_count and MISSING_POLICY == "raise":
        raise ValueError(
            f"'{TARGET_COLUMN}' contains {missing_count} missing/non-numeric values. "
            "Rows are not silently dropped because that would compress the time axis. "
            "Clean the series or set MISSING_POLICY='forward_fill'."
        )
    elif missing_count:
        raise ValueError("MISSING_POLICY must be 'raise' or 'forward_fill'.")

    values = series.to_numpy(dtype=np.float32)
    if np.any(values < 0):
        raise ValueError(f"'{TARGET_COLUMN}' contains negative interruption counts.")
    time_values = (
        dataframe.loc[series.index, TIME_COLUMN].to_numpy()
        if TIME_COLUMN is not None else None
    )
    minimum_length = LOOKBACK + 3 * HORIZON
    if len(values) < minimum_length:
        raise ValueError(
            f"At least {minimum_length} observations are required; found {len(values)}."
        )
    return values, time_values


def make_windows(segment: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    number_of_windows = len(segment) - LOOKBACK - HORIZON + 1
    if number_of_windows <= 0:
        raise ValueError("A split is too short for LOOKBACK and HORIZON.")

    inputs = np.empty((number_of_windows, LOOKBACK, 1), dtype=np.float32)
    targets = np.empty((number_of_windows, HORIZON), dtype=np.float32)
    for index in range(number_of_windows):
        inputs[index, :, 0] = segment[index:index + LOOKBACK]
        targets[index] = segment[index + LOOKBACK:index + LOOKBACK + HORIZON]
    return torch.from_numpy(inputs), torch.from_numpy(targets)


raw_values, time_values = load_univariate_series(DATA_FILE)
number_of_observations = len(raw_values)
train_end = int(number_of_observations * TRAIN_FRACTION)
validation_end = int(
    number_of_observations * (TRAIN_FRACTION + VALIDATION_FRACTION)
)

if train_end <= LOOKBACK + HORIZON:
    raise ValueError("The training split is too short for the selected LOOKBACK.")
if validation_end - train_end < HORIZON:
    raise ValueError("The validation split is shorter than HORIZON.")
if number_of_observations - validation_end < HORIZON:
    raise ValueError("The test split is shorter than HORIZON.")

# Only the chronological training split determines global normalization.
training_mean = float(raw_values[:train_end].mean())
training_std = float(raw_values[:train_end].std())
if not np.isfinite(training_std) or training_std < EPSILON:
    raise ValueError("The training standard deviation is zero or invalid.")

standardized_values = (raw_values - training_mean) / training_std
train_values = standardized_values[:train_end]
validation_values = standardized_values[train_end - LOOKBACK:validation_end]
test_values = standardized_values[validation_end - LOOKBACK:]

x_train, y_train = make_windows(train_values)
x_validation, y_validation = make_windows(validation_values)
x_test, y_test = make_windows(test_values)

train_dataset = TensorDataset(x_train, y_train)
validation_dataset = TensorDataset(x_validation, y_validation)
test_dataset = TensorDataset(x_test, y_test)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=device.type == "cuda",
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=device.type == "cuda",
)

# -----------------------------------------------------------------------------
# Publication-style vector figures (PDF only; no plotting dependency on torch).
# -----------------------------------------------------------------------------
INK = "#172B4D"
SLATE = "#64748B"
MUTED = "#9AAABD"
TEAL = "#0F8B8D"
GREEN = "#168A76"
CORAL = "#D96B5F"
AMBER = "#D8A14A"
BORDER = "#DFE6EE"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.labelcolor": INK, "xtick.color": SLATE, "ytick.color": SLATE,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
    "axes.unicode_minus": False,
})


def style_axis(ax, grid_axis=None):
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(axis="both", which="both", length=0, pad=7)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color=BORDER, linewidth=0.7, alpha=0.65)


def plot_heading(ax, title, subtitle):
    ax.set_title(title, loc="left", color=INK, fontsize=15, fontweight="bold", pad=38)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color=SLATE)


def save_plot(fig, filename):
    fig.savefig(OUTPUT_DIR / filename, format="pdf", bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


def fmt_count(value, _):
    return f"{value / 1000:,.1f}k" if abs(value) >= 1000 else f"{value:,.0f}"


samples = time_values if time_values is not None else np.arange(number_of_observations)
rolling = pd.Series(raw_values).rolling(13, center=True, min_periods=6).median().to_numpy()
fig, ax = plt.subplots(figsize=(13.3, 5.25), layout="constrained")
for first, last, color in [
    (0, train_end, TEAL),
    (train_end, validation_end, AMBER),
    (validation_end, number_of_observations - 1, CORAL),
]:
    ax.axvspan(samples[first], samples[last], color=color, alpha=0.075, linewidth=0)
ax.fill_between(samples, 0, raw_values, color=MUTED, alpha=0.14, linewidth=0)
ax.plot(samples, raw_values, color=MUTED, linewidth=0.95, alpha=0.83,
        label="Weekly observations", zorder=3)
ax.plot(samples, rolling, color=INK, linewidth=2.2,
        label="13-week rolling median", zorder=4)
for index, color in [(train_end, TEAL), (validation_end, CORAL)]:
    ax.axvline(samples[index], color=color, linestyle=(0, (4, 4)), linewidth=1.5)
ax.set(xlabel="Week" if time_values is not None else "Week index",
       ylabel="Interrupted customer units")
ax.set_xlim(samples[0], samples[-1]); ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_count))
if time_values is not None:
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plot_heading(ax, "Weekly interruption intensity",
             f"{number_of_observations:,} observations  |  chronological split: "
             f"{TRAIN_FRACTION:.0%} train / {VALIDATION_FRACTION:.0%} validation / {TEST_FRACTION:.0%} test")
style_axis(ax)
ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=9,
          handlelength=2.6, columnspacing=2.5)
save_plot(fig, "univariate_series_splits.pdf")

# Calendar-month distribution is descriptive; no seasonal data are added to the model.
if time_values is not None:
    dates = pd.DatetimeIndex(time_values)
    by_month = [raw_values[dates.month == m] for m in range(1, 13)]
    medians = [float(np.median(v)) for v in by_month]
    fig, ax = plt.subplots(figsize=(12.7, 4.95), layout="constrained")
    bp = ax.boxplot(by_month, positions=np.arange(1, 13), widths=0.56,
                    patch_artist=True, showfliers=False,
                    medianprops={"color": TEAL, "linewidth": 2},
                    boxprops={"color": BORDER, "linewidth": 1.0},
                    whiskerprops={"color": MUTED, "linewidth": 1},
                    capprops={"color": MUTED, "linewidth": 1})
    for patch in bp["boxes"]:
        patch.set_facecolor("#E9F4F3")
    rng = np.random.default_rng(2026)
    for month, values in enumerate(by_month, 1):
        jitter = rng.uniform(-0.19, 0.19, size=len(values))
        ax.scatter(np.full(len(values), month) + jitter, values, s=7,
                   color=INK, alpha=0.15, linewidths=0, zorder=2)
    ax.plot(np.arange(1, 13), medians, color=TEAL, linewidth=1.75,
            marker="o", markersize=4.5, zorder=4, label="Monthly median")
    ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set(xlabel="Calendar month", ylabel="Interrupted customer units")
    ax.set_xlim(0.4, 12.6); ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_count))
    plot_heading(ax, "Seasonal distribution across the calendar",
                 "Each point is one week  |  boxes show the interquartile range  |  entire observed period")
    style_axis(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    save_plot(fig, "weekly_seasonality.pdf")


# -----------------------------------------------------------------------------
# Shared architecture: 104 weekly steps -> RevIN -> patches -> encoder -> 7 weeks -> inverse RevIN
# -----------------------------------------------------------------------------

class RevIN(nn.Module):
    """Normalize each input window; invert its own observed mean and scale."""

    def __init__(self, affine: bool = True, epsilon: float = 1e-5):
        super().__init__()
        self.affine = affine
        self.epsilon = epsilon
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, 1, 1))

    def normalize(self, x: torch.Tensor):
        center = x.mean(dim=1, keepdim=True).detach()
        scale = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.epsilon
        ).detach()
        normalized = (x - center) / scale
        if self.affine:
            normalized = normalized * self.weight + self.bias
        return normalized, (center, scale)

    def denormalize(self, y: torch.Tensor, statistics):
        center, scale = statistics
        y = y.unsqueeze(-1)
        if self.affine:
            # Preserve the learned affine transform's sign during inversion.
            safe_weight = torch.where(
                self.weight >= 0,
                self.weight.clamp_min(self.epsilon),
                self.weight.clamp_max(-self.epsilon),
            )
            y = (y - self.bias) / safe_weight
        return (y * scale + center).squeeze(-1)


class EncoderBlock(nn.Module):
    """Pre-norm Transformer block with independently switchable sublayers."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        use_attention: bool = True,
        use_ffn: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        if not (use_attention or use_ffn):
            raise ValueError("An encoder block needs attention or a feed-forward sublayer.")
        self.use_attention = use_attention
        self.use_ffn = use_ffn
        self.use_residual = use_residual
        if use_attention:
            self.attention_norm = nn.LayerNorm(d_model)
            self.attention = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
            self.attention_dropout = nn.Dropout(dropout)
        if use_ffn:
            self.ffn_norm = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_attention:
            z = self.attention_norm(x)
            z, _ = self.attention(z, z, z, need_weights=False)
            z = self.attention_dropout(z)
            x = x + z if self.use_residual else z
        if self.use_ffn:
            z = self.ffn(self.ffn_norm(x))
            x = x + z if self.use_residual else z
        return x


def sinusoidal_positions(tokens: int, channels: int) -> torch.Tensor:
    position = torch.arange(tokens, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        -math.log(10000.0) * torch.arange(0, channels, 2) / channels
    )
    angles = position * frequency
    encoding = torch.zeros(tokens, channels)
    encoding[:, 0::2] = angles.sin()
    encoding[:, 1::2] = angles.cos()[:, :channels // 2]
    return encoding.unsqueeze(0)


class PatchTransformer(nn.Module):
    """One deterministic univariate architecture; variants change its modules."""

    def __init__(
        self,
        use_revin: bool = True,
        revin_affine: bool = True,
        use_patches: bool = True,
        patch_stride: int = PATCH_STRIDE,
        position_kind: str = "learned",
        use_attention: bool = True,
        use_ffn: bool = True,
        use_residual: bool = True,
        use_output_norm: bool = True,
        head_kind: str = "flatten",
        use_decomposition: bool = False,
    ):
        super().__init__()
        if position_kind not in {"learned", "sinusoidal", "none"}:
            raise ValueError("Unknown position encoding.")
        if head_kind not in {"flatten", "mean"}:
            raise ValueError("Unknown forecast head.")
        if use_decomposition and DECOMPOSITION_WINDOW % 2 != 1:
            raise ValueError("DECOMPOSITION_WINDOW must be odd.")

        self.revin = RevIN(affine=revin_affine) if use_revin else None
        self.use_decomposition = use_decomposition
        self.position_kind = position_kind
        self.head_kind = head_kind
        width = PATCH_SIZE if use_patches else 1
        stride = patch_stride if use_patches else 1
        if LOOKBACK < width or stride < 1 or stride > width:
            raise ValueError("Patch width and stride must fit the lookback.")
        if (LOOKBACK - width) % stride:
            raise ValueError("Patch stride must cover the full lookback without truncation.")
        self.patch_width = width
        self.patch_stride = stride
        self.n_tokens = 1 + (LOOKBACK - width) // stride

        self.patch_projection = nn.Linear(width, D_MODEL)
        if position_kind == "learned":
            self.position = nn.Parameter(torch.empty(1, self.n_tokens, D_MODEL))
            nn.init.trunc_normal_(self.position, std=0.02)
        elif position_kind == "sinusoidal":
            self.register_buffer(
                "position", sinusoidal_positions(self.n_tokens, D_MODEL),
                persistent=False,
            )

        self.blocks = nn.ModuleList(
            EncoderBlock(
                D_MODEL, N_HEADS, DROPOUT,
                use_attention=use_attention,
                use_ffn=use_ffn,
                use_residual=use_residual,
            )
            for _ in range(N_LAYERS)
        )
        self.output_norm = nn.LayerNorm(D_MODEL) if use_output_norm else nn.Identity()
        head_width = self.n_tokens * D_MODEL if head_kind == "flatten" else D_MODEL
        self.forecast_head = nn.Linear(head_width, HORIZON)
        if use_decomposition:
            # Historical trend only, computed within the observed input window.
            self.trend_head = nn.Linear(LOOKBACK, HORIZON)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x, statistics = self.revin.normalize(inputs)
        else:
            x, statistics = inputs, None
        observed = x.squeeze(-1)

        if self.use_decomposition:
            pad = DECOMPOSITION_WINDOW // 2
            trend = F.avg_pool1d(
                F.pad(observed.unsqueeze(1), (pad, pad), mode="replicate"),
                kernel_size=DECOMPOSITION_WINDOW,
                stride=1,
            ).squeeze(1)
            observed = observed - trend

        tokens = observed.unfold(1, self.patch_width, self.patch_stride)
        tokens = self.patch_projection(tokens)
        if self.position_kind != "none":
            tokens = tokens + self.position
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.output_norm(tokens)
        representation = (
            tokens.flatten(start_dim=1)
            if self.head_kind == "flatten" else tokens.mean(dim=1)
        )
        forecast = self.forecast_head(representation)
        if self.use_decomposition:
            forecast = forecast + self.trend_head(trend)
        if statistics is not None:
            forecast = self.revin.denormalize(forecast, statistics)
        return forecast


# -----------------------------------------------------------------------------
# Predefined module interventions: all settings otherwise match the reference
# -----------------------------------------------------------------------------

REFERENCE_NAME = "Position-Encoded RevIN Patch Transformer"


@dataclass(frozen=True)
class ExperimentSpec:
    change_type: str
    module: str
    change: str
    factory: Callable[[], PatchTransformer]


EXPERIMENTS: "OrderedDict[str, ExperimentSpec]" = OrderedDict([
    (REFERENCE_NAME, ExperimentSpec(
        "Reference", "All modules", "Reference: 13 non-overlapping 8-week patches, learned positions, pre-norm encoder and flattened 7-week head",
        lambda: PatchTransformer(),
    )),
    ("Without RevIN", ExperimentSpec(
        "Removal", "RevIN", "Remove window normalization and inverse normalization",
        lambda: PatchTransformer(use_revin=False),
    )),
    ("RevIN without affine", ExperimentSpec(
        "Removal", "RevIN affine", "Keep RevIN but remove learned scale and shift",
        lambda: PatchTransformer(revin_affine=False),
    )),
    ("Without patching", ExperimentSpec(
        "Removal", "Patching", "Use 104 individual weeks as tokens instead of 13 patches; head size and attention cost change",
        lambda: PatchTransformer(use_patches=False),
    )),
    ("Without position embeddings", ExperimentSpec(
        "Removal", "Learned positions", "Remove additive positions; the flattened head still keeps token locations distinct",
        lambda: PatchTransformer(position_kind="none"),
    )),
    ("Without self-attention", ExperimentSpec(
        "Removal", "Self-attention", "Remove attention in both encoder layers, retaining feed-forward blocks",
        lambda: PatchTransformer(use_attention=False),
    )),
    ("Without feed-forward blocks", ExperimentSpec(
        "Removal", "Feed-forward", "Remove the GELU feed-forward sublayer in both encoder layers",
        lambda: PatchTransformer(use_ffn=False),
    )),
    ("Without encoder residual skips", ExperimentSpec(
        "Removal", "Encoder skips", "Remove both residual additions in each encoder layer",
        lambda: PatchTransformer(use_residual=False),
    )),
    ("Without final LayerNorm", ExperimentSpec(
        "Removal", "Final LayerNorm", "Remove normalization after the last encoder layer",
        lambda: PatchTransformer(use_output_norm=False),
    )),
    ("Mean-pooled forecast head", ExperimentSpec(
        "Replacement", "Forecast head", "Replace flattened token-specific head with mean pooling and a 64-to-7 head",
        lambda: PatchTransformer(head_kind="mean"),
    )),
    ("Fixed sinusoidal positions", ExperimentSpec(
        "Replacement", "Position encoding", "Replace learned positions with fixed sinusoidal positions",
        lambda: PatchTransformer(position_kind="sinusoidal"),
    )),
    ("Overlapping patches (stride 4)", ExperimentSpec(
        "Replacement", "Patch stride", "Replace stride 8 with stride 4, making 25 overlapping patches",
        lambda: PatchTransformer(patch_stride=4),
    )),
    ("With moving-average decomposition", ExperimentSpec(
        "Addition", "Trend branch", "Separate a 25-step smoothed trend; encode the residual and add a learned trend forecast",
        lambda: PatchTransformer(use_decomposition=True),
    )),
])

pd.DataFrame([
    {"Variant": name, "Change_type": spec.change_type,
     "Module": spec.module, "Exact_change": spec.change}
    for name, spec in EXPERIMENTS.items()
]).to_csv(OUTPUT_DIR / "module_ablation_design.csv", index=False)

with open(OUTPUT_DIR / "experiment_settings.json", "w", encoding="utf-8") as file:
    json.dump({
        **asdict(SETTINGS),
        "reference": REFERENCE_NAME,
        "variants_in_run_order": list(EXPERIMENTS),
        "seeds": SEEDS,
        "device": str(device),
        "torch_version": torch.__version__,
        "series_length": number_of_observations,
        "train_end": train_end,
        "validation_end": validation_end,
        "training_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "test_windows": len(test_dataset),
        "normalization": "global z-score fitted to train observations only",
        "forecast_constraint": "nonnegative at validation and test evaluation",
    }, file, indent=2)


# -----------------------------------------------------------------------------
# Training, early stopping and chronological evaluation
# -----------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_train_loader(seed: int) -> DataLoader:
    # Matching seed produces the same batch order across variants.
    return DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=NUM_WORKERS, pin_memory=device.type == "cuda",
    )


@torch.no_grad()
def validation_mse(model: PatchTransformer, loader: DataLoader) -> float:
    model.eval()
    total_error = 0.0
    total_values = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=AMP_ENABLED):
            predictions = model(inputs)
        # Counts cannot be negative. Apply the same constraint at selection and evaluation.
        predictions = predictions.float().clamp_min(-training_mean / training_std)
        total_error += F.mse_loss(
            predictions.float(), targets.float(), reduction="sum"
        ).item()
        total_values += targets.numel()
    return total_error / total_values


def fit_model(model: PatchTransformer, seed: int):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, PATIENCE // 3),
        min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
    history = {"train_mse": [], "validation_rmse": []}
    best_validation_mse = float("inf")
    best_epoch, best_state, stale_epochs = 0, None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        squared_error, count = 0.0, 0
        for inputs, targets in make_train_loader(seed + epoch - 1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=AMP_ENABLED):
                predictions = model(inputs)
                loss = F.mse_loss(predictions.float(), targets.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            squared_error += loss.detach().item() * targets.numel()
            count += targets.numel()

        history["train_mse"].append(squared_error / count)
        current = validation_mse(model, validation_loader)
        history["validation_rmse"].append(math.sqrt(current) * training_std)
        scheduler.step(current)
        if current < best_validation_mse - 1e-10:
            best_validation_mse = current
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch >= MIN_EPOCHS and stale_epochs >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("No validation-selected model checkpoint was retained.")
    model.load_state_dict(best_state)
    return history, best_epoch


def metric_values(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    errors = predictions - targets
    denominator = np.maximum(
        np.abs(predictions) + np.abs(targets), EPSILON
    )
    return {
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "MAE": float(np.mean(np.abs(errors))),
        "SMAPE": float(100 * np.mean(2 * np.abs(errors) / denominator)),
    }


@torch.no_grad()
def evaluate_model(model: PatchTransformer, loader: DataLoader):
    model.eval()
    predictions, targets = [], []
    for inputs, batch_targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=AMP_ENABLED):
            outputs = model(inputs)
        predictions.append(outputs.float().cpu())
        targets.append(batch_targets)
    prediction_array = torch.cat(predictions).numpy() * training_std + training_mean
    prediction_array = np.maximum(prediction_array, 0.0)  # Nonnegative count forecasts.
    target_array = torch.cat(targets).numpy() * training_std + training_mean
    overall = metric_values(prediction_array, target_array)
    horizons = [
        {"Horizon": step + 1, **metric_values(
            prediction_array[:, step], target_array[:, step]
        )}
        for step in range(HORIZON)
    ]
    return overall, horizons


all_runs, all_horizons = [], []
all_histories = {name: [] for name in EXPERIMENTS}

for name, spec in EXPERIMENTS.items():
    print(f"\nModule intervention: {name}")
    for run_index, seed in enumerate(SEEDS, start=1):
        set_seed(seed)
        model = spec.factory().to(device)
        parameters = count_parameters(model)
        start = time.perf_counter()
        history, best_epoch = fit_model(model, seed)
        train_seconds = time.perf_counter() - start
        validation, _ = evaluate_model(model, validation_loader)
        test, horizons = evaluate_model(model, test_loader)
        all_histories[name].append(history)
        all_runs.append({
            "Change_type": spec.change_type, "Module": spec.module,
            "Variant": name, "Seed": seed, "Parameters": parameters,
            "Best_epoch": best_epoch, "Training_seconds": train_seconds,
            "Validation_RMSE": validation["RMSE"], **test,
        })
        all_horizons.extend([
            {"Change_type": spec.change_type, "Variant": name, "Seed": seed, **item}
            for item in horizons
        ])
        print(
            f"Run {run_index}/{RUNS} | seed={seed} | epoch={best_epoch} | "
            f"val RMSE={validation['RMSE']:.6g} | test RMSE={test['RMSE']:.6g} | "
            f"MAE={test['MAE']:.6g} | SMAPE={test['SMAPE']:.3f}%"
        )
        pd.DataFrame(all_runs).to_csv(
            OUTPUT_DIR / "module_ablation_individual_runs_partial.csv", index=False
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Descriptive summaries, paired seed comparisons, and figures
# -----------------------------------------------------------------------------

runs_df = pd.DataFrame(all_runs)
horizons_df = pd.DataFrame(all_horizons)
summary_df = (
    runs_df.groupby(["Change_type", "Module", "Variant"], as_index=False)
    .agg(
        Parameters=("Parameters", "first"),
        Best_epoch_mean=("Best_epoch", "mean"),
        Training_seconds_mean=("Training_seconds", "mean"),
        Validation_RMSE_mean=("Validation_RMSE", "mean"),
        Validation_RMSE_std=("Validation_RMSE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        SMAPE_mean=("SMAPE", "mean"), SMAPE_std=("SMAPE", "std"),
    )
    .sort_values("Validation_RMSE_mean").reset_index(drop=True)
)
horizon_summary_df = (
    horizons_df.groupby(["Variant", "Horizon"], as_index=False)
    .agg(
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        SMAPE_mean=("SMAPE", "mean"), SMAPE_std=("SMAPE", "std"),
    )
)
selected_on_validation = summary_df.iloc[0]

reference_runs = runs_df.loc[
    runs_df["Variant"].eq(REFERENCE_NAME), ["Seed", "RMSE"]
].rename(columns={"RMSE": "Reference_RMSE"})
paired_rows = []
for name in EXPERIMENTS:
    if name == REFERENCE_NAME:
        continue
    paired = runs_df.loc[
        runs_df["Variant"].eq(name), ["Seed", "RMSE"]
    ].merge(reference_runs, on="Seed", validate="one_to_one")
    if len(paired) != RUNS:
        raise RuntimeError(f"Missing paired seeds for {name}.")
    difference = paired["RMSE"].to_numpy() - paired["Reference_RMSE"].to_numpy()
    percent = 100 * difference / paired["Reference_RMSE"].to_numpy()
    if len(paired) < 2:
        statistic, p_value = np.nan, np.nan
    elif np.all(difference == 0):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    paired_rows.append({
        "Variant": name, "Paired_runs": len(paired),
        "Mean_test_RMSE_difference": float(difference.mean()),
        "Mean_test_RMSE_change_percent": float(percent.mean()),
        "Wilcoxon_statistic": statistic, "p_value": p_value,
    })
paired_df = pd.DataFrame(paired_rows)

# Holm's step-down correction applies only to the prespecified paired tests.
valid = paired_df["p_value"].notna()
if valid.any():
    raw = paired_df.loc[valid, "p_value"].to_numpy()
    order = np.argsort(raw)
    corrected = np.empty(len(raw))
    corrected[order] = np.minimum(
        1, np.maximum.accumulate((len(raw) - np.arange(len(raw))) * raw[order])
    )
    paired_df.loc[valid, "p_value_Holm"] = corrected
else:
    paired_df["p_value_Holm"] = np.nan

runs_df.to_csv(OUTPUT_DIR / "module_ablation_individual_runs.csv", index=False)
summary_df.to_csv(OUTPUT_DIR / "module_ablation_summary.csv", index=False)
horizons_df.to_csv(OUTPUT_DIR / "horizon_individual_runs.csv", index=False)
horizon_summary_df.to_csv(OUTPUT_DIR / "horizon_summary.csv", index=False)
paired_df.to_csv(OUTPUT_DIR / "module_ablation_paired_wilcoxon_holm.csv", index=False)


# All figures use the same colors; the reference is navy and the
# validation-selected variant is teal. Test values never decide selection.
selected_name = str(selected_on_validation["Variant"])
color_of = lambda name: (INK if name == REFERENCE_NAME else
                         TEAL if name == selected_name else MUTED)

# 1. Horizontal, validation-ordered RMSE comparison with seed variability.
plot_df = summary_df.iloc[::-1].reset_index(drop=True)
y = np.arange(len(plot_df))
fig, ax = plt.subplots(figsize=(13.0, 7.9), layout="constrained")
values = plot_df["RMSE_mean"].to_numpy(dtype=float)
errors = plot_df["RMSE_std"].fillna(0).to_numpy(dtype=float)
for i, row in plot_df.iterrows():
    ax.barh(i, row["RMSE_mean"], color=color_of(row["Variant"]),
            alpha=0.92 if row["Variant"] in (REFERENCE_NAME, selected_name) else 0.72,
            height=0.68, zorder=2)
ax.errorbar(values, y, xerr=errors, fmt="none", ecolor=INK,
            elinewidth=1.1, capsize=2.7, alpha=0.8, zorder=4)
for i, (value, err) in enumerate(zip(values, errors)):
    ax.text(value + err + max(values) * 0.018, i, f"{value:,.0f}",
            va="center", fontsize=9, color=INK)
ax.set_yticks(y, plot_df["Variant"])
ax.set_xlim(0, float(np.max(values + errors)) * 1.18)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_count))
ax.set_xlabel("Test RMSE (interrupted customer units)")
plot_heading(ax, "Module ablation | forecast error",
             "Ordered by validation RMSE (top = best); bars show test RMSE and whiskers show seed SD")
style_axis(ax)
save_plot(fig, "module_ablation_rmse.pdf")

# 2. Diverging change plot; green = improvement over reference.
changes = paired_df.set_index("Variant").loc[
    [name for name in EXPERIMENTS if name != REFERENCE_NAME]
]
deltas = changes["Mean_test_RMSE_change_percent"].to_numpy(dtype=float)
y = np.arange(len(changes))
fig, ax = plt.subplots(figsize=(12.7, 7.3), layout="constrained")
ax.axvline(0, color=INK, linewidth=1.5, zorder=3)
for i, delta in enumerate(deltas):
    color = GREEN if delta < 0 else CORAL
    ax.plot([0, delta], [i, i], color=color, lw=4.5,
            alpha=0.55, solid_capstyle="round", zorder=2)
    ax.scatter(delta, i, color=color, s=82, edgecolors="white",
               linewidths=1.5, zorder=4)
    ax.annotate(f"{delta:+.2f}%", (delta, i),
                xytext=(9 if delta >= 0 else -9, 0), textcoords="offset points",
                ha="left" if delta >= 0 else "right", va="center",
                color=color, fontsize=9, fontweight="semibold")
pad = max(1.2, float(np.max(np.abs(deltas))) * 0.20)
ax.set_xlim(float(min(0, deltas.min()) - 2 * pad),
            float(max(0, deltas.max()) + 2 * pad))
ax.set_yticks(y, changes.index)
ax.invert_yaxis()
ax.set_xlabel("Paired change in test RMSE relative to reference (%)")
plot_heading(ax, "Contribution of each architectural choice",
             "Negative values indicate lower error; positive values indicate higher error")
style_axis(ax)
save_plot(fig, "module_ablation_relative_rmse.pdf")

# 3. Accuracy / parameter complexity: categorical groups, key labels only.
classes = {"Reference": (INK, "D"), "Removal": (MUTED, "o"),
           "Replacement": (TEAL, "s"), "Addition": (AMBER, "^")}
fig, ax = plt.subplots(figsize=(11.6, 6.2), layout="constrained")
for change_type, (color, marker) in classes.items():
    group = summary_df.loc[summary_df["Change_type"].eq(change_type)]
    if len(group):
        ax.scatter(group["Parameters"], group["RMSE_mean"], s=110,
                   color=color, marker=marker, edgecolors="white",
                   linewidths=1.5, alpha=0.88, zorder=4, label=change_type)
for name, offset in [(REFERENCE_NAME, (16, 24)),
                     (selected_name, (-12, -21)),
                     ("Without patching", (-12, 10))]:
    row = summary_df.loc[summary_df["Variant"].eq(name)].iloc[0]
    text_name = ("Reference" if name == REFERENCE_NAME else
                 "Validation choice" if name == selected_name else "No patching")
    ax.annotate(text_name, (row["Parameters"], row["RMSE_mean"]),
                xytext=offset, textcoords="offset points", fontsize=9,
                ha="right" if offset[0] < 0 else "left", va="center",
                color=INK, arrowprops={"arrowstyle": "-", "color": MUTED,
                                       "linewidth": 0.8})
ax.set_xlabel("Number of trainable parameters")
ax.set_ylabel("Test RMSE (interrupted customer units)")
ax.set_ylim(bottom=max(0, float(summary_df["RMSE_mean"].min()) * 0.82))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda value, _: f"{value / 1000:,.0f}k"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_count))
plot_heading(ax, "Accuracy vs. model complexity",
             "Lower error and fewer parameters are preferred; colors denote intervention type")
style_axis(ax)
ax.legend(loc="upper left", ncol=2, fontsize=9, frameon=False,
          handletextpad=0.4, columnspacing=1.5)
save_plot(fig, "accuracy_vs_parameters.pdf")

# 4. Learning curves: emphasize validation choice + fixed reference.
highlight = list(dict.fromkeys([selected_name, REFERENCE_NAME] +
                               summary_df["Variant"].head(3).tolist()))
fig, ax = plt.subplots(figsize=(12.0, 5.8), layout="constrained")
for name in EXPERIMENTS:
    curves = [np.asarray(h["validation_rmse"], dtype=float)
              for h in all_histories[name]]
    length = max(map(len, curves))
    padded = np.full((len(curves), length), np.nan)
    for i, curve in enumerate(curves):
        padded[i, :len(curve)] = curve
    mean = np.nanmean(padded, axis=0)
    epoch = np.arange(1, length + 1)
    if name not in highlight:
        ax.plot(epoch, mean, color=MUTED, alpha=0.38, linewidth=1.1,
                marker="o" if length == 1 else None, markersize=3.5, zorder=1)
        continue
    color = color_of(name) if name in (selected_name, REFERENCE_NAME) else AMBER
    label = ("Reference" if name == REFERENCE_NAME else
             "Validation choice" if name == selected_name else name)
    ax.plot(epoch, mean, color=color, linewidth=2.2, alpha=0.92,
            marker="o" if length == 1 else None, markersize=5,
            label=label, zorder=3)
    if len(curves) >= 2:
        lo = np.nanpercentile(padded, 25, axis=0)
        hi = np.nanpercentile(padded, 75, axis=0)
        ax.fill_between(epoch, lo, hi, color=color, alpha=0.10, linewidth=0)
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation RMSE (interrupted customer units)")
ax.set_ylim(bottom=0)
max_curve_epoch = max(len(h["validation_rmse"])
                      for histories in all_histories.values() for h in histories)
ax.set_xlim(0.55, max(2, max_curve_epoch) + 0.45)
if max_curve_epoch <= 10:
    ax.set_xticks(np.arange(1, max_curve_epoch + 1))
else:
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=9))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_count))
plot_heading(ax, "Training dynamics across model variants",
             "Faint lines show other variants; shaded bands show interquartile seed variation")
style_axis(ax)
ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
save_plot(fig, "validation_convergence.pdf")

# A one-epoch smoke test cannot show learning dynamics. Use a dot ranking instead.
if max_curve_epoch == 1:
    snapshot = summary_df.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12.5, 7.25), layout="constrained")
    for i, row in snapshot.iterrows():
        ax.scatter(row["Validation_RMSE_mean"], i, s=90,
                   color=color_of(row["Variant"]), edgecolor="white",
                   linewidth=1.4, zorder=3)
    ax.set_yticks(np.arange(len(snapshot)), snapshot["Variant"])
    lo = float(snapshot["Validation_RMSE_mean"].min())
    hi = float(snapshot["Validation_RMSE_mean"].max())
    ax.set_xlim(max(0, lo - (hi - lo) * 0.18), hi + (hi - lo) * 0.18)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_count))
    ax.set_xlabel("Validation RMSE after one epoch (interrupted units)")
    plot_heading(ax, "Single-epoch validation snapshot",
                 "Smoke mode performs only one epoch; run quick/complete for convergence curves")
    style_axis(ax)
    save_plot(fig, "validation_convergence.pdf")

# 5. Error progression across the seven forecast horizons.
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), layout="constrained")
for ax, metric, title in zip(axes, ["RMSE", "SMAPE"],
                             ["RMSE by forecast week", "SMAPE by forecast week"]):
    col = f"{metric}_mean"
    sd_col = f"{metric}_std"
    for name in EXPERIMENTS:
        rows = horizon_summary_df.loc[horizon_summary_df["Variant"].eq(name)]
        h = rows["Horizon"].to_numpy(dtype=float)
        v = rows[col].to_numpy(dtype=float)
        if name in (REFERENCE_NAME, selected_name):
            color = color_of(name)
            ax.plot(h, v, color=color, lw=2.4, marker="o", markersize=4.5,
                    label="Reference" if name == REFERENCE_NAME else "Validation choice",
                    zorder=4)
            if RUNS > 1:
                sd = rows[sd_col].fillna(0).to_numpy(dtype=float)
                ax.fill_between(h, np.maximum(0, v - sd), v + sd,
                                alpha=0.12, color=color, linewidth=0)
        else:
            ax.plot(h, v, color=MUTED, lw=1.05, alpha=0.35)
    ax.set_xticks(np.arange(1, HORIZON + 1))
    ax.set_xlim(1, HORIZON); ax.set_ylim(bottom=0)
    ax.set_xlabel("Forecast horizon (weeks ahead)")
    ax.set_ylabel("RMSE (interrupted units)" if metric == "RMSE" else "SMAPE (%)")
    if metric == "RMSE":
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_count))
    plot_heading(ax, title, "Reference and validation selection highlighted")
    style_axis(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
save_plot(fig, "forecast_horizon_profile.pdf")

print("\nModule study (ranked by mean validation RMSE):")
print(summary_df.to_string(index=False))
print("\nConfiguration selected using validation only:")
print(selected_on_validation.to_string())
print("\nPaired seed-level test comparisons against the fixed reference:")
print(paired_df.to_string(index=False))
print("The paired tests quantify seed-to-seed variation on this same test set; ")
print("overlapping forecast windows and runs are not independent datasets.")
print(f"\nResults saved in: {OUTPUT_DIR.resolve()}")
