"""Weekly interruption forecasting: selected Patch Transformer + optional covariates.

Target: QTDE UC EQPTO INTERROMPIDA
Optional historical co-feature: PRECIPITACAO_TOTAL_MM (observations BEFORE the forecast origin).
Optional seasonality: known calendar features for history and the seven target weeks.

Set the switches below, or override with environment variables. No command-line parser.

  FORECAST_COMPARE_ALL=1 FORECAST_MODE=quick python selected_weekly_transformer_features.py
  FORECAST_COMPARE_ALL=0 FORECAST_RAIN=1 FORECAST_SEASONALITY=1 python selected_weekly_transformer_features.py
  FORECAST_DATA_FILE=Pre-processed/DEC_INMET_week.csv FORECAST_RUNS=50 python selected_weekly_transformer_features.py

When COMPARE_ALL=True, four combinations are trained using identical splits and seeds;
the best configuration is chosen ONLY by mean validation RMSE, never by test RMSE.
Rain from future forecast weeks is NEVER used, even in backtesting.
No target-based seasonal statistics are fitted on the full series.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

# CUDA visibility must be set before importing torch.
USE_GPU = os.getenv("FORECAST_USE_GPU", "1").lower() not in {"0", "false", "no"}
GPU_NUMBER = os.getenv("FORECAST_GPU", "3")
if USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUMBER

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def env_bool(name, default):
    value = os.getenv(name, str(default)).strip().lower()
    if value not in {"1", "0", "true", "false", "yes", "no"}:
        raise ValueError(f"{name} must be true/false (or 1/0), received {value!r}")
    return value in {"1", "true", "yes"}


# ---------------------------------- OPTIONS ----------------------------------
USE_PRECIPITATION = env_bool("FORECAST_RAIN", False)
USE_SEASONALITY = env_bool("FORECAST_SEASONALITY", False)
COMPARE_ALL = env_bool("FORECAST_COMPARE_ALL", True)  # All 4 combinations; flags above used if False.

MODE = os.getenv("FORECAST_MODE", "complete").lower()
MODES = {"complete": (50, 100, 30, 30), "quick": (2, 10, 2, 3), "smoke": (1, 1, 1, 1)}
if MODE not in MODES:
    raise ValueError("FORECAST_MODE must be 'smoke', 'quick', or 'complete'.")
RUNS, MAX_EPOCHS, MIN_EPOCHS, PATIENCE = MODES[MODE]
RUNS = int(os.getenv("FORECAST_RUNS", RUNS))
if RUNS < 1:
    raise ValueError("FORECAST_RUNS must be at least 1")

DEFAULT_DATA = Path("Pre-processed/DEC_INMET_week.csv")
if not DEFAULT_DATA.exists():
    DEFAULT_DATA = Path(__file__).with_name("DEC_INMET_week.csv")
DATA_FILE = Path(os.getenv("FORECAST_DATA_FILE", str(DEFAULT_DATA)))
OUTPUT_DIR = Path(os.getenv("FORECAST_OUTPUT_DIR", "Results_weekly_interruptions_features"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COLUMN = "QTDE UC EQPTO INTERROMPIDA"
RAIN_COLUMN = "PRECIPITACAO_TOTAL_MM"
HOURS_COLUMN = "HORAS_INMET_VALIDAS"  # Used if present to flag missing meteorological measurements.
TIME_COLUMN = "DATA"
SHEET = os.getenv("FORECAST_SHEET", "Plan1")
CSV_SEPARATOR, CSV_DECIMAL = ",", "."
TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION = 0.70, 0.10, 0.20
LOOKBACK, HORIZON = 104, 1
PATCH_SIZE, PATCH_STRIDE = 8, 8
D_MODEL, N_HEADS, N_LAYERS, DROPOUT = 64, 4, 2, 0.10
BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY = 32, 1e-3, 1e-4
GRADIENT_CLIP, NUM_WORKERS, USE_AMP = 1.0, 0, True
EPSILON = 1e-8

assert math.isclose(TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION, 1.0)
assert (LOOKBACK - PATCH_SIZE) % PATCH_STRIDE == 0
assert D_MODEL % N_HEADS == 0
device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
AMP_ENABLED = USE_AMP and device.type == "cuda"

VARIANTS = [
    ("target_only", False, False),
    ("precipitation_only", True, False),
    ("seasonality_only", False, True),
    ("precipitation_and_seasonality", True, True),
] if COMPARE_ALL else [
    ("selected_configuration", USE_PRECIPITATION, USE_SEASONALITY)
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def to_numeric(values):
    return pd.to_numeric(
        values.astype("string").str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def load_frame(path, need_rain):
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=SHEET)
    elif path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path, sep=CSV_SEPARATOR, decimal=CSV_DECIMAL)
    else:
        raise ValueError("Expected CSV, TXT, XLS or XLSX.")
    required = [TIME_COLUMN, TARGET_COLUMN] + ([RAIN_COLUMN] if need_rain else [])
    missing_columns = [c for c in required if c not in df]
    if missing_columns:
        raise KeyError(f"Missing columns: {missing_columns}. Available: {df.columns.tolist()}")
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="raise")
    df = df.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)
    if df[TIME_COLUMN].isna().any() or df[TIME_COLUMN].duplicated().any():
        raise ValueError("Weekly dates must be nonmissing and unique")
    if not df[TIME_COLUMN].diff().dropna().eq(pd.Timedelta(days=7)).all():
        raise ValueError("Weekly DATA timestamps must be consecutive, 7 days apart")
    target = to_numeric(df[TARGET_COLUMN]).to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Target values must be finite and >= 0; no silent row removal")
    if len(target) < LOOKBACK + 3 * HORIZON:
        raise ValueError(f"Need at least {LOOKBACK + 3 * HORIZON} weekly observations")
    return df, target.astype(np.float32), pd.DatetimeIndex(df[TIME_COLUMN])


def prepare_rain(df, training_end):
    """Log1p rain standardized on observed TRAIN data only; missing values get a mask.

    If HORAS_INMET_VALIDAS exists, zero/unknown measurement hours make rain missing,
    and a [0, 1] measurement-coverage feature is appended (168 h per week).
    Missing rain is filled with the train mean in normalized space (0), not with
    zero precipitation. This avoids treating missing readings as dry weather.
    """
    rain = to_numeric(df[RAIN_COLUMN]).to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(rain) & (rain >= 0)
    has_hours = HOURS_COLUMN in df
    coverage = None
    if has_hours:
        hours = to_numeric(df[HOURS_COLUMN]).to_numpy(dtype=float, na_value=np.nan)
        valid &= np.isfinite(hours) & (hours > 0)
        coverage = np.where(np.isfinite(hours), np.clip(hours / 168., 0., 1.), 0.)
    if not valid[:training_end].any():
        raise ValueError("No valid precipitation measurements in the training split")
    logged = np.zeros(len(rain), dtype=np.float64)
    logged[valid] = np.log1p(rain[valid])
    mu = float(logged[:training_end][valid[:training_end]].mean())
    sigma = float(logged[:training_end][valid[:training_end]].std())
    if not np.isfinite(sigma) or sigma < EPSILON:
        sigma = 1.0
    standardized = np.zeros(len(rain), dtype=np.float32)
    standardized[valid] = ((logged[valid] - mu) / sigma).astype(np.float32)
    columns = [standardized, valid.astype(np.float32)]
    if has_hours:
        columns.append(np.asarray(coverage, dtype=np.float32))
    features = np.column_stack(columns).astype(np.float32)
    return features, {"log1p_train_mean": mu, "log1p_train_std": sigma,
                      "valid_training_weeks": int(valid[:training_end].sum()),
                      "missing_weeks_total": int((~valid).sum()),
                      "uses_hours_coverage": has_hours}


def calendar_features(dates):
    """Four deterministic, known-in-advance features; no statistics from target."""
    d = pd.DatetimeIndex(dates)
    annual_phase = 2 * np.pi * (d.dayofyear.to_numpy() - 1) / np.where(d.is_leap_year, 366., 365.)
    month_phase = 2 * np.pi * (d.month.to_numpy() - 1) / 12.
    return np.column_stack([
        np.sin(annual_phase), np.cos(annual_phase),
        np.sin(month_phase), np.cos(month_phase),
    ]).astype(np.float32)


def make_windows(standardized_target, rain_features, season_features, start, end, rain, season):
    """Forecast targets are fully within [start,end); inputs end BEFORE each target."""
    starts = range(max(start, LOOKBACK), end - HORIZON + 1)
    if len(starts) == 0:
        raise ValueError(f"Split [{start},{end}) is too short for HORIZON={HORIZON}")
    inputs, future, targets = [], [], []
    for t in starts:
        parts = [standardized_target[t - LOOKBACK:t, None]]
        if rain:
            parts.append(rain_features[t - LOOKBACK:t])  # Historical rain ONLY.
        if season:
            parts.append(season_features[t - LOOKBACK:t])
            future.append(season_features[t:t + HORIZON].reshape(-1))  # Future calendar ONLY.
        else:
            future.append(np.empty(0, dtype=np.float32))
        inputs.append(np.concatenate(parts, axis=1))
        targets.append(standardized_target[t:t + HORIZON])
    return TensorDataset(
        torch.from_numpy(np.stack(inputs).astype(np.float32)),
        torch.from_numpy(np.stack(future).astype(np.float32)),
        torch.from_numpy(np.stack(targets).astype(np.float32)),
    )


class EncoderBlock(nn.Module):
    """Architecture chosen from earlier module ablation: no residual additions."""
    def __init__(self):
        super().__init__()
        self.norm_attention = nn.LayerNorm(D_MODEL)
        self.attention = nn.MultiheadAttention(D_MODEL, N_HEADS, dropout=DROPOUT, batch_first=True)
        self.attention_dropout = nn.Dropout(DROPOUT)
        self.norm_ffn = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(4 * D_MODEL, D_MODEL), nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        z = self.norm_attention(x)
        z, _ = self.attention(z, z, z, need_weights=False)
        x = self.attention_dropout(z)
        return self.ffn(self.norm_ffn(x))


class SelectedPatchTransformer(nn.Module):
    """Multivariate history patches + optional known-future calendar -> 7 counts."""
    def __init__(self, input_channels, seasonality=False):
        super().__init__()
        n_tokens = 1 + (LOOKBACK - PATCH_SIZE) // PATCH_STRIDE
        self.input_channels = input_channels
        self.seasonality = seasonality
        self.patch_projection = nn.Linear(PATCH_SIZE * input_channels, D_MODEL)
        position = torch.arange(n_tokens, dtype=torch.float32)[:, None]
        frequency = torch.exp(-math.log(10000.) * torch.arange(0, D_MODEL, 2) / D_MODEL)
        angles = position * frequency
        encoding = torch.zeros(1, n_tokens, D_MODEL)
        encoding[0, :, 0::2] = angles.sin()
        encoding[0, :, 1::2] = angles.cos()
        self.register_buffer("positions", encoding, persistent=True)
        self.blocks = nn.ModuleList(EncoderBlock() for _ in range(N_LAYERS))
        future_width = HORIZON * 4 if seasonality else 0
        self.forecast_head = nn.Linear(D_MODEL + future_width, HORIZON)

    def forward(self, history, future_calendar):
        batch = history.size(0)
        # [batch, lookback, channels] -> [batch, patches, patch_width * channels].
        patches = history.unfold(1, PATCH_SIZE, PATCH_STRIDE)
        patches = patches.permute(0, 1, 3, 2).contiguous().reshape(batch, -1, PATCH_SIZE * self.input_channels)
        x = self.patch_projection(patches) + self.positions
        for block in self.blocks:
            x = block(x)
        pooled = x.mean(dim=1)  # Mean head; no final LayerNorm or RevIN.
        if self.seasonality:
            pooled = torch.cat([pooled, future_calendar], dim=-1)
        return self.forecast_head(pooled)


def loader(dataset, seed=None):
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=seed is not None,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )


def scores(prediction, actual):
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    denom = np.maximum(np.abs(prediction) + np.abs(actual), EPSILON)
    return {
        "RMSE": float(np.sqrt(np.mean(difference ** 2))),
        "MAE": float(np.mean(np.abs(difference))),
        "SMAPE": float(200 * np.mean(np.abs(difference) / denom)),
    }


@torch.no_grad()
def evaluate(model, data, target_mean, target_std):
    model.eval()
    predicted, actual = [], []
    for x, known_future, y in loader(data):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=AMP_ENABLED):
            out = model(x.to(device, non_blocking=True), known_future.to(device, non_blocking=True))
        predicted.append(out.float().cpu().numpy())
        actual.append(y.numpy())
    # Physical nonnegativity is used identically for validation and test.
    prediction = np.maximum(np.concatenate(predicted) * target_std + target_mean, 0.)
    truth = np.concatenate(actual) * target_std + target_mean
    return prediction, truth


def train_model(model, training, validation, target_mean, target_std, seed):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=max(1, PATIENCE // 3), min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
    best_rmse, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        squared_error, count = 0., 0
        for x, known_future, y in loader(training, seed + epoch - 1):
            x = x.to(device, non_blocking=True)
            known_future = known_future.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=AMP_ENABLED):
                output = model(x, known_future)
                loss = F.mse_loss(output.float(), y.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            squared_error += loss.detach().item() * y.numel()
            count += y.numel()
        predicted, truth = evaluate(model, validation, target_mean, target_std)
        val_rmse = scores(predicted, truth)["RMSE"]
        scheduler.step(val_rmse)
        history.append({"Seed": seed, "Epoch": epoch,
                        "Training_MSE_standardized": squared_error / count,
                        "Validation_RMSE": val_rmse})
        if np.isfinite(val_rmse) and val_rmse < best_rmse - 1e-8:
            best_rmse, best_epoch, stale = val_rmse, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was found")
    model.load_state_dict(best_state)
    return history, best_epoch, best_rmse


def plot_pdf(name):
    plt.savefig(OUTPUT_DIR / name, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close()


# ------------------------ CHRONOLOGICAL DATA PREPARATION ------------------------
need_rain = any(rain for _, rain, _ in VARIANTS)
frame, target, dates = load_frame(DATA_FILE, need_rain)
n = len(target)
train_end = int(n * TRAIN_FRACTION)
validation_end = int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))
if train_end <= LOOKBACK + HORIZON or min(validation_end - train_end, n - validation_end) < HORIZON:
    raise ValueError("One of the chronological splits is too short for LOOKBACK / HORIZON")
target_mean, target_std = float(target[:train_end].mean()), float(target[:train_end].std())
if not np.isfinite(target_std) or target_std < EPSILON:
    raise ValueError("Target variance in training split must be positive")
standardized_target = ((target - target_mean) / target_std).astype(np.float32)
rain_array, rain_stats = prepare_rain(frame, train_end) if need_rain else (None, None)
calendar = calendar_features(dates)

settings = {
    "data_file": str(DATA_FILE), "device": str(device), "seed_count": RUNS,
    "mode": MODE, "epochs_max": MAX_EPOCHS, "epochs_min": MIN_EPOCHS,
    "patience": PATIENCE, "comparison_enabled": COMPARE_ALL,
    "use_precipitation_single": USE_PRECIPITATION,
    "use_seasonality_single": USE_SEASONALITY,
    "target_column": TARGET_COLUMN, "rain_column": RAIN_COLUMN,
    "lookback": LOOKBACK, "horizon": HORIZON,
    "train_fraction": TRAIN_FRACTION, "validation_fraction": VALIDATION_FRACTION,
    "test_fraction": TEST_FRACTION, "train_end": train_end,
    "validation_end": validation_end, "train_mean": target_mean,
    "train_std": target_std, "rain_stats": rain_stats,
    "calendar_features": ["annual_sin", "annual_cos", "month_sin", "month_cos"],
    "precipitation_timing": "historical-only, within lookback; no future observations",
    "selection_rule": "lowest mean validation RMSE across variants",
    "variants": [dict(name=name, rain=rain, seasonality=season) for name, rain, season in VARIANTS],
    "architecture": dict(patch_size=PATCH_SIZE, patch_stride=PATCH_STRIDE, d_model=D_MODEL,
                         n_heads=N_HEADS, layers=N_LAYERS, dropout=DROPOUT,
                         mean_head=True, residual=False, final_layernorm=False,
                         revin=False, moving_average_decomposition=False),
}
(OUTPUT_DIR / "model_configuration.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
print(f"Device: {device} | records: {n} | train/val/test: {train_end}/{validation_end-train_end}/{n-validation_end}")

run_rows, horizon_rows, history_rows = [], [], []
checkpoints = {}
selected_predictions = {}
for name, use_rain, use_calendar in VARIANTS:
    training = make_windows(standardized_target, rain_array, calendar,
                            0, train_end, use_rain, use_calendar)
    validation = make_windows(standardized_target, rain_array, calendar,
                              train_end, validation_end, use_rain, use_calendar)
    testing = make_windows(standardized_target, rain_array, calendar,
                           validation_end, n, use_rain, use_calendar)
    input_channels = training.tensors[0].shape[-1]
    best_seed_rmse = float("inf")
    print(f"\nVariant: {name} | input channels={input_channels}")
    for seed in range(1, RUNS + 1):
        set_seed(seed)
        model = SelectedPatchTransformer(input_channels, use_calendar).to(device)
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tic = time.perf_counter()
        history, best_epoch, validation_rmse = train_model(
            model, training, validation, target_mean, target_std, seed
        )
        duration = time.perf_counter() - tic
        pred, actual = evaluate(model, testing, target_mean, target_std)
        test = scores(pred, actual)
        run_rows.append(dict(Variant=name, Use_precipitation=use_rain,
                             Use_seasonality=use_calendar, Seed=seed, Parameters=parameter_count,
                             Best_epoch=best_epoch, Training_seconds=duration,
                             Validation_RMSE=validation_rmse, **test))
        history_rows.extend({"Variant": name, **line} for line in history)
        horizon_rows.extend({"Variant": name, "Seed": seed, "Horizon": h+1,
                             **scores(pred[:, h], actual[:, h])} for h in range(HORIZON))
        if validation_rmse < best_seed_rmse:
            best_seed_rmse = validation_rmse
            checkpoints[name] = dict(state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                     seed=seed, channels=input_channels, season=use_calendar,
                                     rain=use_rain, val_rmse=validation_rmse)
            selected_predictions[name] = (pred.copy(), actual.copy())
        print(f"  seed={seed}/{RUNS}  val RMSE={validation_rmse:.3f}  test RMSE={test['RMSE']:.3f}  "
              f"MAE={test['MAE']:.3f}  SMAPE={test['SMAPE']:.2f}%")
        pd.DataFrame(run_rows).to_csv(OUTPUT_DIR / "individual_runs_partial.csv", index=False)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

runs = pd.DataFrame(run_rows)
horizons = pd.DataFrame(horizon_rows)
runs.to_csv(OUTPUT_DIR / "individual_runs.csv", index=False)
horizons.to_csv(OUTPUT_DIR / "horizon_individual_runs.csv", index=False)
pd.DataFrame(history_rows).to_csv(OUTPUT_DIR / "training_history.csv", index=False)
horizon_summary = horizons.groupby(["Variant", "Horizon"], as_index=False).agg(
    RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
    MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
    SMAPE_mean=("SMAPE", "mean"), SMAPE_std=("SMAPE", "std"),
)
horizon_summary.to_csv(OUTPUT_DIR / "horizon_summary.csv", index=False)
comparison = runs.groupby(["Variant", "Use_precipitation", "Use_seasonality"], as_index=False).agg(
    Parameters=("Parameters", "first"), Best_epoch_mean=("Best_epoch", "mean"),
    Training_seconds_mean=("Training_seconds", "mean"),
    Validation_RMSE_mean=("Validation_RMSE", "mean"),
    Validation_RMSE_std=("Validation_RMSE", "std"),
    RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
    MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
    SMAPE_mean=("SMAPE", "mean"), SMAPE_std=("SMAPE", "std"),
).sort_values("Validation_RMSE_mean").reset_index(drop=True)

if COMPARE_ALL:
    baseline = comparison.loc[comparison.Variant.eq("target_only")].iloc[0]
    comparison["Delta_validation_RMSE_percent"] = 100 * (comparison.Validation_RMSE_mean / baseline.Validation_RMSE_mean - 1)
    comparison["Delta_test_RMSE_percent"] = 100 * (comparison.RMSE_mean / baseline.RMSE_mean - 1)
    # Explicitly paired differences use exactly the same seed for every pair.
    base_runs = runs[runs.Variant.eq("target_only")][["Seed", "Validation_RMSE", "RMSE"]]
    paired = []
    for name in comparison.Variant:
        subset = runs[runs.Variant.eq(name)][["Seed", "Validation_RMSE", "RMSE"]].merge(
            base_runs, on="Seed", suffixes=("", "_baseline"), validate="one_to_one"
        )
        paired.extend({"Variant": name, "Seed": int(row.Seed),
                       "Delta_validation_RMSE": row.Validation_RMSE - row.Validation_RMSE_baseline,
                       "Delta_test_RMSE": row.RMSE - row.RMSE_baseline}
                      for row in subset.itertuples(index=False))
    pd.DataFrame(paired).to_csv(OUTPUT_DIR / "paired_differences.csv", index=False)
comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
selected_variant = str(comparison.iloc[0]["Variant"])  # VALIDATION ONLY.
selected = checkpoints[selected_variant]
settings.update(selected_variant=selected_variant, selected_seed=selected["seed"])
(OUTPUT_DIR / "model_configuration.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

model = SelectedPatchTransformer(selected["channels"], selected["season"]).to(device)
model.load_state_dict(selected["state"])
model.eval()
torch.save({"model_state_dict": selected["state"], "configuration": settings,
            "input_channels": selected["channels"], "uses_seasonality": selected["season"],
            "uses_precipitation": selected["rain"]}, OUTPUT_DIR / "selected_model.pt")

# Save selected validation run's predictions across all test origins/horizons.
selected_pred, selected_actual = selected_predictions[selected_variant]
rows = []
for origin in range(len(selected_pred)):
    for h in range(HORIZON):
        rows.append({"Variant": selected_variant, "Seed": selected["seed"],
                     "Forecast_origin": str(dates[validation_end + origin - 1].date()),
                     "Target_week": str(dates[validation_end + origin + h].date()),
                     "Horizon": h + 1, "Actual": float(selected_actual[origin, h]),
                     "Predicted": float(selected_pred[origin, h])})
pd.DataFrame(rows).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

# Truly unseen seven-week forecast. Past weather is observed; future calendar is known.
future_dates = dates[-1] + pd.to_timedelta(np.arange(1, HORIZON + 1) * 7, unit="D")
features = [standardized_target[-LOOKBACK:, None]]
if selected["rain"]:
    features.append(rain_array[-LOOKBACK:])
if selected["season"]:
    features.append(calendar[-LOOKBACK:])
input_x = torch.tensor(np.concatenate(features, axis=1)[None], dtype=torch.float32, device=device)
future_x = torch.tensor(calendar_features(future_dates).reshape(1, -1) if selected["season"]
                        else np.empty((1, 0), dtype=np.float32), dtype=torch.float32, device=device)
with torch.no_grad():
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=AMP_ENABLED):
        future = model(input_x, future_x).float().cpu().numpy()[0]
future = np.maximum(future * target_std + target_mean, 0.)
pd.DataFrame({"Week": future_dates.date.astype(str), "Forecast": future}).to_csv(
    OUTPUT_DIR / "next_7_weeks_forecast.csv", index=False
)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.facecolor": "white", "figure.facecolor": "white",
                     "pdf.fonttype": 42, "axes.grid": False})

if COMPARE_ALL:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    positions = np.arange(len(comparison))
    ax.bar(positions - .19, comparison.Validation_RMSE_mean, width=.38, label="Validation RMSE")
    ax.bar(positions + .19, comparison.RMSE_mean, width=.38, label="Test RMSE")
    ax.set_xticks(positions, comparison.Variant.str.replace("_", " "), rotation=12, ha="right")
    ax.set_ylabel("Interrupted consumer units (RMSE)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(); plot_pdf("feature_ablation_rmse.pdf")

fig, ax = plt.subplots(figsize=(10, 4.3))
for name in comparison.Variant:
    line = horizon_summary[horizon_summary.Variant.eq(name)]
    ax.plot(line.Horizon, line.RMSE_mean, marker="o", linewidth=1.6,
            label=name.replace("_", " "))
ax.set_xticks(np.arange(1, HORIZON + 1))
ax.set_xlabel("Forecast horizon (weeks)")
ax.set_ylabel("Test RMSE (interrupted consumer units)")
ax.set_ylim(bottom=0)
ax.legend(loc="upper left", frameon=False, fontsize=8)
fig.tight_layout(); plot_pdf("forecast_horizon_profile.pdf")

fig, ax = plt.subplots(figsize=(11, 4.2))
test_dates = dates[validation_end:validation_end + len(selected_pred)]
ax.plot(test_dates, selected_actual[:, 0], linewidth=1.6, label="Observed")
ax.plot(test_dates, selected_pred[:, 0], linewidth=1.4, label="Predicted (week +1)")
ax.set_xlabel("Week"); ax.set_ylabel("Interrupted consumer units")
ax.set_ylim(bottom=0)
ax.legend(loc="upper left", frameon=False)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.tight_layout(); plot_pdf("test_forecast_h1.pdf")

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(dates[-26:], target[-26:], label="Historical counts", linewidth=1.5)
ax.plot([dates[-1], *future_dates], [target[-1], *future], marker="o",
        label="Next seven weeks", linewidth=1.7)
ax.axvline(dates[-1], color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Week"); ax.set_ylabel("Interrupted consumer units")
ax.set_ylim(bottom=0)
ax.legend(loc="upper left", frameon=False)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.tight_layout(); plot_pdf("future_forecast.pdf")

print("\nComparison (ordered by mean validation RMSE):")
print(comparison[["Variant", "Validation_RMSE_mean", "RMSE_mean", "MAE_mean", "SMAPE_mean"]].to_string(index=False))
print(f"\nValidation-selected configuration: {selected_variant}; seed={selected['seed']}")
print(f"Results saved in: {OUTPUT_DIR.resolve()}")
