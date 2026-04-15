# -*- coding: utf-8 -*-
"""
Сравнение модели и эксперимента с расчетом ошибок по нормализованным значениям.
+ Отдельный график RMS самоиндукции (model_self_rms_V) из *_results_model_raw.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===================== ПУТИ И БАЗОВЫЕ ПАРАМЕТРЫ =====================
# MODEL_CSV = r"D:\PROJECTs\magnet\harvester\graphs\experiment_2025-10-16_13-35-09\2025-10-16_13-35-09_results_model_raw.csv"
MODEL_CSV = r"D:\PROJECTs\magnet\harvester\graphs\experiment_2025-10-22_11-45-42\2025-10-22_11-45-42_results_model_raw.csv"
EXP_CSV   = r"experiments\harvester_80mm\rms_compact_freq_rms.csv"

RANGE_SELECT = (7.0, 12.0)
OUT_DIR = Path("graphs/comparison_results"); OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SCALE = 1.0
EXP_SCALE   = 1.0
AUTO_SCALE = 'model_to_exp'  # или "model_to_exp" | "exp_to_model"

EPS_FRAC = 0.05
THRESH_NORM = 0.0

# ===================== УТИЛИТЫ =====================
def load_csv_generic(path: str, scale: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    low = [c.lower() for c in df.columns]

    if "model_rms_v" in low:
        freq_col = next(c for c in df.columns if "freq" in c.lower())
        rms_col  = next(c for c in df.columns if "rms"  in c.lower())
        df = df.rename(columns={freq_col: "freq_Hz", rms_col: "rms_V"})
    elif "rms_emf_mv" in low:
        freq_col = next(c for c in df.columns if "freq" in c.lower())
        rms_col  = next(c for c in df.columns if "rms"  in c.lower())
        df = df.rename(columns={freq_col: "freq_Hz"})
        df["rms_V"] = pd.to_numeric(df[rms_col], errors="coerce") * 1e-3
    elif "rms (v)" in low:
        df = df.rename(columns={"Frequency (Hz)": "freq_Hz", "RMS (V)": "rms_V"})
    else:
        freq_col = next((c for c in df.columns if "freq" in c.lower()), None)
        rms_col  = next((c for c in df.columns if "rms"  in c.lower()), None)
        if not freq_col or not rms_col:
            raise ValueError(f"Не распознан формат CSV: {path}")
        scale_guess = 1e-3 if "mv" in rms_col.lower() else 1.0
        df = df.rename(columns={freq_col: "freq_Hz"})
        df["rms_V"] = pd.to_numeric(df[rms_col], errors="coerce") * scale_guess

    df = df[["freq_Hz", "rms_V"]].dropna().sort_values("freq_Hz").reset_index(drop=True)
    df["rms_V"] = df["rms_V"] * float(scale)
    return df

def load_model_self_column(path: str) -> pd.DataFrame:
    """
    Читает колонку model_self_rms_V из summary CSV модели (если есть).
    Возвращает DF: ['freq_Hz','model_self_rms_V'].
    """
    raw = pd.read_csv(path)
    cols = {c.lower(): c for c in raw.columns}
    # Определяем колонку частоты
    if "freq_hz" in cols:
        fcol = cols["freq_hz"]
    else:
        fcol = next(c for c in raw.columns if "freq" in c.lower())
    # Определяем колонку самоиндукции
    if "model_self_rms_v" in cols:
        scol = cols["model_self_rms_v"]
    else:
        # если нет — создаём пустой столбец и вернём пустой DF после dropna
        scol = None

    if scol is None or scol not in raw.columns:
        # Нет колонки самоиндукции — вернём пустой DF
        return pd.DataFrame(columns=["freq_Hz", "model_self_rms_V"])

    df = raw[[fcol, scol]].rename(columns={fcol: "freq_Hz", scol: "model_self_rms_V"})
    df = df.dropna().sort_values("freq_Hz").reset_index(drop=True)
    return df

def minmax_norm(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    if y.size == 0: return y
    y_min, y_max = np.min(y), np.max(y)
    return np.zeros_like(y) if np.isclose(y_min, y_max) else (y - y_min) / (y_max - y_min)

def corr_R(exp_vals: np.ndarray, mod_vals: np.ndarray) -> float:
    e_mean = np.mean(exp_vals); m_mean = np.mean(mod_vals)
    num = np.sum((exp_vals - e_mean) * (mod_vals - m_mean))
    den = np.sqrt(np.sum((exp_vals - e_mean)**2) * np.sum((mod_vals - m_mean)**2))
    return num / den if den != 0 else np.nan

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b)**2))) if a.size else np.nan

def mre_clipped_pct(y_true: np.ndarray, y_pred: np.ndarray, eps_frac: float) -> float:
    if y_true.size == 0: return np.nan
    m = np.max(np.abs(y_true)) if np.isfinite(y_true).any() else 0.0
    eps = max(eps_frac * m, np.finfo(float).eps)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)

def smape_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, np.finfo(float).eps, denom)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)

def fit_scale(y_ref: np.ndarray, y_to_scale: np.ndarray) -> float:
    num = np.sum(y_ref * y_to_scale)
    den = np.sum(y_to_scale**2)
    return float(num / den) if den > 0 else 1.0
# ---- Автомасштаб (двухпараметрический: e ≈ a*m + b) ----
AUTO_SCALE_MODE = "affine"  # "affine" | "none"
applied_scale = 1.0
applied_offset = 0.0

def fit_affine(y_ref, y_pred, mask, w=None):
    """
    Находит a, b из y_ref ≈ a*y_pred + b на пересечении mask.
    Можно передать веса w (та же длина, >=0). Если w=None — обычная МНК.
    """
    x = y_pred[mask].astype(float)
    y = y_ref[mask].astype(float)
    if w is not None:
        w = np.sqrt(np.clip(w[mask].astype(float), 0.0, np.inf))
        X = np.vstack([x, np.ones_like(x)]).T
        Xw = X * w[:, None]
        yw = y * w
        a, b = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    else:
        X = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(a), float(b)


# ===================== ОСНОВНОЙ ХОД =====================
model_df = load_csv_generic(MODEL_CSV, scale=MODEL_SCALE)
exp_df   = load_csv_generic(EXP_CSV,   scale=EXP_SCALE)

# Загрузим самоиндукцию из summary CSV модели
model_self_df = load_model_self_column(MODEL_CSV)

if RANGE_SELECT:
    lo_r, hi_r = RANGE_SELECT
    model_df     = model_df[(model_df["freq_Hz"] >= lo_r) & (model_df["freq_Hz"] <= hi_r)]
    exp_df       = exp_df[(exp_df["freq_Hz"]   >= lo_r) & (exp_df["freq_Hz"]   <= hi_r)]
    if not model_self_df.empty:
        model_self_df = model_self_df[(model_self_df["freq_Hz"] >= lo_r) & (model_self_df["freq_Hz"] <= hi_r)]

mf = model_df["freq_Hz"].to_numpy()
ef = exp_df["freq_Hz"].to_numpy(); ev = exp_df["rms_V"].to_numpy()
lo = max(np.min(mf), np.min(ef)); hi = min(np.max(mf), np.max(ef))
mask_overlap = (mf >= lo) & (mf <= hi)

exp_interp_V = np.full_like(mf, np.nan, dtype=float)
exp_interp_V[mask_overlap] = np.interp(mf[mask_overlap], ef, ev)
model_V = model_df["rms_V"].to_numpy() # - 0.0000275

# ---- Автомасштаб ----
applied_scale = 1.0
if AUTO_SCALE_MODE == "affine":
    # Пример: сильнее «весим» область с большими значениями эксперимента
    eN = minmax_norm(exp_interp_V)              # 0..1
    weights = (eN ** 3) * mask_overlap         # p=3; можно 2..4

    a, b = fit_affine(exp_interp_V, model_V, mask_overlap, w=weights)
    model_V = a * model_V + b                  # применяем подгонку к модели
    applied_scale, applied_offset = a, b
    print(f"[AUTO_SCALE affine] a={a:.6f}, b={b:.6e} (model -> a*model + b)")


model_norm = minmax_norm(model_V)
exp_norm   = minmax_norm(exp_interp_V)

use_mask = mask_overlap
if THRESH_NORM > 0:
    use_mask = use_mask & (exp_norm >= THRESH_NORM)

mV = model_V[use_mask]; eV = exp_interp_V[use_mask]
mN = model_norm[use_mask]; eN = exp_norm[use_mask]

# === Метрики ===
R   = corr_R(eV, mV); R2  = R**2 if np.isfinite(R) else np.nan
RMSE_V    = rmse(eV, mV)
RMSE_norm = rmse(eN, mN)
MRE_pct   = mre_clipped_pct(eV, mV, EPS_FRAC)
SMAPE_pct_val = smape_pct(eV, mV)

# Дополнительные метрики по нормализованным данным
RMSE_norm2 = rmse(eN, mN)
MRE_pct_norm2 = mre_clipped_pct(eN, mN, EPS_FRAC)
SMAPE_pct_norm2 = smape_pct(eN, mN)

# === Вывод ===
print("=== Метрики согласованности ===")
if RANGE_SELECT: print(f"Диапазон: {RANGE_SELECT[0]}–{RANGE_SELECT[1]} Гц")
print(f"Использовано точек: {use_mask.sum()} (из {mf.size})")
print(f"Масштабы: MODEL_SCALE={MODEL_SCALE}, EXP_SCALE={EXP_SCALE}, "
      f"AUTO_SCALE={AUTO_SCALE}, applied_scale={applied_scale:.6f}")
print(f"R   = {R:.5f}"  if np.isfinite(R)  else "R   = n/a")
print(f"R^2 = {R2:.5f}" if np.isfinite(R2) else "R^2 = n/a")
print("--- Метрики (в В) ---")
print(f"RMSE_V         = {RMSE_V:.6f}")
print(f"RMSE_norm      = {RMSE_norm:.6f}")
print(f"MRE_clipped %  = {MRE_pct:.2f}%")
print(f"SMAPE %        = {SMAPE_pct_val:.2f}%")
print("--- Метрики (min–max норм.) ---")
print(f"RMSE_norm2     = {RMSE_norm2:.6f}")
print(f"MRE_norm2 %    = {MRE_pct_norm2:.2f}%")
print(f"SMAPE_norm2 %  = {SMAPE_pct_norm2:.2f}%")

# Сохранение метрик
metrics = {
    "range_low_Hz": None if not RANGE_SELECT else RANGE_SELECT[0],
    "range_high_Hz": None if not RANGE_SELECT else RANGE_SELECT[1],
    "N_used": int(use_mask.sum()),
    "MODEL_SCALE": MODEL_SCALE,
    "EXP_SCALE": EXP_SCALE,
    "AUTO_SCALE": AUTO_SCALE,
    "applied_scale": applied_scale,
    "R": R, "R2": R2,
    "RMSE_V": RMSE_V, "RMSE_norm": RMSE_norm,
    "MRE_pct_clipped": MRE_pct,
    "SMAPE_pct": SMAPE_pct_val,
    "RMSE_norm2": RMSE_norm2,
    "MRE_pct_norm2": MRE_pct_norm2,
    "SMAPE_pct_norm2": SMAPE_pct_norm2,
    "eps_frac": EPS_FRAC, "thresh_norm": THRESH_NORM
}
pd.DataFrame([metrics]).to_csv(OUT_DIR / "correlation_metrics.csv", index=False)

# ===================== ОФОРМЛЕНИЕ ГРАФИКОВ =====================
FIGSIZE_INCH = (6, 3)    # Соотношение 2:1 (ширина:высота)
SAVE_DPI = 1200
FONT_FAMILY = "Arial"     # если нет — возьмётся ближайший sans-serif
FONT_SIZE = 12
XTICK_STEP = 0.5          # Шаг делений по оси X в Гц
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": FONT_SIZE,
    "savefig.dpi": SAVE_DPI,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
})

# === График: эксперимент vs модель (ненормированные) ===
plt.figure(figsize=FIGSIZE_INCH)
ax = plt.gca()
ax.plot(mf[mask_overlap], exp_interp_V, "o-" , label="Physical Experiment")
ax.plot(mf,  model_V, "^--", label="Mathematical Model")
ax.set_title("")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("RMS EMF (V)")
ax.grid(True)
ax.legend()

xmin = np.floor(mf.min())
xmax = np.ceil(mf.max())
ax.set_xlim(xmin, xmax)
ax.set_xticks(np.arange(xmin, xmax + XTICK_STEP, XTICK_STEP))

plt.tight_layout()
png_path = OUT_DIR / "correlation_plot.png"
plt.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
plt.show()

print(f"\n[OK] График: {png_path}")
print(f"[OK] Метрики: {OUT_DIR/'correlation_metrics.csv'}")

# === ОТДЕЛЬНЫЙ ГРАФИК: RMS самоиндукции модели ===
if not model_self_df.empty:
    # Совместим по частоте с моделью (если нужно)
    # Здесь просто рисуем как есть — частоты уже отфильтрованы по RANGE_SELECT
    sf = model_self_df["freq_Hz"].to_numpy()
    sV = model_self_df["model_self_rms_V"].to_numpy()

    plt.figure(figsize=FIGSIZE_INCH)
    ax2 = plt.gca()
    ax2.plot(sf, sV, "s-", label="Self-Induction EMF")
    ax2.set_title("")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("RMS EMF (V)")
    ax2.grid(True)
    ax2.legend()

    xmin2 = np.floor(sf.min()) if sf.size else 0.0
    xmax2 = np.ceil(sf.max())  if sf.size else 1.0
    ax2.set_xlim(xmin2, xmax2)
    ax2.set_xticks(np.arange(xmin2, xmax2 + XTICK_STEP, XTICK_STEP))

    plt.tight_layout()
    png_self = OUT_DIR / "self_induction_rms_plot.png"
    plt.savefig(png_self, dpi=SAVE_DPI, bbox_inches="tight")
    plt.show()
    print(f"[OK] График самоиндукции: {png_self}")
else:
    print("[WARN] В файле модели не найдена колонка 'model_self_rms_V' — график самоиндукции пропущен.")
