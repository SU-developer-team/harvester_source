# -*- coding: utf-8 -*-
"""
Visualization of experiment results over d (magnet spacing) and f (frequency) from saved files.
- Prefer *.npy (d_values, freqs, vp, pout, rhoE). Fall back to CSV matrices if needed.
- Plots:
    1) 3D surfaces: Vp(d, f), ρ_E(d, f)
    2) 2D: Vp(f) for each d
    3) 2D: Pout_max(d), ρE_max(d)
    4) Heatmaps: Vp, Pout, ρ_E
- Saves figures next to the data (PNG @600dpi + PDF).
- Prints a brief summary and the path to the resonance CSV if found.

Usage:
    python visualize_experiment_results.py --dir <path_to_experiment_YYYY-MM-DD_HH-MM-SS>
    # Переопределить сетку частот и d (мм):
    python visualize_experiment_results.py --dir "<path>" --freqs "3,20,0.1" --d "40:49:1"

Опции:
    --freqs "start,stop,step"       Например: "3,20,0.1"
    --d "start:stop:step"           Например: "40:49:1"
         или список через запятую:  "40,42,45,48"
    --no-pub-style                  Не применять публикационные стили оформления
    --show                          Показать интерактивные окна для 2D/heatmap (по умолчанию только 3D show)
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import logging

logger = logging.getLogger(__name__)  # noqa: E402


# =========================
# Publication plotting style
# =========================
PUB_DPI = 300
PUB_FIGSIZE_2D = (4, 4)     # inches (≈ 165 x 102 mm) — typical 1-column
PUB_FIGSIZE_3D = (15, 4.5)      # чуть выше для 3D
PUB_FONT_FAMILY = "Arial"       # fallback на DejaVu Sans, если нет
PUB_FONT_SIZE = 5              # при необходимости 10

DEBUG = True

def apply_publication_style():
    mpl.rcParams.update({
        "figure.dpi": PUB_DPI,
        "savefig.dpi": PUB_DPI,
        "font.family": PUB_FONT_FAMILY,
        "font.size": PUB_FONT_SIZE,
        "axes.labelsize": PUB_FONT_SIZE,
        "axes.titlesize": PUB_FONT_SIZE,   # без title в графиках
        "xtick.labelsize": PUB_FONT_SIZE,
        "ytick.labelsize": PUB_FONT_SIZE,
        "legend.fontsize": PUB_FONT_SIZE,
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.alpha": 0.35,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.spines.top": True,
        "axes.spines.right": True,
    })

# ---------- Latest experiment folder helper ----------
def find_latest_experiment_dir(base="experiments/harvester_50mm/exp_2/graphs") -> Optional[str]:
    if not os.path.isdir(base):
        return None
    candidates = [os.path.join(base, d) for d in os.listdir(base) if d.startswith("experiment_")]
    candidates = [d for d in candidates if os.path.isdir(d)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

# ---------- File picking ----------
def _pick_one(patterns):
    """
    Возвращает самый свежий файл, подходящий под ЛЮБОЙ из переданных шаблонов.
    Шаблоны могут быть как с wildcard, так и точными именами.
    """
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def load_arrays(folder: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[str], Optional[str]]:
    """
    Пытается загрузить d_values, freqs, vp, pout, rhoE.
    1) Сначала *.npy (и с суффиксом, и без).
    2) Если не нашли — CSV-матрицы (и с суффиксом, и без).
    Возвращает: (d_values, freqs, vp, pout, rhoE, ts_str, resonance_csv)
    """
    # --- пробуем NPY ---
    d_path    = _pick_one([os.path.join(folder, "d_values_"
    "*.npy"),
                           os.path.join(folder, "d_values.npy")])
    f_path    = _pick_one([os.path.join(folder, "freqs_*.npy"),
                           os.path.join(folder, "freqs.npy")])
    vp_path   = _pick_one([os.path.join(folder, "vp_values_*.npy"),
                           os.path.join(folder, "vp_values.npy")])
    pout_path = _pick_one([os.path.join(folder, "pout_values_*.npy"),
                           os.path.join(folder, "pout_values.npy")])
    rho_path  = _pick_one([os.path.join(folder, "rho_e_values_*.npy"),
                           os.path.join(folder, "rho_e_values.npy")])

    ts = None
    for p in [d_path, f_path, vp_path, pout_path, rho_path]:
        if p:
            m = re.search(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.npy$", p)
            if m:
                ts = m.group(1)
                break

    resonance_csv = _pick_one([
        os.path.join(folder, "resonance_params_*.csv"),
        os.path.join(folder, "resonance_params.csv"),
    ])

    if all([d_path, f_path, vp_path, pout_path, rho_path]):
        print("[i] Using NPY files:")
        for p in [d_path, f_path, vp_path, pout_path, rho_path]:
            print("   ", os.path.basename(p))
        d_values = np.load(d_path)
        freqs    = np.load(f_path)
        vp       = np.load(vp_path)
        pout     = np.load(pout_path)
        rhoE     = np.load(rho_path)
        return d_values, freqs, vp, pout, rhoE, ts, resonance_csv

    vp_csv   = _pick_one([os.path.join(folder, "vp_values_*.csv"),
                          os.path.join(folder, "vp_values.csv")])
    pout_csv = _pick_one([os.path.join(folder, "pout_values_*.csv"),
                          os.path.join(folder, "pout_values.csv")])
    rho_csv  = _pick_one([os.path.join(folder, "rho_e_values_*.csv"),
                          os.path.join(folder, "rho_e_values.csv")])

    if all([vp_csv, pout_csv, rho_csv]):
        for p in [vp_csv, pout_csv, rho_csv]:
            m = re.search(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv$", p)
            if m:
                ts = m.group(1)
                break

        print("[i] Using CSV matrices:")
        for p in [vp_csv, pout_csv, rho_csv]:
            print("   ", os.path.basename(p))

        vp_df   = pd.read_csv(vp_csv, index_col=0, header=0)
        pout_df = pd.read_csv(pout_csv, index_col=0, header=0)
        rho_df  = pd.read_csv(rho_csv, index_col=0, header=0)

        # индексы и колонки — строки -> переводим в float
        try:
            d_values = vp_df.index.astype(float).values
        except Exception:
            d_values = pd.to_numeric(vp_df.index, errors="coerce").values
        try:
            freqs = vp_df.columns.astype(float).values
        except Exception:
            freqs = pd.to_numeric(pd.Index(vp_df.columns), errors="coerce").values

        vp   = vp_df.values.astype(float)
        pout = pout_df.values.astype(float)
        rhoE = rho_df.values.astype(float)

        return d_values, freqs, vp, pout, rhoE, ts, resonance_csv

    # --- ничего не нашли ---
    searched = [
        "d_values.npy / d_values_*.npy",
        "freqs.npy / freqs_*.npy",
        "vp_values.npy / vp_values_*.npy  ИЛИ  vp_values.csv / vp_values_*.csv",
        "pout_values.npy / pout_values_*.npy  ИЛИ  pout_values.csv / pout_values_*.csv",
        "rho_e_values.npy / rho_e_values_*.npy  ИЛИ  rho_e_values.csv / rho_e_values_*.csv",
    ]
    msg = "Expected data files were not found.\nLooked for:\n  - " + "\n  - ".join(searched) + f"\nIn folder: {folder}"
    raise FileNotFoundError(msg)

def parse_affine_arg(arg: str) -> Tuple[float, float]:
    a_str, b_str = [s.strip() for s in arg.split(",")]
    return float(a_str), float(b_str)

# ---------- Resampling helpers ----------
def resample_matrix_along_axis(matrix: np.ndarray,
                               old_axis_vals: np.ndarray,
                               new_axis_vals: np.ndarray,
                               axis: int = 1,
                               compute_from_vp: bool = False,
                               r_load_ohm: float = 1000.0,
                               volume_cm3: float = 1.0,
                               vp_is_peak: bool = True) -> np.ndarray:
    """
    Универсальная функция:
    - Если compute_from_vp=False (по умолчанию): интерполяция матрицы (NaN-safe).
    - Если compute_from_vp=True: рассчитывает мощность (мВт) и плотность энергии (мВт/см³)
      из матрицы амплитуд напряжений (Vp, мВ).

    axis=1: интерполяция по частотам
    axis=0: интерполяция по d
    """

    matrix = np.asarray(matrix, float)
    old_axis_vals = np.asarray(old_axis_vals, float)
    new_axis_vals = np.asarray(new_axis_vals, float)

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")

    # ---------- Если нужно рассчитать мощность и энергоёмкость ----------
    if compute_from_vp:
        vp_V = matrix / 1e3  # мВ → В
        v_rms = vp_V / np.sqrt(2.0) if vp_is_peak else vp_V
        pout_W = (v_rms ** 2) / r_load_ohm
        pout_mW = pout_W * 1e3
        rhoE_mW_cm3 = pout_mW / volume_cm3
        return pout_mW, rhoE_mW_cm3

    # ---------- Обычная интерполяция ----------
    if axis == 1:
        out = np.full((matrix.shape[0], len(new_axis_vals)), np.nan, dtype=float)
        for i in range(matrix.shape[0]):
            y = matrix[i, :]
            mask = np.isfinite(y) & np.isfinite(old_axis_vals)
            if mask.sum() >= 2:
                out[i, :] = np.interp(new_axis_vals, old_axis_vals[mask], y[mask],
                                      left=np.nan, right=np.nan)
        return out

    elif axis == 0:
        out = np.full((len(new_axis_vals), matrix.shape[1]), np.nan, dtype=float)
        for j in range(matrix.shape[1]):
            y = matrix[:, j]
            mask = np.isfinite(y) & np.isfinite(old_axis_vals)
            if mask.sum() >= 2:
                out[:, j] = np.interp(new_axis_vals, old_axis_vals[mask], y[mask],
                                      left=np.nan, right=np.nan)
        return out

    else:
        raise ValueError("axis must be 0 or 1")


# ---------- Save helpers ----------
def _save_fig(fig: mpl.figure.Figure, base_path_no_ext: str, show: bool = False) -> None:
    """Save figure as PNG (600 dpi) and PDF (vector) with a tight bbox. Optionally show."""
    fig.tight_layout()
    fig.savefig(base_path_no_ext + ".png", bbox_inches="tight")
    fig.savefig(base_path_no_ext + ".pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

# ---------- Plotting ----------
def plot_3d_surface(X_d, Y_f, Z, xlabel, ylabel, zlabel, out_base,
                    cmap="viridis", elev=18, azim=-35, z_aspect=0.70,
                    mark_peak=False, show=False):

    D, F = np.meshgrid(X_d, Y_f)
    fig = plt.figure(figsize=PUB_FIGSIZE_3D)
    ax  = fig.add_subplot(111, projection="3d")

    # 1) фиксируем нормировку по всему Z
    vmin, vmax = float(np.nanmin(Z)), float(np.nanmax(Z))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # 2) красим «по вершинам» и выключаем шейдинг
    facecolors = plt.get_cmap(cmap)(norm(Z.T))
    step = max(1, len(X_d) // 6)                 # оставить ~8 подписей
    ax.set_xticks(X_d[::step])
    ax.set_xticklabels([f"{x:g}" for x in X_d[::step]])

    surf = ax.plot_surface(
        D, F, Z.T,
        facecolors=facecolors,    # <- явные цвета
        cmap=None,                # <- чтобы не перекрашивал
        shade=False,              # <- без освещения
        antialiased=False,        # <- меньше «размазывания»
        rcount=len(Y_f),          # <- ПОЛНОЕ разрешение по частоте
        ccount=len(X_d),          # <- ПОЛНОЕ разрешение по d
        linewidth=0
    )
    # цветовая шкала от той же нормировки
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=14).set_label(zlabel)
    for lab in ax.get_xticklabels():
        lab.set_rotation(10)
        lab.set_horizontalalignment("right")
        lab.set_verticalalignment("center_baseline")
    ax.tick_params(axis="x", pad=2)  
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel('')
    ax.view_init(elev=elev, azim=azim)
    try: ax.set_box_aspect((1.0, 1.0, z_aspect))
    except Exception: pass

    # отметка пика
    zmax = np.nanmax(Z)
    if mark_peak and np.isfinite(zmax):
        i_d, i_f = np.unravel_index(np.nanargmax(Z), Z.shape)
        x_peak, y_peak, z_peak = float(X_d[i_d]), float(Y_f[i_f]), float(Z[i_d, i_f])
        ax.scatter([x_peak], [y_peak], [z_peak], s=60, c="red", depthshade=False, zorder=5)
        ax.text(x_peak, y_peak, z_peak*1.02, f"d={x_peak:.0f} mm, f={y_peak:.1f} Hz, {zlabel.split()[0] if zlabel=='Output power (mW)' else zlabel.split()[0]}={z_peak:.2f} {'mW' if zlabel=='Output power (mW)' else 'mW/cm³'}",
                ha="center", va="bottom")

    _save_fig(fig, out_base, show=show)



def plot_2d_vp_vs_freq(freqs, vp_matrix, d_values, out_base, show=False):
    """
    Строит график Vp(f) для разных d, начиная строго с 2 Гц.
    Пустые участки (NaN) маскируются, а ось X ограничивается снизу 2 Гц.
    """

    # --- фильтрация по частоте ---
    valid_cols = np.isfinite(vp_matrix).any(axis=0)
    idx = (freqs >= 2) & valid_cols
    fi = freqs[idx]

    if fi.size < 2:
        raise ValueError("Недостаточно точек по частоте ≥ 2 Гц.")

    # --- обрезаем матрицу ---
    Z = vp_matrix[:, idx]

    # --- создаем фигуру ---
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    # --- построение линий ---
    for i, d in enumerate(d_values):
        yi = Z[i, :]
        ax.plot(fi, yi, marker="o", linewidth=1.0, label=f"d={d:g} mm")

        # отмечаем максимум
        if np.any(np.isfinite(yi)):
            j = int(np.nanargmax(yi))
            ax.plot(fi[j], yi[j], "x", markersize=8)

    # --- оформление ---
    ax.set_xlim(2, fi[-1])  # гарантируем начало с 2 Гц
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("Vp (mV)")
    ax.grid(True, alpha=0.35)
    ax.legend(ncol=2, frameon=False)

    # --- сохранение ---
    _save_fig(fig, out_base, show=show)


def plot_2d_max_vs_d(d_values, matrix, ylabel, out_base, show=False):
    y = np.nanmax(matrix, axis=1)
    fig = plt.figure(figsize=PUB_FIGSIZE_2D)
    ax = fig.add_subplot(111)
    ax.plot(d_values, y, marker="o", linewidth=1.0)
    ax.set_xlabel("d (mm)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    _save_fig(fig, out_base, show=show)

def plot_heatmap(matrix, x_vals, y_vals, xlabel, ylabel, cbar_label, out_base, cmap="viridis", show=False):
    fig = plt.figure(figsize=PUB_FIGSIZE_2D)
    ax = fig.add_subplot(111)
    # Для корректного extent x_vals и y_vals должны быть возрастающими
    x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
    y_min, y_max = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap=cmap,
        interpolation="nearest"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(im)
    cbar.set_label(cbar_label)
    _save_fig(fig, out_base, show=show)

# ---------- CLI parsing helpers ----------
def parse_freqs_arg(arg: str) -> np.ndarray:
    """
    --freqs "start,stop,step" | "start,stop" (тогда 200 точек)
    """
    parts = [p.strip() for p in arg.split(",")]
    nums = [float(x) for x in parts]
    if len(nums) == 3:
        start, stop, step = nums
        # включим правую границу с малым эпсилон
        return np.arange(start, stop + 1e-12, step)
    elif len(nums) == 2:
        start, stop = nums
        return np.linspace(start, stop, 200)
    else:
        raise ValueError("Use format 'start,stop,step' or 'start,stop'")

def parse_d_arg(arg: str) -> np.ndarray:
    """
    --d "start:stop:step" или список "v1,v2,v3"
    """
    if ":" in arg:
        a, b, s = [float(x) for x in arg.split(":")]
        return np.arange(a, b + 1e-12, s)
    else:
        return np.array([float(x) for x in arg.split(",")], dtype=float)
    
def recompute_pout_rhoe_from_vp(
    vp_mV: np.ndarray,
    r_load_ohm: float,
    volume_cm3: float,
    vp_is_peak: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Пересчёт выходной мощности и плотности энергии из матрицы Vp.
    vp_mV     — матрица Vp в мВ (той же формы, что и vp)
    r_load_ohm — сопротивление нагрузки, Ом
    volume_cm3 — активный объём устройства, см^3
    vp_is_peak — True: значения Vp — амплитуды (V_peak), False: RMS

    Возвращает:
        pout_mW  — матрица мощности в мВт
        rhoE_mW_cm3 — матрица плотности энергии в мВт/см^3
    """
    vp_V = vp_mV / 1e3  # мВ -> В
    if vp_is_peak:
        v_rms = vp_V / np.sqrt(2.0)
    else:
        v_rms = vp_V

    pout_W = (v_rms ** 2) / r_load_ohm
    pout_mW = pout_W * 1e3
    rhoE_mW_cm3 = pout_mW / volume_cm3
    return pout_mW, rhoE_mW_cm3

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=None,
                    help="Path to experiment_YYYY-MM-DD_HH-MM-SS folder. If omitted, latest in exp_2/graphs is used.")
    ap.add_argument("--freqs", type=str, default=None,
                    help='Override frequency grid, e.g. " " or "5,25"')
    ap.add_argument("--d", type=str, default=None,
                    help='Override d-list in mm, e.g. "40:49:1" or "40,42,45,48"')
    ap.add_argument("--no-pub-style", action="store_true", help="Disable publication plotting style.")
    ap.add_argument("--show", action="store_true", help="Show interactive windows for 2D/heatmaps too.")
    ap.add_argument("--affine", type=str, default=None,
                help='Применить аффинный скейл к данным: "a,b"  (y -> a*y + b) '
                     'и подписать на всех графиках.')

    args = ap.parse_args()
    affine_text = None
    a = b = None


    if not args.no_pub_style:
        apply_publication_style()

    # Выбор папки
    if args.dir is None:
        base_graphs = r"D:\PROJECTs\magnet\harvester\last_exp\teory_results\2025-10-25"
        exp_dir = find_latest_experiment_dir(base_graphs)
        if exp_dir is None:
            raise SystemExit("Could not find the latest experiment folder. Specify with --dir")
        print(f"[i] Using latest folder: {exp_dir}")
    else:
        exp_dir = args.dir
        if not os.path.isdir(exp_dir):
            raise SystemExit(f"Folder not found: {exp_dir}")

    # Загрузка исходных массивов
    d_values, freqs, vp, pout, rhoE, ts, resonance_csv = load_arrays(exp_dir)

    # Отладочная печать
    print("\n[DEBUG] Loaded shapes:")
    print(f"d_values: {d_values.shape}")
    print(f"freqs   : {freqs.shape}")
    print(f"vp      : {vp.shape}   (expected: {len(d_values)} x {len(freqs)})")
    print(f"pout    : {pout.shape}")
    print(f"rhoE    : {rhoE.shape}")
    if ts:
        print(f"[i] Timestamp: {ts}")
    if resonance_csv:
        print(f"[i] Resonance CSV: {resonance_csv}")

    # ---------- Переопределение сеток (ресэмплинг) ----------
    # Частоты
    if args.freqs:
        try:
            f_new = parse_freqs_arg(args.freqs)
            if f_new.ndim != 1 or len(f_new) < 2:
                raise ValueError("Parsed frequency grid is invalid.")
            vp   = resample_matrix_along_axis(vp,   freqs, f_new, axis=1)
            pout = resample_matrix_along_axis(pout, freqs, f_new, axis=1)
            rhoE = resample_matrix_along_axis(rhoE, freqs, f_new, axis=1)
            freqs = f_new
            print(f"[i] Frequencies overridden via --freqs: {freqs[0]}..{freqs[-1]} (N={len(freqs)})")
        except Exception as e:
            print(f"[warn] --freqs ignored: {e}")

    # d (мм)
    if args.d:
        try:
            d_new = parse_d_arg(args.d)
            if d_new.ndim != 1 or len(d_new) < 2:
                raise ValueError("Parsed d grid is invalid.")
            vp   = resample_matrix_along_axis(vp,   d_values, d_new, axis=0)
            

            pout = resample_matrix_along_axis(pout, d_values, d_new, axis=0)
            rhoE = resample_matrix_along_axis(rhoE, d_values, d_new, axis=0)
            d_values = d_new
            print(f"[i] d-values overridden via --d: {d_values[0]}..{d_values[-1]} (N={len(d_values)})")
        except Exception as e:
            print(f"[warn] --d ignored: {e}")
    if args.affine:
        a, b = parse_affine_arg(args.affine)
        # Если нужно применять к значениям (обычно к Vp):
        vp   = a * vp   + b
        affine_text = f"[AUTO_SCALE] y = {a:.6g}·y + {b:.3e} (applied)"
        print("\n" + affine_text) 
    # pout_mcW, rhoE_mcW_cm3 = resample_matrix_along_axis(
    #         vp,
    #         old_axis_vals=None, new_axis_vals=None,  # не нужны
    #         compute_from_vp=True,
    #         r_load_ohm=1.9,        # например 1.9 Ом
    #         volume_cm3=0.95,       # объём устройства
    #         vp_is_peak=True
    #     )
    pout_mcW, rhoE_mcW_cm3 = pout, rhoE


    # Проверка NaN-долей (полезно понимать “пустоты” после ресэмплинга)
    print("\n[DEBUG] NaN share:")
    for name, m in [("vp", vp), ("pout", pout), ("rhoE", rhoE)]:
        nshare = float(np.isnan(m).mean())
        print(f"  {name}: {nshare:.3f}")

    # ----- Plotting -----
    base_tag = ts or "latest"

    # 3D surfaces
    vp_3d_base  = os.path.join(exp_dir, f"viz_vp_3d_{base_tag}")
    rho_3d_base = os.path.join(exp_dir, f"viz_rhoE_3d_{base_tag}")
    pout_3d_base = os.path.join(exp_dir, f"viz_pout_3d_{base_tag}")
    plot_3d_surface(
        d_values, freqs, vp,
        "d (mm)", "Frequency (Hz)", "Vp (mV)",
        vp_3d_base, cmap="viridis",
        elev=18, azim=-35, z_aspect=0.70, mark_peak=False, show=False  # 3D показываем всегда
    )

    plot_3d_surface(
        d_values, freqs, rhoE_mcW_cm3,
        "d (mm)", "Frequency (Hz)", "Energy density (mW/cm³)",
        rho_3d_base, cmap="plasma",
        elev=18, azim=-35, z_aspect=0.70, mark_peak=False, show=False
    )

    plot_3d_surface(
        d_values, freqs, pout_mcW,
        "d (mm)", "Frequency (Hz)", "Output power (mW)",
        pout_3d_base, cmap="viridis",
        elev=18, azim=-35, z_aspect=0.70, mark_peak=False, show=False
    )

    plot_3d_surface(
        d_values, freqs, pout_mcW,
        "d (mm)", "Frequency (Hz)", "Output power (mW)",
        pout_3d_base, cmap="plasma",
        elev=18, azim=-35, z_aspect=0.70, mark_peak=False, show=False
    )

    vp_vs_f_base = os.path.join(exp_dir, f"viz_vp_vs_freq_{base_tag}")
    plot_2d_vp_vs_freq(freqs, vp, d_values, vp_vs_f_base, show=args.show)

    pout_vs_d_base = os.path.join(exp_dir, f"viz_pout_max_vs_d_{base_tag}")
    rho_vs_d_base  = os.path.join(exp_dir, f"viz_rhoE_max_vs_d_{base_tag}")
    plot_2d_max_vs_d(d_values, pout_mcW, "Output power (mW)", pout_vs_d_base, show=args.show)
    plot_2d_max_vs_d(d_values, rhoE_mcW_cm3, "Energy density (mW/cm³)", rho_vs_d_base, show=args.show)

    # Heatmaps
    hm_vp_base   = os.path.join(exp_dir, f"viz_heatmap_vp_{base_tag}")
    hm_pout_base = os.path.join(exp_dir, f"viz_heatmap_pout_{base_tag}")
    hm_rho_base  = os.path.join(exp_dir, f"viz_heatmap_rhoE_{base_tag}")
    plot_heatmap(vp,   freqs, d_values, "Frequency (Hz)", "d (mm)", "Vp (mV)",           hm_vp_base,   cmap="viridis", show=args.show)
    plot_heatmap(pout_mcW, freqs, d_values, "Frequency (Hz)", "d (mm)", "Output power (mW)", hm_pout_base, cmap="viridis",   show=args.show)
    plot_heatmap(rhoE_mcW_cm3, freqs, d_values, "Frequency (Hz)", "d (mm)", "Energy density (mW/cm³)",      hm_rho_base,  cmap="plasma",  show=args.show)

    # Summary / optimum by ρ_E
    rho_max_per_d = np.nanmax(rhoE, axis=1)
    if np.any(np.isfinite(rho_max_per_d)):
        opt_idx = int(np.nanargmax(rho_max_per_d))
        print("\n==== Summary ====")
        print(f"Optimal d ≈ {d_values[opt_idx]:.0f} mm, ρ_E_max ≈ {rho_max_per_d[opt_idx]:.3f} mW/cm³")
    else:
        print("\n==== Summary ====\nCould not determine optimal d (all ρ_E are NaN).")

    print("\nSaved figures (PNG+PDF for each):")
    for base in [vp_3d_base, rho_3d_base, vp_vs_f_base, pout_vs_d_base, rho_vs_d_base, hm_vp_base, hm_pout_base, hm_rho_base]:
        print(" -", base + ".png")
        print("   ", base + ".pdf")

    if resonance_csv and os.path.isfile(resonance_csv):
        try:
            df = pd.read_csv(resonance_csv)
            print("\nResonance table preview (first 8 rows):")
            print(df.head(150).to_string(index=False))
        except Exception as e:
            print(f"[warn] Could not read {resonance_csv}: {e}")

if __name__ == "__main__":
    main()
