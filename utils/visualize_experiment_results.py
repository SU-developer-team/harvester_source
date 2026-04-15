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

# =========================
# Publication plotting style
# =========================
PUB_DPI = 600
PUB_FIGSIZE_2D = (6.5, 4.0)     # inches (≈ 165 x 102 mm) — typical 1-column
PUB_FIGSIZE_3D = (14, 4.5)      # чуть выше для 3D
PUB_FONT_FAMILY = "Arial"       # fallback на DejaVu Sans, если нет
PUB_FONT_SIZE = 11              # при необходимости 10

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
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

# ---------- Load arrays ----------
def load_arrays(folder: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[str], Optional[str]]:
    """
    Tries to load d_values, freqs, vp (d x f), pout (d x f), rhoE (d x f) from *.npy.
    Falls back to CSV matrices (rows=d, cols=f).
    Returns (d_values, freqs, vp, pout, rhoE, ts_str, resonance_csv_or_None)
    """
    d_path    = _pick_one([os.path.join(folder, "d_values_*.npy")])
    f_path    = _pick_one([os.path.join(folder, "freqs_*.npy")])
    vp_path   = _pick_one([os.path.join(folder, "vp_values_*.npy")])
    pout_path = _pick_one([os.path.join(folder, "pout_values_*.npy")])
    rho_path  = _pick_one([os.path.join(folder, "rho_e_values_*.npy")])

    # Extract timestamp if possible
    ts = None
    for p in [d_path, f_path, vp_path, pout_path, rho_path]:
        if p:
            m = re.search(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.npy$", p)
            if m:
                ts = m.group(1)
                break

    resonance_csv = _pick_one([os.path.join(folder, "resonance_params_*.csv")])

    if all([d_path, f_path, vp_path, pout_path, rho_path]):
        d_values = np.load(d_path)
        freqs    = np.load(f_path)
        vp       = np.load(vp_path)
        pout     = np.load(pout_path)
        rhoE     = np.load(rho_path)
        return d_values, freqs, vp, pout, rhoE, ts, resonance_csv

    # Fallback to CSV matrices
    vp_csv   = _pick_one([os.path.join(folder, "vp_values_*.csv")])
    pout_csv = _pick_one([os.path.join(folder, "pout_values_*.csv")])
    rho_csv  = _pick_one([os.path.join(folder, "rho_e_values_*.csv")])

    if not all([vp_csv, pout_csv, rho_csv]):
        raise FileNotFoundError("Expected *.npy or matrix *.csv files (vp/pout/rho_e).")

    for p in [vp_csv, pout_csv, rho_csv]:
        m = re.search(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv$", p)
        if m:
            ts = m.group(1)
            break

    vp_df   = pd.read_csv(vp_csv, index_col=0)
    pout_df = pd.read_csv(pout_csv, index_col=0)
    rho_df  = pd.read_csv(rho_csv, index_col=0)

    d_values = vp_df.index.astype(float).values
    freqs    = vp_df.columns.astype(float).values
    vp       = vp_df.values.astype(float)
    pout     = pout_df.values.astype(float)
    rhoE     = rho_df.values.astype(float)

    return d_values, freqs, vp, pout, rhoE, ts, resonance_csv

# ---------- Resampling helpers ----------
def resample_matrix_along_axis(matrix: np.ndarray,
                               old_axis_vals: np.ndarray,
                               new_axis_vals: np.ndarray,
                               axis: int = 1) -> np.ndarray:
    """
    Интерполирует matrix (NaN-safe) вдоль указанной оси.
    axis=1: для каждой строки интерполируем по столбцам (частоты).
    axis=0: для каждого столбца интерполируем по строкам (d).
    Значения вне диапазона old_axis_vals -> NaN (чтобы не плодить фиктивную экстраполяцию).
    """
    matrix = np.asarray(matrix, float)
    old_axis_vals = np.asarray(old_axis_vals, float)
    new_axis_vals = np.asarray(new_axis_vals, float)

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")

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
def plot_3d_surface(
    X_d, Y_f, Z,
    xlabel, ylabel, zlabel,
    out_base,
    cmap="viridis",
    elev=18,         # угол камеры (пониже взгляд)
    azim=-35,        # повернуть к оси частоты
    z_aspect=0.70,   # “понизить” высоту (0.6..0.8)
    mark_peak=False, # отметить максимум
    show=False
):
    """
    Рисует 3D-поверхность с публикационным стилем (без title).
    - elev/azim: угол камеры (ax.view_init)
    - z_aspect: относительный масштаб оси Z (ax.set_box_aspect)
    - mark_peak: отметить глобальный максимум маркером и подписью
    """
    D, F = np.meshgrid(X_d, Y_f)
    fig = plt.figure(figsize=PUB_FIGSIZE_3D)
    ax  = fig.add_subplot(111, projection="3d")

    # ВНИМАНИЕ: Z размером [len(d) x len(f)] -> для plot_surface используем Z.T
    surf = ax.plot_surface(D, F, Z.T, cmap=cmap, linewidth=0, antialiased=True)

    # подписи осей (англ., без title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    # камера и пропорции
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect((1.0, 1.0, z_aspect))
    except Exception:
        pass  # старые MPL могут не поддерживать

    # разумный потолок по Z (чуть выше пика)
    zmax = float(np.nanmax(Z))
    zmin = float(np.nanmin(Z))
    if np.isfinite(zmin) and np.isfinite(zmax):
        ax.set_zlim(zmin, zmax * 1.05)

    # цветовая шкала
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=14)

    # отметка пика
    if mark_peak and np.isfinite(zmax):
        i_d, i_f = np.unravel_index(np.nanargmax(Z), Z.shape)  # Z: [len(d) x len(f)]
        x_peak = float(X_d[i_d])
        y_peak = float(Y_f[i_f])
        z_peak = float(Z[i_d, i_f])
        ax.scatter([x_peak], [y_peak], [z_peak], s=50, c="k", depthshade=False, zorder=5)
        ax.text(x_peak, y_peak, z_peak * 1.02,
                f"d={x_peak:.0f} mm, f={y_peak:.1f} Hz, {zlabel.split()[0]}={z_peak:.1f}",
                ha="center", va="bottom")

    _save_fig(fig, out_base, show=show)

def plot_2d_vp_vs_freq(freqs, vp_matrix, d_values, out_base, show=False):
    fig = plt.figure(figsize=PUB_FIGSIZE_2D)
    ax = fig.add_subplot(111)
    for i, d in enumerate(d_values):
        y = vp_matrix[i, :]
        ax.plot(freqs, y, marker="o", linewidth=1.0, label=f"d={d:g} mm")
        if np.any(np.isfinite(y)):
            j = int(np.nanargmax(y))
            ax.plot(freqs[j], y[j], "x", markersize=8)
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("Vp (mV)")
    ax.grid(True, alpha=0.35)
    ax.legend(ncol=2, frameon=False)
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
    args = ap.parse_args()

    if not args.no_pub_style:
        apply_publication_style()

    # Выбор папки
    if args.dir is None:
        base_graphs = "experiments/harvester_50mm/exp_2/graphs"
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
    plot_3d_surface(
        d_values, freqs, vp,
        "d (mm)", "Frequency (Hz)", "Vp (mV)",
        vp_3d_base, cmap="viridis",
        elev=18, azim=-35, z_aspect=0.70, mark_peak=True, show=True  # 3D показываем всегда
    )

    plot_3d_surface(
        d_values, freqs, rhoE,
        "Magnet gap d (mm)", "Frequency (Hz)", "Energy density ρ_E (mW/cm³)",
        rho_3d_base, cmap="plasma",
        elev=18, azim=-35, z_aspect=0.70, mark_peak=True, show=True
    )

    # 2D curves
    vp_vs_f_base = os.path.join(exp_dir, f"viz_vp_vs_freq_{base_tag}")
    plot_2d_vp_vs_freq(freqs, vp, d_values, vp_vs_f_base, show=args.show)

    # Max vs d
    pout_vs_d_base = os.path.join(exp_dir, f"viz_pout_max_vs_d_{base_tag}")
    rho_vs_d_base  = os.path.join(exp_dir, f"viz_rhoE_max_vs_d_{base_tag}")
    plot_2d_max_vs_d(d_values, pout, "Output power P_out (mW)", pout_vs_d_base, show=args.show)
    plot_2d_max_vs_d(d_values, rhoE, "Energy density ρ_E (mW/cm³)", rho_vs_d_base, show=args.show)

    # Heatmaps
    hm_vp_base   = os.path.join(exp_dir, f"viz_heatmap_vp_{base_tag}")
    hm_pout_base = os.path.join(exp_dir, f"viz_heatmap_pout_{base_tag}")
    hm_rho_base  = os.path.join(exp_dir, f"viz_heatmap_rhoE_{base_tag}")
    plot_heatmap(vp,   freqs, d_values, "Frequency (Hz)", "Magnet gap d (mm)", "Vp (mV)",           hm_vp_base,   cmap="viridis", show=args.show)
    plot_heatmap(pout, freqs, d_values, "Frequency (Hz)", "Magnet gap d (mm)", "Output power (mW)", hm_pout_base, cmap="magma",   show=args.show)
    plot_heatmap(rhoE, freqs, d_values, "Frequency (Hz)", "Magnet gap d (mm)", "ρ_E (mW/cm³)",      hm_rho_base,  cmap="plasma",  show=args.show)

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
            print(df.head(8).to_string(index=False))
        except Exception as e:
            print(f"[warn] Could not read {resonance_csv}: {e}")

if __name__ == "__main__":
    main()
