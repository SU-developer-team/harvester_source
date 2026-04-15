# -*- coding: utf-8 -*-
"""
Batch runner:
freq = 2..30 Hz (step 1)
velocity = 0..1.2 (step 0.05)

Каждая точка:
model_exp_data/f{freq}hz_v{vel}/
  - timeseries.csv
  - summary_ranges.csv
  - meta.txt

Параллельно через multiprocessing.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from datetime import datetime

from model import ElectromagneticHarvesterID80mm
from tqdm import tqdm


# =========================
# ПАРАМЕТРЫ СЕТКИ
# =========================
# FREQS = np.arange(5, 35.25, 0.25)              # 2..30 Hz
# VELOCITIES = np.arange(0.0, 1.21, 0.01)  # 0..1.20

FREQS = [10.6]              # 2..30 Hz
VELOCITIES = [0, 0.6 , 1.2]  # 0..1.20
T_MAX_S = 120.0
FS = 1000.0

COIL_R_OHM = 1.0
LOAD_R_OHM = 1.0

N_PROC = os.cpu_count() - 1   # <= твои 32 потока


BASE_DIR = Path('D:/PROJECTs/magnet/harvester/experiments/harvester_80mm_v2')
OUT_ROOT = BASE_DIR / 'exp3'
OUT_ROOT.mkdir(exist_ok=True)


# =========================
# ХЕЛПЕРЫ
# =========================
def calc_ranges(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"min": np.nan, "max": np.nan, "ptp": np.nan}
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "ptp": float(x.max() - x.min())
    }


# =========================
# ОДНА ТОЧКА РАСЧЁТА
# =========================
def run_point(args):
    freq, velocity = args

    folder = OUT_ROOT / f"f{int(freq)}hz_v{int(round(velocity*100)):03d}"

    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "summary_ranges.csv").exists():
        return True


    try:
        dev = ElectromagneticHarvesterID80mm()
        dev.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)
        dev.set_frequency(float(freq))
        dev.set_base_velocity(float(velocity))  # <<< КЛЮЧЕВО

        steps = int(T_MAX_S * FS)
        t_eval = np.linspace(0, T_MAX_S, steps, endpoint=False)

        t, z, v, i, emf_open, emf_self, forces, v_term = dev.solve_all(
            t_eval_s=t_eval,
            rtol=1e-5,
            atol=1e-8,
            clamp_to_base=False,
            dense_output=False
        )

        # --- timeseries ---
        df = pd.DataFrame({
            "t_s": t,
            "z_m": z,
            "v_mps": v,
            "emf_open_V": emf_open,
            "emf_self_V": emf_self,
            "v_term_V": v_term,
            "i_A": i
        })
        df.to_csv(folder / "timeseries.csv", index=False)

        # --- ranges ---
        summary = pd.DataFrame([
            {"name": "position_z_m", "unit": "m",   **calc_ranges(z)},
            {"name": "velocity_mps", "unit": "m/s", **calc_ranges(v)},
            {"name": "emf_V",   "unit": "V",   **calc_ranges(emf_open)},
            {"name": "emf_self_V",   "unit": "V",   **calc_ranges(emf_self)},
            {"name": "v_term_V",     "unit": "V",   **calc_ranges(v_term)},
        ])
        summary.to_csv(folder / "summary_ranges.csv", index=False)

        # --- meta ---
        with open(folder / "meta.txt", "w", encoding="utf-8") as f:
            f.write(
                f"freq_hz = {freq}\n"
                f"base_velocity = {velocity}\n"
                f"T_MAX_S = {T_MAX_S}\n"
                f"FS = {FS}\n"
                f"timestamp = {datetime.now()}\n"
            )

        return True

    except Exception as e:
        with open(folder / "error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))
        return False


# =========================
# MAIN
# =========================
def main():
    grid = [(f, v) for f in FREQS for v in VELOCITIES]
    total = len(grid)

    print(f"Всего точек: {total}")
    print(f"Параллельно: {N_PROC} процессов")
    print(f"Выходная папка: {OUT_ROOT}")

    with Pool(processes=N_PROC) as pool:
        list(
            tqdm(
                pool.imap_unordered(run_point, grid),
                total=total,
                desc="Simulation",
                unit="point",
                smoothing=0.1
            )
        )

    print("Готово. Все данные в:", OUT_ROOT)



if __name__ == "__main__":
    main()
