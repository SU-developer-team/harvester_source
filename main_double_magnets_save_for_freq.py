# -*- coding: utf-8 -*-
"""
Одна частота -> сохраняем ЭДС катушки (time series), позицию/скорость магнита и диапазоны.
- Симуляция только по одной частоте FREQ_HZ
- Сохраняем CSV временного ряда (не RMS)
- Сохраняем summary с min/max/ptp для z, v, emf_open, emf_self, v_term
- Рисуем графики

Зависимости: numpy, pandas, matplotlib
Модель: ElectromagneticHarvesterID80mm_generator.ElectromagneticHarvesterID80mm
"""

import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import ElectromagneticHarvesterID80mm
import time

# =========================
# НАСТРОЙКИ
# =========================
# Одна частота (шейкер)
FREQ_HZ = 30.0

# Длительность и дискретизация
T_MAX_S = 10.0
MODEL_FS_HZ = 1000.0

# Электрика
COIL_R_OHM = 1.0
LOAD_R_OHM = 1.0

# Сохранение
SAVE_BASE_DIR = None  # None -> папка рядом со скриптом


# =========================
# Хелперы
# =========================
def create_out_dir(base_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = base_dir / "graphs" / f"single_freq_{ts}_f{str(FREQ_HZ).replace('.','_')}Hz"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def calc_ranges(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"min": np.nan, "max": np.nan, "ptp": np.nan}
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "ptp": float(np.max(x) - np.min(x)),
    }


def main():
    now = time.time()
    base_dir = Path(SAVE_BASE_DIR) if SAVE_BASE_DIR else Path(os.path.dirname(os.path.abspath(__file__)))
    out_dir = create_out_dir(base_dir)

    # --- 1) Настройка модели ---
    device = ElectromagneticHarvesterID80mm()
    device.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)
    device.set_frequency(float(FREQ_HZ))

    steps = int(T_MAX_S * MODEL_FS_HZ)
    t_eval = np.linspace(0.0, T_MAX_S, steps, endpoint=False)

    # solve_all: t_s, z, v, i, emf_open_v, emf_self_v, forces, v_term_v
    t_s, z_m, v_mps, i_A, emf_open_V, emf_self_V, forces, v_term_V = device.solve_all(
        t_eval_s=t_eval, rtol=1e-5, atol=1e-8, clamp_to_base=False, dense_output=False
    )

    # --- 2) Сохраняем временной ряд (CSV) ---
    # emf_open_V = ЭДС катушки (open-circuit)
    # emf_self_V = ЭДС самоиндукции (если модель так определяет)
    # v_term_V   = напряжение на клеммах/нагрузке (если нужно)
    df_ts = pd.DataFrame({
        "t_s": t_s,
        "z_m": z_m,
        "v_mps": v_mps,
        "emf_open_V": emf_open_V,   # ЭДС катушки (основное)
        "emf_self_V": emf_self_V,   # ЭДС самоиндукции
        "v_term_V": v_term_V,       # клеммное напряжение
        "i_A": i_A,                 # ток
    })

    ts_csv = out_dir / "timeseries_emf_position_velocity.csv"
    df_ts.to_csv(ts_csv, index=False, encoding="utf-8")
    print(f"CSV временного ряда сохранён: {ts_csv}")
    print(f"Время расчёта: {time.time() - now:.2f} сек")

    # --- 3) Диапазоны (min/max/ptp) ---
    z_rng = calc_ranges(z_m)
    v_rng = calc_ranges(v_mps)
    emf_open_rng = calc_ranges(emf_open_V)
    emf_self_rng = calc_ranges(emf_self_V)
    v_term_rng = calc_ranges(v_term_V)

    summary = pd.DataFrame([
        {"name": "position_z_m",          "min": z_rng["min"], "max": z_rng["max"], "ptp": z_rng["ptp"], "unit": "m"},
        {"name": "velocity_v_mps",        "min": v_rng["min"], "max": v_rng["max"], "ptp": v_rng["ptp"], "unit": "m/s"},
        {"name": "coil_emf_open_V",       "min": emf_open_rng["min"], "max": emf_open_rng["max"], "ptp": emf_open_rng["ptp"], "unit": "V"},
        {"name": "self_induction_emf_V",  "min": emf_self_rng["min"], "max": emf_self_rng["max"], "ptp": emf_self_rng["ptp"], "unit": "V"},
        {"name": "terminal_voltage_V",    "min": v_term_rng["min"], "max": v_term_rng["max"], "ptp": v_term_rng["ptp"], "unit": "V"},
    ])

    summary_csv = out_dir / "summary_ranges.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8")
    print(f"Summary (диапазоны) сохранён: {summary_csv}")

    print("\n=== Диапазоны ===")
    print(summary.to_string(index=False))

    # --- 4) Графики ---
    # (а) Позиция и скорость
    plt.figure(figsize=(12, 6))
    plt.plot(t_s, z_m, lw=1.0, label="z(t) позиция, m")
    plt.plot(t_s, v_mps, lw=1.0, label="v(t) скорость, m/s")
    plt.title(f"Позиция и скорость (f = {FREQ_HZ} Hz)")
    plt.xlabel("t, s")
    plt.grid(True, alpha=0.3)
    plt.legend()
    p1 = out_dir / "position_velocity.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=200)
    plt.show()
    print(f"График позиции/скорости сохранён: {p1}")

    # (б) ЭДС/напряжение
    plt.figure(figsize=(12, 6))
    plt.plot(t_s, emf_open_V, lw=1.0, label="EMF open (ЭДС катушки), V")
    plt.plot(t_s, emf_self_V, lw=1.0, label="EMF self (самоиндукция), V")
    # plt.plot(t_s, v_term_V, lw=1.0, label="V_terminal (клеммы), V")
    plt.title(f"ЭДС/напряжение (f = {FREQ_HZ} Hz)")
    plt.xlabel("t, s")
    plt.grid(True, alpha=0.3)
    plt.legend()
    p2 = out_dir / "emf_and_terminal_voltage.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=200)
    plt.show()
    print(f"График ЭДС сохранён: {p2}")

    # (в) Ток (если хочешь видеть)
    plt.figure(figsize=(12, 4))
    plt.plot(t_s, i_A, lw=1.0, label="i(t), A")
    plt.title(f"Ток (f = {FREQ_HZ} Hz)")
    plt.xlabel("t, s")
    plt.grid(True, alpha=0.3)
    plt.legend()
    p3 = out_dir / "current.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=200)
    plt.show()
    print(f"График тока сохранён: {p3}")

    print(f"\nГотово. Папка результатов: {out_dir}")


if __name__ == "__main__":
    main()
