# -*- coding: utf-8 -*-
"""
Sweep по межмагнитному расстоянию d с ЛОГИКОЙ "мейна" (freq-RMS):
- читаем ЭКСПЕРИМЕНТАЛЬНУЮ ЭДС, интерполируем на сетку модели;
- синтез шейкера: a(t) = -W^2 * X0 * cos(W t);
- model: total_emf = emf_open + emf_self;
- RMS/Power считаем от total_emf (RMS от суммы!);
- калибровка для графиков: Vrms*14200 + 1, Vp*14200 + 1.

Сохраняет:
  • metrics_d{d}.csv (на каждой частоте: Vrms/Vp/P модели + Vrms/P эксперимента),
  • summary.csv по резонансам и ρ_E,
  • 3D-матрицы (npy + csv) совместимые с visualize_experiment_results.py.
"""

import os
import csv
import math
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from ElectromagneticHarvesterID50mm_ERZHANAT import ElectromagneticHarvesterAbsolute

# =========================
# Конфигурация
# =========================
BASE_DIR = r"experiments/harvester_50mm/exp_2"
EXP_EMF_PATH_TEMPLATE = os.path.join(BASE_DIR, "data", "{freq}.csv")

# Частоты, время, дискретизация
FREQS = np.arange(2, 36, 1)          # 2..35 Гц
EMF_FS_HZ = 1000.0
T_SLICE = 5.0                         # анализируем последние 5 секунд

# Диапазон d (мм)
D_VALUES_MM = np.arange(49, 51, 1)    # пример: 49..50 мм

# Парсинг CSV с ЭДС (эксперимент)
EMF_COL_IDX = 3
EMF_SEP = ";"
EMF_DECIMAL = ","
EMF_UNITS = "mV"                      # 'V' | 'mV' | 'uV'

# Электрика
COIL_R_OHM = 0.001
LOAD_R_OHM = 1.0

# Калибровка под графики
CAL_GAIN_EMF = 14200.0
CAL_OFFSET_mV = 1.0

# Амплитуды X0 по частотам (м)
X0_FOR_FREQ = {
    2: 0.001, 3: 0.006, 4: 0.006, 5: 0.006, 6: 0.006, 7: 0.006, 8: 0.006,
    9: 0.008, 10: 0.013, 11: 0.013, 12: 0.0145, 13: 0.0155, 14: 0.0177,
    15: 0.0195, 16: 0.0215, 17: 0.0235, 18: 0.024, 19: 0.0175, 20: 0.0092,
    21: 0.0067, 22: 0.0052, 23: 0.0042, 24: 0.0036, 25: 0.0031, 26: 0.0027,
    27: 0.00235, 28: 0.002, 29: 0.0019, 30: 0.0018, 31: 0.0018, 32: 0.0018,
    33: 0.00183, 34: 0.0017, 35: 0.0015
}

# =========================
# Утилиты ввода/вывода
# =========================
def create_experiment_folder(base_dir: str) -> Tuple[str, str]:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(base_dir, "graphs", f"experiment_d_sweep_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, ts

def _to_numeric_series(raw, decimal_hint: Optional[str]) -> pd.Series:
    s = pd.Series(raw, copy=True).astype(str)
    s = s.str.replace('\u00A0', '', regex=False)  # NBSP
    s = s.str.replace(' ', '', regex=False)
    s = s.str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce')

def load_emf_no_time(
    file_path: str,
    emf_col_idx: int,
    fs_hz: float,
    sep: str = ";",
    decimal: str = ",",
    units: str = "mV",
    normalize_time: bool = True,
) -> Tuple[np.ndarray, np.ndarray, interp1d]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path, sep=sep, header=None, dtype=str)
    if not (0 <= emf_col_idx < df.shape[1]):
        raise ValueError(f"В файле {os.path.basename(file_path)} нет столбца с индексом {emf_col_idx}")

    emf_raw = df.iloc[:, emf_col_idx]
    emf = _to_numeric_series(emf_raw, decimal_hint=decimal).to_numpy(dtype=float)

    scale = {"V": 1.0, "mV": 1e-3, "uV": 1e-6}[units]
    emf_v = emf * scale

    n = int(len(emf_v))
    t_s = np.arange(n, dtype=float) / float(fs_hz)
    if normalize_time and n > 0:
        t_s = t_s - t_s[0]

    # линейная интерполяция дыр
    bad = ~np.isfinite(emf_v)
    if bad.any():
        s = pd.Series(emf_v, index=t_s).interpolate(method="linear", limit_direction="both")
        if s.isna().sum() > 0:
            raise ValueError("В ЭДС после интерполяции остались NaN. Проверьте файл.")
        emf_v = s.to_numpy()

    f = interp1d(t_s, emf_v, kind="linear", bounds_error=False, fill_value=(emf_v[0], emf_v[-1]))
    return t_s, emf_v, f

def write_temp_shaker_csv(t_s: np.ndarray, a_s: np.ndarray) -> str:
    fd, path = tempfile.mkstemp(prefix="shaker_", suffix=".csv")
    os.close(fd)
    pd.DataFrame({"t": t_s, "acc_mps2": a_s}).to_csv(path, sep=";", index=False, header=True, float_format="%.9f")
    return path

def rms(x: np.ndarray, center: bool = True) -> float:
    x = np.asarray(x, dtype=float)
    if center:
        x = x - np.mean(x)
    return float(np.sqrt(np.mean(x**2)))

# =========================
# Геометрия и объём
# =========================
def apply_geometry_for_d(device: ElectromagneticHarvesterAbsolute, d_mm: float) -> None:
    """
    coil_height = d; пересчёт числа витков и относительных отметок.
    """
    d_m = d_mm / 1000.0
    device.coil_height_m = d_m
    device.turns_N = max(1, int(round(device.coil_height_m / device.wire_diam_m)))
    device._precompute_coil_geometry()

    device.coil_z_bottom_rel = device.magnet_height_m + 0.002
    device.coil_z_top_rel = device.coil_z_bottom_rel + device.coil_height_m
    device.top_magnet_z_bottom_rel = device.coil_z_top_rel + 0.002
    device.top_magnet_z_top_rel = device.top_magnet_z_bottom_rel + device.magnet_height_m
    device.top_magnet_center_rel = 0.5 * (device.top_magnet_z_top_rel + device.top_magnet_z_bottom_rel)
    device.bottom_magnet_center_rel = 0.5 * (device.bottom_magnet_z_top_rel + device.bottom_magnet_z_bottom_rel)

def calculate_volume_cm3(device: ElectromagneticHarvesterAbsolute, d_mm: float) -> float:
    d_m = d_mm / 1000.0
    coil_outer_radius_m = device.coil_diam_m / 2.0
    volume_m3 = np.pi * coil_outer_radius_m**2 * d_m
    return volume_m3 * 1e6  # в см³

# =========================
# Шейкер
# =========================
def get_x0_for_freq(freq_hz: int) -> float:
    if freq_hz in X0_FOR_FREQ:
        return float(X0_FOR_FREQ[freq_hz])
    raise ValueError(f"Нет X0 для {freq_hz} Гц")

def synthesize_base_accel(freq_hz: float, t_s: np.ndarray) -> np.ndarray:
    X0 = get_x0_for_freq(int(round(freq_hz)))
    W = 2.0 * math.pi * float(freq_hz)
    return - (W ** 2) * X0 * np.cos(W * t_s)

# =========================
# Worker по частоте (для фиксированного d) — ЛОГИКА МЕЙНА
# =========================
def _process_one_freq_for_d(args):
    """
    Возвращает:
      (freq_hz,
       Vrms_total_cal_mV, Vp_total_cal_mV, P_total_mW,
       Vrms_exp_cal_mV,    P_exp_mW)
    """
    d_mm, freq_hz = args
    try:
        # Модель
        device = ElectromagneticHarvesterAbsolute()
        device.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)
        apply_geometry_for_d(device, d_mm)

        # Экспериментальная ЭДС и окно последних T_SLICE сек
        emf_csv = EXP_EMF_PATH_TEMPLATE.format(freq=freq_hz)
        t_emf_s, emf_exp_v, emf_exp_interp = load_emf_no_time(
            emf_csv, emf_col_idx=EMF_COL_IDX, fs_hz=EMF_FS_HZ,
            sep=EMF_SEP, decimal=EMF_DECIMAL, units=EMF_UNITS, normalize_time=True
        )
        if len(t_emf_s) == 0:
            raise ValueError("Пустые данные ЭДС")

        t_max = float(t_emf_s[-1])
        mask = t_emf_s >= (t_max - T_SLICE)
        t_window = t_emf_s[mask]
        if t_window.size < 3:
            # fallback — возьмём как есть
            t_window = t_emf_s

        # Синтетический шейкер на этом же окне
        a_base = synthesize_base_accel(freq_hz=float(freq_hz), t_s=t_window)
        tmp_csv = write_temp_shaker_csv(t_window, a_base)
        try:
            device.load_base_from_csv(tmp_csv, time_col="t", accel_col="acc_mps2",
                                      sep=";", decimal=".", normalize_time=True)
        finally:
            try: os.remove(tmp_csv)
            except OSError: pass

        # Решение модели на сетке t_window
        t_s, z, v, i, emf_open_v, emf_self_v, forces, v_term_v, *rest = device.solve_all(
            t_eval_s=t_window, rtol=1e-5, atol=1e-8, clamp_to_base=True
        )

        # Эксперимент на сетке модели
        emf_exp_on_grid = emf_exp_interp(t_s)

        # Полная ЭДС (как в main)
        total_emf_v = emf_open_v + emf_self_v

        # Фильтрация NaN/inf синхронно
        ok = (np.isfinite(total_emf_v) & np.isfinite(emf_exp_on_grid))
        if not np.all(ok):
            total_emf_v     = total_emf_v[ok]
            emf_exp_on_grid = emf_exp_on_grid[ok]
            if total_emf_v.size == 0:
                raise ValueError("Нет валидных точек после фильтрации")

        # === Метрики: RMS от суммы сигналов (исправлено относительно старой версии) ===
        vrms_total_v = rms(total_emf_v, center=True)
        p_total_mW   = (vrms_total_v**2) / LOAD_R_OHM * 1000.0
        vp_total_v   = np.max(np.abs(total_emf_v - np.mean(total_emf_v)))

        # Эксперимент для справки/таблиц: RMS/Power на той же сетке
        vrms_exp_v = rms(emf_exp_on_grid, center=True)
        p_exp_mW   = (vrms_exp_v**2) / LOAD_R_OHM * 1000.0

        # Калибровка под графики (как в main)
        vrms_total_cal_mV = vrms_total_v * CAL_GAIN_EMF + CAL_OFFSET_mV
        vp_total_cal_mV   = vp_total_v   * CAL_GAIN_EMF + CAL_OFFSET_mV
        vrms_exp_cal_mV   = vrms_exp_v   * CAL_GAIN_EMF + CAL_OFFSET_mV

        return (int(freq_hz),
                float(vrms_total_cal_mV), float(vp_total_cal_mV), float(p_total_mW),
                float(vrms_exp_cal_mV),   float(p_exp_mW))

    except Exception as e:
        print(f"[d={d_mm} мм] Ошибка на f={freq_hz} Гц: {e}")
        return (int(freq_hz), np.nan, np.nan, np.nan, np.nan, np.nan)

# =========================
# Сохранение 3D-матриц/CSV
# =========================
def _dump_mat_csv(out_dir, ts, name, mat, idx, cols):
    p = os.path.join(out_dir, f"{name}_{ts}.csv")
    pd.DataFrame(mat, index=idx, columns=cols).to_csv(p, encoding="utf-8", float_format="%.6g")
    return p

# =========================
# Основной сценарий
# =========================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir, ts = create_experiment_folder(base_dir)

    num_proc = min(mp.cpu_count(), max(1, len(FREQS)))
    print(f"Используем {num_proc} процессов; sweep по d={list(D_VALUES_MM)}")

    summary_rows = []

    # Матрицы: строки — d, столбцы — f
    VRMS_model_mat = np.full((len(D_VALUES_MM), len(FREQS)), np.nan, dtype=float)  # калибр., мВ
    VP_model_mat   = np.full((len(D_VALUES_MM), len(FREQS)), np.nan, dtype=float)  # калибр., мВ
    P_model_mat    = np.full((len(D_VALUES_MM), len(FREQS)), np.nan, dtype=float)  # мВт
    VRMS_exp_mat   = np.full((len(D_VALUES_MM), len(FREQS)), np.nan, dtype=float)  # калибр., мВ
    P_exp_mat      = np.full((len(D_VALUES_MM), len(FREQS)), np.nan, dtype=float)  # мВт
    RHO_E_mat      = np.full((len(D_VALUES_MM), len(FREQS)), np.nan, dtype=float)  # мВт/см^3

    with mp.Pool(processes=num_proc) as pool:
        for i_d, d_mm in enumerate(D_VALUES_MM):
            print(f"\n=== Обработка d = {d_mm} мм ===")
            # Для объёма
            dev_ref = ElectromagneticHarvesterAbsolute()
            dev_ref.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)
            apply_geometry_for_d(dev_ref, d_mm)
            vol_cm3 = calculate_volume_cm3(dev_ref, d_mm)

            args = [(float(d_mm), int(f)) for f in FREQS]
            results = pool.map(_process_one_freq_for_d, args)

            # Таблица по d
            df_d = pd.DataFrame(results, columns=[
                "freq_hz",
                "Vrms_total_cal_mV", "Vp_total_cal_mV", "P_total_mW",
                "Vrms_exp_cal_mV",   "P_exp_mW"
            ]).sort_values("freq_hz")

            d_csv_path = os.path.join(out_dir, f"metrics_d{int(d_mm)}_{ts}.csv")
            df_d.to_csv(d_csv_path, index=False, encoding="utf-8")
            print(f"[d={d_mm}] сохранено: {d_csv_path}")

            # Заполнение матриц
            for j, f in enumerate(FREQS):
                row = df_d[df_d["freq_hz"] == f]
                if row.empty:
                    continue
                VRMS_model_mat[i_d, j] = row["Vrms_total_cal_mV"].values[0]
                VP_model_mat[i_d,   j] = row["Vp_total_cal_mV"].values[0]
                P_model_mat[i_d,    j] = row["P_total_mW"].values[0]
                VRMS_exp_mat[i_d,   j] = row["Vrms_exp_cal_mV"].values[0]
                P_exp_mat[i_d,      j] = row["P_exp_mW"].values[0]
                RHO_E_mat[i_d,      j] = (P_model_mat[i_d, j] / vol_cm3) if np.isfinite(P_model_mat[i_d, j]) and vol_cm3>0 else np.nan

            # Резонанс по «мейну»: максимум мощности модели (и для диагностики — по Vp)
            if np.any(np.isfinite(P_model_mat[i_d, :])):
                idx_pmax = int(np.nanargmax(P_model_mat[i_d, :]))
                f_res_power = int(FREQS[idx_pmax]); p_max = float(P_model_mat[i_d, idx_pmax])
            else:
                f_res_power, p_max = np.nan, np.nan

            if np.any(np.isfinite(VP_model_mat[i_d, :])):
                idx_vpmax = int(np.nanargmax(VP_model_mat[i_d, :]))
                f_res_vp = int(FREQS[idx_vpmax]); vp_max = float(VP_model_mat[i_d, idx_vpmax])
            else:
                f_res_vp, vp_max = np.nan, np.nan

            rho_e_max = float(np.nanmax(RHO_E_mat[i_d, :])) if np.any(np.isfinite(RHO_E_mat[i_d, :])) else np.nan

            summary_rows.append({
                "d_mm": d_mm,
                "F_res_by_power_Hz": f_res_power,
                "Pout_max_mW": p_max,
                "F_res_by_Vp_Hz": f_res_vp,
                "Vp_max_cal_mV": vp_max,
                "rho_E_max_mW_per_cm3": rho_e_max
            })

    # === Сводные сохранения ===
    d_idx  = D_VALUES_MM
    f_cols = FREQS

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(out_dir, f"summary_{ts}.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    print(f"\nСводка по d сохранена: {summary_csv}")

    # NPY
    np.save(os.path.join(out_dir, f"vrms_model_cal_mV_{ts}.npy"), VRMS_model_mat)
    np.save(os.path.join(out_dir, f"vp_model_cal_mV_{ts}.npy"),   VP_model_mat)
    np.save(os.path.join(out_dir, f"p_model_mW_{ts}.npy"),        P_model_mat)
    np.save(os.path.join(out_dir, f"vrms_exp_cal_mV_{ts}.npy"),   VRMS_exp_mat)
    np.save(os.path.join(out_dir, f"p_exp_mW_{ts}.npy"),          P_exp_mat)
    np.save(os.path.join(out_dir, f"rho_E_mW_per_cm3_{ts}.npy"),  RHO_E_mat)

    # CSV-матрицы
    paths_csv = [
        _dump_mat_csv(out_dir, ts, "vrms_model_cal_mV", VRMS_model_mat, d_idx, f_cols),
        _dump_mat_csv(out_dir, ts, "vp_model_cal_mV",   VP_model_mat,   d_idx, f_cols),
        _dump_mat_csv(out_dir, ts, "p_model_mW",        P_model_mat,    d_idx, f_cols),
        _dump_mat_csv(out_dir, ts, "vrms_exp_cal_mV",   VRMS_exp_mat,   d_idx, f_cols),
        _dump_mat_csv(out_dir, ts, "p_exp_mW",          P_exp_mat,      d_idx, f_cols),
        _dump_mat_csv(out_dir, ts, "rho_E_mW_per_cm3",  RHO_E_mat,      d_idx, f_cols),
    ]
    print("CSV-матрицы сохранены:\n - " + "\n - ".join(paths_csv))

    print("\nГотово. Папка эксперимента:", out_dir)
                                                                                                                    