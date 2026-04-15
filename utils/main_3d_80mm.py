# -*- coding: utf-8 -*-
"""
Инкрементальное сохранение и возобновление расчётов по d:
- После каждого d обновляет/добавляет строки в CSV (vp, pout, rho_e, resonance).
- При повторном запуске скрипт продолжает с того места, где остановились
  (или перезаписывает строки, если OVERWRITE=True).

Файлы в папке эксперимента:
    vp_values.csv         # строки = d (мм), столбцы = FREQS (Гц), значения = Vp (мВ)
    pout_values.csv       # ... значения = Pout (мВт)
    rho_e_values.csv      # ... значения = ρ_E (мВт/см³)
    resonance_params.csv  # сводка по d: F_res, Vp_max, Pout_max, ρ_E_max

При желании можно указать EXISTING_OUT_DIR, чтобы продолжить в уже созданной папке.
"""

import os
import math
from datetime import datetime
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import multiprocessing as mp

from model import ElectromagneticHarvesterID80mm

# =========================
# Конфигурация эксперимента
# =========================
BASE_DIR = "D:/PROJECTs/magnet/harvester/teory_results"   # корень для экспериментов
EXISTING_OUT_DIR: Optional[str] = None  # укажи путь к папке эксперимента, чтобы продолжить; иначе None
OVERWRITE = False        # True — пересчитывать и перезаписывать строки для уже посчитанных d
SKIP_DONE = True         # True — пропускать уже посчитанные d (если OVERWRITE=False)

T_SLICE    = 5.0         # длительность моделирования (с)
BASE_FS_HZ = 1000.0      # дискретизация времени

# Частоты 2..30 Гц с шагом 0.1 Гц
FREQS = np.arange(2.0, 30.0, 0.1)
FREQS = np.round(np.asarray(FREQS, float), 6)  # стабильная ось

# Диапазон зазора d (мм)
D_VALUES_MM = range(21, 201, 1)

# Электрические параметры
COIL_R_OHM = 0.001     # Ом (сопротивление катушки)
LOAD_R_OHM = 1.0       # Ом (нагрузка)

# =========================
# Хелперы
# =========================
def fkey(x: float) -> float:
    """Стабильный ключ частоты (округление до 6 знаков)."""
    return float(np.round(float(x), 6))

def create_experiment_folder(base_dir: str) -> Tuple[str, str]:
    """
    Создаёт иерархию: base_dir / YYYY-MM-DD / <timestamp>
    Возвращает (out_dir, timestamp).
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    date_dir = os.path.join(base_dir, today_str)
    os.makedirs(date_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(date_dir, f"{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, ts

def get_x0_for_freq(freq_hz: float) -> float:
    """Амплитуда базового смещения шейкера X0 (м) как функция частоты."""
    # TODO: заменить на реальную таблицу/интерполяцию при появлении данных
    return 0.0025

def calculate_volume(device: ElectromagneticHarvesterID80mm, d_mm: float) -> float:
    """Рабочий объём ~ цилиндр: радиус = внешний радиус катушки, высота = d."""
    d_m = d_mm / 1000.0
    coil_outer_radius_m = device.coil_outer_diam_m / 2.0
    volume_m3 = math.pi * (coil_outer_radius_m ** 2) * d_m
    return volume_m3 * 1e6  # в см³

def calculate_metrics(v_term_v: np.ndarray, i_model: np.ndarray, t_s: np.ndarray) -> dict:
    res = {}
    res["Vp_model_mV"] = float(np.nanmax(np.abs(v_term_v)) * 1000.0)
    res["Vrms_model_mV"] = float(np.sqrt(np.mean(v_term_v**2)) * 1000.0)
    p_inst = v_term_v * i_model
    res["Pout_model_mW"] = float(np.mean(p_inst) * 1000.0) if np.any(np.isfinite(p_inst)) else float("nan")
    if len(t_s) > 1:
        dt = float(np.mean(np.diff(t_s)))
        T = float(t_s[-1] - t_s[0])
        res["Pavg_model_mW"] = float((1.0 / T) * np.sum(p_inst) * dt * 1000.0)
    else:
        res["Pavg_model_mW"] = float("nan")
    return res

def print_parameters_table(device: ElectromagneticHarvesterID80mm, d_mm: float):
    print(f"\n=== Параметры для d = {d_mm} мм ===")
    params = {
        "m_kg": device.m_kg,
        "g_mps2": device.g_mps2,
        "mu0_Hpm": device.mu0_Hpm,
        "B_T": device.B_T,
        "magnet_radius_m": device.magnet_radius_m,
        "magnet_height_m": device.magnet_height_m,
        "coil_height_m": device.coil_height_m,
        "coil_inner_diam_m": device.coil_inner_diam_m,
        "coil_outer_diam_m": device.coil_outer_diam_m,
        "wire_diam_m": device.wire_diam_m,
        "turns_N": device.turns_N,
        "coil_resistance_ohm": device.coil_resistance_ohm,
        "load_resistance_ohm": device.load_resistance_ohm,
        "coil_z_bottom_m": device.coil_z_bottom_m,
        "coil_z_top_m": device.coil_z_top_m,
        "top_magnet_center_m": device.top_magnet_center_m,
        "bottom_magnet_center_m": device.bottom_magnet_center_m,
        "z0_m": device.z0_m,
        "v0_mps": device.v0_mps,
    }
    df_params = pd.DataFrame(list(params.items()), columns=["Параметр", "Значение"])
    print(df_params.to_string(index=False))

def reconfigure_geometry_for_d(device: ElectromagneticHarvesterID80mm, d_mm: float):
    """
    Перестраиваем геометрию под заданное межмагнитное расстояние d:
    - высота катушки = d
    - пересчитываем coil_z_top и координаты магнитов
    - пересчитываем turns_N по высоте/диаметру провода
    - пересчитываем индуктивность/сопротивление
    """
    d_m = d_mm / 1000.0
    device.device_height = d_m + device.magnet_height_m * 2 + 0.004  # с зазорами по 2 мм сверху/снизу
    device.coil_height_m = d_m
    device.turns_N = max(1, int(round(device.coil_height_m / max(device.wire_diam_m, 1e-6))))
    device.coil_z_bottom_m = device.magnet_height_m + 0.002  # от нижнего края до низа катушки 2 мм
    device.coil_z_top_m = device.coil_z_bottom_m + device.coil_height_m

    device.top_magnet_z_bottom_m = device.coil_z_top_m + 0.002
    device.top_magnet_z_top_m    = device.top_magnet_z_bottom_m + device.magnet_height_m
    device.bottom_magnet_z_top_m = device.coil_z_bottom_m - 0.002
    device.bottom_magnet_z_bottom_m = device.bottom_magnet_z_top_m - device.magnet_height_m
    device.top_magnet_center_m      = 0.5 * (device.top_magnet_z_top_m + device.top_magnet_z_bottom_m)
    device.bottom_magnet_center_m   = 0.5 * (device.bottom_magnet_z_top_m + device.bottom_magnet_z_bottom_m)

    device._precompute_coil_geometry()

# ========= Инкрементальные таблицы и сводка =========
def _csv_paths(out_dir: str) -> Tuple[str, str, str, str]:
    return (
        os.path.join(out_dir, "vp_values.csv"),
        os.path.join(out_dir, "pout_values.csv"),
        os.path.join(out_dir, "rho_e_values.csv"),
        os.path.join(out_dir, "resonance_params.csv"),
    )

def _ensure_freq_columns(df: pd.DataFrame, freqs: np.ndarray) -> pd.DataFrame:
    cols = [float(f) for f in freqs]
    df = df.copy()
    # Если пусто — просто выставим колонки
    if df.empty:
        df = pd.DataFrame(columns=cols)
    else:
        # Приведём колонки к float и отсортируем в порядке FREQS
        try:
            df.columns = [float(c) for c in df.columns]
        except Exception:
            # если вдруг строки — попробуем coercion
            df.columns = [float(str(c).replace(',', '.')) for c in df.columns]
        # Добавим недостающие столбцы
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        # Удалим лишние столбцы (не из FREQS)
        df = df[cols]
    return df

def load_or_init_tables(out_dir: str, freqs: np.ndarray):
    vp_csv, pout_csv, rho_csv, res_csv = _csv_paths(out_dir)
    if os.path.exists(vp_csv):
        vp_df = pd.read_csv(vp_csv, index_col=0)
    else:
        vp_df = pd.DataFrame()

    if os.path.exists(pout_csv):
        pout_df = pd.read_csv(pout_csv, index_col=0)
    else:
        pout_df = pd.DataFrame()

    if os.path.exists(rho_csv):
        rho_df = pd.read_csv(rho_csv, index_col=0)
    else:
        rho_df = pd.DataFrame()

    # Убедимся, что колонки совпадают с freqs
    vp_df = _ensure_freq_columns(vp_df, freqs)
    pout_df = _ensure_freq_columns(pout_df, freqs)
    rho_df = _ensure_freq_columns(rho_df, freqs)

    # Сводная
    if os.path.exists(res_csv):
        res_df = pd.read_csv(res_csv)
        # Нормализуем название колонок (на случай старых файлов)
        res_df.columns = [str(c).strip() for c in res_df.columns]
    else:
        res_df = pd.DataFrame(columns=['d, мм', 'F_res, Гц', 'Vp, мВ', 'Pout, мВт', 'ρ_E, мВт/см³'])

    # Индекс d должен быть float/мм
    def _fix_index(df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty:
            df.index = [float(i) for i in df.index]
            df.sort_index(inplace=True)
        return df
    vp_df   = _fix_index(vp_df)
    pout_df = _fix_index(pout_df)
    rho_df  = _fix_index(rho_df)

    return vp_df, pout_df, rho_df, res_df

def save_tables(out_dir: str, vp_df: pd.DataFrame, pout_df: pd.DataFrame, rho_df: pd.DataFrame, res_df: pd.DataFrame):
    vp_csv, pout_csv, rho_csv, res_csv = _csv_paths(out_dir)
    vp_df.sort_index().to_csv(vp_csv)
    pout_df.sort_index().to_csv(pout_csv)
    rho_df.sort_index().to_csv(rho_csv)
    res_df.sort_values(by='d, мм').to_csv(res_csv, index=False)

def upsert_row_for_d(
    d_mm: float,
    freqs: np.ndarray,
    vp_freq: np.ndarray,
    pout_freq: np.ndarray,
    rho_e_freq: np.ndarray,
    vp_df: pd.DataFrame,
    pout_df: pd.DataFrame,
    rho_df: pd.DataFrame,
    res_df: pd.DataFrame
):
    # Обновляем/добавляем строки в матрицы
    row_index = float(d_mm)
    cols = [float(f) for f in freqs]

    s_vp   = pd.Series(vp_freq,   index=cols, dtype=float)
    s_pout = pd.Series(pout_freq, index=cols, dtype=float)
    s_rho  = pd.Series(rho_e_freq,index=cols, dtype=float)

    vp_df.loc[row_index, cols]   = s_vp.values
    pout_df.loc[row_index, cols] = s_pout.values
    rho_df.loc[row_index, cols]  = s_rho.values

    # Пересчёт сводных параметров
    vp_max   = float(np.nanmax(vp_freq))   if np.any(np.isfinite(vp_freq))   else float('nan')
    pout_max = float(np.nanmax(pout_freq)) if np.any(np.isfinite(pout_freq)) else float('nan')
    rho_max  = float(np.nanmax(rho_e_freq))if np.any(np.isfinite(rho_e_freq))else float('nan')
    if np.any(np.isfinite(vp_freq)):
        f_res = float(freqs[np.nanargmax(vp_freq)])
    else:
        f_res = float('nan')

    # Апсерт строки в сводной таблице
    mask = (res_df['d, мм'] == row_index)
    if mask.any():
        res_df.loc[mask, ['F_res, Гц', 'Vp, мВ', 'Pout, мВт', 'ρ_E, мВт/см³']] = [f_res, vp_max, pout_max, rho_max]
    else:
        res_df.loc[len(res_df)] = [row_index, f_res, vp_max, pout_max, rho_max]

    return vp_df, pout_df, rho_df, res_df

# =========================
# Worker по одной частоте для данного d
# =========================
def _process_one_freq_for_d(args):
    d_mm, freq_hz = args
    try:
        device = ElectromagneticHarvesterID80mm()
        # Электрика
        device.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)

        # Геометрия под d
        reconfigure_geometry_for_d(device, d_mm)

        # Шейкер: частота и амплитуда
        device.set_frequency(float(freq_hz))
        device.base_amp_x0_m = get_x0_for_freq(float(freq_hz))

        # Временная сетка (модель сама генерит a_base по этим параметрам)
        t_eval = np.linspace(0.0, T_SLICE, int(T_SLICE * BASE_FS_HZ))

        # Решение модели
        t_s, z, v, i, emf_open_v, emf_self_v, forces, v_term_v = device.solve_all(
            t_eval_s=t_eval,
            rtol=1e-5, atol=1e-8,
            dense_output=False
        )

        # Метрики
        metrics = calculate_metrics(v_term_v, i, t_s)
        vp_mV   = metrics["Vp_model_mV"]
        pout_mW = metrics["Pout_model_mW"]

        # Энергетическая плотность
        volume_cm3 = calculate_volume(device, d_mm)
        rho_e = pout_mW / volume_cm3 if volume_cm3 > 0 else np.nan

        return fkey(freq_hz), float(vp_mV), float(pout_mW), float(rho_e)

    except Exception as e:
        print(f"[Worker d={d_mm} мм] Ошибка на f={freq_hz} Гц: {e}")
        return fkey(freq_hz), np.nan, np.nan, np.nan

# =========================
# Основной сценарий
# =========================
if __name__ == "__main__":
    if EXISTING_OUT_DIR and os.path.isdir(EXISTING_OUT_DIR):
        out_dir = EXISTING_OUT_DIR
        ts = os.path.basename(out_dir)
        print(f"Продолжаем в существующей папке эксперимента: {out_dir}")
    else:
        out_dir, ts = create_experiment_folder(BASE_DIR)
        print(f"Создана новая папка эксперимента: {out_dir}")

    # Загружаем/инициализируем таблицы
    vp_df, pout_df, rho_df, res_df = load_or_init_tables(out_dir, FREQS)

    num_proc = min(mp.cpu_count(), max(1, len(FREQS)))
    # num_proc = 26
    print(f"Используем {num_proc} процессов")

    with mp.Pool(processes=num_proc) as pool:
        for d_mm in D_VALUES_MM:
            d_mm = float(d_mm)
            already_done = (not vp_df.empty) and (d_mm in vp_df.index)

            if already_done and SKIP_DONE and not OVERWRITE:
                print(f"\n=== d = {d_mm:.0f} мм === (уже есть, пропускаю)")
                continue

            print(f"\n=== d = {d_mm:.0f} мм ===")

            # Печать параметров для контроля
            device_ref = ElectromagneticHarvesterID80mm()
            device_ref.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)
            reconfigure_geometry_for_d(device_ref, d_mm)
            print_parameters_table(device_ref, d_mm)

            # Параллельно по частотам
            args = [(float(d_mm), float(f)) for f in FREQS]
            results = pool.map(_process_one_freq_for_d, args)

            # Пустые строки под текущий d
            vp_freq    = np.full(len(FREQS), np.nan, dtype=float)
            pout_freq  = np.full(len(FREQS), np.nan, dtype=float)
            rho_e_freq = np.full(len(FREQS), np.nan, dtype=float)

            # Индексация по «нормализованной» частоте
            freq_to_idx = {fkey(f): idx for idx, f in enumerate(FREQS)}
            for freq_hz, vp_mV, pout_mW, rho_e in results:
                j = freq_to_idx.get(fkey(freq_hz))
                if j is None:
                    j = int(np.argmin(np.abs(FREQS - float(freq_hz))))
                vp_freq[j]    = vp_mV
                pout_freq[j]  = pout_mW
                rho_e_freq[j] = rho_e

            # === Инкрементально обновляем CSV после каждого d ===
            vp_df, pout_df, rho_df, res_df = upsert_row_for_d(
                d_mm, FREQS, vp_freq, pout_freq, rho_e_freq, vp_df, pout_df, rho_df, res_df
            )
            save_tables(out_dir, vp_df, pout_df, rho_df, res_df)
            print(f"[✔] Сохранено для d={d_mm:.0f} мм -> vp_values.csv, pout_values.csv, rho_e_values.csv, resonance_params.csv")

    # После цикла можно построить итоговые графики на основе CSV
    # и сохранить 3D-массивы (опционально)
    try:
        # Восстановим массивы из актуальных CSV
        d_values = np.array(sorted(vp_df.index), dtype=float)
        vp_values = vp_df.sort_index().to_numpy(dtype=float)
        pout_values = pout_df.sort_index().to_numpy(dtype=float)
        rho_e_values = rho_df.sort_index().to_numpy(dtype=float)

        # Резонансные частоты (по Vp) — из сводной таблицы
        res_sorted = res_df.sort_values(by='d, мм')
        f_res_values = res_sorted['F_res, Гц'].to_numpy(dtype=float)

        # Сохраним npy (удобно для последующей визуализации)
        np.save(os.path.join(out_dir, "d_values.npy"), d_values)
        np.save(os.path.join(out_dir, "freqs.npy"), FREQS)
        np.save(os.path.join(out_dir, "vp_values.npy"), vp_values)
        np.save(os.path.join(out_dir, "pout_values.npy"), pout_values)
        np.save(os.path.join(out_dir, "rho_e_values.npy"), rho_e_values)

        # Небольшая сводка
        rho_e_max_per_d = np.nanmax(rho_e_values, axis=1) if rho_e_values.size else np.array([])
        if rho_e_max_per_d.size and np.any(np.isfinite(rho_e_max_per_d)):
            optimal_idx = int(np.nanargmax(rho_e_max_per_d))
            optimal_d = d_values[optimal_idx]
            optimal_rho = rho_e_max_per_d[optimal_idx]
            print(f"\nОптимальное d = {optimal_d:.0f} мм, ρ_E = {optimal_rho:.3f} мВт/см³")
        else:
            print("\nНе удалось определить оптимальное d (все ρ_E — NaN).")

        print("\n==== Сводка ====")
        print(f"Папка эксперимента: {out_dir}")
        print("Файлы CSV: vp_values.csv, pout_values.csv, rho_e_values.csv, resonance_params.csv")
        print("Файлы NPY: d_values.npy, freqs.npy, vp_values.npy, pout_values.npy, rho_e_values.npy")
    except Exception as e:
        print(f"[WARN] Итоговая сводка/NPY не сохранены: {e}")
