# -*- coding: utf-8 -*-
"""
Фиксируем d = D_VALUE и считаем сетку по (z0, v0) с инкрементальным сохранением:

Сохраняются карты (строки=z0 [м], столбцы=v0 [м/с]):

    vpmax_map.csv     # Vp_max (мВ) по частотам для каждой (z0, v0)
    poutmax_map.csv   # Pout_max (мВт)
    rhoemax_map.csv   # ρ_E_max (мВт/см^3)
    fres_map.csv      # F_res (Гц) — частота максимума Vp

ВАЖНО:
- Перед каждым запуском solve_all() выставляем:
    device.z0_m   = z0
    device.v0_mps = v0
- Параллелим по частотам через multiprocessing.Pool (как у тебя было).
- Сохранение делаем после каждого z0 (чтобы можно было продолжать).
"""

import os
import math
from datetime import datetime
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import multiprocessing as mp

# 👉️ ВАЖНО: импортируем твою модель
from model import ElectromagneticHarvesterID80mm

# =========================
# Конфигурация эксперимента
# =========================
BASE_DIR = "D:/PROJECTs/magnet/harvester/delta_z0_v0"  # поменяй на Linux-путь, если запускаешь на Ubuntu
EXISTING_OUT_DIR: Optional[str] = None  # укажи путь к папке эксперимента, чтобы продолжить; иначе None

OVERWRITE = False  # True — пересчитывать ячейки даже если они уже заполнены
SKIP_DONE = True   # True — пропускать уже посчитанные (z0, v0), если OVERWRITE=False

T_SLICE    = 5.0       # длительность моделирования (с)
BASE_FS_HZ = 1000.0    # дискретизация времени (Гц)

# Частоты 2..30 Гц с шагом 0.1 Гц
FREQS = np.arange(2.0, 30.0, 0.1)
FREQS = np.round(np.asarray(FREQS, float), 6)  # стабильная ось

# Фиксированный зазор d (мм)
D_VALUE = 80.0

# Электрические параметры
COIL_R_OHM = 0.001  # Ом
LOAD_R_OHM = 1.0    # Ом

# Сетка начальных условий
Z0_VALUES_M   = np.arange(0.022, D_VALUE, 0.001)    # -10..10 мм шаг 0.5 мм
V0_VALUES_MPS = np.arange(0, 0.5 + 1e-12, 0.02)     # -0.5..0.5 м/с шаг 0.02


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
    res["Vrms_model_mV"] = float(np.sqrt(np.mean(v_term_v ** 2)) * 1000.0)
    p_inst = v_term_v * i_model
    res["Pout_model_mW"] = float(np.mean(p_inst) * 1000.0) if np.any(np.isfinite(p_inst)) else float("nan")
    if len(t_s) > 1:
        dt = float(np.mean(np.diff(t_s)))
        T = float(t_s[-1] - t_s[0])
        res["Pavg_model_mW"] = float((1.0 / T) * np.sum(p_inst) * dt * 1000.0)
    else:
        res["Pavg_model_mW"] = float("nan")
    return res


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
    device.top_magnet_z_top_m = device.top_magnet_z_bottom_m + device.magnet_height_m
    device.bottom_magnet_z_top_m = device.coil_z_bottom_m - 0.002
    device.bottom_magnet_z_bottom_m = device.bottom_magnet_z_top_m - device.magnet_height_m
    device.top_magnet_center_m = 0.5 * (device.top_magnet_z_top_m + device.top_magnet_z_bottom_m)
    device.bottom_magnet_center_m = 0.5 * (device.bottom_magnet_z_top_m + device.bottom_magnet_z_bottom_m)

    device._precompute_coil_geometry()


# =========================
# Карты z0 x v0 (CSV)
# =========================
def _grid_csv_paths(out_dir: str) -> Tuple[str, str, str, str]:
    return (
        os.path.join(out_dir, "vpmax_map.csv"),
        os.path.join(out_dir, "poutmax_map.csv"),
        os.path.join(out_dir, "rhoemax_map.csv"),
        os.path.join(out_dir, "fres_map.csv"),
    )


def _init_grid_df(index_vals: np.ndarray, col_vals: np.ndarray) -> pd.DataFrame:
    idx = [float(x) for x in index_vals]
    cols = [float(x) for x in col_vals]
    return pd.DataFrame(index=idx, columns=cols, dtype=float)


def load_or_init_grids(out_dir: str, z0_vals: np.ndarray, v0_vals: np.ndarray):
    vp_csv, pout_csv, rho_csv, fres_csv = _grid_csv_paths(out_dir)

    def load_or_empty(path: str) -> pd.DataFrame:
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            df.index = [float(i) for i in df.index]
            df.columns = [float(c) for c in df.columns]
            return df
        return _init_grid_df(z0_vals, v0_vals)

    vp_df = load_or_empty(vp_csv)
    pout_df = load_or_empty(pout_csv)
    rho_df = load_or_empty(rho_csv)
    fres_df = load_or_empty(fres_csv)

    # гарантируем наличие всех строк/столбцов, если сетка изменилась
    target_idx = [float(x) for x in z0_vals]
    target_cols = [float(x) for x in v0_vals]

    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        # add missing rows
        for z in target_idx:
            if z not in df.index:
                df.loc[z, :] = np.nan
        # add missing cols
        for v in target_cols:
            if v not in df.columns:
                df[v] = np.nan

        df.sort_index(inplace=True)
        df = df[target_cols]  # порядок колонок
        return df

    vp_df = normalize(vp_df)
    pout_df = normalize(pout_df)
    rho_df = normalize(rho_df)
    fres_df = normalize(fres_df)

    return vp_df, pout_df, rho_df, fres_df


def save_grids(out_dir: str, vp_df: pd.DataFrame, pout_df: pd.DataFrame, rho_df: pd.DataFrame, fres_df: pd.DataFrame):
    vp_csv, pout_csv, rho_csv, fres_csv = _grid_csv_paths(out_dir)
    vp_df.sort_index().to_csv(vp_csv)
    pout_df.sort_index().to_csv(pout_csv)
    rho_df.sort_index().to_csv(rho_csv)
    fres_df.sort_index().to_csv(fres_csv)


# =========================
# Worker: одна частота для заданных (d, z0, v0)
# =========================
def _process_one_freq_for_state(args):
    d_mm, z0_m, v0_mps, freq_hz = args
    try:
        device = ElectromagneticHarvesterID80mm()

        # Электрика
        device.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)

        # Геометрия под фиксированный d
        reconfigure_geometry_for_d(device, float(d_mm))

        # ✅ КЛЮЧЕВОЕ: начальные условия ПЕРЕД запуском solve_all()
        device.z0_m = float(z0_m)
        device.v0_mps = float(v0_mps)

        # Шейкер: частота и амплитуда
        device.set_frequency(float(freq_hz))
        device.base_amp_x0_m = get_x0_for_freq(float(freq_hz))

        # Временная сетка
        n = int(T_SLICE * BASE_FS_HZ)
        if n < 2:
            n = 2
        t_eval = np.linspace(0.0, T_SLICE, n)

        # Решение модели
        t_s, z, v, i, emf_open_v, emf_self_v, forces, v_term_v = device.solve_all(
            t_eval_s=t_eval,
            rtol=1e-5, atol=1e-8,
            dense_output=False
        )

        # Метрики
        metrics = calculate_metrics(v_term_v, i, t_s)
        vp_mV = metrics["Vp_model_mV"]
        pout_mW = metrics["Pout_model_mW"]

        # Энергетическая плотность
        volume_cm3 = calculate_volume(device, float(d_mm))
        rho_e = pout_mW / volume_cm3 if volume_cm3 > 0 else np.nan

        return fkey(freq_hz), float(vp_mV), float(pout_mW), float(rho_e)

    except Exception as e:
        print(f"[Worker d={d_mm} z0={z0_m} v0={v0_mps}] Ошибка на f={freq_hz} Гц: {e}")
        return fkey(freq_hz), np.nan, np.nan, np.nan


# =========================
# Основной сценарий
# =========================
if __name__ == "__main__":
    # Папка эксперимента
    if EXISTING_OUT_DIR and os.path.isdir(EXISTING_OUT_DIR):
        out_dir = EXISTING_OUT_DIR
        ts = os.path.basename(out_dir)
        print(f"Продолжаем в существующей папке эксперимента: {out_dir}")
    else:
        out_dir, ts = create_experiment_folder(BASE_DIR)
        print(f"Создана новая папка эксперимента: {out_dir}")

    # Грузим/инициализируем карты
    vpmax_df, poutmax_df, rhoemax_df, fres_df = load_or_init_grids(out_dir, Z0_VALUES_M, V0_VALUES_MPS)

    # Параллель по частотам
    num_proc = min(mp.cpu_count(), max(1, len(FREQS)))
    print(f"Используем {num_proc} процессов для распараллеливания по частотам")

    d_mm = float(D_VALUE)
    print(f"\n=== Фиксируем d = {d_mm:.0f} мм. Считаем сетку по z0 и v0 ===")

    # Небольшая проверка геометрии (один раз)
    device_ref = ElectromagneticHarvesterID80mm()
    device_ref.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)
    reconfigure_geometry_for_d(device_ref, d_mm)
    print(f"[info] turns_N={device_ref.turns_N}, coil_height_m={device_ref.coil_height_m:.6f}")

    with mp.Pool(processes=num_proc) as pool:
        for z0_m in Z0_VALUES_M:
            z0_m = float(z0_m)
            print(f"\n=== z0 = {z0_m:.6f} m ===")

            for v0_mps in V0_VALUES_MPS:
                v0_mps = float(v0_mps)

                # пропуск уже посчитанных ячеек
                prev = vpmax_df.loc[z0_m, v0_mps]
                already_done = bool(np.isfinite(prev))
                if already_done and SKIP_DONE and not OVERWRITE:
                    continue

                print(f"  -> v0 = {v0_mps:.6f} m/s")

                # Параллельно по частотам
                args = [(d_mm, z0_m, v0_mps, float(f)) for f in FREQS]
                results = pool.map(_process_one_freq_for_state, args)

                # массивы по частоте
                vp_freq = np.full(len(FREQS), np.nan, dtype=float)
                pout_freq = np.full(len(FREQS), np.nan, dtype=float)
                rho_e_freq = np.full(len(FREQS), np.nan, dtype=float)

                freq_to_idx = {fkey(f): idx for idx, f in enumerate(FREQS)}
                for freq_hz, vp_mV, pout_mW, rho_e in results:
                    j = freq_to_idx.get(fkey(freq_hz))
                    if j is None:
                        j = int(np.argmin(np.abs(FREQS - float(freq_hz))))
                    vp_freq[j] = vp_mV
                    pout_freq[j] = pout_mW
                    rho_e_freq[j] = rho_e

                # сводные по частоте для (z0, v0)
                vp_max = float(np.nanmax(vp_freq)) if np.any(np.isfinite(vp_freq)) else np.nan
                pout_max = float(np.nanmax(pout_freq)) if np.any(np.isfinite(pout_freq)) else np.nan
                rho_max = float(np.nanmax(rho_e_freq)) if np.any(np.isfinite(rho_e_freq)) else np.nan
                f_res = float(FREQS[np.nanargmax(vp_freq)]) if np.any(np.isfinite(vp_freq)) else np.nan

                # записываем в карты
                vpmax_df.loc[z0_m, v0_mps] = vp_max
                poutmax_df.loc[z0_m, v0_mps] = pout_max
                rhoemax_df.loc[z0_m, v0_mps] = rho_max
                fres_df.loc[z0_m, v0_mps] = f_res

            # Инкрементально сохраняем после каждого z0
            save_grids(out_dir, vpmax_df, poutmax_df, rhoemax_df, fres_df)
            print(f"[✔] Сохранено для z0={z0_m:.6f}: vpmax_map.csv, poutmax_map.csv, rhoemax_map.csv, fres_map.csv")

    # Итоговая сводка
    try:
        # найдём глобальный максимум rho_e
        rho_vals = rhoemax_df.to_numpy(dtype=float)
        if rho_vals.size and np.any(np.isfinite(rho_vals)):
            idx_flat = int(np.nanargmax(rho_vals))
            iz, iv = np.unravel_index(idx_flat, rho_vals.shape)
            best_z0 = float(rhoemax_df.index[iz])
            best_v0 = float(rhoemax_df.columns[iv])
            best_rho = float(rho_vals[iz, iv])
            best_fres = float(fres_df.loc[best_z0, best_v0])
            print("\n==== ЛУЧШАЯ ТОЧКА (по ρ_E_max) ====")
            print(f"d = {d_mm:.0f} мм")
            print(f"z0 = {best_z0:.6f} m")
            print(f"v0 = {best_v0:.6f} m/s")
            print(f"ρ_E_max = {best_rho:.6f} мВт/см³")
            print(f"F_res = {best_fres:.3f} Гц")
        else:
            print("\n[WARN] Не удалось найти максимум: все значения ρ_E — NaN")

        print("\n==== ГОТОВО ====")
        print(f"Папка эксперимента: {out_dir}")
        print("Файлы: vpmax_map.csv, poutmax_map.csv, rhoemax_map.csv, fres_map.csv")
    except Exception as e:
        print(f"[WARN] Итоговая сводка не выполнена: {e}")