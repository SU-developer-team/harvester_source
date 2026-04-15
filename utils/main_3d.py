# -*- coding: utf-8 -*-
"""
Сценарий для анализа влияния межмагнитного расстояния d на резонансные параметры.
- Шейкер синтезируется по формуле: a(t) = -W^2 * X0 * cos(W * t), W = 2*pi*f, X0 — из таблицы амплитуд.
- Для совместимости с ElectromagneticHarvesterID50 база ускорений загружается через load_base_from_csv из временного CSV.
- Параллельная обработка частот для каждого d (multiprocessing.Pool).
Сохраняет 3D-массивы в .npy и CSV файлы.
Выводит таблицу резонансных параметров в формате: d, мм | F_res, Гц | Vp, мВ | Pout, мВт | ρ_E, мВт/см³.
Добавлен расчёт энергетической плотности ρ_E и 3D-график ρ_E(f, d).
"""

import os
import csv
import math
import tempfile
from datetime import datetime
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (нужно для 3D)
import multiprocessing as mp

from ElectromagneticHarvester50 import ElectromagneticHarvesterID50

# =========================
# Конфигурация эксперимента
# =========================
BASE_DIR = "experiments/harvester_50mm/exp_2"

T_MAX_S = 20.0             # максимум длительности моделирования (информативно)
T_SLICE = 20.0             # длительность моделирования (сек), используем последние 20 с
BASE_FS_HZ = 1000.0        # частота дискретизации времени для синтетического шейкера

FREQS = np.arange(10, 26, 1)  # частоты 10..25 Гц
D_VALUES_MM = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
               42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72,
               74, 76, 78, 80]  # межмагнитные расстояния, мм

# Электрические параметры
COIL_R_OHM = 0.001         # сопротивление катушки, Ом
LOAD_R_OHM = 1.0           # сопротивление нагрузки, Ом

# Амплитуды X0 (м) для частот 10..25 Гц (пример — адаптируйте под свои данные)
# Индексация: X0_for_freq[f] => амплитуда для частоты f Гц
# X0_FOR_FREQ = {
#     10: 0.018, 11: 0.019, 12: 0.022, 13: 0.0225, 14: 0.0235, 15: 0.024,
#     16: 0.0254, 17: 0.025, 18: 0.0245, 19: 0.0195, 20: 0.0125, 21: 0.009,
#     22: 0.007, 23: 0.0055, 24: 0.0047, 25: 0.0042  # 25 — примерная добавка
# }
X0_FOR_FREQ = {
    2: 0.001, 3: 0.006, 4: 0.006, 5: 0.006, 6: 0.006, 7: 0.006, 8: 0.006,
    9: 0.008, 10: 0.013, 11: 0.013, 12: 0.0145, 13: 0.0155, 14: 0.0177,
    15: 0.0195, 16: 0.0215, 17: 0.0235, 18: 0.024, 19: 0.0175, 20: 0.0092,
    21: 0.0067, 22: 0.0052, 23: 0.0042, 24: 0.0036, 25: 0.0031, 26: 0.0027,
    27: 0.00235, 28: 0.002, 29: 0.0019, 30: 0.0018, 31: 0.0018, 32: 0.0018,
    33: 0.00183, 34: 0.0017, 35: 0.0015
}
# =========================
# Хелперы
# =========================
def create_experiment_folder(base_dir: str) -> Tuple[str, str]:
    """Создать папку для графиков/файлов с таймстампом."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(base_dir, "graphs", f"experiment_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, ts

def calculate_volume(device, d_mm: float) -> float:
    """Рассчитывает рабочий объём устройства в см³ на основе межмагнитного расстояния d."""
    d_m = d_mm / 1000.0  # Перевод в метры
    coil_outer_radius_m = device.coil_diam_m / 2
    volume_m3 = np.pi * coil_outer_radius_m**2 * d_m
    volume_cm3 = volume_m3 * 1e6  # Перевод из м³ в см³
    return volume_cm3

def calculate_metrics(
    v_term_v: np.ndarray,
    i_model: np.ndarray,
    t_s: np.ndarray
) -> dict:
    """
    Вычисляет ключевые метрики.
    P(t) = U(t) * I(t), P_out = mean(P(t)).
    """
    results = {}
    vp_model = np.max(np.abs(v_term_v))
    results["Vp_model_mV"] = vp_model * 1000.0

    vrms_model = np.sqrt(np.mean(v_term_v**2))
    results["Vrms_model_mV"] = vrms_model * 1000.0

    p_model = v_term_v * i_model
    pout_model = np.mean(p_model) * 1000.0 if np.any(np.isfinite(p_model)) else np.nan
    results["Pout_model_mW"] = pout_model

    if len(t_s) > 1:
        dt = np.mean(np.diff(t_s))
        T = t_s[-1] - t_s[0]
        p_model_int = (1.0 / T) * np.sum(v_term_v * i_model) * dt * 1000.0
    else:
        p_model_int = np.nan
    results["Pavg_model_mW"] = p_model_int

    return results

def print_parameters_table(device, d_mm: float):
    """Выводит таблицу основных параметров и коэффициентов."""
    print(f"\n=== Параметры для d = {d_mm} мм ===")
    params = {
        "Масса магнита (m_kg)": device.m_kg,
        "Гравитация (g_mps2)": device.g_mps2,
        "Магнитная постоянная (mu0_Hpm)": device.mu0_Hpm,
        "Характерная индукция (B_T)": device.B_T,
        "Коэффициент ускорения (k_accel)": device.k_accel,
        "Коэффициент ЭДС (k_emf)": device.k_emf,
        "Радиус магнита (magnet_radius_m)": device.magnet_radius_m,
        "Высота магнита (magnet_height_m)": device.magnet_height_m,
        "Высота катушки (coil_height_m)": device.coil_height_m,
        "Внутренний диаметр катушки (coil_inner_diam_m)": device.coil_inner_diam_m,
        "Внешний диаметр катушки (coil_diam_m)": device.coil_diam_m,
        "Диаметр провода (wire_diam_m)": device.wire_diam_m,
        "Количество витков (turns_N)": device.turns_N,
        "Сопротивление катушки (coil_resistance_ohm)": device.coil_resistance_ohm,
        "Сопротивление нагрузки (load_resistance_ohm)": device.load_resistance_ohm,
        "Нижняя граница катушки (coil_z_bottom_m)": device.coil_z_bottom_m,
        "Верхняя граница катушки (coil_z_top_m)": device.coil_z_top_m,
        "Центр верхнего магнита (top_magnet_center_m)": device.top_magnet_center_m,
        "Центр нижнего магнита (bottom_magnet_center_m)": device.bottom_magnet_center_m,
        "Начальное положение (z0_m)": device.z0_m,
        "Начальная скорость (v0_mps)": device.v0_mps,
        "Коэффициент демпфирования (c_damping)": device.c_damping,
        "Максимальная сила трения (F_c)": device.F_c,
        "Коэффициент чувствительности трения (beta)": device.beta,
        "Межмагнитное расстояние (d_mm)": d_mm,
    }
    df_params = pd.DataFrame(list(params.items()), columns=["Параметр", "Значение"])
    print(df_params.to_string(index=False))

def save_3d_arrays(save_dir: str, timestamp: str, d_values: np.ndarray, freqs: np.ndarray, vp: np.ndarray, pout: np.ndarray, rho_e: np.ndarray) -> List[str]:
    """Сохраняет 3D-массивы в .npy и CSV."""
    output_paths = []

    np.save(os.path.join(save_dir, f"d_values_{timestamp}.npy"), d_values)
    output_paths.append(os.path.join(save_dir, f"d_values_{timestamp}.npy"))

    np.save(os.path.join(save_dir, f"freqs_{timestamp}.npy"), freqs)
    output_paths.append(os.path.join(save_dir, f"freqs_{timestamp}.npy"))

    np.save(os.path.join(save_dir, f"vp_values_{timestamp}.npy"), vp)
    output_paths.append(os.path.join(save_dir, f"vp_values_{timestamp}.npy"))

    np.save(os.path.join(save_dir, f"pout_values_{timestamp}.npy"), pout)
    output_paths.append(os.path.join(save_dir, f"pout_values_{timestamp}.npy"))

    np.save(os.path.join(save_dir, f"rho_e_values_{timestamp}.npy"), rho_e)
    output_paths.append(os.path.join(save_dir, f"rho_e_values_{timestamp}.npy"))

    # CSV матрицы (строки — d, столбцы — f)
    vp_df = pd.DataFrame(vp, index=d_values, columns=freqs)
    vp_df.to_csv(os.path.join(save_dir, f"vp_values_{timestamp}.csv"))
    output_paths.append(os.path.join(save_dir, f"vp_values_{timestamp}.csv"))

    pout_df = pd.DataFrame(pout, index=d_values, columns=freqs)
    pout_df.to_csv(os.path.join(save_dir, f"pout_values_{timestamp}.csv"))
    output_paths.append(os.path.join(save_dir, f"pout_values_{timestamp}.csv"))

    rho_e_df = pd.DataFrame(rho_e, index=d_values, columns=freqs)
    rho_e_df.to_csv(os.path.join(save_dir, f"rho_e_values_{timestamp}.csv"))
    output_paths.append(os.path.join(save_dir, f"rho_e_values_{timestamp}.csv"))

    return output_paths

def plot_results(d_values: np.ndarray, f_res: np.ndarray, vp: np.ndarray, pout: np.ndarray, rho_e: np.ndarray, save_dir: str, timestamp: str) -> List[str]:
    """Строит 3D-графики Vp и ρ_E от d и частоты, а также 2D-графики Vp, Pout и ρ_E."""
    output_paths = []

    D, F = np.meshgrid(d_values, FREQS)

    # 3D Vp(d,f)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(D, F, vp.T, cmap='viridis')
    ax.set_xlabel("Межмагнитное расстояние, мм")
    ax.set_ylabel("Частота, Гц")
    ax.set_zlabel("Vp, мВ")
    ax.set_title("Амплитуда ЭДС vs Межмагнитное расстояние и Частота")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    vp_3d_plot = os.path.join(save_dir, f"vp_3d_{timestamp}.png")
    plt.savefig(vp_3d_plot)
    plt.close()
    output_paths.append(vp_3d_plot)

    # 3D rho_E(d,f)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(D, F, rho_e.T, cmap='plasma')
    ax.set_xlabel("Межмагнитное расстояние, мм")
    ax.set_ylabel("Частота, Гц")
    ax.set_zlabel("ρ_E, мВт/см³")
    ax.set_title("Энергетическая плотность vs Межмагнитное расстояние и Частота")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    rho_e_3d_plot = os.path.join(save_dir, f"rho_e_3d_{timestamp}.png")
    plt.savefig(rho_e_3d_plot)
    plt.close()
    output_paths.append(rho_e_3d_plot)

    # 2D: Vp(f) для каждого d
    plt.figure(figsize=(11, 6))
    for i, d in enumerate(d_values):
        vp_freqs = vp[i, :]
        plt.plot(FREQS, vp_freqs, marker='o', label=f"d={d} мм")
        if np.any(np.isfinite(vp_freqs)):
            j = np.nanargmax(vp_freqs)
            plt.plot(FREQS[j], vp_freqs[j], 'x', markersize=10, label=None)
    plt.xlabel("Частота, Гц")
    plt.ylabel("Vp, мВ")
    plt.title("Амплитуда ЭДС vs Частота (для разных d)")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    vp_freq_plot = os.path.join(save_dir, f"vp_vs_freq_{timestamp}.png")
    plt.savefig(vp_freq_plot)
    plt.close()
    output_paths.append(vp_freq_plot)

    # 2D: Pout(d) — максимум по частоте для каждого d
    plt.figure(figsize=(11, 6))
    pout_max = np.nanmax(pout, axis=1)
    plt.plot(d_values, pout_max, marker='o', label="Выходная мощность (мВт)")
    plt.xlabel("Межмагнитное расстояние, мм")
    plt.ylabel("Pout, мВт")
    plt.title("Выходная мощность vs Межмагнитное расстояние")
    plt.grid(True)
    plt.legend()
    pout_plot = os.path.join(save_dir, f"pout_vs_d_{timestamp}.png")
    plt.savefig(pout_plot)
    plt.close()
    output_paths.append(pout_plot)

    # 2D: rho_E(d) — максимум по частоте для каждого d
    plt.figure(figsize=(11, 6))
    rho_e_max = np.nanmax(rho_e, axis=1)
    plt.plot(d_values, rho_e_max, marker='o', label="Энергетическая плотность (мВт/см³)")
    plt.xlabel("Межмагнитное расстояние, мм")
    plt.ylabel("ρ_E, мВт/см³")
    plt.title("Энергетическая плотность vs Межмагнитное расстояние")
    plt.grid(True)
    plt.legend()
    rho_e_plot = os.path.join(save_dir, f"rho_e_vs_d_{timestamp}.png")
    plt.savefig(rho_e_plot)
    plt.close()
    output_paths.append(rho_e_plot)

    return output_paths

def save_results_csv(save_dir: str, timestamp: str, d_values: np.ndarray, f_res: np.ndarray, vp: np.ndarray, pout: np.ndarray, rho_e: np.ndarray) -> str:
    """Сохраняет сводный CSV: d, мм | F_res, Гц | Vp, мВ | Pout, мВт | ρ_E, мВт/см³."""
    # Контроль форм
    if (vp.shape != (len(d_values), len(FREQS)) or
        pout.shape != (len(d_values), len(FREQS)) or
        rho_e.shape != (len(d_values), len(FREQS))):
        raise ValueError(f"Неправильная форма массивов: vp={vp.shape}, pout={pout.shape}, rho_e={rho_e.shape}")

    vp_max = np.nanmax(vp, axis=1)
    pout_max = np.nanmax(pout, axis=1)
    rho_e_max = np.nanmax(rho_e, axis=1)

    out = os.path.join(save_dir, f"resonance_params_{timestamp}.csv")
    df = pd.DataFrame({
        'd, мм': d_values,
        'F_res, Гц': f_res,
        'Vp, мВ': vp_max,
        'Pout, мВт': pout_max,
        'ρ_E, мВт/см³': rho_e_max
    })
    df.to_csv(out, index=False, encoding='utf-8')
    return out

# =========================
# Синтетический шейкер
# =========================
def get_x0_for_freq(freq_hz: int) -> float:
    if freq_hz in X0_FOR_FREQ:
        return float(X0_FOR_FREQ[freq_hz])
    raise ValueError(f"Нет X0 для частоты {freq_hz} Гц")

def synthesize_base_accel(freq_hz: float, t_s: np.ndarray) -> np.ndarray:
    """
    a(t) = -W^2 * X0 * cos(W t),   W = 2*pi*f,  X0 — амплитуда (м)
    """
    X0 = get_x0_for_freq(int(round(freq_hz)))
    W = 2.0 * math.pi * float(freq_hz)
    return - (W ** 2) * X0 * np.cos(W * t_s)

def write_temp_shaker_csv(t_s: np.ndarray, a_s: np.ndarray) -> str:
    """
    Временный CSV с колонками t;acc_mps2 (десятичная точка).
    Загружается далее через load_base_from_csv().
    """
    fd, path = tempfile.mkstemp(prefix="shaker_", suffix=".csv")
    os.close(fd)
    df = pd.DataFrame({"t": t_s, "acc_mps2": a_s})
    df.to_csv(path, sep=";", index=False, header=True, float_format="%.9f")
    return path

# =========================
# Параллельная обработка ЧАСТОТ для одного d
# =========================
def _process_one_freq_for_d(args) -> Tuple[int, float, float, float]:
    """
    Worker: обрабатывает одну частоту для фиксированного d.
    Возвращает (freq_hz, Vp_mV, Pout_mW, rho_E_mW_per_cm3)
    """
    d_mm, freq_hz = args
    try:
        # Устройство и электрические параметры
        device = ElectromagneticHarvesterID50()
        device.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)

        # Геометрия/коил под заданный d
        d_m = d_mm / 1000.0
        device.coil_height_m = d_m
        device.turns_N = int(device.coil_height_m / device.wire_diam_m)  # пример автоподсчёта
        device.coil_z_top_m = device.coil_z_bottom_m + device.coil_height_m
        device._precompute_coil_geometry()

        # Объём (для rho_E)
        volume_cm3 = calculate_volume(device, d_mm)

        # Сетка по времени
        t_eval = np.linspace(0.0, T_SLICE, int(T_SLICE * BASE_FS_HZ))

        # Синтетический шейкер
        a_base = synthesize_base_accel(freq_hz=freq_hz, t_s=t_eval)
        tmp_csv = write_temp_shaker_csv(t_eval, a_base)
        try:
            device.load_base_from_csv(
                tmp_csv,
                time_col="t",
                accel_col="acc_mps2",
                sep=";",
                decimal=".",     # мы писали с точкой
                normalize_time=True
            )
        finally:
            try:
                os.remove(tmp_csv)
            except OSError:
                pass

        # Решение модели
        # NB: сигнатура solve_all в вашей версии возвращает 8 значений — используем их.
        t_s, z, v, i, emf_open_v, emf_self_v, forces, v_term_v = device.solve_all(
            t_eval_s=t_eval,
            rtol=1e-5, atol=1e-8,
            clamp_to_base=True
        )

        # Метрики
        metrics = calculate_metrics(v_term_v, i, t_s)
        vp_mV = metrics["Vp_model_mV"]
        pout_mW = metrics["Pout_model_mW"]
        rho_e = pout_mW / volume_cm3 if volume_cm3 > 0 else np.nan

        return int(freq_hz), float(vp_mV), float(pout_mW), float(rho_e)

    except Exception as e:
        print(f"[Worker d={d_mm} мм] Ошибка на частоте {freq_hz} Гц: {e}")
        return int(freq_hz), np.nan, np.nan, np.nan

# =========================
# Основной сценарий
# =========================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir, ts = create_experiment_folder(base_dir)

    # Результаты (по d)
    f_res_values = []
    vp_values = []
    pout_values = []
    rho_e_values = []

    # Пул процессов (один раз, переиспользуем для всех d)
    num_proc = min(mp.cpu_count(), max(1, len(FREQS)))
    print(f"Используем {num_proc} процессов для параллельной обработки частот")
    with mp.Pool(processes=num_proc) as pool:
        for d_mm in D_VALUES_MM:
            print(f"\n=== Обработка межмагнитного расстояния d = {d_mm} мм ===")

            # Для контроля — создадим "референсное" устройство и выведем параметры
            device_ref = ElectromagneticHarvesterID50()
            device_ref.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)
            d_m = d_mm / 1000.0
            device_ref.coil_height_m = d_m
            device_ref.turns_N = int(device_ref.coil_height_m / device_ref.wire_diam_m)
            device_ref.coil_z_top_m = device_ref.coil_z_bottom_m + device_ref.coil_height_m
            device_ref._precompute_coil_geometry()
            print_parameters_table(device_ref, d_mm)

            # Параллельно считаем по всем частотам для текущего d
            args = [(d_mm, int(freq)) for freq in FREQS]
            results = pool.map(_process_one_freq_for_d, args)

            # Разворачиваем в массивы по FREQS (в правильном порядке)
            vp_freq = np.full(shape=len(FREQS), fill_value=np.nan, dtype=float)
            pout_freq = np.full(shape=len(FREQS), fill_value=np.nan, dtype=float)
            rho_e_freq = np.full(shape=len(FREQS), fill_value=np.nan, dtype=float)

            freq_to_idx = {int(f): idx for idx, f in enumerate(FREQS)}
            for freq_hz, vp_mV, pout_mW, rho_e in results:
                if freq_hz in freq_to_idx:
                    j = freq_to_idx[freq_hz]
                    vp_freq[j] = vp_mV
                    pout_freq[j] = pout_mW
                    rho_e_freq[j] = rho_e

            # Резонанс по максимуму Vp
            if np.any(np.isfinite(vp_freq)):
                j_max = np.nanargmax(vp_freq)
                f_res = FREQS[j_max]
            else:
                f_res = np.nan

            f_res_values.append(f_res)
            vp_values.append(vp_freq)
            pout_values.append(pout_freq)
            rho_e_values.append(rho_e_freq)

    # В массивы
    d_values = np.array(D_VALUES_MM, dtype=float)
    f_res_values = np.array(f_res_values, dtype=float)
    vp_values = np.array(vp_values, dtype=float)
    pout_values = np.array(pout_values, dtype=float)
    rho_e_values = np.array(rho_e_values, dtype=float)

    # Контроль форм
    print("\n[DEBUG] Форма массивов перед сохранением:")
    print(f"d_values shape: {d_values.shape}")
    print(f"f_res_values shape: {f_res_values.shape}")
    print(f"vp_values shape: {vp_values.shape}")
    print(f"pout_values shape: {pout_values.shape}")
    print(f"rho_e_values shape: {rho_e_values.shape}")

    # Сохранение 3D-массивов
    array_paths = save_3d_arrays(out_dir, ts, d_values, FREQS, vp_values, pout_values, rho_e_values)

    # Графики
    plot_paths = plot_results(d_values, f_res_values, vp_values, pout_values, rho_e_values, out_dir, ts)

    # Сводный CSV
    csv_path = save_results_csv(out_dir, ts, d_values, f_res_values, vp_values, pout_values, rho_e_values)

    # Оптимальное d по максимальной ρ_E
    rho_e_max_per_d = np.nanmax(rho_e_values, axis=1)
    optimal_idx = int(np.nanargmax(rho_e_max_per_d)) if np.any(np.isfinite(rho_e_max_per_d)) else None
    if optimal_idx is not None:
        optimal_d = d_values[optimal_idx]
        optimal_rho_e = rho_e_max_per_d[optimal_idx]
        print(f"\nОптимальное d = {optimal_d:.0f} мм с ρ_E = {optimal_rho_e:.3f} мВт/см³")
    else:
        print("\nНе удалось определить оптимальное d (все значения ρ_E — NaN).")

    # Сводка
    print("\n==== Сводка ====")
    print(f"Папка эксперимента: {out_dir}")
    print(f"Графики: {', '.join(plot_paths)}")
    print(f"Сохранённые массивы: {', '.join(array_paths)}")
    print(f"CSV с результатами: {csv_path}")

    # Таблица в консоль
    df_results = pd.DataFrame({
        'd, мм': d_values,
        'F_res, Гц': f_res_values,
        'Vp, мВ': np.nanmax(vp_values, axis=1),
        'Pout, мВт': np.nanmax(pout_values, axis=1),
        'ρ_E, мВт/см³': rho_e_max_per_d
    })
    print("\nТаблица резонансных параметров:")
    print(df_results.to_string(index=False))
