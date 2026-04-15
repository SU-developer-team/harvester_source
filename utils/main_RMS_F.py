
# -*- coding: utf-8 -*-
"""
Параллельный сценарий для анализа RMS напряжения и координаты z модели и эксперимента по частотам 2–34 Гц.
- Обрабатывает выборочные частоты параллельно с использованием multiprocessing.
- Строит график: частота (Гц) vs RMS (В) для основной ЭДС, полной ЭДС и ЭДС самоиндукции (Рисунок 6).
- Строит график: частота (Гц) vs RMS координаты z (м).
- Строит отдельный график: частота (Гц) vs RMS ЭДС самоиндукции (В).
- Строит график: частота (Гц) vs средняя выходная мощность (мВт) для основной и полной ЭДС (Рисунок 7).
- Формирует и сохраняет Таблицу 5 (верификация) и Таблицу 6 (влияние самоиндукции) в CSV.
- Сохраняет результаты RMS (включая z) в CSV и графики в PNG.
"""

import os
from datetime import datetime
from typing import Tuple, Optional
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from ElectromagneticHarvesterID50mm_ERZHANAT import ElectromagneticHarvesterAbsolute

# =========================
# Конфигурация эксперимента
# =========================
BASE_DIR = r"D:\PROJECTs\magnet\harvester\experiments\harvester_50mm\exp_2"
SHAKER_PATH_TEMPLATE = os.path.join(BASE_DIR, "synthesized_data", "{freq}.csv")
EXP_EMF_PATH_TEMPLATE = os.path.join(BASE_DIR, "data", "{freq}.csv")

T_MAX_S = 20.0              # максимум длительности моделирования
T_SLICE = 5                 # ограничение времени для шейкера и ЭДС (последние 5 секунд)

FREQS = [i for i in range(2, 50)]    # выборочные частоты для всех таблиц и графиков

# ЭДС параметры
EMF_COL_IDX = 3             # ЭДС в 4-м столбце (индексация с 0)
EMF_FS_HZ = 1000.0          # частота дискретизации, Гц
EMF_UNITS = "mV"            # единицы: "V", "mV", "uV"
EMF_SEP = ";"
EMF_DECIMAL = ","

# Электрические параметры
COIL_R_OHM = 0.001          # сопротивление катушки, Ом
LOAD_R_OHM = 1.0            # сопротивление нагрузки, Ом

amplitudes = [ 0.005 for _ in range(len(FREQS))]

# =========================
# Хелперы
# =========================
def create_experiment_folder(base_dir: str) -> Tuple[str, str]:
    """Создать папку для графиков/файлов с таймстампом."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(base_dir, "graphs", f"experiment_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, ts

def _to_numeric_series(raw, decimal_hint: Optional[str]) -> pd.Series:
    """Надёжное приведение строк с запятой/пробелами к float. Мусор -> NaN."""
    s = pd.Series(raw, copy=True).astype(str)
    s = s.str.replace('\u00A0', '', regex=False)   # неразрывный пробел
    s = s.str.replace(' ', '', regex=False)
    s = s.str.replace(',', '.', regex=False)       # унифицируем десятичный знак
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
    """
    Читает CSV без столбца времени, строит t[k] = k / fs_hz.
    Возвращает (t_s, emf_V, интерполяция emf(t)).
    """
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

    bad = ~np.isfinite(emf_v)
    if bad.any():
        s = pd.Series(emf_v, index=t_s)
        s = s.interpolate(method="linear", limit_direction="both")
        if s.isna().sum() > 0:
            raise ValueError("В ЭДС после интерполяции остались NaN. Проверьте файл.")
        emf_v = s.to_numpy()

    f = interp1d(t_s, emf_v, kind="linear", bounds_error=False, fill_value=(emf_v[0], emf_v[-1]))

    print(f"[EMF] {n} точек, fs={fs_hz} Гц, длит. ~{t_s[-1]:.3f} с, диапазон {np.nanmin(emf_v):.6g}..{np.nanmax(emf_v):.6g} В")
    return t_s, emf_v, f

def rms(x: np.ndarray, center: bool = True) -> float:
    """Вычисляет RMS сигнала, с центрированием или без."""
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        nan_count = np.isnan(x).sum()
        inf_count = np.isinf(x).sum()
        raise ValueError(f"Обнаружены нечисловые значения: {nan_count} NaN, {inf_count} inf")
    if center:
        x = x - np.mean(x)
    return float(np.sqrt(np.mean(x**2)))

def calculate_rms_error(rms_model: np.ndarray, rms_exp: np.ndarray) -> float:
    """Вычисляет среднюю процентную ошибку между моделью и экспериментом."""
    valid_mask = (np.isfinite(rms_model) & np.isfinite(rms_exp) & (rms_exp != 0))
    if valid_mask.any():
        rms_error_percent = 100 * np.abs(rms_model[valid_mask] - rms_exp[valid_mask]) / rms_exp[valid_mask]
        return float(np.mean(rms_error_percent))
    return float("nan")

def plot_rms_vs_frequency(freqs: np.ndarray, rms_model: np.ndarray, rms_total_emf: np.ndarray, rms_self_induction: np.ndarray, rms_z: np.ndarray, save_dir: str, timestamp: str) -> str:
    """Строит два графика рядом: RMS ЭДС vs частота и RMS z vs частота (Рисунок 6)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # График RMS ЭДС
    ax1.plot(freqs, rms_model, label="Основная ЭДС: RMS (В)", marker='o')
    ax1.plot(freqs, rms_total_emf, label="Полная ЭДС (основная + самоиндукция): RMS (В)", marker='^', alpha=0.7)
    ax1.plot(freqs, rms_self_induction, label="ЭДС самоиндукции: RMS (В)", marker='s', alpha=0.85)
    ax1.set_xlabel("Частота, Гц")
    ax1.set_ylabel("RMS, В")
    ax1.set_title("RMS ЭДС: основная vs полная vs самоиндукция")
    ax1.grid(True)
    ax1.legend()

    # График RMS z
    ax2.plot(freqs, rms_z * 1000, label="RMS z (мм)", marker='x', color='purple')
    ax2.set_xlabel("Частота, Гц")
    ax2.set_ylabel("RMS z, мм")
    ax2.set_title("RMS координаты z центрального магнита")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    out = os.path.join(save_dir, f"rms_vs_freq_{timestamp}.png")
    plt.savefig(out)
    plt.show()
    plt.close()
    return out

def plot_self_induction_rms(freqs: np.ndarray, rms_self_induction: np.ndarray, save_dir: str, timestamp: str) -> str:
    """Строит отдельный график RMS ЭДС самоиндукции vs частота."""
    plt.figure(figsize=(11, 6))
    plt.plot(freqs, rms_self_induction, label="ЭДС самоиндукции: RMS (В)", marker='s', color='red')
    plt.xlabel("Частота, Гц")
    plt.ylabel("RMS, В")
    plt.title("RMS ЭДС самоиндукции")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(save_dir, f"self_induction_rms_{timestamp}.png")
    plt.savefig(out)
    plt.close()
    return out

def save_rms_csv(save_dir: str, timestamp: str, freqs: np.ndarray, rms_model: np.ndarray, rms_total_emf: np.ndarray, rms_self_induction: np.ndarray, rms_z: np.ndarray) -> str:
    """Сохраняет RMS данные (включая z) в CSV."""
    out = os.path.join(save_dir, f"rms_data_{timestamp}.csv")
    df = pd.DataFrame({
        'freq_hz': freqs,
        'rms_main_emf_v': rms_model,
        'rms_total_emf_v': rms_total_emf,
        'rms_self_induction_v': rms_self_induction,
        'rms_z_m': rms_z
    })
    df.to_csv(out, index=False)
    return out

def plot_power_vs_frequency(freqs: np.ndarray, p_with: np.ndarray, p_without: np.ndarray, save_dir: str, timestamp: str) -> str:
    """Строит график средней выходной мощности vs частота для основной и полной ЭДС (Рисунок 7)."""
    plt.figure(figsize=(11, 6))
    plt.plot(freqs, p_with, label="Полная ЭДС", marker='o')
    plt.plot(freqs, p_without, label="Основная ЭДС", marker='s', alpha=0.85)
    plt.xlabel("Частота, Гц")
    plt.ylabel("Средняя выходная мощность, мВт")
    plt.title("Средняя выходная мощность: основная ЭДС vs полная ЭДС")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(save_dir, f"power_vs_freq_{timestamp}.png")
    plt.savefig(out)
    plt.close()
    return out

def find_fwhm(freqs: np.ndarray, powers: np.ndarray) -> float:
    """Находит ширину резонансного пика на уровне половинной мощности (FWHM)."""
    valid = np.isfinite(powers)
    freqs = freqs[valid]
    powers = powers[valid]
    if len(freqs) < 3:
        return np.nan
    max_p = np.max(powers)
    f_max = freqs[np.argmax(powers)]
    half = max_p / 2

    left_mask = freqs < f_max
    if np.any(left_mask):
        interp_left = interp1d(powers[left_mask], freqs[left_mask], kind='linear', fill_value='extrapolate')
        f_left = interp_left(half)
    else:
        f_left = np.nan

    right_mask = freqs > f_max
    if np.any(right_mask):
        interp_right = interp1d(powers[right_mask], freqs[right_mask], kind='linear', fill_value='extrapolate')
        f_right = interp_right(half)
    else:
        f_right = np.nan

    if np.isnan(f_left) or np.isnan(f_right):
        return np.nan
    return f_right - f_left

def process_frequency(freq: int) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    """Обработка одной частоты для параллельного выполнения."""
    print(f"[Process {mp.current_process().name}] Обработка частоты {freq} Гц")
    try:
        # Инициализация устройства для каждого процесса
        device = ElectromagneticHarvesterAbsolute()
        device.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)

        shaker_csv = SHAKER_PATH_TEMPLATE.format(freq=freq)
        emf_csv = EXP_EMF_PATH_TEMPLATE.format(freq=freq)

        # Проверка наличия файлов
        if not os.path.exists(shaker_csv):
            raise FileNotFoundError(f"Файл шейкера {shaker_csv} не найден")
        if not os.path.exists(emf_csv):
            raise FileNotFoundError(f"Файл ЭДС {emf_csv} не найден")

        # 1) Загрузка шейкера
        device.load_base_from_csv(
            shaker_csv,
            time_col="t", accel_col="acc_mps2",
            sep=";", decimal=".", normalize_time=True
        )

        # Проверка длительности данных шейкера
        if device.base_time_s is not None:
            shaker_duration = device.base_time_s[-1] - device.base_time_s[0]
            print(f"[Process {mp.current_process().name}] [Shaker] Duration: {shaker_duration:.3f} s")
            if shaker_duration < T_SLICE:
                print(f"[Process {mp.current_process().name}] [Warning] Shaker data duration ({shaker_duration:.3f} s) is less than T_SLICE ({T_SLICE} s)")

        # 2) Загрузка экспериментальной ЭДС
        t_emf_s, emf_exp_v, emf_exp_interp = load_emf_no_time(
            emf_csv,
            emf_col_idx=EMF_COL_IDX,
            fs_hz=EMF_FS_HZ,
            sep=EMF_SEP,
            decimal=EMF_DECIMAL,
            units=EMF_UNITS,
            normalize_time=True
        )

        # Проверка длительности данных ЭДС
        emf_duration = t_emf_s[-1] - t_emf_s[0] if len(t_emf_s) > 0 else 0
        print(f"[Process {mp.current_process().name}] [EMF] Duration: {emf_duration:.3f} s")
        if emf_duration < T_SLICE:
            print(f"[Process {mp.current_process().name}] [Warning] EMF data duration ({emf_duration:.3f} s) is less than T_SLICE ({T_SLICE} s)")

        # 3) СЕТКА: последние T_SLICE секунд
        if device.base_time_s is not None:
            t_window = np.union1d(device.base_time_s, t_emf_s)
        else:
            t_window = t_emf_s
        t_max = float(t_window[-1])
        mask = t_window >= (t_max - T_SLICE)
        t_window = t_window[mask]
        print(f"[Process {mp.current_process().name}] [grid] points in last {T_SLICE}s: {len(t_window)}")

        # 4) Решаем модель
        t_s, z, v, i, emf_open_v, emf_self_v, forces, v_term_v, z_shaker, v_shaker, z_bottom, v_bottom, z_top, v_top = device.solve_all(
            t_eval_s=t_window,
            rtol=1e-5, atol=1e-8,
            clamp_to_base=True
        )

        # 5) Интерполяция экспериментальной ЭДС на сетку модели
        emf_exp_on_grid = emf_exp_interp(t_s)

        # 6) Вычисление полного ЭДС
        total_emf_v = emf_open_v + emf_self_v

        # 7) Проверка на NaN/inf и согласование длины
        ok = np.isfinite(emf_exp_on_grid) & np.isfinite(emf_open_v) & np.isfinite(total_emf_v) & np.isfinite(emf_self_v) & np.isfinite(z)
        if not np.all(ok):
            bad = (~ok).sum()
            print(f"[Process {mp.current_process().name}] [RMS] предупреждение: {bad} NaN/inf — точки отброшены")
            emf_exp_on_grid = emf_exp_on_grid[ok]
            emf_open_v = emf_open_v[ok]
            total_emf_v = total_emf_v[ok]
            emf_self_v = emf_self_v[ok]
            z = z[ok]

        if len(emf_open_v) == 0:
            raise ValueError("Нет валидных точек для RMS.")

        # 8) RMS для основной ЭДС, эксперимента, полной ЭДС, ЭДС самоиндукции и координаты z
        rms_main_emf = rms(emf_open_v, center=True)
        rms_exp = rms(emf_exp_on_grid, center=True)
        rms_total_emf = rms(total_emf_v, center=True)
        rms_self_induction = rms(emf_self_v, center=True)
        rms_z = rms(z, center=True)

        # 9) Мощность для основной ЭДС, полной ЭДС и эксперимента
        p_main_emf = (rms_main_emf ** 2) / LOAD_R_OHM * 1000  # мВт
        p_total_emf = (rms_total_emf ** 2) / LOAD_R_OHM * 1000  # мВт
        p_exp = (rms_exp ** 2) / LOAD_R_OHM * 1000  # мВт

        # 10) Относительные ошибки
        err_rms = calculate_rms_error(np.array([rms_main_emf]), np.array([rms_exp]))
        err_p = 100 * np.abs(p_main_emf - p_exp) / p_exp if p_exp != 0 else np.nan

        return (freq, rms_main_emf, rms_exp, rms_total_emf, rms_self_induction,
                p_main_emf, p_exp, err_rms, err_p, p_total_emf, rms_z)

    except Exception as e:
        print(f"[Process {mp.current_process().name}] Ошибка при обработке частоты {freq} Гц: {e}")
        return (freq, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

# =========================
# Основной сценарий
# =========================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir, ts = create_experiment_folder(base_dir)

    # Параллельная обработка частот
    num_processes = min(mp.cpu_count(), len(FREQS))  # Не больше, чем доступных ядер или частот
    print(f"Запуск {num_processes} процессов для обработки {len(FREQS)} частот")
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_frequency, FREQS)

    # Сбор результатов
    freqs_processed = []
    rms_main_emf_values = []
    rms_exp_values = []
    rms_total_emf_values = []
    rms_self_induction_values = []
    p_main_emf_values = []
    p_exp_values = []
    err_rms_values = []
    err_p_values = []
    p_total_emf_values = []
    rms_z_values = []

    for result in results:
        (freq, rms_main_emf, rms_exp, rms_total_emf, rms_self_induction,
         p_main_emf, p_exp, err_rms, err_p, p_total_emf, rms_z) = result
        freqs_processed.append(freq)
        rms_main_emf_values.append(rms_main_emf)
        rms_exp_values.append(rms_exp)
        rms_total_emf_values.append(rms_total_emf)
        rms_self_induction_values.append(rms_self_induction)
        p_main_emf_values.append(p_main_emf)
        p_exp_values.append(p_exp)
        err_rms_values.append(err_rms)
        err_p_values.append(err_p)
        p_total_emf_values.append(p_total_emf)
        rms_z_values.append(rms_z)

    # Конвертация в массивы
    freqs_processed = np.array(freqs_processed)
    rms_main_emf_values = np.array(rms_main_emf_values)
    rms_exp_values = np.array(rms_exp_values)
    rms_total_emf_values = np.array(rms_total_emf_values)
    rms_self_induction_values = np.array(rms_self_induction_values)
    p_main_emf_values = np.array(p_main_emf_values)
    p_exp_values = np.array(p_exp_values)
    err_rms_values = np.array(err_rms_values)
    err_p_values = np.array(err_p_values)
    p_total_emf_values = np.array(p_total_emf_values)
    rms_z_values = np.array(rms_z_values)

    # Сортировка результатов по частоте
    sort_idx = np.argsort(freqs_processed)
    freqs_processed = freqs_processed[sort_idx]
    rms_main_emf_values = rms_main_emf_values[sort_idx]
    rms_exp_values = rms_exp_values[sort_idx]
    rms_total_emf_values = rms_total_emf_values[sort_idx]
    rms_self_induction_values = rms_self_induction_values[sort_idx]
    p_main_emf_values = p_main_emf_values[sort_idx]
    p_exp_values = p_exp_values[sort_idx]
    err_rms_values = err_rms_values[sort_idx]
    err_p_values = err_p_values[sort_idx]
    p_total_emf_values = p_total_emf_values[sort_idx]
    rms_z_values = rms_z_values[sort_idx]

    # Рисунок 6: Построение графика RMS (ЭДС и z)
    plot_rms_path = plot_rms_vs_frequency(freqs_processed, rms_main_emf_values, rms_total_emf_values, rms_self_induction_values, rms_z_values, out_dir, ts)

    # Рисунок: Отдельный график RMS ЭДС самоиндукции
    plot_self_induction_path = plot_self_induction_rms(freqs_processed, rms_self_induction_values, out_dir, ts)

    # Сохранение RMS в CSV (включая z)
    csv_path = save_rms_csv(out_dir, ts, freqs_processed, rms_main_emf_values, rms_total_emf_values, rms_self_induction_values, rms_z_values)

    # Таблица 5: Верификация
    table5_df = pd.DataFrame({
        'Частота возбуждения, Гц': freqs_processed,
        'V_RMSexp, мВ': rms_exp_values * 1000,
        'V_RMS основная ЭДС, мВ': rms_main_emf_values * 1000,
        'P_outexp, мВт': p_exp_values,
        'P_out основная ЭДС, мВт': p_main_emf_values,
        'Относ. ошибка по RMS, %': err_rms_values,
        'Относ. ошибка по Pout, %': err_p_values
    })
    table5_path = os.path.join(out_dir, f"table5_{ts}.csv")
    table5_df.to_csv(table5_path, index=False)

    # Рисунок 7: Построение графика мощности
    plot_power_path = plot_power_vs_frequency(freqs_processed, p_total_emf_values, p_main_emf_values, out_dir, ts)

    # Таблица 6: Влияние самоиндукции
    max_p_total = np.nanmax(p_total_emf_values)
    f_res_total = freqs_processed[np.nanargmax(p_total_emf_values)] if np.any(np.isfinite(p_total_emf_values)) else np.nan
    width_total = find_fwhm(freqs_processed, p_total_emf_values)
    # Амплитуда на резонансе
    if not np.isnan(f_res_total):
        try:
            device = ElectromagneticHarvesterAbsolute()
            device.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)
            device.load_base_from_csv(SHAKER_PATH_TEMPLATE.format(freq=f_res_total), time_col="t", accel_col="acc_mps2", sep=";", decimal=".", normalize_time=True)
            t_res = np.linspace(0, T_SLICE, int(T_SLICE * EMF_FS_HZ))
            t_res, _, _, _, emf_open_v_res, emf_self_v_res, _, _, _, _, _, _, _, _ = device.solve_all(t_eval_s=t_res, rtol=1e-5, atol=1e-8, clamp_to_base=True)
            total_emf_res = emf_open_v_res + emf_self_v_res
            amp_total = np.max(np.abs(total_emf_res - np.mean(total_emf_res)))
        except Exception as e:
            print(f"Ошибка при пересимуляции для f_res_total={f_res_total} Гц: {e}")
            amp_total = np.nan
    else:
        amp_total = np.nan

    max_p_main = np.nanmax(p_main_emf_values)
    f_res_main = freqs_processed[np.nanargmax(p_main_emf_values)] if np.any(np.isfinite(p_main_emf_values)) else np.nan
    width_main = find_fwhm(freqs_processed, p_main_emf_values)
    # Амплитуда на резонансе
    if not np.isnan(f_res_main):
        try:
            device = ElectromagneticHarvesterAbsolute()
            device.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)
            device.load_base_from_csv(SHAKER_PATH_TEMPLATE.format(freq=f_res_main), time_col="t", accel_col="acc_mps2", sep=";", decimal=".", normalize_time=True)
            t_res = np.linspace(0, T_SLICE, int(T_SLICE * EMF_FS_HZ))
            t_res, _, _, _, emf_open_v_res, _, _, _, _, _, _, _, _, _ = device.solve_all(t_eval_s=t_res, rtol=1e-5, atol=1e-8, clamp_to_base=True)
            amp_main = np.max(np.abs(emf_open_v_res - np.mean(emf_open_v_res)))
        except Exception as e:
            print(f"Ошибка при пересимуляции для f_res_main={f_res_main} Гц: {e}")
            amp_main = np.nan
    else:
        amp_main = np.nan

    change_p = 100 * (max_p_total - max_p_main) / max_p_main if max_p_main != 0 else np.nan
    change_amp = 100 * (amp_total - amp_main) / amp_main if amp_main != 0 else np.nan
    change_width = 100 * (width_total - width_main) / width_main if width_main != 0 else np.nan
    change_f_res = f_res_total - f_res_main if not (np.isnan(f_res_total) or np.isnan(f_res_main)) else np.nan

    table6_df = pd.DataFrame({
        'Показатель': [
            'Средняя мощность на нагрузке, мВт',
            'Амплитуда ЭДС (Vp), В',
            'Ширина резонансного пика по мощности, Гц',
            'Максимум частоты резонанса, Гц'
        ],
        'Основная ЭДС': [max_p_main, amp_main, width_main, f_res_main],
        'Полная ЭДС': [max_p_total, amp_total, width_total, f_res_total],
        'Изменение': [change_p, change_amp, change_width, change_f_res]
    })
    table6_path = os.path.join(out_dir, f"table6_{ts}.csv")
    table6_df.to_csv(table6_path, index=False)

    # Сводка
    print("\n==== Сводка ====")
    print(f"Папка эксперимента: {out_dir}")
    print(f"Рисунок 6 (RMS ЭДС и z vs freq): {plot_rms_path}")
    print(f"Рисунок (RMS самоиндукции): {plot_self_induction_path}")
    print(f"CSV с RMS (включая z): {csv_path}")
    print(f"Таблица 5: {table5_path}")
    print(f"Рисунок 7 (Power vs freq): {plot_power_path}")
    print(f"Таблица 6: {table6_path}")
    print(f"Обработано частот: {len(freqs_processed)}")
    for freq, rm, rt, rsi, rz in zip(freqs_processed, rms_main_emf_values, rms_total_emf_values, rms_self_induction_values, rms_z_values):
        print(f"Частота {freq} Гц: RMS основная ЭДС = {rm:.6f} В, RMS полная ЭДС = {rt:.6f} В, RMS самоиндукции = {rsi:.6f} В, RMS z = {rz:.6f} м")
