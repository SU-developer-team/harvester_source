# -*- coding: utf-8 -*-
"""
Параллельный сценарий для анализа RMS напряжения и координаты z модели и эксперимента по частотам.
- Шейкер синтезируется по формуле: a(t) = -W^2 * X0 * cos(W * t), W = 2*pi*f, X0 — из списка amplitudes.
- Для совместимости с ElectromagneticHarvesterAbsolute база ускорений загружается через load_base_from_csv из временного CSV.
- Остальная логика (RMS, мощности, Таблица 5/6, графики) без изменений по смыслу.
"""

import os
import csv
import math
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from scipy.interpolate import interp1d
from ElectromagneticHarvesterID50mm_ERZHANAT import ElectromagneticHarvesterAbsolute

# =========================
# Конфигурация эксперимента
# =========================
BASE_DIR = r"D:\PROJECTs\magnet\harvester\experiments\harvester_50mm\exp_2"
EXP_EMF_PATH_TEMPLATE = os.path.join(BASE_DIR, "data", "{freq}.csv")

T_MAX_S = 20.0
T_SLICE = 5

# Диапазон частот, под который заданы амплитуды (см. ниже base_freq=5)
FREQS = [i for i in range(5, 35, 1)]

# ЭДС параметры
EMF_COL_IDX = 3
EMF_FS_HZ = 1000.0
EMF_UNITS = "mV"
EMF_SEP = ";"
EMF_DECIMAL = ","

# Электрические параметры
COIL_R_OHM = 0.001
LOAD_R_OHM = 1.0

# Амплитуды X0 по частотам, начиная с 5 Гц (base_freq=5)
# amplitudes = [
#     0.001,    # 2
#     0.006,    # 3
#     0.006,    # 4
#     0.006,    # 5
#     0.006,    # 6
#     0.006,    # 7
#     0.006,    # 8
#     0.008,    # 9
#     0.013,    # 10
#     0.013,    # 11
#     0.0145,   # 12
#     0.0155,   # 13
#     0.0177,   # 14
#     0.0195,   # 15
#     0.0215,   # 16
#     0.0235,   # 17
#     0.024,    # 18
#     0.0175,   # 19
#     0.0092,   # 20
#     0.0067,   # 21
#     0.0052,   # 22
#     0.0042,   # 23
#     0.0036,   # 24
#     0.0031,   # 25
#     0.0027,   # 26
#     0.00235,  # 27
#     0.002,    # 28
#     0.0019,   # 29
#     0.0018,   # 30
#     0.0018,   # 31
#     0.0018,   # 32
#     0.00183,  # 33
#     0.0017,   # 34
#     0.0015    # 35
# ]
amplitudes = [ 0.01 for _ in range(len(FREQS))]

def plot_amplitude_vs_frequency(freqs: np.ndarray, amplitudes: np.ndarray, save_dir: str, timestamp: str) -> str:
    plt.figure(figsize=(11, 6))
    plt.plot(freqs, amplitudes, label="Амплитуда X0", marker='o', color='blue')
    plt.xlabel("Частота, Гц")
    plt.ylabel("Амплитуда X0, м")
    plt.title("Амплитуда X0 от частоты")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(save_dir, f"amplitude_vs_freq_{timestamp}.png")
    plt.savefig(out)
    plt.close()
    return out
plot_amplitude_vs_frequency(FREQS, amplitudes, '', 1)
# =========================
# Хелперы
# =========================
def create_experiment_folder(base_dir: str) -> Tuple[str, str]:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(base_dir, "graphs", f"experiment_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, ts

def _to_numeric_series(raw, decimal_hint: Optional[str]) -> pd.Series:
    s = pd.Series(raw, copy=True).astype(str)
    s = s.str.replace('\u00A0', '', regex=False)
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
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        nan_count = np.isnan(x).sum()
        inf_count = np.isinf(x).sum()
        raise ValueError(f"Обнаружены нечисловые значения: {nan_count} NaN, {inf_count} inf")
    if center:
        x = x - np.mean(x)
    return float(np.sqrt(np.mean(x**2)))

def calculate_rms_error(rms_model: np.ndarray, rms_exp: np.ndarray) -> float:
    valid_mask = (np.isfinite(rms_model) & np.isfinite(rms_exp) & (rms_exp != 0))
    if valid_mask.any():
        rms_error_percent = 100 * np.abs(rms_model[valid_mask] - rms_exp[valid_mask]) / rms_exp[valid_mask]
        return float(np.mean(rms_error_percent))
    return float("nan")

def get_amplitude_for_freq(freq: int, base_freq: int = 2) -> float:
    idx = freq - base_freq
    if 0 <= idx < len(amplitudes):
        return float(amplitudes[idx])
    raise ValueError(
        f"Для частоты {freq} Гц нет соответствующей амплитуды X0. "
        f"Ожидался диапазон {base_freq}..{base_freq + len(amplitudes) - 1} Гц."
    )

def write_temp_shaker_csv(t_s: np.ndarray, a_s: np.ndarray) -> str:
    """
    Создаёт временный CSV с колонками: t;acc_mps2 (десятичная точка),
    возвращает путь. Файл удаляется вызывающей стороной.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="shaker_", suffix=".csv")
    os.close(tmp_fd)
    df = pd.DataFrame({"t": t_s, "acc_mps2": a_s})
    df.to_csv(tmp_path, sep=";", index=False, header=True, float_format="%.9f")
    return tmp_path

PLOT_RMS_EMF_AND_AMPLITUDE = False
DISPLAY_RANGE = (7.0, 50.0)
_TARGET_W_PX, _TARGET_H_PX, _DPI = 2200, 1600, 600
_FIGSIZE = (6,3)


def plot_rms_vs_frequency(
    freqs: np.ndarray,
    rms_model: np.ndarray,
    rms_total_emf: np.ndarray,
    rms_self_induction: np.ndarray,
    rms_z: np.ndarray,
    save_dir: str,
    timestamp: str
) -> str:
    os.makedirs(save_dir, exist_ok=True)

    coeff = 14200.0

    # ВСЁ переводим в numpy перед маскированием
    freqs = np.asarray(freqs, dtype=float)
    rms_model = np.asarray(rms_model, dtype=float)
    rms_total_emf = np.asarray(rms_total_emf, dtype=float)
    rms_self_induction = np.asarray(rms_self_induction, dtype=float)
    rms_z = np.asarray(rms_z, dtype=float)

    # Применяем ваш масштаб
    rms_model_with_coef = rms_model * coeff + 1.0
    rms_total_emf_with_coef = rms_total_emf * coeff + 1.0
    rms_self_induction_with_coef = rms_self_induction * coeff + 1.0  # на будущее, если понадобится

    for freq, rms_val in zip(freqs, rms_total_emf_with_coef):
        print(f"Частота: {freq:.0f} Гц, RMS: {rms_val}")

    if PLOT_RMS_EMF_AND_AMPLITUDE:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # ВАЖНО: чтобы подписи соответствовали данным
        ax1.plot(freqs, rms_total_emf_with_coef, '^--', linewidth=0.8, label="Total EMF RMS")
        ax1.plot(freqs, rms_model_with_coef, 's-', linewidth=0.8, label="Induction EMF RMS")

        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("RMS EMF (mV)")
        ax1.grid(True, which='both', linestyle='-', alpha=0.35)
        ax1.legend()

        ax2.plot(freqs, rms_z * 1000.0, 'x-', label="RMS z (мм)")
        ax2.set_xlabel("Частота, Гц")
        ax2.set_ylabel("RMS z, мм")
        ax2.set_title("RMS координаты z центрального магнита")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        out = os.path.join(save_dir, f"rms_vs_freq_{timestamp}.png")
        plt.savefig(out, dpi=300)
        plt.close()
    else:
        lo_disp, hi_disp = sorted(DISPLAY_RANGE)
        mask_disp = (freqs >= lo_disp) & (freqs <= hi_disp)

        x_plot = freqs[mask_disp]
        y_plot_total = rms_total_emf_with_coef[mask_disp]      # теперь это np.ndarray → ОК
        y_plot_induction = rms_model_with_coef[mask_disp]      # тоже ОК

        plt.figure(figsize=_FIGSIZE, dpi=_DPI)
        # Подписи соответствуют кривым
        plt.plot(x_plot, y_plot_total, 's-', linewidth=0.8, label="Total EMF")
        plt.plot(x_plot, y_plot_induction, '^--', linewidth=0.8, label="Without Self-Induction EMF")

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('RMS EMF (mV)')
        plt.xticks(np.arange(np.floor(lo_disp), np.ceil(hi_disp) + 1, 1))
        plt.grid(True, which='both', linestyle='-', alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.xticks(np.arange(np.floor(lo_disp), np.ceil(hi_disp) + 1, 1))
        plt.yticks(np.arange(1, 14 + 1, 1))

        out = os.path.join(save_dir, f"rms_emf_total_vs_induction_{timestamp}.png")
        print(f"Сохранен в { out}")
        plt.savefig(out)
        plt.show()
        plt.close()

    # CSV с логами: гарантируем, что папка есть
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_file = logs_dir / f"{timestamp}_results.csv"
    with output_file.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frequency (Hz)", "Mean Total EMF RMS (mV)"])
        writer.writerows(zip(freqs, rms_total_emf_with_coef))
    print(f"Сохранен в {output_file}")
    return out 

def plot_self_induction_rms(
    freqs: np.ndarray,
    rms_self_induction: np.ndarray,
    save_dir: str,
    timestamp: str
) -> str:
    # Папка точно существует
    os.makedirs(save_dir, exist_ok=True)

    # Приводим к numpy
    freqs = np.asarray(freqs, dtype=float)
    rms_self_induction = np.asarray(rms_self_induction, dtype=float)

    # Фильтруем нечисловые
    valid = np.isfinite(freqs) & np.isfinite(rms_self_induction)
    freqs = freqs[valid]
    rms_self_induction = rms_self_induction[valid]

    # Масштабирование как в plot_rms_vs_frequency
    coeff = 14200.0
    rms_self_induction_mv = rms_self_induction * coeff + 1.0

    # Отображаем только выбранный диапазон частот
    lo_disp, hi_disp = sorted(DISPLAY_RANGE)
    mask_disp = (freqs >= lo_disp) & (freqs <= hi_disp)
    x_plot = freqs[mask_disp]
    y_plot = rms_self_induction_mv[mask_disp]

    if x_plot.size == 0:
        raise ValueError("Нет точек в DISPLAY_RANGE для построения графика самоиндукции.")

    # Рисуем с общими параметрами
    plt.figure(figsize=_FIGSIZE, dpi=_DPI)
    plt.plot(x_plot, y_plot, 'o-', linewidth=0.8, label="Self-Induction EMF")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("RMS EMF (mV)")
    plt.grid(True, which='both', linestyle='-', alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.xticks(np.arange(np.floor(lo_disp), np.ceil(hi_disp) + 1, 1))

    out = os.path.join(save_dir, f"self_induction_rms_{timestamp}.png")
    plt.savefig(out)
    plt.close()

    return out


def save_rms_csv(save_dir: str, timestamp: str, freqs: np.ndarray, rms_model: np.ndarray, rms_total_emf: np.ndarray, rms_self_induction: np.ndarray, rms_z: np.ndarray) -> str:
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

def synthesize_base_accel(freq_hz: float, t_s: np.ndarray) -> np.ndarray:
    """
    Возвращает массив ускорений шейкера по формуле:
        a(t) = -W^2 * X0 * cos(W t),   W = 2*pi*f,  X0 = амплитуда для данной частоты (м)
    """
    X0 = get_amplitude_for_freq(int(round(freq_hz)), base_freq=2)
    W = 2.0 * math.pi * float(freq_hz)
    return - (W ** 2) * X0 * np.cos(W * t_s)

# =========================
# Параллельная обработка частоты
# =========================
def process_frequency(freq: int) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    print(f"[Process {mp.current_process().name}] Обработка частоты {freq} Гц")
    try:
        device = ElectromagneticHarvesterAbsolute()
        device.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)

        emf_csv = EXP_EMF_PATH_TEMPLATE.format(freq=freq)

        # 1) Загрузка экспериментальной ЭДС
        t_emf_s, emf_exp_v, emf_exp_interp = load_emf_no_time(
            emf_csv,
            emf_col_idx=EMF_COL_IDX,
            fs_hz=EMF_FS_HZ,
            sep=EMF_SEP,
            decimal=EMF_DECIMAL,
            units=EMF_UNITS,
            normalize_time=True
        )

        # Проверка длительности ЭДС
        emf_duration = t_emf_s[-1] - t_emf_s[0] if len(t_emf_s) > 0 else 0.0
        print(f"[Process {mp.current_process().name}] [EMF] Duration: {emf_duration:.3f} s")
        if emf_duration < T_SLICE:
            print(f"[Process {mp.current_process().name}] [Warning] EMF data duration ({emf_duration:.3f} s) is less than T_SLICE ({T_SLICE} s)")

        # 2) Сетка моделирования: последние T_SLICE секунд по времени ЭДС
        if len(t_emf_s) == 0:
            raise ValueError("Пустые данные ЭДС — нечего интерполировать и строить сетку.")
        t_max = float(t_emf_s[-1])
        mask = t_emf_s >= (t_max - T_SLICE)
        t_window = t_emf_s[mask]
        print(f"[Process {mp.current_process().name}] [grid] points in last {T_SLICE}s: {len(t_window)}")

        # 3) Синтетический шейкер
        a_base = synthesize_base_accel(freq_hz=freq, t_s=t_window)

        # 4) Загружаем шейкер ЧЕРЕЗ CSV (родной метод класса)
        tmp_csv = write_temp_shaker_csv(t_window, a_base)
        try:
            device.load_base_from_csv(
                tmp_csv,
                time_col="t",
                accel_col="acc_mps2",
                sep=";",
                decimal=".",      # пишем и читаем с точкой
                normalize_time=True
            )
        finally:
            try:
                os.remove(tmp_csv)
            except OSError:
                pass

        # 5) Решение модели
        t_s, z, v, i, emf_open_v, emf_self_v, forces, v_term_v, z_shaker, v_shaker, z_bottom, v_bottom, z_top, v_top = device.solve_all(
            t_eval_s=t_window,
            rtol=1e-5, atol=1e-8,
            clamp_to_base=True
        )

        # 6) Интерполяция экспериментальной ЭДС на сетку модели
        emf_exp_on_grid = emf_exp_interp(t_s)

        # 7) Полная ЭДС
        total_emf_v = emf_open_v + emf_self_v

        # 8) Фильтрация NaN/inf
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

        # 9) RMS и мощность
        rms_main_emf = rms(emf_open_v, center=True)
        rms_exp = rms(emf_exp_on_grid, center=True)
        rms_self_induction = rms(emf_self_v, center=True)
        rms_total_emf = rms_main_emf + rms_self_induction
        rms_z = rms(z, center=True)

        p_main_emf = (rms_main_emf ** 2) / LOAD_R_OHM * 1000  # мВт
        p_total_emf = (rms_total_emf ** 2) / LOAD_R_OHM * 1000  # мВт
        p_exp = (rms_exp ** 2) / LOAD_R_OHM * 1000  # мВт

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

    num_processes = min(mp.cpu_count(), len(FREQS))
    print(f"Запуск {num_processes} процессов для обработки {len(FREQS)} частот")
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_frequency, FREQS)

    # Сбор результатов
    freqs_processed, rms_main_emf_values, rms_exp_values = [], [], []
    rms_total_emf_values, rms_self_induction_values = [], []
    p_main_emf_values, p_exp_values, err_rms_values, err_p_values = [], [], [], []
    p_total_emf_values, rms_z_values = [], []

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

    # В массивы и сортировка
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

    # Рисунки и CSV
    plot_rms_path = plot_rms_vs_frequency(freqs_processed, rms_main_emf_values, rms_total_emf_values, rms_self_induction_values, rms_z_values, out_dir, ts)
    plot_self_induction_path = plot_self_induction_rms(freqs_processed, rms_self_induction_values, out_dir, ts)
    csv_path = save_rms_csv(out_dir, ts, freqs_processed, rms_main_emf_values, rms_total_emf_values, rms_self_induction_values, rms_z_values)

    # Таблица 5
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

    # Рисунок 7
    plot_power_path = plot_power_vs_frequency(freqs_processed, p_total_emf_values, p_main_emf_values, out_dir, ts)

    # Таблица 6 (пересимуляция на резонансах)
    max_p_total = np.nanmax(p_total_emf_values)
    f_res_total = freqs_processed[np.nanargmax(p_total_emf_values)] if np.any(np.isfinite(p_total_emf_values)) else np.nan
    width_total = find_fwhm(freqs_processed, p_total_emf_values)

    if not np.isnan(f_res_total):
        try:
            device = ElectromagneticHarvesterAbsolute()
            device.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)
            t_res = np.linspace(0, T_SLICE, int(T_SLICE * EMF_FS_HZ))
            a_res = synthesize_base_accel(freq_hz=float(f_res_total), t_s=t_res)
            tmp_csv = write_temp_shaker_csv(t_res, a_res)
            try:
                device.load_base_from_csv(tmp_csv, time_col="t", accel_col="acc_mps2", sep=";", decimal=".", normalize_time=True)
            finally:
                try: os.remove(tmp_csv)
                except OSError: pass
            t_res, _, _, _, emf_open_v_res, emf_self_v_res, _, _, _, _, _, _, _, _ = device.solve_all(
                t_eval_s=t_res, rtol=1e-5, atol=1e-8, clamp_to_base=True
            )
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

    if not np.isnan(f_res_main):
        try:
            device = ElectromagneticHarvesterAbsolute()
            device.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)
            t_res = np.linspace(0, T_SLICE, int(T_SLICE * EMF_FS_HZ))
            a_res = synthesize_base_accel(freq_hz=float(f_res_main), t_s=t_res)
            tmp_csv = write_temp_shaker_csv(t_res, a_res)
            try:
                device.load_base_from_csv(tmp_csv, time_col="t", accel_col="acc_mps2", sep=";", decimal=".", normalize_time=True)
            finally:
                try: os.remove(tmp_csv)
                except OSError: pass
            t_res, _, _, _, emf_open_v_res, _, _, _, _, _, _, _, _, _ = device.solve_all(
                t_eval_s=t_res, rtol=1e-5, atol=1e-8, clamp_to_base=True
            )
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
    # for freq, rm, rt, rsi, rz in zip(freqs_processed, rms_main_emf_values, rms_total_emf_values, rms_self_induction_values, rms_z_values):
    #     print(f"Частота {freq} Гц: RMS основная ЭДС = {rm:.6f} В, RMS полная ЭДС = {rt:.6f} В, RMS самоиндукции = {rsi:.6f} В, RMS z = {rz:.6f} м")
