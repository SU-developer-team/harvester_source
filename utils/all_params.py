# -*- coding: utf-8 -*-
"""
Параллельный сценарий для анализа RMS напряжения и координаты z модели по частотам 2–34 Гц с перебором параметров Br и c.
- Перебирает значения Br_values и c_values в обычном цикле.
- Обрабатывает выборочные частоты параллельно с использованием multiprocessing.
- Строит график: частота (Гц) vs RMS (В) для основной ЭДС, полной ЭДС и ЭДС самоиндукции (Рисунок 6).
- Строит график: частота (Гц) vs RMS координаты z (м).
- Строит отдельный график: частота (Гц) vs RMS ЭДС самоиндукции (В).
- Строит график: частота (Гц) vs средняя выходная мощность (мВт) для основной и полной ЭДС (Рисунок 7).
- Формирует и сохраняет Таблицу 5 (верификация) и Таблицу 6 (влияние самоиндукции) в CSV.
- Сохраняет результаты RMS (включая z) в CSV и графики в PNG.
- Записывает все результаты в текстовый файл.
"""

import os
from datetime import datetime
from typing import Tuple, Optional
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# =========================
# Конфигурация эксперимента
# =========================
BASE_DIR = r"D:\PROJECTs\magnet\harvester\experiments\harvester_50mm\exp_2"
SHAKER_PATH_TEMPLATE = os.path.join(BASE_DIR, "synthesized_data", "{freq}.csv")
EXP_EMF_PATH_TEMPLATE = os.path.join(BASE_DIR, "data", "{freq}.csv")

T_MAX_S = 20.0              # максимум длительности моделирования
T_SLICE = 5                 # ограничение времени для шейкера и ЭДС (последние 5 секунд)

FREQS = [i for i in range(2, 35)]    # выборочные частоты для всех таблиц и графиков

# ЭДС параметры
EMF_COL_IDX = 3             # ЭДС в 4-м столбце (индексация с 0)
EMF_FS_HZ = 1000.0          # частота дискретизации, Гц
EMF_UNITS = "mV"            # единицы: "V", "mV", "uV"
EMF_SEP = ";"
EMF_DECIMAL = ","

# Электрические параметры
COIL_R_OHM = 0.001          # сопротивление катушки, Ом
LOAD_R_OHM = 1.0            # сопротивление нагрузки, Ом

# Массивы параметров для перебора
R = 0.0195 / 2   # радиус магнита, м
Br_values = [Br for Br in np.arange(0.4, 2, 0.1)]  # значения остаточной магнитной индукции, Тл
c_values = [R * c for c in np.arange(0.4, 2, 0.05)]  # значения параметра c (R * [0.45, 0.6, 0.8])
c_values_d = [c for c in np.arange(0.4, 2, 0.05)]  # значения параметра c (R * [0.45, 0.6, 0.8])

# --- Параметры системы ---
m = 0.020        # масса магнита, кг
rho = 1.2        # плотность воздуха, кг/м³
S = np.pi * R**2 # площадь поперечного сечения: S = π·R²
g = 9.81         # ускорение свободного падения, м/с²
h = 0.0572       # расстояние до верхнего магнита, м
L = 0.01         # длина цилиндрического магнита, м
mu0 = 1.25663706e-6  # магнитная постоянная, Гн/м
alpha = 0.01     # параметр alpha, А/м
K_emf = 0.129    # коэффициент ЭДС, В·с/м
r_device = 0.01225 # радиус устройства
r_wire = 0.00045
R_coil = r_device + r_wire  # радиус катушки, м
Cf = 0.82        # коэффициент аэродинамического сопротивления
h_coil = 0.05
# --- Электрическая часть ---
N_turns = 56     # число витков катушки
dc = h / N_turns # шаг между витками катушки
z_coil = np.linspace(0.0036, h-0.0036, N_turns)  # координаты колец катушки
R_load = LOAD_R_OHM
r_coil = COIL_R_OHM
R_resistance = R_load + r_coil  # сопротивление R, Ом
S_coil = np.pi * R_coil**2  # площадь поперечного сечения катушки
L_inductance = (mu0 * S_coil * h_coil) / (r_wire*2)**2  # индуктивность катушки

# --- Начальные условия ---
z0 = 0.025
v0 = 0.0
I0 = 0.0       # начальный ток

conf_1 = 1
conf_2 = 1
conf_3 = 1
conf_4 = 1

# =========================
# Хелперы
# =========================
def create_experiment_folder(base_dir: str, Br: float, c: float) -> Tuple[str, str]:
    """Создать папку для графиков/файлов с таймстампом и параметрами Br, c."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(base_dir, "graphs", f"experiment_Br_{Br:.7f}_c_{(c/R):.5f}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, ts

def _to_numeric_series(raw, decimal_hint: Optional[str]) -> pd.Series:
    """Надёжное приведение строк с запятой/пробелами к float. Мусор -> NaN."""
    s = pd.Series(raw, copy=True).astype(str)
    s = s.str.replace('\u00A0', '', regex=False)   # неразрывный пробел
    s = s.str.replace(' ', '', regex=False)
    s = s.str.replace(',', '.', regex=False)       # унифицируем десятичный знак
    return pd.to_numeric(s, errors='coerce')

def load_shaker_data(file_path: str, time_col: str = "t", accel_col: str = "acc_mps2",
                     sep: str = ";", decimal: str = ".", normalize_time: bool = True) -> Tuple[np.ndarray, np.ndarray, interp1d]:
    """Загрузить акселерограмму шейкера из CSV и построить интерполяцию."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
    
    df = pd.read_csv(file_path, sep=sep, decimal=decimal)
    if time_col not in df or accel_col not in df:
        raise ValueError(f"В {file_path} должны быть колонки '{time_col}' и '{accel_col}'")
    
    df = df.groupby(time_col, as_index=False)[accel_col].mean().sort_values(time_col)
    t_s = df[time_col].to_numpy(dtype=float)
    a_mps2 = _to_numeric_series(df[accel_col], decimal).to_numpy(dtype=float)
    
    if normalize_time:
        t_s -= t_s[0]
    
    # Ограничим данные до T_MAX_S
    mask = t_s <= T_MAX_S
    t_s, a_mps2 = t_s[mask], a_mps2[mask]
    
    # Проверка на NaN и минимальное количество точек
    if np.any(~np.isfinite(a_mps2)) or np.any(~np.isfinite(t_s)):
        raise ValueError(f"В данных акселерограммы {file_path} есть NaN или нечисловые значения")
    if len(t_s) < 2:
        raise ValueError(f"В файле {file_path} недостаточно данных для интерполяции (менее 2 точек)")
    
    # Проверка монотонности времени
    if not np.all(np.diff(t_s) > 0):
        raise ValueError(f"Временной столбец в {file_path} не монотонно возрастает")
    
    # Интерполяция
    accel_interp = interp1d(t_s, a_mps2, kind="linear", bounds_error=False,
                           fill_value=(a_mps2[0], a_mps2[-1]))
    
    print(f"[Shaker {os.path.basename(file_path)}] {len(t_s)} точек, длит. ~{t_s[-1]:.3f} с, диапазон {np.min(a_mps2):.3f}..{np.max(a_mps2):.3f} м/с²")
    return t_s, a_mps2, accel_interp

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

# --- Функции модели ---
def F1(z, Bc, c):
    return Bc * (1 / (c + (h - z + L/2))**2 + 1 / (2*L + c + (h - z + L/2))**2 - 2 / (L + c + (h - z + L/2))**2) * conf_4

def F3(z, Bc, c):
    return Bc * (1 / (c + z - L/2)**2 + 1 / (2*L + c + z - L/2)**2 - 2 / (L + c + z - L/2)**2) * conf_4

def dB_dz(z, z_coil, Br):
    dh = np.abs(z - z_coil)
    term_1 = R**2 / ((dh+L/2)**2 + R**2)**(3/2)
    term_2 = R**2 / ((dh-L/2)**2 + R**2)**(3/2)
    db_dz = Br / 2 * (term_1 - term_2) * K_emf
    return db_dz * 0.129

def total_dB_dz(z, z_coil, Br):
    """Вычисляет сумму dB/dz для скалярного z."""
    if not np.isscalar(z):
        raise ValueError(f"total_dB_dz ожидает скалярное z, получено {z.shape}")
    return np.sum(dB_dz(z, z_coil, Br))

def ode_system(t, y, accel_values, t_eval, Bc, c, Br, L_inductance):
    z, v, I = y
    if not np.isscalar(z):
        raise ValueError(f"ode_system ожидает скалярное z, получено {z.shape}")
    # Ускорение от шейкера с использованием интерполяции по ближайшей точке
    idx = np.searchsorted(t_eval, t, side='right') - 1
    idx = np.clip(idx, 0, len(accel_values) - 1)
    a_base = accel_values[idx] * conf_3
    F_shaker = -m * a_base
    F_aero = -(Cf * rho * S * v * abs(v)) / 2
    F_gravity = -m * g
    dv_dt = (F3(z, Bc, c) - F1(z, Bc, c) + F_shaker + F_aero + F_gravity) / m
    dI_dt = (total_dB_dz(z, z_coil, Br) * v - R_resistance * I) / L_inductance
    return np.array([v, dv_dt, dI_dt])

def process_frequency(args: Tuple[int, float, float, float, float]) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    """Обработка одной частоты для параллельного выполнения."""
    freq, Br, c, Bc, L_inductance = args
    print(f"[Process {mp.current_process().name}] Обработка частоты {freq} Гц, Br={Br:.4f}, c={c:.5f}")
    try:
        shaker_csv = SHAKER_PATH_TEMPLATE.format(freq=freq)
        emf_csv = EXP_EMF_PATH_TEMPLATE.format(freq=freq)

        # Проверка наличия файлов
        if not os.path.exists(shaker_csv):
            raise FileNotFoundError(f"Файл шейкера {shaker_csv} не найден")
        if not os.path.exists(emf_csv):
            raise FileNotFoundError(f"Файл ЭДС {emf_csv} не найден")

        # 1) Загрузка шейкера
        t_shaker_s, a_shaker_mps2, accel_interp = load_shaker_data(
            shaker_csv, time_col="t", accel_col="acc_mps2", sep=";", decimal=".", normalize_time=True
        )

        # Проверка длительности данных шейкера
        shaker_duration = t_shaker_s[-1] - t_shaker_s[0]
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
        t_window = np.union1d(t_shaker_s, t_emf_s)
        t_max = float(t_window[-1])
        mask = t_window >= (t_max - T_SLICE)
        t_window = t_window[mask]
        print(f"[Process {mp.current_process().name}] [grid] points in last {T_SLICE}s: {len(t_window)}")

        # 4) Решаем модель
        accel_values = accel_interp(t_window)
        sol = solve_ivp(
            lambda t, y: ode_system(t, y, accel_values, t_window, Bc, c, Br, L_inductance),
            t_span=(t_window[0], t_window[-1]),
            y0=[z0, v0, I0],
            t_eval=t_window,
            method='RK45',
            rtol=1e-5,
            atol=1e-8
        )

        if not sol.success:
            print(f"[Process {mp.current_process().name}] Предупреждение: Решение ОДУ для {freq} Гц не удалось: {sol.message}")
            return (freq, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

        z = sol.y[0]
        v = sol.y[1]
        I = sol.y[2]
        dv_dt = np.zeros_like(v)

        # Вычисление сил
        F1_vals = np.array([F1(zz, Bc, c) for zz in z])
        F3_vals = np.array([F3(zz, Bc, c) for zz in z])
        F_shaker_vals = -m * accel_values * conf_3
        F_aero_vals = -(Cf * rho * S * v * np.abs(v)) / 2
        F_gravity_vals = np.full_like(z, -m * g)

        for i in range(len(t_window)):
            idx = np.clip(np.searchsorted(t_window, t_window[i], side='right') - 1, 0, len(accel_values) - 1)
            a_base = accel_values[idx]
            F_shaker = -m * a_base
            dv_dt[i] = (F3(z[i], Bc, c) - F1(z[i], Bc, c) + F_shaker - (Cf * rho * S * v[i] * abs(v[i])) / 2 - m * g) / m

        total_dB = np.array([total_dB_dz(zz, z_coil, Br) for zz in z])
        d_emfi_dt = - S_coil * dv_dt * total_dB - S_coil * v**2 * total_dB**2
        e_i = - S_coil * v * total_dB
        e_self = L_inductance * d_emfi_dt / R_resistance
        e_tot = e_i + e_self

        # 5) Интерполяция экспериментальной ЭДС на сетку модели
        emf_exp_on_grid = emf_exp_interp(t_window)

        # 6) Проверка на NaN/inf и согласование длины
        ok = np.isfinite(emf_exp_on_grid) & np.isfinite(e_i) & np.isfinite(e_tot) & np.isfinite(e_self) & np.isfinite(z)
        if not np.all(ok):
            bad = (~ok).sum()
            print(f"[Process {mp.current_process().name}] [RMS] предупреждение: {bad} NaN/inf — точки отброшены")
            emf_exp_on_grid = emf_exp_on_grid[ok]
            e_i = e_i[ok]
            e_tot = e_tot[ok]
            e_self = e_self[ok]
            z = z[ok]

        if len(e_i) == 0:
            raise ValueError("Нет валидных точек для RMS.")

        # 7) RMS для основной ЭДС, эксперимента, полной ЭДС, ЭДС самоиндукции и координаты z
        rms_main_emf = rms(e_i, center=True)
        rms_exp = rms(emf_exp_on_grid, center=True)
        rms_total_emf = rms(e_tot, center=True)
        rms_self_induction = rms(e_self, center=True)
        rms_z = rms(z, center=True)

        # 8) Мощность для основной ЭДС, полной ЭДС и эксперимента
        p_main_emf = (rms_main_emf ** 2) / LOAD_R_OHM * 1000  # мВт
        p_total_emf = (rms_total_emf ** 2) / LOAD_R_OHM * 1000  # мВт
        p_exp = (rms_exp ** 2) / LOAD_R_OHM * 1000  # мВт

        # 9) Относительные ошибки
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
    results_txt_path = os.path.join(base_dir, "graphs", f"parameter_sweep_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")

    # Открываем текстовый файл для записи результатов
    with open(results_txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write("=== Результаты перебора параметров Br и c ===\n\n")

        # Перебор значений Br и c
        for Br in Br_values:
            for c in c_values:
                # Пересчёт Bc для текущего Br
                Bc = (np.pi * Br**2 * R**4) / (4 * mu0)
                print(f"\n=== Обработка Br={Br:.4f}, c={c:.5f} ===\n")
                txt_file.write(f"\n=== Br={Br:.4f}, c={c:.5f} ===\n")

                # Создание папки для текущей комбинации
                out_dir, ts = create_experiment_folder(base_dir, Br, c)

                # Параллельная обработка частот
                num_processes = min(mp.cpu_count(), len(FREQS))  # Не больше, чем доступных ядер или частот
                print(f"Запуск {num_processes} процессов для обработки {len(FREQS)} частот")
                txt_file.write(f"Запуск {num_processes} процессов для обработки {len(FREQS)} частот\n")
                with mp.Pool(processes=num_processes) as pool:
                    args = [(freq, Br, c, Bc, L_inductance) for freq in FREQS]
                    results = pool.map(process_frequency, args)

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
                max_p_total = np.nanmax(p_total_emf_values) if np.any(np.isfinite(p_total_emf_values)) else np.nan
                f_res_total = freqs_processed[np.nanargmax(p_total_emf_values)] if np.any(np.isfinite(p_total_emf_values)) else np.nan
                width_total = find_fwhm(freqs_processed, p_total_emf_values)
                # Амплитуда на резонансе
                if not np.isnan(f_res_total):
                    try:
                        t_shaker_s, a_shaker_mps2, accel_interp = load_shaker_data(
                            SHAKER_PATH_TEMPLATE.format(freq=f_res_total), time_col="t", accel_col="acc_mps2", sep=";", decimal=".", normalize_time=True
                        )
                        t_res = np.linspace(t_shaker_s[-1] - T_SLICE, t_shaker_s[-1], int(T_SLICE * EMF_FS_HZ))
                        accel_values = accel_interp(t_res)
                        sol = solve_ivp(
                            lambda t, y: ode_system(t, y, accel_values, t_res, Bc, c, Br, L_inductance),
                            t_span=(t_res[0], t_res[-1]),
                            y0=[z0, v0, I0],
                            t_eval=t_res,
                            method='RK45',
                            rtol=1e-5,
                            atol=1e-8
                        )
                        if not sol.success:
                            raise ValueError(f"Решение ОДУ для f_res_total={f_res_total} Гц не удалось: {sol.message}")
                        z = sol.y[0]
                        v = sol.y[1]
                        total_dB = np.array([total_dB_dz(zz, z_coil, Br) for zz in z])
                        dv_dt = np.zeros_like(v)
                        for i in range(len(t_res)):
                            idx = np.clip(np.searchsorted(t_res, t_res[i], side='right') - 1, 0, len(accel_values) - 1)
                            a_base = accel_values[idx]
                            F_shaker = -m * a_base
                            dv_dt[i] = (F3(z[i], Bc, c) - F1(z[i], Bc, c) + F_shaker - (Cf * rho * S * v[i] * abs(v[i])) / 2 - m * g) / m
                        d_emfi_dt = - S_coil * dv_dt * total_dB - S_coil * v**2 * total_dB**2
                        e_i = - S_coil * v * total_dB
                        e_self = L_inductance * d_emfi_dt / R_resistance
                        total_emf_res = e_i + e_self
                        amp_total = np.max(np.abs(total_emf_res - np.mean(total_emf_res)))
                    except Exception as e:
                        print(f"Ошибка при пересимуляции для f_res_total={f_res_total} Гц: {e}")
                        amp_total = np.nan
                else:
                    amp_total = np.nan

                max_p_main = np.nanmax(p_main_emf_values) if np.any(np.isfinite(p_main_emf_values)) else np.nan
                f_res_main = freqs_processed[np.nanargmax(p_main_emf_values)] if np.any(np.isfinite(p_main_emf_values)) else np.nan
                width_main = find_fwhm(freqs_processed, p_main_emf_values)
                # Амплитуда на резонансе
                if not np.isnan(f_res_main):
                    try:
                        t_shaker_s, a_shaker_mps2, accel_interp = load_shaker_data(
                            SHAKER_PATH_TEMPLATE.format(freq=f_res_main), time_col="t", accel_col="acc_mps2", sep=";", decimal=".", normalize_time=True
                        )
                        t_res = np.linspace(t_shaker_s[-1] - T_SLICE, t_shaker_s[-1], int(T_SLICE * EMF_FS_HZ))
                        accel_values = accel_interp(t_res)
                        sol = solve_ivp(
                            lambda t, y: ode_system(t, y, accel_values, t_res, Bc, c, Br, L_inductance),
                            t_span=(t_res[0], t_res[-1]),
                            y0=[z0, v0, I0],
                            t_eval=t_res,
                            method='RK45',
                            rtol=1e-5,
                            atol=1e-8
                        )
                        if not sol.success:
                            raise ValueError(f"Решение ОДУ для f_res_main={f_res_main} Гц не удалось: {sol.message}")
                        z = sol.y[0]
                        v = sol.y[1]
                        total_dB = np.array([total_dB_dz(zz, z_coil, Br) for zz in z])
                        e_i = - S_coil * v * total_dB
                        amp_main = np.max(np.abs(e_i - np.mean(e_i)))
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

                # Запись результатов в текстовый файл
                txt_file.write(f"\nРезультаты обработки:\n")
                txt_file.write(f"Рисунок 6 (RMS ЭДС и z vs freq): {plot_rms_path}\n")
                txt_file.write(f"Рисунок (RMS самоиндукции): {plot_self_induction_path}\n")
                txt_file.write(f"CSV с RMS (включая z): {csv_path}\n")
                txt_file.write(f"Таблица 5: {table5_path}\n")
                txt_file.write(f"Рисунок 7 (Power vs freq): {plot_power_path}\n")
                txt_file.write(f"Таблица 6: {table6_path}\n")
                txt_file.write(f"Обработано частот: {len(freqs_processed)}\n")
                txt_file.write("\nRMS и мощности по частотам:\n")
                txt_file.write("Частота (Гц) | RMS основная ЭДС (В) | RMS эксперимент (В) | RMS полная ЭДС (В) | RMS самоиндукции (В) | RMS z (м) | P_main (мВт) | P_exp (мВт) | P_total (мВт) | Ошибка RMS (%) | Ошибка Pout (%)\n")
                txt_file.write("-" * 120 + "\n")
                for freq, rm, re, rt, rsi, rz, pm, pe, pt, er, ep in zip(
                    freqs_processed, rms_main_emf_values, rms_exp_values, rms_total_emf_values,
                    rms_self_induction_values, rms_z_values, p_main_emf_values, p_exp_values,
                    p_total_emf_values, err_rms_values, err_p_values
                ):
                    txt_file.write(f"{freq:.1f} | {rm:.6f} | {re:.6f} | {rt:.6f} | {rsi:.6f} | {rz:.6f} | {pm:.6f} | {pe:.6f} | {pt:.6f} | {er:.2f} | {ep:.2f}\n")
                
                txt_file.write("\nТаблица 6: Влияние самоиндукции\n")
                txt_file.write(table6_df.to_string(index=False) + "\n")
                txt_file.write("\n" + "-" * 120 + "\n")

                # Сводка в консоль
                print("\n==== Сводка ====")
                print(f"Br={Br:.4f}, c={c:.5f}")
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

    print(f"\nВсе результаты сохранены в текстовый файл: {results_txt_path}")