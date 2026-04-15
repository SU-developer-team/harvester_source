# -*- coding: utf-8 -*-
"""
Параллельный сценарий для анализа RMS напряжения и координаты z модели и эксперимента по частотам.
- Шейкер синтезируется по формуле: a(t) = -W^2 * X0 * cos(W * t), W = 2*pi*f.
- Эксп. ЭДС приводится к Вольтам.
- Автоподбор X0 на каждой частоте: модель масштабируется на k = SCALE_COEFF при сравнении с эксперементом.
- Метрики (MRE, ошибка по мощности) считаются с учётом масштабирования.
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
from typing import Tuple, Optional, Dict
from scipy.interpolate import interp1d
from ElectromagneticHarvesterID50mm_ERZHANAT import ElectromagneticHarvesterAbsolute

# =========================
# Конфигурация эксперимента
# =========================
BASE_DIR = r"D:\PROJECTs\magnet\harvester\experiments\harvester_50mm\exp_2"
EXP_EMF_PATH_TEMPLATE = os.path.join(BASE_DIR, "data", "{freq}.csv")

T_MAX_S = 20.0
T_SLICE = 5

# Частоты для обработки
FREQS = [i for i in range(5, 25, 1)]

# ЭДС параметры (эксп.)
EMF_COL_IDX = 3
EMF_FS_HZ = 1000.0
EMF_UNITS = "mV"       # входные единицы в CSV
EMF_SEP = ";"
EMF_DECIMAL = ","

# Электрические параметры
COIL_R_OHM = 0.001
LOAD_R_OHM = 1.0

# ---------- Масштабирование модели для сравнения с экспериментом ----------
SCALE_COEFF = 13500.0   # k: модельная RMS умножается на k перед сравнением
APPLY_SCALE_IN_OPT = True  # True: используем k в оптимизации и метриках

# ---------- Настройки автоподбора амплитуды ----------
AMPL_MIN = 0.0001      # м
AMPL_MAX = 0.030      # м
COARSE_STEP = 0.0001   # м
FINE_STEP = 0.0001    # м
FINE_SPAN = 0.000015    # м вокруг лучшего coarse
TARGET_SIGNAL = "main"   # 'main' (основная ЭДС) или 'total' (полная ЭДС)
OPTIMIZE_BY   = "abs"    # 'abs' -> |k*model-exp|, 'mre' -> |k*model-exp|/|exp|
_EPS = 1e-12
best_amplitudes = []
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

    # Приводим к Вольтам (эксп. у нас в мВ -> умножаем на 1e-3)
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

def calculate_rms_error_percent_single(rms_model_scaled: float, rms_exp: float) -> float:
    """Возвращает |model_scaled - exp|/|exp| * 100, %."""
    denom = abs(rms_exp) if abs(rms_exp) > _EPS else _EPS
    return float(100.0 * abs(rms_model_scaled - rms_exp) / denom)

def write_temp_shaker_csv(t_s: np.ndarray, a_s: np.ndarray) -> str:
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="shaker_", suffix=".csv")
    os.close(tmp_fd)
    df = pd.DataFrame({"t": t_s, "acc_mps2": a_s})
    df.to_csv(tmp_path, sep=";", index=False, header=True, float_format="%.9f")
    return tmp_path

def synthesize_base_accel(freq_hz: float, t_s: np.ndarray, X0: float) -> np.ndarray:
    """a(t) = -W^2 * X0 * cos(W t), W = 2*pi*f."""
    W = 2.0 * math.pi * float(freq_hz)
    return - (W ** 2) * X0 * np.cos(W * t_s)

def run_model_once(device: ElectromagneticHarvesterAbsolute,
                   freq: float,
                   t_window: np.ndarray,
                   X0: float) -> Dict[str, float]:
    """Запускает модель на одной амплитуде X0 и возвращает метрики RMS/мощности (RAW, без масштабирования)."""
    a_base = synthesize_base_accel(freq_hz=freq, t_s=t_window, X0=X0)

    tmp_csv = write_temp_shaker_csv(t_window, a_base)
    try:
        device.load_base_from_csv(
            tmp_csv, time_col="t", accel_col="acc_mps2",
            sep=";", decimal=".", normalize_time=True
        )
    finally:
        try:
            os.remove(tmp_csv)
        except OSError:
            pass

    (t_s, z, v, i, emf_open_v, emf_self_v, forces,
     v_term_v, z_shaker, v_shaker, z_bottom, v_bottom, z_top, v_top) = device.solve_all(
        t_eval_s=t_window, rtol=1e-5, atol=1e-8, clamp_to_base=True
    )

    total_emf_v = emf_open_v + emf_self_v

    # RMS (RAW)
    rms_main_emf = rms(emf_open_v, center=True)
    rms_total_emf = rms(total_emf_v, center=True)
    rms_self_induction = rms(emf_self_v, center=True)
    rms_z = rms(z, center=True)

    # Выходные мощности на нагрузке (RAW, мВт)
    p_main_emf = (rms_main_emf ** 2) / LOAD_R_OHM * 1000.0
    p_total_emf = (rms_total_emf ** 2) / LOAD_R_OHM * 1000.0

    return dict(
        rms_main_emf=rms_main_emf,
        rms_total_emf=rms_total_emf,
        rms_self_induction=rms_self_induction,
        rms_z=rms_z,
        p_main_emf=p_main_emf,
        p_total_emf=p_total_emf
    )

def error_value(model_rms_raw: float, exp_rms: float, mode: str = "abs", use_scale: bool = True) -> float:
    """
    'abs' -> |k*model - exp|,
    'mre' -> |k*model - exp| / |exp|,
    где k = SCALE_COEFF (если use_scale=True).
    """
    model = model_rms_raw * SCALE_COEFF if use_scale else model_rms_raw
    if mode == "mre":
        denom = abs(exp_rms) if abs(exp_rms) > _EPS else _EPS
        return abs(model - exp_rms) / denom
    return abs(model - exp_rms)

def find_best_amplitude(device: ElectromagneticHarvesterAbsolute,
                        freq: int,
                        t_window: np.ndarray,
                        rms_exp: float,
                        target_signal: str = "main",
                        optimize_by: str = "abs") -> Tuple[float, Dict[str, float]]:
    """
    Двухпроходный поиск X0:
      1) coarse: [AMPL_MIN..AMPL_MAX] с шагом COARSE_STEP
      2) fine:   вокруг лучшего ±FINE_SPAN с шагом FINE_STEP
    Возвращает (X0_best, metrics_best) — метрики RAW (без k).
    """
    assert target_signal in ("main", "total")
    assert optimize_by in ("abs", "mre")

    def pick_model_rms(m: Dict[str, float]) -> float:
        return m["rms_main_emf"] if target_signal == "main" else m["rms_total_emf"]

    # --- coarse ---
    X0_candidates = np.arange(AMPL_MIN, AMPL_MAX + 0.5 * COARSE_STEP, COARSE_STEP)
    best = (None, None, float("inf"))  # (X0, metrics_raw, err)
    for X0 in X0_candidates:
        metrics = run_model_once(device, freq, t_window, X0)
        cur_err = error_value(pick_model_rms(metrics), rms_exp, optimize_by, use_scale=APPLY_SCALE_IN_OPT)
        if cur_err < best[2]:
            best = (X0, metrics, cur_err)

    X0_coarse, metrics_coarse, _ = best

    # --- fine ---
    lo = max(AMPL_MIN, X0_coarse - FINE_SPAN)
    hi = min(AMPL_MAX, X0_coarse + FINE_SPAN)
    X0_fine_candidates = np.arange(lo, hi + 0.5 * FINE_STEP, FINE_STEP)

    best = (X0_coarse, metrics_coarse,
            error_value(pick_model_rms(metrics_coarse), rms_exp, optimize_by, use_scale=APPLY_SCALE_IN_OPT))
    for X0 in X0_fine_candidates:
        metrics = run_model_once(device, freq, t_window, X0)
        cur_err = error_value(pick_model_rms(metrics), rms_exp, optimize_by, use_scale=APPLY_SCALE_IN_OPT)
        if cur_err < best[2]:
            best = (X0, metrics, cur_err)

    return best[0], best[1]

# =========================
# Плоттеры и сохранение
# =========================
def plot_rms_vs_frequency(freqs: np.ndarray,
                          rms_model_raw: np.ndarray,
                          rms_total_emf_raw: np.ndarray,
                          rms_self_induction_raw: np.ndarray,
                          rms_z: np.ndarray,
                          save_dir: str, timestamp: str) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    coeff = SCALE_COEFF

    rms_model_scaled = rms_model_raw * coeff + 1.0
    rms_total_emf_scaled = rms_total_emf_raw * coeff + 1.0
    rms_self_induction_scaled = rms_self_induction_raw * coeff + 1.0

    for freq, r in zip(freqs, rms_total_emf_scaled):
        print(f"Частота: {freq} Гц, RMS(полная, с k)+1: {r}")

    ax1.plot(freqs, rms_model_scaled, label=f"Основная ЭДС: RMS (В) ×k +1, k={coeff:g}", marker='o')
    ax1.plot(freqs, rms_total_emf_scaled, label="Полная ЭДС: RMS (В) ×k +1", marker='^', alpha=0.7)
    ax1.plot(freqs, rms_self_induction_scaled, label="Самоиндукция: RMS (В) ×k +1", marker='s', alpha=0.85)
    ax1.set_xlabel("Частота, Гц")
    ax1.set_ylabel("RMS, В (условная шкала)")
    ax1.set_title("RMS ЭДС (модель, масштабированная)")
    ax1.grid(True)
    ax1.legend()

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

    # Логи (как раньше)
    output_file = Path(f"logs/{timestamp}_results.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frequency (Hz)", "Mean Self-Induction EMF (V_scaled_plus1)"])
        writer.writerows(zip(freqs, rms_total_emf_scaled))
    print(f"Сохранен в {output_file}")
    return out

def plot_self_induction_rms(freqs: np.ndarray, rms_self_induction_raw: np.ndarray, save_dir: str, timestamp: str) -> str:
    plt.figure(figsize=(11, 6))
    plt.plot(freqs, rms_self_induction_raw * SCALE_COEFF, label=f"Самоиндукция: RMS (В) ×k, k={SCALE_COEFF:g}", marker='s', color='red')
    plt.xlabel("Частота, Гц")
    plt.ylabel("RMS, В (масштабировано)")
    plt.title("RMS ЭДС самоиндукции (масштабированная)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(save_dir, f"self_induction_rms_{timestamp}.png")
    plt.savefig(out)
    plt.close()
    return out

def save_rms_csv(save_dir: str, timestamp: str,
                 freqs: np.ndarray,
                 rms_model_raw: np.ndarray,
                 rms_total_emf_raw: np.ndarray,
                 rms_self_induction_raw: np.ndarray,
                 rms_z: np.ndarray,
                 x0_best: np.ndarray) -> str:
    out = os.path.join(save_dir, f"rms_data_{timestamp}.csv")
    df = pd.DataFrame({
        'freq_hz': freqs,
        'rms_main_emf_v_raw': rms_model_raw,
        'rms_total_emf_v_raw': rms_total_emf_raw,
        'rms_self_induction_v_raw': rms_self_induction_raw,
        'rms_z_m': rms_z,
        'X0_best_m': x0_best,
        'X0_best_mm': x0_best * 1000.0
    })
    df.to_csv(out, index=False)
    return out

def plot_power_vs_frequency(freqs: np.ndarray, p_with_raw: np.ndarray, p_without_raw: np.ndarray, save_dir: str, timestamp: str) -> str:
    plt.figure(figsize=(11, 6))
    plt.plot(freqs, p_with_raw, label="Полная ЭДС (RAW)", marker='o')
    plt.plot(freqs, p_without_raw, label="Основная ЭДС (RAW)", marker='s', alpha=0.85)
    plt.xlabel("Частота, Гц")
    plt.ylabel("Средняя выходная мощность, мВт (RAW)")
    plt.title("Средняя выходная мощность: основная vs полная (RAW)")
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

    from scipy.interpolate import interp1d as _interp1d
    left_mask = freqs < f_max
    if np.any(left_mask):
        interp_left = _interp1d(powers[left_mask], freqs[left_mask], kind='linear', fill_value='extrapolate')
        f_left = interp_left(half)
    else:
        f_left = np.nan

    right_mask = freqs > f_max
    if np.any(right_mask):
        interp_right = _interp1d(powers[right_mask], freqs[right_mask], kind='linear', fill_value='extrapolate')
        f_right = interp_right(half)
    else:
        f_right = np.nan

    if np.isnan(f_left) or np.isnan(f_right):
        return np.nan
    return float(f_right - f_left)

# =========================
# Параллельная обработка частоты
# =========================
def process_frequency(freq: int) -> Tuple[float, float, float, float, float, float, float, float, float, float, float, float, float, float]:
    """
    Возвращает:
      (freq,
       rms_main_raw, rms_exp, rms_total_raw, rms_self_raw,
       p_main_raw, p_exp, mre_rms_percent, err_p_percent,
       p_total_raw, rms_z, X0_best,
       rms_model_scaled_for_target, p_model_scaled_for_target)
    """
    print(f"[{mp.current_process().name}] Обработка частоты {freq} Гц")
    try:
        device = ElectromagneticHarvesterAbsolute()
        device.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)

        emf_csv = EXP_EMF_PATH_TEMPLATE.format(freq=freq)

        # 1) Загрузка ЭДС (эксп.)
        t_emf_s, emf_exp_v, emf_exp_interp = load_emf_no_time(
            emf_csv, EMF_COL_IDX, EMF_FS_HZ, sep=EMF_SEP, decimal=EMF_DECIMAL, units=EMF_UNITS, normalize_time=True
        )

        if len(t_emf_s) == 0:
            raise ValueError("Пустые данные ЭДС — нечего интерполировать и строить сетку.")

        emf_duration = t_emf_s[-1] - t_emf_s[0]
        if emf_duration < T_SLICE:
            print(f"[{mp.current_process().name}] [Warning] EMF duration {emf_duration:.3f} s < T_SLICE {T_SLICE} s")

        # 2) Сетка: последние T_SLICE секунд
        t_max = float(t_emf_s[-1])
        mask = t_emf_s >= (t_max - T_SLICE)
        t_window = t_emf_s[mask]

        # 3) Эксп. RMS на окне
        rms_exp = rms(emf_exp_interp(t_window), center=True)

        # 4) Автоподбор X0 по целевому сигналу
        X0_best, metrics_best_raw = find_best_amplitude(
            device=device,
            freq=freq,
            t_window=t_window,
            rms_exp=rms_exp,
            target_signal=TARGET_SIGNAL,
            optimize_by=OPTIMIZE_BY
        )

        # 5) Ошибки и метрики с учётом масштабирования
        if TARGET_SIGNAL == "main":
            rms_model_raw_for_err = metrics_best_raw["rms_main_emf"]
        else:
            rms_model_raw_for_err = metrics_best_raw["rms_total_emf"]

        rms_model_scaled_for_target = rms_model_raw_for_err * (SCALE_COEFF if APPLY_SCALE_IN_OPT else 1.0)
        mre_rms_percent = calculate_rms_error_percent_single(rms_model_scaled_for_target, rms_exp)

        p_exp = (rms_exp ** 2) / LOAD_R_OHM * 1000.0  # мВт
        p_model_scaled_for_target = ((rms_model_scaled_for_target ** 2) / LOAD_R_OHM) * 1000.0  # мВт
        err_p_percent = 100.0 * abs(p_model_scaled_for_target - p_exp) / (abs(p_exp) if abs(p_exp) > _EPS else _EPS)

        print(f"[{mp.current_process().name}] f={freq} Гц -> X0_best={X0_best:.6f} м ({X0_best*1000:.3f} мм), "
              f"RMS_exp={rms_exp:.6g} В, RMS_model_scaled={rms_model_scaled_for_target:.6g} В, MRE={mre_rms_percent:.3f}%")
        best_amplitudes.append(X0_best)
        return (freq,
                metrics_best_raw["rms_main_emf"],        # RAW
                rms_exp,
                metrics_best_raw["rms_total_emf"],       # RAW
                metrics_best_raw["rms_self_induction"],  # RAW
                metrics_best_raw["p_main_emf"],          # RAW
                p_exp,
                mre_rms_percent,
                err_p_percent,
                metrics_best_raw["p_total_emf"],         # RAW
                metrics_best_raw["rms_z"],
                X0_best,
                rms_model_scaled_for_target,
                p_model_scaled_for_target)

    except Exception as e:
        print(f"[{mp.current_process().name}] Ошибка на f={freq} Гц: {e}")
        return (freq, *(np.nan,)*12, np.nan, np.nan)

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
    (freqs_processed,
     rms_main_raw_values, rms_exp_values, rms_total_raw_values, rms_self_raw_values,
     p_main_raw_values, p_exp_values, mre_rms_values, err_p_values,
     p_total_raw_values, rms_z_values, x0_best_values,
     rms_model_scaled_values, p_model_scaled_values) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [])

    for result in results:
        (freq, rms_main_raw, rms_exp, rms_total_raw, rms_self_raw,
         p_main_raw, p_exp, mre_rms_percent, err_p_percent,
         p_total_raw, rms_z, x0_best, rms_model_scaled, p_model_scaled) = result

        freqs_processed.append(freq)
        rms_main_raw_values.append(rms_main_raw)
        rms_exp_values.append(rms_exp)
        rms_total_raw_values.append(rms_total_raw)
        rms_self_raw_values.append(rms_self_raw)
        p_main_raw_values.append(p_main_raw)
        p_exp_values.append(p_exp)
        mre_rms_values.append(mre_rms_percent)
        err_p_values.append(err_p_percent)
        p_total_raw_values.append(p_total_raw)
        rms_z_values.append(rms_z)
        x0_best_values.append(x0_best)
        rms_model_scaled_values.append(rms_model_scaled)
        p_model_scaled_values.append(p_model_scaled)

    # В массивы и сортировка
    freqs_processed = np.array(freqs_processed)
    sort_idx = np.argsort(freqs_processed)

    def _sort(a): return np.array(a)[sort_idx]

    freqs_processed          = freqs_processed[sort_idx]
    rms_main_raw_values      = _sort(rms_main_raw_values)
    rms_exp_values           = _sort(rms_exp_values)
    rms_total_raw_values     = _sort(rms_total_raw_values)
    rms_self_raw_values      = _sort(rms_self_raw_values)
    p_main_raw_values        = _sort(p_main_raw_values)
    p_exp_values             = _sort(p_exp_values)
    mre_rms_values           = _sort(mre_rms_values)
    err_p_values             = _sort(err_p_values)
    p_total_raw_values       = _sort(p_total_raw_values)
    rms_z_values             = _sort(rms_z_values)
    x0_best_values           = _sort(x0_best_values)
    rms_model_scaled_values  = _sort(rms_model_scaled_values)
    p_model_scaled_values    = _sort(p_model_scaled_values)

    # Рисунки и CSV
    plot_rms_path = plot_rms_vs_frequency(freqs_processed,
                                          rms_main_raw_values,
                                          rms_total_raw_values,
                                          rms_self_raw_values,
                                          rms_z_values, out_dir, ts)
    plot_self_induction_path = plot_self_induction_rms(freqs_processed, rms_self_raw_values, out_dir, ts)
    csv_path = save_rms_csv(out_dir, ts, freqs_processed,
                            rms_main_raw_values, rms_total_raw_values, rms_self_raw_values, rms_z_values, x0_best_values)

    # Таблица 5 (включая X0_best и метрики с учётом масштабирования)
    table5_df = pd.DataFrame({
        'Частота, Гц': freqs_processed,
        'X0_best, мм': x0_best_values * 1000.0,
        'V_RMSexp, мВ': rms_exp_values * 1000.0,
        'V_RMS модель (RAW основная), мВ': rms_main_raw_values * 1000.0,
        'V_RMS модель (RAW полная), мВ': rms_total_raw_values * 1000.0,
        'V_RMS модель (целевой, ×k), мВ': (rms_model_scaled_values * 1000.0) if APPLY_SCALE_IN_OPT else np.nan,
        'P_outexp, мВт': p_exp_values,
        'P_out модель (RAW целевой), мВт': (p_main_raw_values if TARGET_SIGNAL == "main" else p_total_raw_values),
        'P_out модель (целевой, ×k), мВт': p_model_scaled_values if APPLY_SCALE_IN_OPT else np.nan,
        'MRE по RMS (целевой, ×k), %': mre_rms_values,
        'Относ. ошибка по Pout (целевой, ×k), %': err_p_values
    })
    table5_path = os.path.join(out_dir, f"table5_{ts}.csv")
    table5_df.to_csv(table5_path, index=False)

    # Рисунок 7
    plot_power_path = plot_power_vs_frequency(freqs_processed, p_total_raw_values, p_main_raw_values, out_dir, ts)

    # Таблица 6 и пересимуляции резонансов — как было (используем RAW мощности)
    max_p_total = np.nanmax(p_total_raw_values)
    f_res_total = freqs_processed[np.nanargmax(p_total_raw_values)] if np.any(np.isfinite(p_total_raw_values)) else np.nan
    width_total = find_fwhm(freqs_processed, p_total_raw_values)

    if not np.isnan(f_res_total):
        try:
            device = ElectromagneticHarvesterAbsolute()
            device.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)
            t_res = np.linspace(0, T_SLICE, int(T_SLICE * EMF_FS_HZ))
            # для пересимуляции используем лучший X0 для этой частоты
            X0_use = float(x0_best_values[np.where(freqs_processed == f_res_total)][0])
            a_res = synthesize_base_accel(freq_hz=float(f_res_total), t_s=t_res, X0=X0_use)
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

    max_p_main = np.nanmax(p_main_raw_values)
    f_res_main = freqs_processed[np.nanargmax(p_main_raw_values)] if np.any(np.isfinite(p_main_raw_values)) else np.nan
    width_main = find_fwhm(freqs_processed, p_main_raw_values)

    if not np.isnan(f_res_main):
        try:
            device = ElectromagneticHarvesterAbsolute()
            device.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)
            t_res = np.linspace(0, T_SLICE, int(T_SLICE * EMF_FS_HZ))
            X0_use = float(x0_best_values[np.where(freqs_processed == f_res_main)][0])
            a_res = synthesize_base_accel(freq_hz=float(f_res_main), t_s=t_res, X0=X0_use)
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
    change_amp = 100 * (amp_total - amp_main) / amp_main if (amp_main is not None and amp_main != 0) else np.nan
    change_width = 100 * (width_total - width_main) / width_main if (width_main is not None and width_main != 0) else np.nan
    change_f_res = f_res_total - f_res_main if not (np.isnan(f_res_total) or np.isnan(f_res_main)) else np.nan

    table6_df = pd.DataFrame({
        'Показатель': [
            'Средняя мощность на нагрузке, мВт (RAW)',
            'Амплитуда ЭДС (Vp), В (RAW)',
            'Ширина резонансного пика по мощности, Гц (RAW)',
            'Максимум частоты резонанса, Гц (RAW)'
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
    print(f"Рисунок (RMS ЭДС и z, масштабированные): {plot_rms_path}")
    print(f"Рисунок (RMS самоиндукции, масштабированная): {plot_self_induction_path}")
    print(f"CSV с RMS RAW (включая z и X0_best): {csv_path}")
    print(f"Таблица 5 (вкл. масштабированные метрики): {table5_path}")
    print(f"Рисунок (Power vs freq, RAW): {plot_power_path}")
    print(f"Таблица 6 (RAW): {table6_path}")
    print(f"Обработано частот: {len(freqs_processed)}")
    print(best_amplitudes)
