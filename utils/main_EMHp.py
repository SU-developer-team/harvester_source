# -*- coding: utf-8 -*-
"""
Сценарий эксперимента с параллельной обработкой нескольких файлов:
- Загружаем акселерограмму шейкера (t, acc_mps2).
- Загружаем ЭДС БЕЗ столбца времени: берём столбец по индексу, время строим из частоты дискретизации.
- Интегрируем модель на объединённой сетке времён.
- Сравниваем напряжение на входе прибора (модель) с экспериментальной ЭДС.
- Опционально подбираем масштаб e(t) (emf_scale) по МНК и пересчитываем.
"""

import os
from datetime import datetime
from typing import Tuple, Optional, List
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from ElectromagneticHarvesterID80mm import ElectromagneticHarvesterID80mm


# =========================
# Конфигурация эксперимента
# =========================
T_MAX_S = 5.0              # максимум длительности моделирования
EMF_COL_IDX = 3            # ЭДС в 4-м столбце файла (индексация с 0)
EMF_FS_HZ = 1000.0         # Гц
EMF_UNITS = "mV"           # "V" | "mV" | "uV"
EMF_SEP = ";"
EMF_DECIMAL = ","          # числа с запятой
COIL_R_OHM = 1.0           # Ω — как померили
LOAD_R_OHM = 1e6           # Ω — вход осциллографа/АЦП
AUTO_FIT_EMF_SCALE = True

# Список экспериментов (пары файлов: акселерограмма + ЭДС)
EXPERIMENTS = [
    {
        "shaker_csv": r"experiments\harvester_80mm\exp_1\synthesized_data\22.csv",
        "emf_csv": r"experiments\harvester_80mm\exp_1\data\22.csv",
        "name": "exp_22"
    },
    # Добавьте другие эксперименты, например:
    # {
    #     "shaker_csv": r"experiments\harvester_80mm\exp_1\synthesized_data\23.csv",
    #     "emf_csv": r"experiments\harvester_80mm\exp_1\data\23.csv",
    #     "name": "exp_23"
    # },
]


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


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2)))


def plot_overlay(t_s: np.ndarray, model_v: np.ndarray, exp_v_on_grid: np.ndarray, save_dir: str, timestamp: str, exp_name: str) -> str:
    """Наложение модельного напряжения и экспериментальной ЭДС."""
    plt.figure(figsize=(11, 6))
    plt.plot(t_s, model_v, label="Модель: напряжение на входе прибора (В)", linewidth=1.5)
    plt.plot(t_s, exp_v_on_grid, label="Экспериментальная ЭДС (В)", alpha=0.85)
    plt.xlabel("Время, с"); plt.ylabel("ЭДС, В")
    plt.title(f"Индукционная ЭДС: модель vs эксперимент ({exp_name})")
    plt.grid(True); plt.legend(); plt.tight_layout()
    out = os.path.join(save_dir, f"emf_model_vs_experiment_{exp_name}_{timestamp}.png")
    plt.savefig(out); plt.close()
    return out


def save_npz(save_dir: str, timestamp: str, exp_name: str, **arrays) -> str:
    out = os.path.join(save_dir, f"harvester_data_{exp_name}_{timestamp}.npz")
    np.savez(out, **arrays)
    return out


def save_timeseries_csv(save_dir: str, timestamp: str, exp_name: str, t: np.ndarray, model_v: np.ndarray, exp_v_on_grid: np.ndarray) -> str:
    out = os.path.join(save_dir, f"timeseries_{exp_name}_{timestamp}.csv")
    df = pd.DataFrame({'t_s': t, 'v_model_V': model_v, 'v_exp_V': exp_v_on_grid})
    df.to_csv(out, index=False)
    return out


# =========================
# Функция для обработки одного эксперимента
# =========================
def process_experiment(exp: dict, out_dir: str, timestamp: str) -> dict:
    """Обработка одного эксперимента."""
    shaker_csv = exp["shaker_csv"]
    emf_csv = exp["emf_csv"]
    exp_name = exp["name"]

    print(f"[Эксперимент {exp_name}] Начало обработки...")

    # 1) Устройство + электрические параметры
    device = ElectromagneticHarvesterID80mm()
    device.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)

    # 2) База шейкера (t, acc_mps2)
    device.load_base_from_csv(
        shaker_csv,
        time_col="t", accel_col="acc_mps2",
        sep=";", decimal=".", normalize_time=True
    )

    # 3) Экспериментальная ЭДС без столбца времени
    t_emf_s, emf_exp_v, emf_exp_interp = load_emf_no_time(
        emf_csv,
        emf_col_idx=EMF_COL_IDX,
        fs_hz=EMF_FS_HZ,
        sep=EMF_SEP,
        decimal=EMF_DECIMAL,
        units=EMF_UNITS,
        normalize_time=True
    )

    # 4) Формируем временную сетку и обрезаем до T_MAX_S
    if device.base_time_s is not None:
        common_t_s = np.union1d(device.base_time_s, t_emf_s)
    else:
        common_t_s = t_emf_s

    t0 = float(common_t_s[0])
    mask = (common_t_s - t0) <= T_MAX_S
    common_t_s = common_t_s[mask]

    print(f"[Эксперимент {exp_name}] Сетка: {len(common_t_s)} точек")

    # 5) Решение модели
    t_s, z_m, v_mps, current_a, emf_open_v, emf_self_v, forces, v_term_v = device.solve_all(
        t_eval_s=common_t_s,
        rtol=1e-5, atol=1e-8,
        clamp_to_base=True
    )

    # 6) Интерполяция эксперимента на сетку модели
    v_exp_on_grid = emf_exp_interp(t_s)

    # 7) Автокалибровка масштаба ЭДС
    if AUTO_FIT_EMF_SCALE:
        valid = np.isfinite(v_term_v) & np.isfinite(v_exp_on_grid)
        if valid.any():
            num = float(np.dot(v_term_v[valid], v_exp_on_grid[valid]))
            den = float(np.dot(v_term_v[valid], v_term_v[valid])) or 1e-12
            k_fit = num / den
            device.emf_scale = k_fit
            print(f"[Эксперимент {exp_name}] Подобран масштаб ЭДС emf_scale ≈ {k_fit:.4g}")
            # Пересчёт с новым масштабом
            t_s, z_m, v_mps, current_a, emf_open_v, emf_self_v, forces, v_term_v = device.solve_all(
                t_eval_s=common_t_s,
                rtol=1e-5, atol=1e-8,
                clamp_to_base=True
            )
        else:
            print(f"[Эксперимент {exp_name}] Нет валидных точек для подбора масштаба — пропускаю.")

    # 8) RMS и графики
    finite = np.isfinite(v_exp_on_grid)
    if not np.all(finite):
        print(f"[Эксперимент {exp_name}] Предупреждение: {(~finite).sum()} нечисловых точек в эксперименте — исключаем из RMS")

    rms_model = rms(v_term_v)
    rms_exp = rms(v_exp_on_grid[finite]) if finite.any() else float("nan")

    overlay_path = plot_overlay(t_s, v_term_v, v_exp_on_grid, out_dir, timestamp, exp_name)

    plt.figure(figsize=(11, 5))
    plt.plot(t_s, z_m, label='z(t)')
    plt.xlabel('Время, с'); plt.ylabel('Положение, м')
    plt.title(f'Свободный магнит: z(t) ({exp_name})')
    plt.grid(True); plt.legend(); plt.tight_layout()
    z_path = os.path.join(out_dir, f"z_of_t_{exp_name}_{timestamp}.png")
    plt.savefig(z_path); plt.close()

    # Разложение сил
    device.plot_forces(t_s, forces, out_dir, f"{exp_name}_{timestamp}")

    # 9) Сохранения
    npz_path = save_npz(
        out_dir, timestamp, exp_name,
        t_s=t_s, z_m=z_m, v_mps=v_mps,
        current_a=current_a,
        emf_open_v=emf_open_v,
        emf_self_v=emf_self_v,
        v_term_v=v_term_v,
        v_exp_on_grid=v_exp_on_grid,
        forces=forces
    )
    csv_path = save_timeseries_csv(out_dir, timestamp, exp_name, t_s, v_term_v, v_exp_on_grid)

    # 10) Сводка
    summary = {
        "exp_name": exp_name,
        "out_dir": out_dir,
        "overlay_path": overlay_path,
        "z_path": z_path,
        "npz_path": npz_path,
        "csv_path": csv_path,
        "R_coil_ohm": COIL_R_OHM,
        "R_load_ohm": LOAD_R_OHM,
        "emf_scale": device.emf_scale,
        "rms_model": rms_model,
        "rms_exp": rms_exp
    }
    print(f"[Эксперимент {exp_name}] Завершено")
    return summary


# =========================
# Основной сценарий
# =========================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir, ts = create_experiment_folder(base_dir)

    # Параллельная обработка экспериментов
    num_processes = min(mp.cpu_count(), len(EXPERIMENTS))  # Не больше CPU или экспериментов
    print(f"Запуск {len(EXPERIMENTS)} экспериментов на {num_processes} процессах...")

    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            process_experiment,
            [(exp, out_dir, ts) for exp in EXPERIMENTS]
        )

    # Сводка по всем экспериментам
    print("\n==== Итоговая сводка ====")
    for res in results:
        print(f"\nЭксперимент: {res['exp_name']}")
        print(f"Папка эксперимента: {res['out_dir']}")
        print(f"График наложения:   {res['overlay_path']}")
        print(f"График z(t):        {res['z_path']}")
        print(f"NPZ с массивами:    {res['npz_path']}")
        print(f"CSV рядов:          {res['csv_path']}")
        print(f"R катушки, Ом:      {res['R_coil_ohm']}")
        print(f"R нагрузки, Ом:     {res['R_load_ohm']}")
        print(f"ЭДС scale:          {res['emf_scale']:.6g}")
        print(f"RMS (модель, В):    {res['rms_model']:.6f}")
        print(f"RMS (эксп., В):     {res['rms_exp']:.6f}")