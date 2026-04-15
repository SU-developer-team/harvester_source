# -*- coding: utf-8 -*-
"""
График max(center_magnet_pos) по частоте для ElectromagneticHarvesterID80mm.

- Берём сетку частот FREQS (и опционально фильтруем по RANGE_SELECT)
- Для каждой частоты считаем модель (параллельно)
- Из временного ряда положения центра магнита (z) берём:
    * z_max_pos_m       = max(z)             [м]
    * z_max_abs_m       = max(|z|)           [м]  (на всякий случай)
- Строим график частота -> max(z) (в миллиметрах)
- Сохраняем CSV с результатами
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp

from model import ElectromagneticHarvesterID80mm

# ====== ПУТИ ======
# Базовая папка проекта — рядом со скриптом
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
# Папка, куда будут складываться графики и CSV
RESULTS_BASE_DIR = BASE_DIR

# =========================
# Конфигурация эксперимента (модель)
# =========================
T_MAX_S = 5.0          # длительность моделирования, с
MODEL_FS_HZ = 1000.0   # шаг дискретизации модели, Гц

# Электрика
COIL_R_OHM = 1.0
LOAD_R_OHM = 1.0

# Базовая сетка частот модели
FREQS = np.arange(3.0, 22.0, 0.1)

# >>> НОВОЕ: выбор диапазонов частот (в Гц)
# Если нужно все частоты, поставь RANGE_SELECT = None
RANGE_SELECT: Optional[List[Tuple[float, float]]] = [(7.0, 12.0)]

# Параллелизм
N_PROCESSES = None  # если None — возьмём os.cpu_count()


# =========================
# Хелперы
# =========================
def create_experiment_folder(base_dir: Path) -> Tuple[Path, str]:
    """
    Создаёт папку:
        base_dir / "graphs_center_pos" / "experiment_YYYY-MM-DD_HH-MM-SS"
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = base_dir / "graphs_center_pos" / f"experiment_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, ts


def save_npz(save_dir: Path, timestamp: str, **arrays) -> Path:
    out = save_dir / f"harvester_center_pos_{timestamp}.npz"
    np.savez(out, **arrays)
    return out


def _in_ranges(x: float, ranges: Optional[Iterable[Tuple[float, float]]]) -> bool:
    """Проверка, попадает ли x в любой из интервалов ranges (включительно)."""
    if not ranges:
        return True
    for lo, hi in ranges:
        if lo <= x <= hi:
            return True
    return False


# =========================
# Рабочая функция (для параллельного процесса)
# =========================
def run_single_frequency(args: Tuple[float, Path, str]) -> Dict[str, Any]:
    """
    Запускает модель для одной частоты и возвращает максимальное смещение центра магнита.
    """
    freq, out_dir, ts = args
    try:
        device = ElectromagneticHarvesterID80mm()
        device.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)
        device.set_frequency(freq)

        steps = int(T_MAX_S * MODEL_FS_HZ)
        t_eval = np.linspace(0.0, T_MAX_S, steps, endpoint=False)

        # t_s, z, v, i, emf_open_v, emf_self_v, forces, v_term_v
        t_s, z, v, i, emf_open_v, emf_self_v, forces, v_term_v = device.solve_all(
            t_eval_s=t_eval, rtol=1e-5, atol=1e-8, clamp_to_base=False, dense_output=False
        )

        z = np.asarray(z, dtype=float)
        if z.size == 0 or not np.all(np.isfinite(z)):
            raise ValueError("Неверные данные z (NaN/Inf или пустой массив)")

        # max(center_magnet_pos) — предполагаем, что это z
        z_max_pos_m = float(np.max(z))          # максимум по z, м
        z_max_abs_m = float(np.max(np.abs(z)))  # максимум по модулю, м (на всякий случай)

        # Сохраним тайм-серию для отладки (опционально)
        freq_out_dir = out_dir / f"freq_{str(freq).replace('.', '_')}"
        freq_out_dir.mkdir(parents=True, exist_ok=True)

        npz_path = save_npz(
            freq_out_dir, f"{ts}_{freq}",
            t_s=t_s,
            z_m=z,
            v_mps=v,
            current_a=i,
            emf_open_v=emf_open_v,
            emf_self_v=emf_self_v,
            v_term_v=v_term_v,
            forces=forces,
        )

        return {
            "freq": float(freq),
            "z_max_pos_m": z_max_pos_m,
            "z_max_abs_m": z_max_abs_m,
            "npz": npz_path,
            "ok": True,
            "error": None,
        }
    except Exception as e:
        return {
            "freq": float(freq),
            "z_max_pos_m": np.nan,
            "z_max_abs_m": np.nan,
            "npz": None,
            "ok": False,
            "error": repr(e),
        }


# =========================
# Основной сценарий
# =========================
def main():
    # Настройка шрифта и качества графиков
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 12

    out_dir, ts = create_experiment_folder(RESULTS_BASE_DIR)

    # 1) Список частот + фильтр по диапазону (если задан)
    freqs_base = sorted(set(map(float, FREQS)))

    if RANGE_SELECT:
        freqs_use = [f for f in freqs_base if _in_ranges(f, RANGE_SELECT)]
    else:
        freqs_use = freqs_base

    if not freqs_use:
        raise ValueError("После применения диапазонов частот список пуст. Измени RANGE_SELECT.")

    print(f"Папка эксперимента: {out_dir}")
    print(f"Диапазоны: {RANGE_SELECT if RANGE_SELECT else 'все частоты'}")
    print(f"Частоты к расчёту (параллельно): {freqs_use} Гц")

    # 2) Параллельный расчёт
    jobs = [(f, out_dir, ts) for f in freqs_use]
    processes = N_PROCESSES or os.cpu_count() or 1
    print(f"Используем процессов: {processes}")

    results: List[Dict[str, Any]] = []
    with mp.get_context("spawn").Pool(processes=processes) as pool:
        for res in pool.imap(run_single_frequency, jobs, chunksize=2):
            if res["ok"]:
                print(
                    f"✓ {res['freq']} Гц: "
                    f"z_max_pos = {res['z_max_pos_m']*1000:.3f} мм; "
                    f"z_max_abs = {res['z_max_abs_m']*1000:.3f} мм"
                )
            else:
                print(f"✗ {res['freq']} Гц: Ошибка: {res['error']}")
            results.append(res)

    # 3) Собираем данные в массивы
    model_ok = [
        (float(r["freq"]), float(r["z_max_pos_m"]), float(r["z_max_abs_m"]))
        for r in results
        if r["ok"] and np.isfinite(r["z_max_pos_m"])
    ]
    model_ok.sort(key=lambda x: x[0])

    if not model_ok:
        raise RuntimeError("Нет корректных точек для построения графика.")

    freqs_arr = np.array([p[0] for p in model_ok], dtype=float)
    z_max_pos_m = np.array([p[1] for p in model_ok], dtype=float)
    z_max_abs_m = np.array([p[2] for p in model_ok], dtype=float)

    # Для графика используем миллиметры
    z_max_pos_mm = z_max_pos_m * 1000.0
    z_max_abs_mm = z_max_abs_m * 1000.0

    # 4) Сохраняем CSV
    summary_csv = out_dir / f"{ts}_center_magnet_pos_vs_freq.csv"
    df = pd.DataFrame(
        {
            "freq_Hz": freqs_arr,
            "z_max_pos_m": z_max_pos_m,
            "z_max_pos_mm": z_max_pos_mm,
            "z_max_abs_m": z_max_abs_m,
            "z_max_abs_mm": z_max_abs_mm,
        }
    )
    df.to_csv(summary_csv, index=False)
    print(f"CSV с max(center_magnet_pos) сохранён: {summary_csv}")

    # 5) График max(center_magnet_pos) vs freq
    plt.figure(figsize=(5, 3))
    plt.plot(freqs_arr, z_max_pos_mm, marker="o", linestyle="-", label="z, мм")
    # plt.plot(freqs_arr, z_max_abs_mm, marker="s", linestyle="--", label="max(|center_magnet_pos|), мм")

    plt.xlabel("frequency, Hz")
    plt.ylabel("z, mm")
    plt.title("")
    plt.grid(True)
    plt.legend()

    plot_path = out_dir / f"center_magnet_pos_vs_freq.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=600)
    plt.show()
    print(f"График max(center_magnet_pos) сохранён: {plot_path}")

    print("\n==== Сводка ====")
    print(f"Папка эксперимента: {out_dir}")
    print(f"Диапазоны: {RANGE_SELECT if RANGE_SELECT else 'все частоты'}")
    print(f"Точек модели: {len(freqs_arr)}")


if __name__ == "__main__":
    main()
