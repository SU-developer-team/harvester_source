# -*- coding: utf-8 -*-
"""
Модель + сравнение с экспериментом + выбор диапазонов частот.
- Загружаем компактный CSV эксперимента (freq,rms_emf_mV ИЛИ Frequency (Hz),RMS (V))
- Объединяем частоты модели с частотами эксперимента
- Фильтруем всё по заданным диапазонам RANGE_SELECT (например, 7–12 Гц)
- Считаем модель параллельно
- Строим один график (min-max нормировка) и сохраняем CSV
- <<< NEW >>>: В summary CSV пишем и RMS самоиндукции; строим отдельные графики по самоиндукции
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import multiprocessing as mp

from model import ElectromagneticHarvesterID80mm

# ====== ПУТИ ======
exp_data = r'experiments\harvester_80mm\rms_compact_freq_rms.csv'

# =========================
# Конфигурация эксперимента (модель)
# =========================
T_MAX_S = 5.0
MODEL_FS_HZ = 1000.0

# Электрика
COIL_R_OHM = 1.0
LOAD_R_OHM = 1.0

# Базовая сетка частот модели (будет расширена частотами из эксперимента)
FREQS = np.arange(3, 22, 0.5)

# >>> НОВОЕ: выбор диапазонов частот (в Гц)
RANGE_SELECT: Optional[List[Tuple[float, float]]] = [(7.0, 12.0)]

# Параллелизм
N_PROCESSES = None  # os.cpu_count() if None


# =========================
# Хелперы
# =========================
def create_experiment_folder(base_dir: Path) -> Tuple[Path, str]:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = base_dir / "graphs" / f"experiment_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, ts


def rms(x: np.ndarray, center: bool = True) -> float:
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("RMS: данные содержат NaN/Inf")
    if center:
        x = x - np.mean(x)
    return float(np.sqrt(np.mean(x**2)))


def save_npz(save_dir: Path, timestamp: str, **arrays) -> Path:
    out = save_dir / f"harvester_data_{timestamp}.npz"
    np.savez(out, **arrays)
    return out


def save_timeseries_csv(save_dir: Path, timestamp: str, t: np.ndarray, model_v: np.ndarray) -> Path:
    out = save_dir / f"timeseries_{timestamp}.csv"
    df = pd.DataFrame({'t_s': t, 'v_model_V': model_v})
    df.to_csv(out, index=False)
    return out


def _parse_freq_value(val: str) -> Optional[float]:
    """
    Парсер частоты:
      '9,2' -> 9.2
      '9.1-9.2' или '9,1-9,2' -> середина диапазона (9.15)
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    s = s.replace(',', '.')
    if '-' in s:
        try:
            a, b = s.split('-', 1)
            a = float(a.strip()); b = float(b.strip())
            return (a + b) / 2.0
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def load_experiment_compact(path: str) -> Optional[pd.DataFrame]:
    """
    Загружает экспериментальный CSV.
    Поддержка:
      1) freq, rms_emf_mV
      2) Frequency (Hz), RMS (V)
    Возвращает DataFrame: ['freq_Hz','rms_V'] (усредняет дубли частот).
    """
    if not os.path.exists(path):
        print(f"[INFO] Файл экспериментальных данных не найден: {path}")
        return None

    df = pd.read_csv(path)
    if df.empty:
        print(f"[WARN] Экспериментальный файл пуст: {path}")
        return None

    cols = {c.strip().lower(): c for c in df.columns}
    freq_col = None
    rms_col = None
    scale_to_volts = 1.0

    if 'freq' in cols and 'rms_emf_mv' in cols:
        freq_col = cols['freq']
        rms_col = cols['rms_emf_mv']
        scale_to_volts = 1e-3
    elif 'frequency (hz)' in cols and 'rms (v)' in cols:
        freq_col = cols['frequency (hz)']
        rms_col = cols['rms (v)']
        scale_to_volts = 1.0
    else:
        possible_freq = [c for c in df.columns if 'freq' in c.lower()]
        possible_rms = [c for c in df.columns if 'rms' in c.lower()]
        if not possible_freq or not possible_rms:
            print("[ERROR] Не удалось определить столбцы частоты/амплитуды.")
            return None
        freq_col = possible_freq[0]
        rms_col = possible_rms[0]
        scale_to_volts = 1e-3 if 'mv' in rms_col.lower() else 1.0

    freqs_parsed = df[freq_col].apply(_parse_freq_value)
    good = freqs_parsed.notna() & df[rms_col].notna()
    df = df.loc[good].copy()

    df['freq_Hz'] = freqs_parsed.loc[good].astype(float)
    df['rms_V'] = pd.to_numeric(df[rms_col].astype(str).str.replace(',', '.'), errors='coerce') * scale_to_volts
    df = df[['freq_Hz', 'rms_V']].dropna()
    df = df.groupby('freq_Hz', as_index=False)['rms_V'].mean().sort_values('freq_Hz')
    return df if not df.empty else None


def minmax_norm(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    y_min = np.min(y); y_max = np.max(y)
    if not np.isfinite(y_min) or not np.isfinite(y_max) or np.isclose(y_max, y_min):
        return np.zeros_like(y)
    return (y - y_min) / (y_max - y_min)


def _in_ranges(x: float, ranges: Optional[Iterable[Tuple[float, float]]]) -> bool:
    """Проверка, попадает ли x в любой из интервалов ranges (включительно)."""
    if not ranges:
        return True
    for lo, hi in ranges:
        if lo <= x <= hi:
            return True
    return False


def _filter_df_by_ranges(df: pd.DataFrame, ranges: Optional[List[Tuple[float, float]]]) -> pd.DataFrame:
    if df is None or ranges is None:
        return df
    mask = np.zeros(len(df), dtype=bool)
    for lo, hi in ranges:
        mask |= (df['freq_Hz'] >= lo) & (df['freq_Hz'] <= hi)
    return df.loc[mask].reset_index(drop=True)


# =========================
# Рабочая функция (в процессе)
# =========================
def run_single_frequency(args: Tuple[float, Path, str]) -> Dict[str, Any]:
    freq, out_dir, ts = args
    try:
        device = ElectromagneticHarvesterID80mm()
        device.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)
        device.set_frequency(freq)

        steps = int(T_MAX_S * MODEL_FS_HZ)
        t_eval = np.linspace(0.0, T_MAX_S, steps, endpoint=False)

        t_s, z, v, i, emf_open_v, emf_self_v, forces, v_term_v = device.solve_all(
            t_eval_s=t_eval, rtol=1e-5, atol=1e-8, clamp_to_base=False, dense_output=False
        )

        rms_model = rms(v_term_v)
        rms_self  = rms(emf_self_v)  # <<< NEW >>> RMS ЭДС самоиндукции

        freq_out_dir = out_dir / f"freq_{str(freq).replace('.', '_')}"
        freq_out_dir.mkdir(parents=True, exist_ok=True)

        npz_path = save_npz(
            freq_out_dir, f"{ts}_{freq}",
            t_s=t_s, z_m=z, v_mps=v, current_a=i,
            emf_open_v=emf_open_v, emf_self_v=emf_self_v,
            v_term_v=v_term_v, forces=forces
        )
        csv_path = save_timeseries_csv(freq_out_dir, f"{ts}_{freq}", t_s, v_term_v)

        return {
            "freq": float(freq),
            "rms": rms_model,
            "rms_self": rms_self,   # <<< NEW >>>
            "npz": npz_path,
            "csv": csv_path,
            "ok": True,
            "error": None
        }
    except Exception as e:
        return {"freq": float(freq), "rms": np.nan, "rms_self": np.nan, "npz": None, "csv": None, "ok": False, "error": repr(e)}


# =========================
# Основной сценарий
# =========================
def main():
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    out_dir, ts = create_experiment_folder(base_dir)

    # 1) Загружаем эксперимент и фильтруем по диапазонам (если заданы)
    exp_df = load_experiment_compact(exp_data)
    if exp_df is not None and RANGE_SELECT:
        exp_df = _filter_df_by_ranges(exp_df, RANGE_SELECT)
        if exp_df is not None and exp_df.empty:
            exp_df = None  # после фильтра ничего не осталось

    freqs_base = list(map(float, FREQS))

    # 2) Формируем объединённый список частот (модель + эксперимент), затем фильтруем по диапазонам
    if exp_df is not None and not exp_df.empty:
        exp_freqs = exp_df['freq_Hz'].tolist()
        freqs_union = sorted(set([round(f, 5) for f in (freqs_base + exp_freqs)]))
    else:
        freqs_union = sorted(set([round(f, 5) for f in freqs_base]))

    # применим фильтр диапазонов к union (если задан)
    if RANGE_SELECT:
        freqs_union = [f for f in freqs_union if _in_ranges(f, RANGE_SELECT)]

    if len(freqs_union) == 0:
        raise ValueError("После применения диапазонов список частот пуст. Измени RANGE_SELECT.")

    print(f"Папка эксперимента: {out_dir}")
    print(f"Диапазоны: {RANGE_SELECT if RANGE_SELECT else 'все частоты'}")
    print(f"Частоты к расчёту (параллельно): {freqs_union} Гц")

    # 3) Параллельный расчёт
    jobs = [(f, out_dir, ts) for f in freqs_union]
    processes = N_PROCESSES or os.cpu_count() or 1
    print(f"Используем процессов: {processes}")

    results: List[Dict[str, Any]] = []
    with mp.get_context("spawn").Pool(processes=processes) as pool:
        for res in pool.imap(run_single_frequency, jobs, chunksize=2):
            if res["ok"]:
                print(f"✓ {res['freq']} Гц: RMS = {res['rms']:.6f} В; RMS_self = {res['rms_self']:.6f} В")
            else:
                print(f"✗ {res['freq']} Гц: Ошибка: {res['error']}")
            results.append(res)

    # 4) CSV модели (сырые значения без нормировки)
    summary_csv = out_dir / f"{ts}_results_model_raw.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # <<< NEW >>> доп. колонка RMS самоиндукции:
        writer.writerow(["freq_Hz", "model_rms_V", "model_self_rms_V"])
        freq_to_rms = {float(r["freq"]): (r["rms"], r.get("rms_self", np.nan)) for r in results}
        for f in freqs_union:
            rms_main, rms_self = freq_to_rms.get(float(f), (np.nan, np.nan))
            writer.writerow([f, rms_main, rms_self])
    print(f"Сводный CSV модели сохранён: {summary_csv}")

    # 5) Данные для графика (min-max)
    model_ok = [(float(r["freq"]), float(r["rms"]), float(r.get("rms_self", np.nan)))
                for r in results if r["ok"] and np.isfinite(r["rms"])]
    model_ok.sort(key=lambda x: x[0])
    model_freqs      = np.array([p[0] for p in model_ok], dtype=float)
    model_rms        = np.array([p[1] for p in model_ok], dtype=float)  # исходная поправка
    model_self_rms   = np.array([p[2] for p in model_ok], dtype=float)  #

    if exp_df is not None and not exp_df.empty:
        exp_freqs_arr = exp_df['freq_Hz'].to_numpy(dtype=float)
        exp_rms_arr   = exp_df['rms_V'].to_numpy(dtype=float)    # ваш текущий масштаб
    else:
        exp_freqs_arr = np.array([], dtype=float)
        exp_rms_arr   = np.array([], dtype=float)

    model_norm = minmax_norm(model_rms)
    exp_norm   = minmax_norm(exp_rms_arr)
    self_norm  = minmax_norm(model_self_rms)  # <<< NEW >>>

    # --- Основной график модель vs эксперимент (min-max) ---
    plt.figure(figsize=(10, 6))
    if model_freqs.size:
        plt.plot(model_freqs, model_norm, marker='o', linestyle='-', label='Модель (RMS, min-max)')
    if exp_freqs_arr.size:
        plt.plot(exp_freqs_arr, exp_norm, marker='o', linestyle='-', label='Эксперимент (RMS, min-max)')
    plt.xlabel('Частота, Гц')
    plt.ylabel('Нормированный RMS (0..1)')
    plt.title('Сравнение модели и эксперимента (min-max, с учётом диапазонов)')
    plt.grid(True)
    plt.legend()
    plot_path = out_dir / f"rms_vs_freq_minmax_{ts}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    print(f"График (min-max) сохранён: {plot_path}")

    # --- <<< NEW >>> Отдельный вывод самоиндукции ---
    # (а) raw, В
    if model_freqs.size:
        plt.figure(figsize=(10, 6))
        plt.plot(model_freqs, model_self_rms, marker='s', linestyle='-', label='ЭДС самоиндукции: RMS (В)')
        plt.xlabel('Частота, Гц')
        plt.ylabel('RMS, В')
        plt.title('RMS ЭДС самоиндукции (raw)')
        plt.grid(True)
        plt.legend()
        self_raw_path = out_dir / f"self_induction_rms_{ts}.png"
        plt.tight_layout()
        plt.savefig(self_raw_path)
        plt.show()
        print(f"График самоиндукции (raw) сохранён: {self_raw_path}")

        # (б) min–max
        plt.figure(figsize=(10, 6))
        plt.plot(model_freqs, self_norm, marker='o', linestyle='-', label='ЭДС самоиндукции: RMS (min–max)')
        plt.xlabel('Частота, Гц')
        plt.ylabel('Нормированный RMS (0..1)')
        plt.title('ЭДС самоиндукции (min–max)')
        plt.grid(True)
        plt.legend()
        self_mm_path = out_dir / f"self_induction_minmax_{ts}.png"
        plt.tight_layout()
        plt.savefig(self_mm_path)
        plt.show()
        print(f"График самоиндукции (min–max) сохранён: {self_mm_path}")

    # 6) Объединённый CSV с нормировкой (для быстрого сравнения)
    merged_csv = out_dir / f"{ts}_model_experiment_minmax.csv"
    merged_df = pd.DataFrame({
        "freq_model_Hz": model_freqs,
        "model_rms_V": model_rms,
        "model_rms_minmax": model_norm,
        "model_self_rms_V": model_self_rms,          # <<< NEW >>>
        "model_self_rms_minmax": self_norm           # <<< NEW >>>
    })
    if exp_freqs_arr.size:
        exp_df_out = pd.DataFrame({
            "freq_exp_Hz": exp_freqs_arr,
            "exp_rms_V": exp_rms_arr,
            "exp_rms_minmax": exp_norm
        })
        max_len = max(len(merged_df), len(exp_df_out))
        merged_df = merged_df.reindex(range(max_len)).reset_index(drop=True)
        exp_df_out = exp_df_out.reindex(range(max_len)).reset_index(drop=True)
        out_block = pd.concat([merged_df, exp_df_out], axis=1)
    else:
        out_block = merged_df
    out_block.to_csv(merged_csv, index=False)
    print(f"Объединённый CSV (min-max) сохранён: {merged_csv}")

    print("\n==== Сводка ====")
    print(f"Папка эксперимента: {out_dir}")
    print(f"Диапазоны: {RANGE_SELECT if RANGE_SELECT else 'все частоты'}")
    print(f"Точек модели: {len(model_freqs)}, точек эксперимента: {len(exp_freqs_arr)}")


if __name__ == "__main__":
    main()
