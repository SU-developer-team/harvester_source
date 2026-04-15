# -*- coding: utf-8 -*-
r"""
Сценарий для обработки всех файлов в experiments\harvester_50mm\exp_2.
- Загружаем акселерограмму шейкера (t, acc_mps2).
- Загружаем ЭДС, пробуя сначала со столбцом времени, затем без.
- Интегрируем модель с оптимизацией параметров conf_1, conf_2, conf_3.
- Минимизируем MSE между RMS модельной и экспериментальной ЭДС.
- Сохраняем результаты и логируем проблемные файлы.
"""

import os
from datetime import datetime
from typing import Tuple, Optional, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import csv
from ElectromagneticHarvesterID80mm import ElectromagneticHarvesterID80mm as ElectromagneticHarvesterID50mm

# =========================
# Пути к данным
# =========================
SHAKER_PATH = r"experiments\harvester_50mm\exp_2\synthesized_data"
EMF_PATH = r"experiments\harvester_50mm\exp_2\data"
SHAKER_FILES = [os.path.join(SHAKER_PATH, f"{i}.csv") for i in range(2, 35)]
EMF_FILES = [os.path.join(EMF_PATH, f"{i}.csv") for i in range(2, 35)]
FREQUENCIES = np.arange(2, 35, 1)

# =========================
# Конфигурация эксперимента
# =========================
T_MAX_S = 5.0              # максимальная длительность моделирования
T_SLICE = 1.0              # ограничение для анализа (1 секунда)

# ЭДС: параметры
EMF_COL_IDX = 3             # ЭДС в 4-м столбце (индексация с 0)
EMF_FS_HZ = 1000.0          # частота дискретизации, Гц
EMF_UNITS = "mV"            # "V" | "mV" | "uV"
EMF_SEP = ";"               # разделитель в файлах ЭДС
EMF_DECIMAL = ","           # десятичный разделитель в файлах ЭДС
TIME_COL = "t"              # имя столбца времени (если есть)

# Электрика
COIL_R_OHM = 1.0            # сопротивление катушки, Ом
LOAD_R_OHM = 1e6            # сопротивление нагрузки, Ом
L_INDUCTANCE = 1e-3         # индуктивность катушки, Гн

# Начальные параметры для оптимизации
INITIAL_CONF = [1.0, 0.94, 0.001]  # [conf_1, conf_2, conf_3]
BOUNDS = [(0.5, 1.5), (0.5, 1.5), (0.0001, 0.01)]  # Ограничения

# Автокалибровка масштаба ЭДС
AUTO_FIT_EMF_SCALE = True

# =========================
# Хелперы
# =========================
def create_experiment_folder(base_dir: str) -> Tuple[str, str]:
    """Создать папку для графиков/файлов с таймстампом."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(base_dir, "graphs", f"experiment_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, ts

def _to_numeric_series(raw, decimal_hint: Optional[str], file_path: str, column_name: str) -> pd.Series:
    """Надёжное приведение строк с запятой/пробелами к float. Мусор -> NaN."""
    s = pd.Series(raw, copy=True).astype(str)
    s = s.str.strip()  # Удаляем пробелы и переносы строк
    s = s.str.replace('\u00A0', '', regex=False)  # Неразрывный пробел
    s = s.str.replace(r'\s+', '', regex=True)  # Все пробелы
    s = s.str.replace(',', '.', regex=False)  # Заменяем запятую на точку
    s = pd.to_numeric(s, errors='coerce')
    if s.isna().any():
        nan_indices = s[s.isna()].index.tolist()
        nan_values = raw[s.isna()].tolist()
        print(f"В файле {file_path}, столбце {column_name} найдены нечисловые значения или NaN на строках: {nan_indices[:10]}")
        print(f"Проблемные значения (первые 10): {nan_values[:10]}")
    return s

def check_file_structure(file_path: str, expected_cols: int, sep: str = ";") -> bool:
    """Проверка структуры CSV-файла."""
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            first_line = f.readline().strip()
            num_cols = len(first_line.split(sep))
            if num_cols < expected_cols:
                print(f"Ошибка: Файл {file_path} содержит {num_cols} столбцов, ожидается минимум {expected_cols}")
                return False
            return True
    except Exception as e:
        print(f"Ошибка проверки структуры файла {file_path}: {str(e)}")
        return False

def load_shaker_data(
    file_path: str,
    time_col: str = "t",
    accel_col: str = "acc_mps2",
    sep: str = ";",
    decimal: str = ".",
    normalize_time: bool = True
) -> Tuple[np.ndarray, np.ndarray, interp1d]:
    """Загрузить акселерограмму шейкера из CSV и построить интерполяцию."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
    
    if not check_file_structure(file_path, expected_cols=3, sep=sep):
        raise ValueError(f"Неверная структура файла {file_path}")
    
    try:
        df = pd.read_csv(file_path, sep=sep, decimal=decimal, header=0, encoding='utf-8-sig')
    except Exception as e:
        raise ValueError(f"Ошибка чтения файла {file_path}: {str(e)}")
    
    if time_col not in df or accel_col not in df:
        raise ValueError(f"В {file_path} должны быть колонки '{time_col}', '{accel_col}'")
    
    if df.empty:
        raise ValueError(f"Файл {file_path} пуст")
    
    df[time_col] = _to_numeric_series(df[time_col], decimal, file_path, time_col)
    df[accel_col] = _to_numeric_series(df[accel_col], decimal, file_path, accel_col)
    
    df = df.dropna(subset=[time_col, accel_col])
    
    if df.empty:
        raise ValueError(f"Файл {file_path} пуст после удаления строк с NaN")
    
    df = df.groupby(time_col, as_index=False).mean().sort_values(time_col)
    
    t_s = df[time_col].to_numpy(dtype=float)
    a_mps2 = df[accel_col].to_numpy(dtype=float)
    
    if normalize_time:
        t_s -= t_s[0]
    
    mask = t_s <= T_MAX_S
    t_s, a_mps2 = t_s[mask], a_mps2[mask]
    
    if np.any(~np.isfinite(a_mps2)) or np.any(~np.isfinite(t_s)):
        raise ValueError(f"В данных {file_path} есть NaN или нечисловые значения")
    if len(t_s) < 2:
        raise ValueError(f"В файле {file_path} недостаточно данных для интерполяции")
    
    if not np.all(np.diff(t_s) > 0):
        raise ValueError(f"Временной столбец в {file_path} не монотонно возрастает")
    
    accel_interp = interp1d(t_s, a_mps2, kind="linear", bounds_error=False,
                           fill_value=(a_mps2[0], a_mps2[-1]))
    
    print(f"[Shaker {os.path.basename(file_path)}] {len(t_s)} точек, длит. ~{t_s[-1]:.3f} с, "
          f"диапазон {np.min(a_mps2):.3f}..{np.max(a_mps2):.3f} м/с²")
    return t_s, a_mps2, accel_interp

def load_emf_data(
    file_path: str,
    emf_col_idx: int = 3,
    fs_hz: float = 1000.0,
    sep: str = ";",
    decimal: str = ",",
    units: str = "mV",
    normalize_time: bool = True,
    time_col: str = "t"
) -> Tuple[np.ndarray, np.ndarray, interp1d]:
    """
    Читает ЭДС из CSV, пробуя сначала со столбцом времени, затем без.
    Возвращает (t_s, emf_V, интерполяция emf(t)).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
    
    if not check_file_structure(file_path, expected_cols=4, sep=sep):
        raise ValueError(f"Неверная структура файла {file_path}")
    
    try:
        df = pd.read_csv(file_path, sep=sep, decimal=decimal, header=None, encoding='utf-8-sig', dtype=str)
    except Exception as e:
        print(f"Ошибка при чтении с utf-8-sig: {str(e)}. Пробуем latin1.")
        df = pd.read_csv(file_path, sep=sep, decimal=decimal, header=None, encoding='latin1', dtype=str)
    
    if df.empty:
        raise ValueError(f"Файл {file_path} пуст")
    
    num_cols = df.shape[1]
    if num_cols < 2:
        raise ValueError(f"Файл {file_path} должен содержать минимум 2 столбца")
    
    try:
        df.columns = [time_col, "unknown1", "unknown2", "emf"] if num_cols == 4 else \
                     [time_col, "unknown1", "unknown2", "unknown3", "emf"]
        
        df[time_col] = _to_numeric_series(df[time_col], decimal, file_path, time_col)
        df["emf"] = _to_numeric_series(df["emf"], decimal, file_path, "emf")
        
        if df["emf"].isna().all():
            raise ValueError(f"Столбец emf в {file_path} содержит только нечисловые значения")
        
        df = df.dropna(subset=[time_col, "emf"])
        
        if df.empty:
            raise ValueError(f"Файл {file_path} пуст после удаления строк с NaN")
        
        t_s = df[time_col].to_numpy(dtype=float)
        emf = df["emf"].to_numpy(dtype=float)
        
        if normalize_time and len(t_s) > 0:
            t_s -= t_s[0]
        
    except Exception as e:
        print(f"Не удалось загрузить столбец времени из {file_path}: {str(e)}. Пробуем без времени.")
        if not (0 <= emf_col_idx < num_cols):
            raise ValueError(f"В файле {file_path} нет столбца с индексом {emf_col_idx}")
        
        emf_raw = df.iloc[:, emf_col_idx]
        emf = _to_numeric_series(emf_raw, decimal, file_path, "emf").to_numpy(dtype=float)
        
        if np.all(np.isnan(emf)):
            raise ValueError(f"Столбец emf в {file_path} содержит только нечисловые значения")
        
        scale = {"V": 1.0, "mV": 1e-3, "uV": 1e-6}[units]
        emf_v = emf * scale
        
        n = len(emf_v)
        t_s = np.arange(n, dtype=float) / float(fs_hz)
        if normalize_time and n > 0:
            t_s -= t_s[0]
        
        bad = ~np.isfinite(emf_v)
        if bad.any():
            s = pd.Series(emf_v, index=t_s)
            s = s.interpolate(method="linear", limit_direction="both")
            if s.isna().sum() > 0:
                raise ValueError(f"В ЭДС {file_path} после интерполяции остались NaN")
            emf_v = s.to_numpy()
        
        t_s, emf = t_s, emf_v
    
    mask = t_s <= T_MAX_S
    t_s, emf = t_s[mask], emf[mask]
    
    if np.any(~np.isfinite(emf)) or np.any(~np.isfinite(t_s)):
        raise ValueError(f"В данных {file_path} есть NaN или нечисловые значения")
    if len(t_s) < 2:
        raise ValueError(f"В файле {file_path} недостаточно данных")
    
    f = interp1d(t_s, emf, kind="linear", bounds_error=False, fill_value=(emf[0], emf[-1]))
    
    print(f"[EMF {os.path.basename(file_path)}] {len(t_s)} точек, длит. ~{t_s[-1]:.3f} с, "
          f"диапазон ЭДС {np.min(emf):.6g}..{np.max(emf):.6g} В")
    return t_s, emf, f

def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2)))

def plot_overlay(t_s: np.ndarray, model_v: np.ndarray, exp_v_on_grid: np.ndarray, save_dir: str, timestamp: str, file_name: str, freq: float) -> str:
    """Наложение модельного напряжения и экспериментальной ЭДС."""
    plt.figure(figsize=(11, 6))
    plt.plot(t_s, model_v, label="Модель: напряжение на входе прибора (В)", linewidth=1.5)
    plt.plot(t_s, exp_v_on_grid, label="Экспериментальная ЭДС (В)", alpha=0.85)
    plt.xlabel("Время, с")
    plt.ylabel("ЭДС, В")
    plt.title(f"Индукционная ЭДС: модель vs эксперимент (файл {file_name}, f={freq} Гц)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(save_dir, f"emf_model_vs_experiment_{file_name}_{timestamp}.png")
    plt.savefig(out)
    plt.close()
    return out

def save_npz(save_dir: str, timestamp: str, file_name: str, **arrays) -> str:
    out = os.path.join(save_dir, f"harvester_data_{file_name}_{timestamp}.npz")
    np.savez(out, **arrays)
    return out

def save_timeseries_csv(save_dir: str, timestamp: str, file_name: str, t: np.ndarray, model_v: np.ndarray, exp_v_on_grid: np.ndarray) -> str:
    out = os.path.join(save_dir, f"timeseries_{file_name}_{timestamp}.csv")
    df = pd.DataFrame({'t_s': t, 'v_model_V': model_v, 'v_exp_V': exp_v_on_grid})
    df.to_csv(out, index=False)
    return out

# =========================
# Обработка одного файла
# =========================
def process_file(args: Tuple[str, str, float, str, str, List[float]]) -> Dict:
    shaker_file, emf_file, freq, out_dir, timestamp, conf = args
    file_name = os.path.basename(shaker_file).replace('.csv', '')
    print(f"\n=== Обработка файлов {file_name} (f={freq} Гц) ===")
    
    try:
        # 1) Устройство + электрические параметры
        device = ElectromagneticHarvesterID50mm()
        device.set_electrical(R_coil_ohm=COIL_R_OHM, R_load_ohm=LOAD_R_OHM)
        
        # Установка параметров conf
        device.conf_1, device.conf_2, device.conf_3 = conf
        
        # 2) Загрузка данных шейкера
        t_shaker_s, a_shaker_mps2, accel_interp = load_shaker_data(
            shaker_file, time_col="t", accel_col="acc_mps2", sep=";", decimal=".", normalize_time=True
        )
        device.load_base_from_csv(
            shaker_file, time_col="t", accel_col="acc_mps2", sep=";", decimal=".", normalize_time=True
        )
        
        # 3) Загрузка экспериментальной ЭДС
        t_emf_s, emf_exp_v, emf_exp_interp = load_emf_data(
            emf_file,
            emf_col_idx=EMF_COL_IDX,
            fs_hz=EMF_FS_HZ,
            sep=EMF_SEP,
            decimal=EMF_DECIMAL,
            units=EMF_UNITS,
            normalize_time=True,
            time_col=TIME_COL
        )
        
        # 4) Формируем временную сетку
        if device.base_time_s is not None:
            common_t_s = np.union1d(device.base_time_s, t_emf_s)
        else:
            common_t_s = t_emf_s
        
        t0 = float(common_t_s[0])
        mask = (common_t_s - t0) <= T_SLICE
        common_t_s = common_t_s[mask]
        
        print(f"[grid {file_name}] точек в {T_SLICE}с: {len(common_t_s)}")
        
        # 5) Решение модели
        t_s, z_m, v_mps, current_a, emf_open_v, emf_self_v, forces, v_term_v = device.solve_all(
            t_eval_s=common_t_s,
            rtol=1e-5,
            atol=1e-8,
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
                print(f"[auto-calib {file_name}] подобран масштаб ЭДС emf_scale ≈ {k_fit:.4g}")
                # Пересчёт с новым масштабом
                t_s, z_m, v_mps, current_a, emf_open_v, emf_self_v, forces, v_term_v = device.solve_all(
                    t_eval_s=common_t_s,
                    rtol=1e-5,
                    atol=1e-8,
                    clamp_to_base=True
                )
            else:
                print(f"[auto-calib {file_name}] нет валидных точек для подбора масштаба — пропускаю.")
        
        # 8) RMS и графики
        finite = np.isfinite(v_exp_on_grid)
        if not np.all(finite):
            print(f"[RMS {file_name}] предупреждение: {(~finite).sum()} нечисловых точек в эксперименте — исключаем из RMS")
        
        rms_model = rms(v_term_v)
        rms_exp = rms(v_exp_on_grid[finite]) if finite.any() else float("nan")
        
        file_out_dir = os.path.join(out_dir, f"file_{file_name}")
        os.makedirs(file_out_dir, exist_ok=True)
        
        overlay_path = plot_overlay(t_s, v_term_v, v_exp_on_grid, file_out_dir, timestamp, file_name, freq)
        
        plt.figure(figsize=(11, 5))
        plt.plot(t_s, z_m, label='z(t)')
        plt.xlabel('Время, с')
        plt.ylabel('Положение, м')
        plt.title(f'Свободный магнит: z(t) (файл {file_name}, f={freq} Гц)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        z_plot_path = os.path.join(file_out_dir, f"z_of_t_{file_name}_{timestamp}.png")
        plt.savefig(z_plot_path)
        plt.close()
        
        # Разложение сил
        device.plot_forces(t_s, forces, file_out_dir, timestamp)
        
        # 9) Сохранения
        npz_path = save_npz(
            file_out_dir, timestamp, file_name,
            t_s=t_s, z_m=z_m, v_mps=v_mps,
            current_a=current_a,
            emf_open_v=emf_open_v,
            emf_self_v=emf_self_v,
            v_term_v=v_term_v,
            v_exp_on_grid=v_exp_on_grid,
            forces=forces
        )
        csv_path = save_timeseries_csv(file_out_dir, timestamp, file_name, t_s, v_term_v, v_exp_on_grid)
        
        # 10) Результат
        return {
            'file': file_name,
            'frequency': freq,
            'rms_model': rms_model,
            'rms_exp': rms_exp,
            'emf_scale': device.emf_scale,
            'success': True,
            'overlay_path': overlay_path,
            'npz_path': npz_path,
            'csv_path': csv_path
        }
    
    except Exception as e:
        error_msg = f"Ошибка при обработке {shaker_file} или {emf_file}: {str(e)}"
        print(error_msg)
        return {
            'file': file_name,
            'frequency': freq,
            'success': False,
            'error': str(e)
        }

# =========================
# Оптимизация параметров
# =========================
def objective_function(conf, shaker_files, emf_files, frequencies, out_dir, timestamp):
    """Целевая функция для оптимизации: минимизация разницы RMS."""
    rms_exp_list = []
    rms_model_list = []
    
    with ProcessPoolExecutor(max_workers=min(multiprocessing.cpu_count(), len(shaker_files))) as executor:
        future_to_file = {
            executor.submit(process_file, (shaker_file, emf_file, freq, out_dir, timestamp, conf)): shaker_file
            for shaker_file, emf_file, freq in zip(shaker_files, emf_files, frequencies)
        }
        
        for future in as_completed(future_to_file):
            result = future.result()
            if result['success']:
                rms_exp_list.append(result['rms_exp'])
                rms_model_list.append(result['rms_model'])
    
    if rms_exp_list and rms_model_list:
        mse = np.mean((np.array(rms_model_list) - np.array(rms_exp_list))**2)
        print(f"Текущие conf={conf}, MSE={mse:.6f}")
        return mse
    return np.inf

# =========================
# Основной сценарий
# =========================
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir, timestamp = create_experiment_folder(base_dir)
    
    problematic_files = []
    summary_results = []
    
    # Фильтрация проблемных файлов
    valid_shaker_files = []
    valid_emf_files = []
    valid_frequencies = []
    
    for shaker_file, emf_file, freq in zip(SHAKER_FILES, EMF_FILES, FREQUENCIES):
        try:
            if check_file_structure(shaker_file, expected_cols=3, sep=";") and \
               check_file_structure(emf_file, expected_cols=4, sep=";"):
                valid_shaker_files.append(shaker_file)
                valid_emf_files.append(emf_file)
                valid_frequencies.append(freq)
            else:
                problematic_files.append((os.path.basename(shaker_file).replace('.csv', ''),
                                        f"Неверная структура файла {shaker_file} или {emf_file}"))
        except Exception as e:
            problematic_files.append((os.path.basename(shaker_file).replace('.csv', ''),
                                    f"Ошибка проверки структуры: {str(e)}"))
    
    # Оптимизация параметров
    print("\n=== Оптимизация параметров conf_1, conf_2, conf_3 ===")
    result = minimize(
        objective_function,
        INITIAL_CONF,
        args=(valid_shaker_files, valid_emf_files, valid_frequencies, out_dir, timestamp),
        method='Nelder-Mead',
        bounds=BOUNDS,
        options={'maxiter': 50, 'disp': True}
    )
    
    optimal_conf = result.x
    min_mse = result.fun
    
    print(f"\nСамый успешный вариант:")
    print(f"Оптимальные параметры: conf_1={optimal_conf[0]:.6f}, conf_2={optimal_conf[1]:.6f}, conf_3={optimal_conf[2]:.6f}")
    print(f"Минимальная MSE: {min_mse:.6f}")
    
    # Повторная обработка с оптимальными параметрами
    print("\n=== Повторная обработка с оптимальными параметрами ===")
    summary_results = []
    max_workers = min(multiprocessing.cpu_count(), len(valid_shaker_files))
    print(f"Используется {max_workers} параллельных процессов")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_file, (shaker_file, emf_file, freq, out_dir, timestamp, optimal_conf)): shaker_file
            for shaker_file, emf_file, freq in zip(valid_shaker_files, valid_emf_files, valid_frequencies)
        }
        
        for future in as_completed(future_to_file):
            shaker_file = future_to_file[future]
            try:
                result = future.result()
                if result['success']:
                    summary_results.append(result)
                else:
                    problematic_files.append((result['file'], result['error']))
            except Exception as e:
                file_name = os.path.basename(shaker_file).replace('.csv', '')
                problematic_files.append((file_name, str(e)))
    
    # Сохранение сводного CSV
    summary_csv_path = os.path.join(out_dir, f'summary_results_{timestamp}.csv')
    with open(summary_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File', 'Frequency (Hz)', 'RMS_Model (V)', 'RMS_Exp (V)', 'EMF_Scale'])
        for res in summary_results:
            writer.writerow([res['file'], res['frequency'], res['rms_model'], res['rms_exp'], res['emf_scale']])
    
    # Сохранение списка проблемных файлов
    if problematic_files:
        problematic_files_path = os.path.join(out_dir, f'problematic_files_{timestamp}.txt')
        with open(problematic_files_path, 'w', encoding='utf-8-sig') as f:
            for file_name, error in problematic_files:
                f.write(f"Файл {file_name}: {error}\n")
    
    # График RMS по частотам
    if summary_results:
        frequencies = [res['frequency'] for res in summary_results]
        rms_model_values = [res['rms_model'] for res in summary_results]
        rms_exp_values = [res['rms_exp'] for res in summary_results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, rms_model_values, label='RMS модельной ЭДС (В)', marker='o', color='#1f77b4')
        plt.plot(frequencies, rms_exp_values, label='RMS экспериментальной ЭДС (В)', marker='x', linestyle='--', color='#ff7f0e')
        plt.xlabel('Частота, Гц')
        plt.ylabel('RMS, В')
        plt.title(f'RMS ЭДС в зависимости от частоты (conf_1={optimal_conf[0]:.3f}, conf_2={optimal_conf[1]:.3f}, conf_3={optimal_conf[2]:.3f})')
        plt.grid(True)
        plt.legend()
        rms_plot_path = os.path.join(out_dir, f'rms_vs_frequency_{timestamp}.png')
        plt.savefig(rms_plot_path)
        plt.close()
    
    # Сводка
    print("\n==== Сводка ====")
    print(f"Папка результатов: {out_dir}")
    print(f"Сводный CSV: {summary_csv_path}")
    print(f"График RMS по частотам: {rms_plot_path if summary_results else 'None'}")
    print(f"Обработано файлов: {len(summary_results)} из {len(SHAKER_FILES)}")
    print(f"Самый успешный вариант: conf_1={optimal_conf[0]:.6f}, conf_2={optimal_conf[1]:.6f}, conf_3={optimal_conf[2]:.6f}, MSE={min_mse:.6f}")
    for res in summary_results:
        print(f"Файл {res['file']} (f={res['frequency']} Гц): "
              f"RMS модельной ЭДС: {res['rms_model']:.6f} В, "
              f"RMS экспериментальной ЭДС: {res['rms_exp']:.6f} В, "
              f"EMF Scale: {res['emf_scale']:.6g}")
    
    if problematic_files:
        print("\n==== Проблемные файлы ====")
        for file_name, error in problematic_files:
            print(f"Файл {file_name}: {error}")