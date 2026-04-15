import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# --- Пути к экспериментам ---
FORWARD_BASE_PATH = r"experiments\harvester_50mm"
REVERSED_BASE_PATH = r"experiments_reversed\harvester_50mm"
FORWARD_EXPERIMENT_FOLDERS = [os.path.join(FORWARD_BASE_PATH, f"exp_{i}") for i in range(1, 4)]
REVERSED_EXPERIMENT_FOLDERS = [os.path.join(REVERSED_BASE_PATH, f"exp_{i}") for i in range(1, 4)]

# --- Параметры ---
ACC_COL_IDX = 2        # 3-й столбец (0-based)
CSV_SEP     = ";"
CSV_DEC     = ","
F_MIN, F_MAX = 2.0, 35.0

# --- Калибровка акселерометра (из Test.Lab Channel Setup) ---
C0 = -1.8       # смещение (bias)
C1 = 0.168      # коэффициент чувствительности (мВ → g)
G_TO_MS2 = 9.80665  # 1 g = 9.80665 м/с²

# --- Функции ---
def is_freq_filename(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False

def load_acc(path: str, col_idx=ACC_COL_IDX, sep=CSV_SEP, decimal=CSV_DEC):
    """Читает столбец ускорения (в мВ)."""
    df = pd.read_csv(path, sep=sep, header=None, dtype=str)
    acc_mv = df.iloc[:, col_idx].str.replace(',', '.', regex=False).astype(float).to_numpy()
    return acc_mv

def calc_amplitudes(acc, freq, reverse_calibration=False):
    """
    Возвращает (амплитуда ускорения, амплитуда смещения).
    Если reverse_calibration=True — выполняется обратное преобразование из откалиброванных данных обратно в мВ.
    """
    if reverse_calibration:
        # Обратное преобразование: из м/с² -> g -> мВ
        a_g = acc / 9.80665
        acc_mv = (a_g + 1.8) / 0.168
        acc = acc_mv  # сохраняем «возвращённые» мВ
  

    # --- амплитуды ---
    a_max = np.max(acc)
    a_min = np.min(acc)
    A_a = (a_max - a_min) / 2.0
    omega = 2 * math.pi * freq
    A_x = A_a / (omega ** 2)
    return A_a, A_x



def collect_current(exp_folder: str, fmin=F_MIN, fmax=F_MAX, col_idx: int = 0):
    """Collect current channel (first column by default) from CSV files in a single experiment."""
    data_dir = os.path.join(exp_folder, "data")
    if not os.path.isdir(data_dir):
        print(f"[WARN] missing folder {data_dir}")
        return np.array([]), []

    freqs, currents = [], []
    for fname in os.listdir(data_dir):
        if not fname.endswith(".csv"):
            continue
        stem = fname[:-4]
        if not is_freq_filename(stem):
            continue
        f = float(stem)
        if f < fmin or f > fmax:
            continue

        fpath = os.path.join(data_dir, fname)
        try:
            current = load_acc(fpath, col_idx=col_idx)
            freqs.append(f)
            currents.append(current)
        except Exception as e:
            print(f"[ERR] {fpath}: {e}")

    if not freqs:
        return np.array([]), []

    order = np.argsort(freqs)
    freqs_sorted = np.asarray(freqs)[order]
    currents_sorted = [currents[i] for i in order]
    return freqs_sorted, currents_sorted


def collect_amplitudes(exp_folder: str, fmin=F_MIN, fmax=F_MAX):
    """Собирает амплитуды по всем CSV-файлам эксперимента."""
    data_dir = os.path.join(exp_folder, "data")
    if not os.path.isdir(data_dir):
        print(f"[WARN] нет папки {data_dir}")
        return np.array([]), np.array([]), np.array([])

    freqs, A_a_list, A_x_list = [], [], []
    for fname in os.listdir(data_dir):
        if not fname.endswith(".csv"):
            continue
        stem = fname[:-4]
        if not is_freq_filename(stem):
            continue
        f = float(stem)
        if f < fmin or f > fmax:
            continue

        fpath = os.path.join(data_dir, fname)
        try:
            acc_mv = load_acc(fpath)
            A_a, A_x = calc_amplitudes(acc_mv, f, True)
            freqs.append(f)
            A_a_list.append(A_a)
            A_x_list.append(A_x)
        except Exception as e:
            print(f"[ERR] {fpath}: {e}")

    if not freqs:
        return np.array([]), np.array([]), np.array([])

    order = np.argsort(freqs)
    return (np.asarray(freqs)[order],
            np.asarray(A_a_list)[order],
            np.asarray(A_x_list)[order])

# --- Построение графиков ---
all_sets = [(FORWARD_EXPERIMENT_FOLDERS, "forward"),
            (REVERSED_EXPERIMENT_FOLDERS, "reversed")]

acc_plot_data, disp_plot_data = [], []

for folders, tag in all_sets:
    for exp_folder in folders:
        freqs, A_a, A_x = collect_amplitudes(exp_folder)
        if freqs.size == 0:
            continue
        label = f"{os.path.basename(exp_folder)} {tag}"
        style = '-' if tag == "forward" else '--'
        acc_plot_data.append((freqs, A_a, label, style))
        disp_plot_data.append((freqs, A_x, label, style))

# --- График 1: амплитуда ускорения ---
plt.figure(figsize=(12, 6))
for freqs, A_a, label, style in acc_plot_data:
    plt.plot(freqs, A_a, style, marker='o', label=label)
plt.xlabel("Частота (Гц)")
plt.ylabel("Амплитуда ускорения Aₐ (м/с²)")
plt.title("Амплитуда ускорения по частоте (с учётом калибровки)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- График 2: амплитуда смещения ---
plt.figure(figsize=(12, 6))
for freqs, A_x, label, style in disp_plot_data:
    plt.plot(freqs, A_x, style, marker='s', label=label)
plt.xlabel("Частота (Гц)")
plt.ylabel("Амплитуда смещения Aₓ (м)")
plt.title("Амплитуда вибрации по частоте (Aₓ = Aₐ / (2πf)², с учётом калибровки)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
