import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# === ГЛОБАЛЬНЫЕ НАСТРОЙКИ ГРАФИКОВ ===
plt.rcParams["font.family"] = "Arial"    # <<< шрифт Arial
plt.rcParams["font.size"] = 12           # <<< размер шрифта 12

# --- Пути к экспериментам ---
FORWARD_BASE_PATH = r"experiments\harvester_80mm"
REVERSED_BASE_PATH = r"experiments_reversed\harvester_80mm"
FORWARD_EXPERIMENT_FOLDERS = [os.path.join(FORWARD_BASE_PATH, f"exp_{i}") for i in range(1, 4)]
REVERSED_EXPERIMENT_FOLDERS = [os.path.join(REVERSED_BASE_PATH, f"exp_{i}") for i in range(1, 4)]

# --- Параметры ---
CURRENT_COL_IDX = 0     # Берём ток из столбца 0 (0-based)
CSV_SEP     = ";"
CSV_DEC     = ","
F_MIN, F_MAX = 2.0, 35.0


# --- Вспомогательные функции ---

def is_freq_filename(name: str) -> bool:
    """Проверка, что имя файла (без .csv) — это число (частота)."""
    try:
        float(name)
        return True
    except ValueError:
        return False


def load_column(path: str, col_idx=CURRENT_COL_IDX, sep=CSV_SEP, decimal=CSV_DEC):
    """
    Читает указанный столбец из CSV как float.
    Здесь столбец 0 — это ток как функция времени.
    """
    df = pd.read_csv(path, sep=sep, header=None, decimal=decimal)
    col = df.iloc[:, col_idx].astype(float).to_numpy()
    return col


def calc_rms(signal: np.ndarray) -> float:
    """RMS (среднеквадратичное) значение сигнала."""
    return float(np.sqrt(np.mean(signal ** 2)))


def collect_rms_current(exp_folder: str, fmin=F_MIN, fmax=F_MAX):
    """
    Собирает RMS-ток по всем CSV-файлам эксперимента.
    Для каждого файла (частоты) берём столбец 0, считаем RMS.
    """
    data_dir = os.path.join(exp_folder, "data")
    if not os.path.isdir(data_dir):
        print(f"[WARN] нет папки {data_dir}")
        return np.array([]), np.array([])

    freqs, I_rms_list = [], []

    for fname in os.listdir(data_dir):
        if not fname.endswith(".csv"):
            continue

        stem = fname[:-4]  # имя без .csv
        if not is_freq_filename(stem):
            continue

        f = float(stem)
        if f < fmin or f > fmax:
            continue

        fpath = os.path.join(data_dir, fname)
        try:
            current_samples = load_column(fpath, col_idx=CURRENT_COL_IDX)
            I_rms = calc_rms(current_samples)
            freqs.append(f)
            I_rms_list.append(I_rms)
        except Exception as e:
            print(f"[ERR] {fpath}: {e}")

    if not freqs:
        return np.array([]), np.array([])

    # сортировка по возрастанию частоты
    order = np.argsort(freqs)
    return np.asarray(freqs)[order], np.asarray(I_rms_list)[order]


# --- Построение графика RMS тока ---

all_sets = [
    (FORWARD_EXPERIMENT_FOLDERS, "forward"),
    (REVERSED_EXPERIMENT_FOLDERS, "reversed")
]

current_plot_data = []

for folders, tag in all_sets:
    for exp_folder in folders:
        freqs, I_rms = collect_rms_current(exp_folder)
        if freqs.size == 0:
            continue
        label = f"{os.path.basename(exp_folder)} {tag}"
        style = '-' if tag == "forward" else '--'
        current_plot_data.append((freqs, I_rms, label, style))

# --- График: RMS тока от частоты ---
fig = plt.figure(figsize=(6, 3), dpi=800)   # <<< dpi у фигуры (для предпросмотра)
ax = fig.add_subplot(111)

for freqs, I_rms, label, style in current_plot_data:
    ax.plot(freqs, I_rms, style, marker='o', label=label)

ax.set_xlabel("Частота (Гц)")
ax.set_ylabel("I RMS (ток)")
ax.set_title("Зависимость RMS тока от частоты")
ax.grid(True)
ax.legend()
fig.tight_layout()

# --- СОХРАНЕНИЕ С ВЫСОКИМ DPI ---
output_path = "harvester_80mm_current_rms.png"
fig.savefig(output_path, dpi=800)   # <<< сохраняем картинку с dpi > 600

plt.show()
print(f"График сохранён в файл: {output_path}")
