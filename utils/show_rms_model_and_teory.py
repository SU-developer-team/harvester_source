import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

# Список папок экспериментов
EXPERIMENT_FOLDERS = [r'D:\PROJECTs\magnet\harvester\experiments\harvester_50mm\exp_1', r'D:\PROJECTs\magnet\harvester\experiments\harvester_80mm\exp_1']
exp_model_csvs = [r'D:\PROJECTs\magnet\harvester\logs\rms_f50mm_20250910_203153.csv', r'D:\PROJECTs\magnet\harvester\logs\rms_f80mm_20250910_203723.csv']

# Параметры для ЭДС
EMF_COL_IDX = 3       # Столбец с ЭДС (4-й столбец, индексация с 0)
EMF_FS_HZ = 1000.0    # Частота дискретизации (Гц)
EMF_UNITS = "mV"      # Единицы измерения (мВ)
EMF_SEP = ";"
EMF_DECIMAL = ","
SELECTED_FREQ = 4.0   # Частота для анализа (Гц)
WINDOW_SIZE = 2000    # Размер скользящего окна (точек, 2 секунды при 1000 Гц)

def load_emf_no_time(file_path, emf_col_idx, fs_hz, sep=";", decimal=",", units="mV"):
    """Загрузка ЭДС без времени, построение временной шкалы."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    
    df = pd.read_csv(file_path, sep=sep, header=None, dtype=str)
    emf_raw = df.iloc[:, emf_col_idx].str.replace(',', '.').astype(float)
    
    scale = {"V": 1.0, "mV": 1.0, "uV": 1e-3}[units]  # Оставляем в мВ, не переводим в В
    emf_v = emf_raw * scale
    
    t_s = np.arange(len(emf_v)) / fs_hz
    return t_s, emf_v

def load_accel(file_path, sep=";", decimal="."):
    """Загрузка ускорения из synthesized_data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    df = pd.read_csv(file_path, sep=sep, decimal=decimal)
    t = df['t'].values
    acc = df['acc_mps2'].values
    return t, acc

def rms_sliding_window(signal, window_size, step=1):
    """Вычисление RMS в скользящем окне."""
    signal = np.asarray(signal, dtype=float)
    if not np.all(np.isfinite(signal)):
        raise ValueError("В данных есть NaN или inf.")
    
    rms_values = []
    for i in range(0, len(signal) - window_size + 1, step):
        window = signal[i:i + window_size]
        rms_val = np.sqrt(np.mean(window**2))
        rms_values.append(rms_val)
    return np.array(rms_values)

# Сбор данных для всех экспериментов
all_freqs = set()
exp_data = {}

for i, exp_folder in enumerate(EXPERIMENT_FOLDERS, start=1):
    data_folder = os.path.join(exp_folder, "data")
    exp_name = os.path.basename(exp_folder)
    
    freqs = []
    rms_values = []
    
    if not os.path.exists(data_folder):
        print(f"Папка {data_folder} не найдена, пропускаем.")
        continue
    
    for fname in os.listdir(data_folder):
        if fname.endswith('.csv') and fname[:-4].replace('.', '').isdigit():
            freq = float(fname[:-4])
            if freq >= 2.0:
                emf_file = os.path.join(data_folder, fname)
                try:
                    t_emf, emf_v = load_emf_no_time(emf_file, EMF_COL_IDX, EMF_FS_HZ, EMF_SEP, EMF_DECIMAL, EMF_UNITS)
                    rms_emf = np.sqrt(np.mean(emf_v**2))  # RMS в мВ
                    freqs.append(freq)
                    rms_values.append(rms_emf)
                    all_freqs.add(freq)
                except Exception as e:
                    print(f"Ошибка в файле {emf_file}: {e}")
    
    sorted_indices = np.argsort(freqs)
    freqs = np.array(freqs)[sorted_indices]
    rms_values = np.array(rms_values)[sorted_indices]
    
    exp_data[f'exp_name_{i}'] = {'freqs': freqs, 'rms': rms_values}
    print(f"Эксперимент {exp_name}: найдено {len(freqs)} частот, отсортировано: {freqs}")

# Чтение данных модели
data_50mm = []
data_80mm = []

for csv_file in exp_model_csvs:
    data = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        header = next(reader, None)  # Пропускаем заголовок
        for row in reader:
            if len(row) >= 2:
                try:
                    value = row[1].replace(',', '.').strip()
                    data.append(float(value))
                except ValueError as e:
                    print(f"Предупреждение: Не удалось преобразовать значение '{row[1]}' в float: {e}. Пропускаем строку.")
    if csv_file == exp_model_csvs[0]:
        data_50mm = data
    else:
        data_80mm = data

# Проверка длины данных
print(f"Длина data_50mm: {len(data_50mm)}, Длина data_80mm: {len(data_80mm)}")

# Убедимся, что модельные данные имеют длину 45
if len(data_50mm) > 45:
    data_50mm = data_50mm[:45]
elif len(data_50mm) < 45:
    print(f"Предупреждение: data_50mm содержит {len(data_50mm)} значений, ожидалось 45. Дополняем нулями.")
    data_50mm.extend([0.0] * (45 - len(data_50mm)))

if len(data_80mm) > 45:
    data_80mm = data_80mm[:45]
elif len(data_80mm) < 45:
    print(f"Предупреждение: data_80mm содержит {len(data_80mm)} значений, ожидалось 45. Дополняем нулями.")
    data_80mm.extend([0.0] * (45 - len(data_80mm)))

# Определяем частоты для модельных данных (2–46 Гц)
freqs_model = np.arange(2, 47)  # 45 значений

# Для экспериментальных данных, если они короче 45, дополним нулями
def extend_to_45(freqs, rms):
    if len(freqs) < 45:
        extended_freqs = np.arange(2, 47)
        extended_rms = np.zeros(45)
        for i, f in enumerate(freqs):
            idx = int(f - 2)  # Сопоставляем частоту с индексом
            extended_rms[idx] = rms[i]
        return extended_freqs, extended_rms
    return freqs, rms

# --- убираем extend_to_45 ---
# Вместо extend_to_45 будем использовать только реальные данные

# Экспериментальные данные
freqs_50mm_exp = exp_data['exp_name_1']['freqs']
rms_50mm_exp = exp_data['exp_name_1']['rms']

freqs_80mm_exp = exp_data['exp_name_2']['freqs']Те
rms_80mm_exp = exp_data['exp_name_2']['rms']

# --- проверка модельных данных ---
# Не обрезаем/не дополняем — рисуем только то, что реально есть
if len(data_50mm) != 45:
    print(f"Предупреждение: data_50mm содержит {len(data_50mm)} значений (не 45)")
if len(data_80mm) != 45:
    print(f"Предупреждение: data_80mm содержит {len(data_80mm)} значений (не 45)")

# --- построение ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
colors = ['blue', 'orange', 'green', 'red']

# Эксперимент
ax1.set_title('RMS индукционной ЭДС от частоты (Эксперимент)')
ax1.set_xlabel('Частота (Гц)')
ax1.set_ylabel('RMS ЭДС (мВ)')
ax1.grid(True)
ax1.plot(freqs_50mm_exp, rms_50mm_exp, label='Эксперимент 50мм', marker='o', color=colors[0])
ax1.plot(freqs_80mm_exp, rms_80mm_exp, label='Эксперимент 80мм', marker='o', color=colors[1])
ax1.legend()

# Модель
ax2.set_title('RMS индукционной ЭДС от частоты (Модель)')
ax2.set_xlabel('Частота (Гц)')
ax2.set_ylabel('RMS ЭДС (мВ)')
ax2.grid(True)
if data_50mm:
    ax2.plot(range(2, 2 + len(data_50mm)), data_50mm, label='Модель 50мм', marker='o', color=colors[2])
if data_80mm:
    ax2.plot(range(2, 2 + len(data_80mm)), data_80mm, label='Модель 80мм', marker='o', color=colors[3])
ax2.legend()

plt.tight_layout()
plt.savefig('logs/rms_emf_experiment_vs_model.png')
plt.show()
