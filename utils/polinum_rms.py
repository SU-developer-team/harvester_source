import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Базовый путь к экспериментам
FORWARD_BASE_PATH = r"experiments\harvester_50mm"
REVERSED_BASE_PATH = r"experiments_reversed\harvester_50mm"

# Список папок экспериментов
FORWARD_EXPERIMENT_FOLDERS = [os.path.join(FORWARD_BASE_PATH, f"exp_{i}") for i in range(1, 4)]
REVERSED_EXPERIMENT_FOLDERS = [os.path.join(REVERSED_BASE_PATH, f"exp_{i}") for i in range(1, 4)]

# Параметры для ЭДС
EMF_COL_IDX = 3       # Столбец с ЭДС (4-й столбец, индексация с 0)
EMF_FS_HZ = 1000.0    # Частота дискретизации (Гц)
EMF_UNITS = "mV"      # Единицы измерения (мВ)
EMF_SEP = ";"
EMF_DECIMAL = ","
WINDOW_SIZE = 2000    # Размер скользящего окна

# Функция аппроксимации: гауссова кривая с зависимостью от h
def gaussian(freq, amp, center, sigma, h_scale, h_shift):
    return amp * np.exp(-((freq - (center + h_shift * h_scale))**2) / (2 * sigma**2)) * h_scale

def load_emf_no_time(file_path, emf_col_idx, fs_hz, sep=";", decimal=",", units="mV"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    
    df = pd.read_csv(file_path, sep=sep, header=None, dtype=str)
    emf_raw = df.iloc[:, emf_col_idx].str.replace(',', '.').astype(float)
    
    scale = {"V": 1.0, "mV": 1.0, "uV": 1e-3}[units]
    emf_v = emf_raw * scale
    
    t_s = np.arange(len(emf_v)) / fs_hz
    return t_s, emf_v

def load_accel(file_path, sep=";", decimal="."):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    df = pd.read_csv(file_path, sep=sep, decimal=decimal)
    t = df['t'].values
    acc = df['acc_mps2'].values
    return t, acc

def rms_sliding_window(signal, window_size, step=1):
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

h = 0.0572  # Начальное значение h

for exp_folder in FORWARD_EXPERIMENT_FOLDERS:
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
                    rms_emf = np.sqrt(np.mean(emf_v**2))
                    freqs.append(freq)
                    rms_values.append(rms_emf)
                    all_freqs.add(freq)
                except Exception as e:
                    print(f"Ошибка в файле {emf_file}: {e}")
    
    sorted_indices = np.argsort(freqs)
    freqs = np.array(freqs)[sorted_indices]
    rms_values = np.array(rms_values)[sorted_indices]
    
    # Аппроксимация
    popt, _ = curve_fit(gaussian, freqs, rms_values, p0=[max(rms_values), np.mean(freqs), 2.0, h, 0.1])
    amp, center, sigma, h_scale, h_shift = popt
    freq_range = np.linspace(min(freqs), max(freqs), 100)
    rms_fit = gaussian(freq_range, amp, center, sigma, h_scale, h_shift)
    
    exp_data[f'{exp_name} forward'] = {'freqs': freqs, 'rms': rms_values, 'fit_freqs': freq_range, 'fit_rms': rms_fit}
    print(f"Эксперимент {exp_name}: найдено {len(freqs)} частот, пиковая RMS (аппрокс.): {max(rms_fit):.2f} мВ")

for exp_folder in REVERSED_EXPERIMENT_FOLDERS:
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
                    rms_emf = np.sqrt(np.mean(emf_v**2))
                    freqs.append(freq)
                    rms_values.append(rms_emf)
                    all_freqs.add(freq)
                except Exception as e:
                    print(f"Ошибка в файле {emf_file}: {e}")
    
    sorted_indices = np.argsort(freqs)
    freqs = np.array(freqs)[sorted_indices]
    rms_values = np.array(rms_values)[sorted_indices]
    
    # Аппроксимация
    popt, _ = curve_fit(gaussian, freqs, rms_values, p0=[max(rms_values), np.mean(freqs), 2.0, h, 0.1])
    amp, center, sigma, h_scale, h_shift = popt
    freq_range = np.linspace(min(freqs), max(freqs), 100)
    rms_fit = gaussian(freq_range, amp, center, sigma, h_scale, h_shift)
    
    exp_data[f'{exp_name} revers'] = {'freqs': freqs, 'rms': rms_values, 'fit_freqs': freq_range, 'fit_rms': rms_fit}
    print(f"Эксперимент {f'{exp_name} rev'}: найдено {len(freqs)} частот, пиковая RMS (аппрокс.): {max(rms_fit):.2f} мВ")

# График RMS vs Частота с аппроксимацией
plt.figure(figsize=(10, 6))
for exp_name, data in exp_data.items():
    plt.plot(data['freqs'], data['rms'], 'o-', label=f'Эксперимент {exp_name} (данные)')
    plt.plot(data['fit_freqs'], data['fit_rms'], '--', label=f'Эксперимент {exp_name} (аппрокс.)')

plt.xlabel('Частота (Гц)')
plt.ylabel('RMS ЭДС (мВ)')
plt.title('Сравнение RMS ЭДС с аппроксимацией')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Предсказание пиковой RMS для нового значения h
new_h = 0.2# Пример нового значения h
for exp_name, data in exp_data.items():
    amp, center, sigma, h_scale, h_shift = curve_fit(gaussian, data['freqs'], data['rms'], p0=[max(data['rms']), np.mean(data['freqs']), 2.0, h, 0.1])[0]
    # Корректировка пика с учетом нового h
    new_h_scale = new_h / h * h_scale
    new_center = center + h_shift * (new_h - h)
    peak_rms_new_h = gaussian(new_center, amp, new_center, sigma, new_h_scale, h_shift)
    print(f"Предсказанная пиковая RMS для {exp_name} при h={new_h}: {peak_rms_new_h:.2f} мВ")