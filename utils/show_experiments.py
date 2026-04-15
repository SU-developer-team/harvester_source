import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
SELECTED_FREQ = 4.0   # Частота для анализа (Гц)
WINDOW_SIZE = 2000    # Размер скользящего окна (точек, 2 секунды при 1000 Гц)
# Диапазон частот для оценки ошибки (включительно). Если None — берём весь доступный диапазон.
EVAL_RANGE = (5.0, 20.0)  # или None

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

def relative_error_percent(exp_freqs, exp_rms, model_freqs, model_rms, freq_range=None, eps=1e-12):
    exp_freqs   = np.asarray(exp_freqs,   float)
    exp_rms     = np.asarray(exp_rms,     float)
    model_freqs = np.asarray(model_freqs, float)
    model_rms   = np.asarray(model_rms,   float)

    # маска попадания в диапазон модели
    mask = (exp_freqs >= model_freqs.min()) & (exp_freqs <= model_freqs.max())

    # маска диапазона оценки (если задан)
    if freq_range is not None:
        lo, hi = sorted(freq_range)  # на всякий случай
        mask &= (exp_freqs >= lo) & (exp_freqs <= hi)

    if mask.sum() == 0:
        return None

    model_interp = np.interp(exp_freqs[mask], model_freqs, model_rms)
    denom = np.maximum(np.abs(model_interp), eps)
    mape = np.mean(np.abs(exp_rms[mask] - model_interp) / denom) * 100.0
    return float(mape)



# Сбор данных для всех экспериментов
all_freqs = set()
exp_data = {}

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
                    rms_emf = np.sqrt(np.mean(emf_v**2))  # RMS в мВ
                    freqs.append(freq)
                    rms_values.append(rms_emf)
                    all_freqs.add(freq)
                except Exception as e:
                    print(f"Ошибка в файле {emf_file}: {e}")
    
    sorted_indices = np.argsort(freqs)
    freqs = np.array(freqs)[sorted_indices]
    rms_values = np.array(rms_values)[sorted_indices]
    
    exp_data[f'{exp_name} forward'] = {'freqs': freqs, 'rms': rms_values}
    print(f"Эксперимент {exp_name}: найдено {len(freqs)} частот, отсортировано: {freqs}")

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
                    rms_emf = np.sqrt(np.mean(emf_v**2))  # RMS в мВ
                    freqs.append(freq)
                    rms_values.append(rms_emf)
                    all_freqs.add(freq)
                except Exception as e:
                    print(f"Ошибка в файле {emf_file}: {e}")
    
    sorted_indices = np.argsort(freqs)
    freqs = np.array(freqs)[sorted_indices]
    rms_values = np.array(rms_values)[sorted_indices]
    
    exp_data[f'{exp_name} revers'] = {'freqs': freqs, 'rms_reversed': rms_values}
    print(f"Эксперимент {f'{exp_name} rev'}: найдено {len(freqs)} частот, отсортировано: {freqs}")
math_model_dir = r'logs\2025-10-08_12-17-33_results.csv'
 
model_available = False
model_freqs = model_rms = None
if os.path.exists(math_model_dir):
    df_model = pd.read_csv(math_model_dir)
    model_freqs = df_model['Frequency (Hz)'].values
    # как у тебя: перевод и калибровка
    model_rms = (df_model['Mean Self-Induction EMF (V)'].values)   # мВ
    model_available = True
else:
    print(f"Модельный файл не найден: {math_model_dir}")

plt.figure(figsize=(10, 6))

for exp_name, data in exp_data.items():
    # forward
    if 'rms' in data and data['rms'] is not None:
        freqs = data['freqs']
        rms   = data['rms']
        # NEW: считаем относительную ошибку, если доступна модель
        if model_available:
            mape = relative_error_percent(freqs, rms, model_freqs, model_rms, freq_range=EVAL_RANGE)
            range_str = f" {EVAL_RANGE[0]:g}-{EVAL_RANGE[1]:g} Гц" if (EVAL_RANGE is not None) else ""
            if mape is not None:
                label = f'Эксперимент {exp_name} (отн. ошибка{range_str} {mape:.1f}%)'
            else:
                label = f'Эксперимент {exp_name}'

        else:
            label = f'Эксперимент {exp_name}'
        plt.plot(freqs, rms, 'o-', label=label)

    # reversed
    elif 'rms_reversed' in data and data['rms_reversed'] is not None:
        freqs = data['freqs']
        rms   = data['rms_reversed']
        # NEW: считаем относительную ошибку, если доступна модель
        if model_available:
            mape = relative_error_percent(freqs, rms, model_freqs, model_rms, freq_range=EVAL_RANGE)
            range_str = f" {EVAL_RANGE[0]:g}-{EVAL_RANGE[1]:g} Гц" if (EVAL_RANGE is not None) else ""
            if mape is not None:
                label = f'Эксперимент {exp_name} (отн. ошибка{range_str} {mape:.1f}%)'
            else:
                label = f'Эксперимент {exp_name}'

        else:
            label = f'Эксперимент {exp_name}'
        plt.plot(freqs, rms, '--', label=label)
if model_available:
    plt.plot(model_freqs, model_rms, 'k--', label='Математическая модель', linewidth=2)

plt.xlabel('Частота (Гц)')
plt.ylabel('RMS ЭДС (мВ)')
plt.title('Сравнение RMS ЭДС для экспериментов')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# График ускорения vs Время для SELECTED_FREQ
plt.figure(figsize=(12, 8))
for exp_folder in FORWARD_EXPERIMENT_FOLDERS:
    synth_folder = os.path.join(exp_folder, "synthesized_data")
    exp_name = os.path.basename(exp_folder)
    accel_file = os.path.join(synth_folder, f"{int(SELECTED_FREQ)}.csv")
    if os.path.exists(accel_file):
        try:
            t_acc, acc = load_accel(accel_file)
            plt.plot(t_acc, acc, label=f'Ускорение {exp_name} ({SELECTED_FREQ} Гц)')
        except Exception as e:
            print(f"Ошибка в файле {accel_file}: {e}")
    else:
        print(f"Файл {accel_file} не найден.")
# plt.xlabel('Время (с)')
# plt.ylabel('Ускорение (м/с²)')
# plt.title(f'Сравнение ускорения по времени для {SELECTED_FREQ} Гц')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# Новые графики: RMS ЭДС по времени для частоты 4 Гц для каждого эксперимента
for exp_folder in FORWARD_EXPERIMENT_FOLDERS:
    data_folder = os.path.join(exp_folder, "data")
    exp_name = os.path.basename(exp_folder)
    emf_file = os.path.join(data_folder, f"{int(SELECTED_FREQ)}.csv")  # Имя файла, например, 4.csv
    
    if os.path.exists(emf_file):
        try:
            t_emf, emf_v = load_emf_no_time(emf_file, EMF_COL_IDX, EMF_FS_HZ, EMF_SEP, EMF_DECIMAL, EMF_UNITS)
            print(f"Файл {emf_file}: {len(emf_v)} точек, {len(emf_v)/EMF_FS_HZ:.2f} секунд")
            
            # Вычисляем RMS в скользящем окне с шагом 1 точка
            rms_values = rms_sliding_window(emf_v, WINDOW_SIZE, step=1)
            # Временная шкала для RMS (шаг 0.001 секунды)
            t_rms = np.arange(len(rms_values)) / EMF_FS_HZ
            
            # Построение графика
            # plt.figure(figsize=(10, 6))
            # plt.plot(t_rms, rms_values, '-', label=f'RMS ЭДС ({exp_name}, {SELECTED_FREQ} Гц)')
            # plt.xlabel('Время (с)')
            # plt.ylabel('RMS ЭДС (мВ)')
            # plt.title(f'Изменение RMS ЭДС по времени для {exp_name} на {SELECTED_FREQ} Гц')
            # plt.grid(True)
            # plt.legend()
            # plt.tight_layout()
            # plt.show()
        except Exception as e:
            print(f"Ошибка в файле {emf_file}: {e}")
    else:
        print(f"Файл {emf_file} не найден.")