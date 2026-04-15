import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict

# ------------------------------- ПАРАМЕТРЫ -------------------------------

# Базовый путь к экспериментам (только forward)
FORWARD_BASE_PATH = r"experiments\harvester_50mm"

# Список папок экспериментов (только forward)
FORWARD_EXPERIMENT_FOLDERS = [os.path.join(FORWARD_BASE_PATH, f"exp_{i}") for i in range(1, 4)]

# Параметры для ЭДС
EMF_COL_IDX = 3       # Столбец с ЭДС (4-й столбец, индексация с 0)
EMF_FS_HZ = 1000.0    # Частота дискретизации (Гц)
EMF_UNITS = "mV"      # 'V' | 'mV' | 'uV'
EMF_SEP = ";"
EMF_DECIMAL = ","

# Диапазон частот для оценки метрик (включительно). Если None — берём весь доступный диапазон.
EVAL_RANGE = (7.0, 32.0)  # или None

# Диапазон отображения на ГРАФИКЕ (включительно)
DISPLAY_RANGE = (7.0, 31.0)

# Путь к CSV с результатами модели
MATH_MODEL_CSV = r'logs\2025-10-08_18-57-44_results.csv'

# Числовая стабильность
_EPS = 1e-12

# ------------------------------- НАСТРОЙКИ ГРАФИКИ -------------------------------
# Хотим 1800x1200 пикселей. Пусть DPI=100, тогда figsize=(18, 12)
_TARGET_W_PX, _TARGET_H_PX, _DPI = 2200, 1600, 600
_FIGSIZE = (6,3)

# Пользователь просит размер шрифта 12 px. В matplotlib шрифт задаётся в pt:
# points = pixels * 72 / dpi
_FONT_SIZE_PT = 12 * 72.0 / _DPI  # = 8.64 pt при dpi=100

mpl.rcParams.update({
    "figure.dpi": _DPI,
    "savefig.dpi": _DPI,
    "font.family": "Arial",   # если Arial не установлен в системе, будет fallback
    "font.size": 10,
})

# ------------------------------- ФУНКЦИИ -------------------------------
def load_emf_no_time(file_path, emf_col_idx, fs_hz, sep=";", decimal=",", units="mV"):
    """Загрузка ЭДС без явного времени, построение временной шкалы, приведение к мВ."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    
    df = pd.read_csv(file_path, sep=sep, header=None, dtype=str)
    # Нормализуем десятичный разделитель для столбца ЭДС
    emf_raw = df.iloc[:, emf_col_idx].str.replace(',', '.').astype(float)

    # Приводим к мВ
    to_mV = {"V": 1000.0, "mV": 1.0, "uV": 1e-3}[units]
    emf_mV = emf_raw * to_mV
    
    t_s = np.arange(len(emf_mV)) / fs_hz
    return t_s, emf_mV

def compute_rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))

def compute_metrics(exp_freqs, exp_rms, model_freqs, model_rms, freq_range=None):
    """
    Метрики между усреднённым экспериментом (exp_*) и моделью (model_*), интерполируем модель при необходимости.
    Возвращает: (rmse, r, mre)
    """
    exp_freqs   = np.asarray(exp_freqs,   dtype=float)
    exp_rms     = np.asarray(exp_rms,     dtype=float)
    model_freqs = np.asarray(model_freqs, dtype=float)
    model_rms   = np.asarray(model_rms,   dtype=float)

    # Пересечение по диапазону частот модели
    mask = (exp_freqs >= model_freqs.min()) & (exp_freqs <= model_freqs.max())

    # Применим пользовательский диапазон, если дан
    if freq_range is not None:
        lo, hi = sorted(freq_range)
        mask &= (exp_freqs >= lo) & (exp_freqs <= hi)

    if mask.sum() == 0:
        return None, None, None

    x = exp_freqs[mask]
    y_exp = exp_rms[mask]
    y_mod = np.interp(x, model_freqs, model_rms)

    rmse = float(np.sqrt(np.mean((y_exp - y_mod) ** 2)))

    # r
    if len(y_exp) >= 2 and np.std(y_exp) > 0 and np.std(y_mod) > 0:
        r = float(np.corrcoef(y_exp, y_mod)[0, 1])
    else:
        r = None

    # MRE относительно эксперимента
    denom = np.maximum(np.abs(y_exp), _EPS)
    mre = float(np.mean(np.abs(y_exp - y_mod) / denom) * 100.0)

    return rmse, r, mre

# ------------------------------- СБОР ДАННЫХ (ТОЛЬКО FORWARD) -------------------------------

# Соберём RMS по всем forward-экспериментам и усредним по одинаковым частотам
from collections import defaultdict
freq_to_rms_list = defaultdict(list)

for exp_folder in FORWARD_EXPERIMENT_FOLDERS:
    data_folder = os.path.join(exp_folder, "data")
    if not os.path.exists(data_folder):
        print(f"Папка {data_folder} не найдена, пропускаем: {data_folder}")
        continue

    for fname in os.listdir(data_folder):
        # Имя файла — частота, например "4.csv" или "12.5.csv"
        if fname.endswith('.csv') and fname[:-4].replace('.', '').isdigit():
            try:
                freq = float(fname[:-4])
            except ValueError:
                continue

            if freq < 2.0:
                continue

            emf_file = os.path.join(data_folder, fname)
            try:
                _, emf_mV = load_emf_no_time(emf_file, EMF_COL_IDX, EMF_FS_HZ,
                                             sep=EMF_SEP, decimal=EMF_DECIMAL, units=EMF_UNITS)
                rms_emf = compute_rms(emf_mV)  # RMS в мВ
                freq_to_rms_list[freq].append(rms_emf)
            except Exception as e:
                print(f"Ошибка при обработке {emf_file}: {e}")

# Усреднение RMS по частоте
if not freq_to_rms_list:
    raise RuntimeError("Не найдено ни одной точки данных в FORWARD_EXPERIMENT_FOLDERS.")

all_freqs = sorted(freq_to_rms_list.keys())
avg_rms_exp = np.array([np.mean(freq_to_rms_list[f]) for f in all_freqs], dtype=float)

print(f"Найдено частот (forward): {len(all_freqs)}")
print("Первые несколько точек (freq -> avg RMS мВ):")
for f in all_freqs[:10]:
    print(f"  {f:g} Гц -> {np.mean(freq_to_rms_list[f]):.6g} мВ  (n={len(freq_to_rms_list[f])})")

# ------------------------------- ЗАГРУЗКА МОДЕЛИ -------------------------------

model_available = False
model_freqs = model_rms_mV = None

if os.path.exists(MATH_MODEL_CSV):
    df_model = pd.read_csv(MATH_MODEL_CSV)

    # Ожидаемые столбцы:
    # 'Frequency (Hz)' и 'Mean Self-Induction EMF (V)' (в Вольтах)
    if ('Frequency (Hz)' in df_model.columns) and ('Mean Self-Induction EMF (V)' in df_model.columns):
        model_freqs = df_model['Frequency (Hz)'].values.astype(float)
        model_rms_V = df_model['Mean Self-Induction EMF (V)'].values.astype(float)
        model_rms_mV = model_rms_V  # В -> мВ (чтобы совпадало с подписью оси)
        model_available = True
    else:
        print(f"В модели {MATH_MODEL_CSV} нет нужных столбцов.")
else:
    print(f"Модельный файл не найден: {MATH_MODEL_CSV}")

# ------------------------------- МЕТРИКИ -------------------------------

rmse = r = mre = None
range_str = ""
if model_available:
    rmse, r, mre = compute_metrics(all_freqs, avg_rms_exp, model_freqs, model_rms_mV, freq_range=EVAL_RANGE)
    if EVAL_RANGE is not None:
        range_str = f" {EVAL_RANGE[0]:g}-{EVAL_RANGE[1]:g} Гц"

    print("\nМетрики (усреднённый эксперимент vs модель):")
    if rmse is not None:
        print(f"  RMSE{range_str}: {rmse:.6g} мВ")
        print(f"  MRE{range_str}:  {mre:.3f} %")
        if r is not None:
            print(f"  r{range_str}:    {r:.6f}")
        else:
            print(f"  r{range_str}:    н/д (мало точек или нулевая дисперсия)")
    else:
        print("  Нет общих точек частоты в заданном диапазоне/пересечении с моделью.")

# ------------------------------- ГРАФИК -------------------------------

# Применяем диапазон отображения
lo_disp, hi_disp = sorted(DISPLAY_RANGE)
mask_disp = (np.array(all_freqs) >= lo_disp) & (np.array(all_freqs) <= hi_disp)
x_plot = np.array(all_freqs)[mask_disp]
y_plot = avg_rms_exp[mask_disp]

plt.figure(figsize=_FIGSIZE, dpi=_DPI)

plt.plot(x_plot, y_plot, 's-', linewidth=0.8, label="Physical Experiment")

# Модель (ограничим к DISPLAY_RANGE)
if model_available:
    m_mask = (model_freqs >= lo_disp) & (model_freqs <= hi_disp)
    plt.plot(model_freqs[m_mask], model_rms_mV[m_mask], '^--', linewidth=0.8, label='Mathematical Model')

plt.xlabel('Frequency (Hz)')
plt.ylabel('RMS EMF (mV)')
plt.xticks(np.arange(np.floor(lo_disp), np.ceil(hi_disp) + 1, 1))
plt.yticks(np.arange(1, 14 + 1, 1))
plt.grid(True, which='both', linestyle='-', alpha=0.35)
plt.legend()
plt.tight_layout()
plt.savefig('rms_exp_vs_model.png')
plt.show()