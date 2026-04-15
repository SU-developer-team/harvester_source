import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

BASE_PATH = r"experiments\harvester_50mm"
EXPERIMENT_FOLDERS = [os.path.join(BASE_PATH, f"exp_{i}") for i in range(1, 4)]

TARGET_COL = "displacement_mm"   # что считаем (ускорение)
SEP_CANDIDATES = [";", ","]  # на случай различий по файлам
DECIMAL_CANDIDATES = [".", ","]

def read_numeric_column(file_path, column_name):
    # Пробуем несколько сочетаний разделителей
    last_err = None
    for sep in SEP_CANDIDATES:
        for decimal in DECIMAL_CANDIDATES:
            try:
                df = pd.read_csv(
                    file_path,
                    sep=sep,
                    engine="python",
                    on_bad_lines="skip"
                )
                # Если имена столбцов с пробелами — подчистим
                df.columns = [c.strip() for c in df.columns]
                if column_name not in df.columns:
                    # Иногда десятичные запятые мешают: попробуем вручную заменить
                    if decimal == "," and df.columns and df.dtypes.apply(lambda x: x==object).any():
                        # Попытка привести нужный столбец, если он есть в другом написании
                        pass
                    raise KeyError(f"Нет столбца '{column_name}' при sep='{sep}', decimal='{decimal}'")
                s = df[column_name].astype(str).str.replace(",", ".", regex=False)
                s = pd.to_numeric(s, errors="coerce")
                s = s.dropna()
                if s.empty:
                    raise ValueError("Столбец пустой после приведения к числу")
                return s.to_numpy()
            except Exception as e:
                last_err = e
    raise last_err if last_err else RuntimeError("Не удалось прочитать столбец")

def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.max(np.abs(x)))


all_freqs = set()
exp_data = {}

for exp_folder in EXPERIMENT_FOLDERS:
    data_folder = os.path.join(exp_folder, "synthesized_data")
    exp_name = os.path.basename(exp_folder)

    freqs, rms_values = [], []
    datas = {}
    if not os.path.exists(data_folder):
        print(f"Папка {data_folder} не найдена, пропускаем.")
        continue

    for fname in os.listdir(data_folder):
        if not fname.endswith(".csv"):
            continue
        name_no_ext = fname[:-4]
        if not name_no_ext.replace(".", "", 1).isdigit():
            continue

        freq = float(name_no_ext)
        if freq < 2.0:
            continue

        fpath = os.path.join(data_folder, fname)
        try:
            col = read_numeric_column(fpath, TARGET_COL)
            freqs.append(freq)
            rms_values.append(rms(col))
            
            all_freqs.add(freq)
        except Exception as e:
            print(f"Ошибка в файле {fpath}: {e}")

    if freqs:
        order = np.argsort(freqs)
        freqs = np.array(freqs)[order]
        rms_values = np.array(rms_values)[order]
    else:
        freqs = np.array([])
        rms_values = np.array([])
    for freq in freqs:
        datas[freq] = rms_values[freqs.tolist().index(freq)]

    # Сохраняем данные в JSON файл
    json_path = os.path.join(exp_folder, "rms_data.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(datas, f, ensure_ascii=False, indent=4)
        print(f"Данные сохранены в {json_path}")
    except Exception as e:
        print(f"Ошибка при сохранении данных в {json_path}: {e}")
    exp_data[f'{exp_name} (RMS {TARGET_COL})'] = {'freqs': freqs, 'rms': rms_values}
    print(f"Эксперимент {exp_name}: найдено {len(freqs)} частот, отсортировано: {freqs}")

plt.figure(figsize=(10, 6))
for exp_name, data in exp_data.items():
    if data['freqs'].size:
        plt.plot(data['freqs'], data['rms'], 'o-', label=exp_name)

plt.xlabel('Частота (Гц)')
plt.ylabel(f'RMS {TARGET_COL}')
plt.title('Сравнение RMS по частоте')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
