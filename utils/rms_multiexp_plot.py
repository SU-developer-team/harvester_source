import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Параметры ===
ROOT_FOLDER = r"experiments\harvester_80mm"
EXPERIMENTS = [f"exp_{i}" for i in range(1, 4)]  # exp_1 ... exp_5

# --- Диапазон частот (имена файлов) ---
FREQ_MIN = 2   # например, 2 Гц (включительно)
FREQ_MAX = 35  # например, 50 Гц (включительно)

plt.figure(figsize=(12, 6))

for exp in EXPERIMENTS:
    data_folder = os.path.join(ROOT_FOLDER, exp, "data")
    rms_list = []
    freq_list = []

    # По всем частотам (файлам)
    for filename in os.listdir(data_folder):
        if not filename.endswith(".csv"):
            continue
        try:
            freq = int(filename.replace(".csv", ""))
        except Exception:
            continue
        if not (FREQ_MIN <= freq <= FREQ_MAX):
            continue

        filepath = os.path.join(data_folder, filename)
        try:
            df = pd.read_csv(filepath, header=None, sep=";", decimal=",")
        except Exception:
            continue
        if df.shape[1] < 4:
            continue

        channel4 = df.iloc[:, 3].astype(float).values
        # --- RMS ---
        rms_val = np.sqrt(np.mean(channel4 ** 2))
        freq_list.append(freq)
        rms_list.append(rms_val)

    # Сортируем по частоте (чтобы линия не прыгала)
    freq_list, rms_list = zip(*sorted(zip(freq_list, rms_list)))
    plt.plot(freq_list, rms_list, marker='o', label=exp)

plt.xlabel("Частота возбуждения, Гц")
plt.ylabel("RMS (4 канал), В")
plt.title("RMS 4-го канала по частотам для всех экспериментов")
plt.grid(True)
plt.legend(title="Эксперимент")
plt.tight_layout()
plt.show()
