import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# НАСТРОЙКИ
# =========================
CSV = r"D:\PROJECTs\magnet\harvester\experiments\model_80mm\f2hz_v100\timeseries.csv"
FS = 1000.0

# "experiment" | "model"
SOURCE = "model"

# для experiment — имя столбца с реальным сигналом
EXPERIMENT_COLUMN = "emf_open_V"

# для model — имя столбца модели
MODEL_COLUMN = "emf_open_V"

# диапазон FFT
FREQ_MIN = 1.0
FREQ_MAX = 200.0

# порог пиков
PEAK_REL_THR = 0.1   # 10% от максимума


# =========================
# ЗАГРУЗКА ДАННЫХ
# =========================
df = pd.read_csv(CSV)   # <-- ВАЖНО: обычный CSV, без sep=";" и decimal=","

if SOURCE == "experiment":
    if EXPERIMENT_COLUMN not in df.columns:
        raise ValueError(f"Нет столбца {EXPERIMENT_COLUMN} в CSV")
    y = df[EXPERIMENT_COLUMN].to_numpy()
    label_src = "Эксперимент"

elif SOURCE == "model":
    if MODEL_COLUMN not in df.columns:
        raise ValueError(f"Нет столбца {MODEL_COLUMN} в CSV")
    y = df[MODEL_COLUMN].to_numpy()
    label_src = "Модель"

else:
    raise ValueError("SOURCE должен быть 'experiment' или 'model'")

t = np.arange(len(y)) / FS
y = y - np.mean(y)   # убрать DC


# =========================
# FFT (окно только для спектра)
# =========================
w = np.hanning(len(y))
yw = y * w

X = np.fft.rfft(yw)
freqs = np.fft.rfftfreq(len(yw), d=1/FS)
amp = np.abs(X)

# диапазон
mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
f = freqs[mask]
a = amp[mask]


# =========================
# ПОИСК ПИКОВ
# =========================
thr = a.max() * PEAK_REL_THR

peaks = np.where(
    (a[1:-1] > thr) &
    (a[1:-1] > a[:-2]) &
    (a[1:-1] > a[2:])
)[0] + 1

pf = f[peaks]
pa = a[peaks]

order = np.argsort(pa)[::-1]
pf = pf[order]
pa = pa[order]

print(f"\nИсточник: {label_src}")
print("Топ пиков (первый ≈ частота шейкера):")
for i in range(min(10, len(pf))):
    print(f"{i+1:2d}) {pf[i]:7.3f} Hz   amp={pa[i]:.3g}")


# =========================
# ГРАФИКИ
# =========================
plt.figure(figsize=(12,4))
plt.plot(t, y, lw=0.7)
plt.title(f"{label_src}: сигнал во времени")
plt.xlabel("Время, сек")
plt.ylabel("Амплитуда")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(f, a, lw=0.9)
plt.scatter(pf[:10], pa[:10], s=40, color="red")
plt.title(f"{label_src}: спектр FFT ({FREQ_MIN}–{FREQ_MAX} Гц)")
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
