import numpy as np
from scipy.integrate import solve_ivp
import csv
import matplotlib.pyplot as plt
from datetime import datetime


# --- Параметры системы ---
m = 0.020        # масса магнита, кг
X0 = 0.001       # амплитуда вынужденных колебаний платформы, м
Cf = 0.82        # коэффициент аэродинамического сопротивления (безразмерный)
rho = 1.2        # плотность воздуха, кг/м³
R = 0.0195 / 2   # радиус магнита, м
S = np.pi * R**2 # площадь поперечного сечения: S = π·R²
g = 9.81         # ускорение свободного падения, м/с²
Br = 0.9399      # остаточная магнитная индукция (Тл)
c = 0.8 * R      # параметр для расчёта сил, c = 0.8R
h = 0.05         # расстояние до верхнего магнита, м
L = 0.01         # длина цилиндрического магнита, м (индуктивность катушки)
mu0 = 1.25663706e-6  # магнитная постоянная, Гн/м
alpha = 0.001    # новый параметр alpha, А/м
R_coil = 0.01225 + 0.00045  # радиус катушки, м
# --- Электрическая часть ---
N_turns = 56     # число витков катушки
dc = h / N_turns  # шаг между витками катушки
z_coil = np.linspace(0, h, N_turns)  # координаты колец катушки
R_resistance = 0.1 # сопротивление R

# --- Начальные условия ---
z0 = 0.025
v0 = 0.0
I0 = 0.0        # начальный ток
t_span = (0, 1)
t_eval = np.linspace(0, 1, 1000)
S_coil = np.pi * R_coil**2  # площадь поперечного сечения катушки
Bc = (np.pi * Br**2 * R**4) / (4 * mu0)
L_inductance = (mu0 *  h * S_coil) / (R_coil*2)**2  # индуктивность катушки


def F1(z):
    return Bc * (1 / (c + (h - z))**2 + 1 / (2*L + c + (h - z))**2 + 1 / (L + c + (h - z))**2)

def F3(z):
    return Bc * (1 / (c + z)**2 + 1 / (2*L + c + z)**2 + 1 / (L + c + z)**2)

def dB_dz(z):
    return 3 * mu0 * Br * R**2 * z / (2 * (R**2 + z**2)**(5/2))

def total_dB_dz(z, z_coil):
    return sum(dB_dz(z - zj) for zj in z_coil)

def ode_system(t, y, w, z_coil):
    z, v, I = y
    dv_dt = (F3(z) - F1(z) + m * X0 * w**2 * np.cos(w * t) - (Cf * rho * S * v**2) / 2 - m * g) / m
    dI_dt = (alpha * total_dB_dz(z, z_coil) - R_resistance * I) / L_inductance  # Учет alpha
    return [v, dv_dt, dI_dt]
 

frequencies = np.arange(2, 36, 1)
amplitudes = []
rms_emf_i = []  # RMS индукционной ЭДС
rms_emf_self = []  # RMS самоиндукции
rms_emf_tot = []  # RMS общей ЭДС

# --- Функция индукционной ЭДС (должна быть определена) ---
def emf_i(z, v):
    total_emf = sum(-S_coil * v * dB_dz(z - zj) for zj in z_coil)
    return total_emf

for f in frequencies:
    print(f"\n=== Симуляция для f = {f} Гц ===")
    w = 2 * np.pi * f

    # Решаем систему для [z, v, I]
    sol = solve_ivp(lambda t, y: ode_system(t, y, w, z_coil),
                    t_span, [z0, v0, I0],
                    t_eval=t_eval, method='RK45')

    z = sol.y[0]
    v = sol.y[1]
    I = sol.y[2]
    dI_dt = np.array([ode_system(t, [zz, vv, ii], w, z_coil)[2] for t, zz, vv, ii in zip(sol.t, z, v, I)])

    # Вычисляем ЭДС
    e_i = np.array([emf_i(zz, vv) for zz, vv in zip(z, v)])  # Индукционная ЭДС
    e_self = -L_inductance * dI_dt  # ЭДС самоиндукции
    e_tot = e_i - e_self  # Общая ЭДС

    # RMS значений

    rms_emf_i.append(np.sqrt(np.mean(e_i**2)))
    rms_emf_self.append(np.sqrt(np.mean(e_self**2)))
    rms_emf_tot.append(np.sqrt(np.mean(e_i**2)) - np.sqrt(np.mean(e_self**2)))

    # Амплитуда смещения
    amplitudes.append(np.max(np.abs(z)))

    
    # --- Графики F1, F3, F_total, v, z, a ---
    plt.figure(figsize=(12, 8))

    # Вычисляем F1, F3, F_total, ускорение a
    F1_values = np.array([F1(zz) for zz in z])
    F3_values = np.array([F3(zz) for zz in z])
    F_total = F3_values - F1_values
    a_values = np.array([(F3(zz) - F1(zz) - (Cf * rho * S * vv**2) / 2 - m * g) / m for zz, vv in zip(z, v)])

    # # График F1
    # plt.subplot(3, 2, 1)
    # plt.plot(sol.t, F1_values, label='F1', color='blue')
    # plt.xlabel('Время (с)')
    # plt.ylabel('F1 (Н)')
    # plt.title('Сила F1')
    # plt.grid(True)

    # # График F3
    # plt.subplot(3, 2, 2)
    # plt.plot(sol.t, F3_values, label='F3', color='green')
    # plt.xlabel('Время (с)')
    # plt.ylabel('F3 (Н)')
    # plt.title('Сила F3')
    # plt.grid(True)

    # # График F_total
    # plt.subplot(3, 2, 3)
    # plt.plot(sol.t, F_total, label='F_total', color='red')
    # plt.xlabel('Время (с)')
    # plt.ylabel('F_total (Н)')
    # plt.title('Суммарная сила F_total')
    # plt.grid(True)

    # # График скорости v
    # plt.subplot(3, 2, 4)
    # plt.plot(sol.t, v, label='v', color='orange')
    # plt.xlabel('Время (с)')
    # plt.ylabel('Скорость v (м/с)')
    # plt.title('Скорость v')
    # plt.grid(True)

    # # График смещения z
    # plt.subplot(3, 2, 5)
    # plt.plot(sol.t, z, label='z', color='purple')
    # plt.xlabel('Время (с)')
    # plt.ylabel('Смещение z (м)')
    # plt.title('Смещение z')
    # plt.grid(True)

    # # График ускорения a
    # plt.subplot(3, 2, 6)
    # plt.plot(sol.t, a_values, label='a', color='brown')
    # plt.xlabel('Время (с)')
    # plt.ylabel('Ускорение a (м/с²)')
    # plt.title('Ускорение a')
    # plt.grid(True)

    # plt.tight_layout()
    # plt.savefig('logs/forces_vs_time_{}.png'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    # plt.show()
    
# --- Графики RMS ---
plt.figure(figsize=(10, 6))
plt.plot(frequencies, rms_emf_i, label='RMS индукционной ЭДС', marker='o')
plt.plot(frequencies, rms_emf_self, label='RMS самоиндукции', marker='o')
plt.plot(frequencies, rms_emf_tot, label='RMS общей ЭДС', marker='o')
plt.xlabel('Частота f (Гц)')
plt.ylabel('RMS (В)')
plt.title('RMS ЭДС в зависимости от частоты')
plt.grid(True)
plt.legend()
plt.savefig('logs/rms_emf_vs_frequency_{}.png'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))
# Сохраняем значения RMS в .env файл
with open('logs/rms_50mm.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Frequency (Hz)', 'RMS_EMF_I', 'RMS_EMF_SELF', 'RMS_EMF_TOT'])
    for f, e_i, e_self, e_tot in zip(frequencies, rms_emf_i, rms_emf_self, rms_emf_tot):
        writer.writerow([f, e_i, e_self, e_tot])
plt.show()


# --- График амплитуды от частоты ---
# plt.figure(figsize=(8, 5))
# plt.plot(frequencies, amplitudes, marker='o')
# plt.xlabel('Частота f (Гц)')
# plt.ylabel('Амплитуда смещения |z| (м)')
# plt.title('Амплитуда смещения в зависимости от частоты')
# plt.grid(True)
# plt.show()
