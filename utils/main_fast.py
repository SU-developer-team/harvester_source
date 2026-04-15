import numpy as np
from scipy.integrate import solve_ivp
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing
import math
# --- Параметры системы ---
m = 0.020        # масса магнита, кг
X0 = 0.01        # амплитуда вынужденных колебаний платформы, м
Cf = 0.82        # коэффициент аэродинамического сопротивления
rho = 1.2        # плотность воздуха, кг/м³
R = 0.0195 / 2   # радиус магнита, м
S = np.pi * R**2 # площадь поперечного сечения
g = 9.81         # ускорение свободного падения, м/с²
Br = 3   # остаточная магнитная индукция, Тл
c = 1.5 * R     # параметр для расчёта сил
h = 0.0572       # расстояние до верхнего магнита, м
L = 0.01         # длина цилиндрического магнита, м
mu0 = 1.25663706e-6  # магнитная постоянная, Гн/м
alpha = 0.001    # параметр alpha, А/м
R_coil = 0.01225 + 0.00045  # радиус катушки, м
N_turns = 56     # число витков катушки
dc = h / N_turns # шаг между витками катушки
z_coil = np.linspace(0.0036, h-0.0036, N_turns)  # координаты колец катушки
R_resistance = 0.1  # сопротивление, Ом
z0 = 0.025
v0 = 0.0
I0 = 0.0
t_span = (0, 2)
t_eval = np.linspace(0, 2, 2000)
S_coil = np.pi * R_coil**2
Bc = (np.pi * Br**2 * R**4) / (4 * mu0)
L_inductance = (mu0 * h * S_coil) / (R_coil*2)**2

def calc_distance(z, static_pos):
    return abs(static_pos - z + L/2)

def F1(z):
    term1 = (1 / (c + calc_distance(z, h))**2)
    term2 = (1 / (2*L + c + calc_distance(z, h))**2)
    term3 = (-2 / (L + c + calc_distance(z, h))**2)
    F = Bc * (term1 + term2 + term3)
    F_max = 50.0 
    
    F_sat = F_max * math.tanh(F / max(F_max, 1e-9))

    return float(F_sat) 

def F3(z):
    term1 = (1 / (c + calc_distance(z, 0))**2)
    term2 = (1 / (2*L + c + calc_distance(z, 0))**2)
    term3 = (-2 / (L + c + calc_distance(z, 0))**2)
    F = Bc * (term1 + term2 + term3)
    F_max = 50.0 
    
    F_sat = F_max * math.tanh(F / max(F_max, 1e-9))

    return float(F_sat) 

def dB_dz(z, zj):
    dh = np.abs(z - zj)
    term_1 = R**2 / ((dh + L/2)**2 + R**2)**(3/2)
    term_2 = R**2 / ((dh - L/2)**2 + R**2)**(3/2)
    return Br / 2 * (term_1 - term_2)

def total_dB_dz(z, z_coil):
    return np.sum([dB_dz(z, zj) for zj in z_coil])

def F_shaker(t, w):
    return m * X0 * w**2 * np.cos(w * t)

def ode_system(t, y, w, z_coil):
    z, v, I = y
    dv_dt = (F3(z) - F1(z) + F_shaker(t, w) - (Cf * rho * S * v**2) / 2 - m * g) / m
    dI_dt = (alpha * total_dB_dz(z, z_coil) - R_resistance * I) / L_inductance
    return [v, dv_dt, dI_dt]

def emf_i(z, v):
    total_emf = np.sum([-S_coil * v * dB_dz(z - zj, zj) for zj in z_coil])
    return total_emf

def compute_for_frequency(f):
    print(f"\n=== Симуляция для f = {f} Гц (в параллельном процессе) ===")
    w = 2 * np.pi * f

    # Определяем функции событий внутри compute_for_frequency, чтобы захватить w и z_coil
    def event_z_max(t, y):
        z = y[0]
        return z - h  # z - h = 0, когда z = h
    event_z_max.terminal = True
    event_z_max.direction = 1

    def event_z_min(t, y):
        z = y[0]
        return z      # z = 0, когда z = 0
    event_z_min.terminal = True
    event_z_min.direction = -1

    sol = solve_ivp(
        lambda t, y: ode_system(t, y, w, z_coil),
        t_span,
        [z0, v0, I0],
        t_eval=t_eval,
        method='LSODA',
        events=[event_z_max, event_z_min]
    )
    if sol.t_events[0].size > 0:
        print(f"Частота {f} Гц: Событие z > h произошло на t={sol.t_events[0][0]:.6f} с, z={sol.y_events[0][0][0]:.6f} м")
    if sol.t_events[1].size > 0:
        print(f"Частота {f} Гц: Событие z < 0 произошло на t={sol.t_events[1][0][0]:.6f} с, z={sol.y_events[1][0][0]:.6f} м")

    z = sol.y[0]
    v = sol.y[1]
    I = sol.y[2]
    dI_dt = np.array([ode_system(t, [zz, vv, ii], w, z_coil)[2] for t, zz, vv, ii in zip(sol.t, z, v, I)])
    
    e_i = np.array([emf_i(zz, vv) for zz, vv in zip(z, v)])
    e_self = -L_inductance * dI_dt
    e_tot = e_i - e_self
    
    rms_i = np.sqrt(np.mean(e_i**2))
    rms_self = np.sqrt(np.mean(e_self**2))
    rms_tot = rms_i - rms_self
    amplitude = np.sqrt(np.mean(z**2))
    
    return f, amplitude, rms_i, rms_self, rms_tot

if __name__ == "__main__":
    frequencies = np.arange(2, 50+1, 0.2)
    num_processes = min(multiprocessing.cpu_count(), len(frequencies))
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(compute_for_frequency, frequencies)
    
    results.sort(key=lambda x: x[0])
    amplitudes = [res[1] for res in results]
    rms_emf_i = [res[2] for res in results]
    rms_emf_self = [res[3] for res in results]
    rms_emf_tot = [res[4] for res in results]
    
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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'logs/rms_h572_50mm_{timestamp}.png')
    plt.show()
    
    # Сохраняем значения RMS в CSV
    with open(f'logs/rms_h572_50mm_{timestamp}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frequency (Hz)', 'RMS_EMF_I', 'RMS_EMF_SELF', 'RMS_EMF_TOT'])
        for f, e_i, e_self, e_tot in zip(frequencies, rms_emf_i, rms_emf_self, rms_emf_tot):
            writer.writerow([f, e_i, e_self, e_tot])
    
    # --- График амплитуды ---
    plt.figure(figsize=(8, 5))
    plt.plot(frequencies, amplitudes, marker='o')
    plt.xlabel('Частота f (Гц)')
    plt.ylabel('Амплитуда смещения |z| (м)')
    plt.title('Амплитуда смещения в зависимости от частоты')
    plt.savefig(f'logs/rms_amplitude_h572_50mm_{timestamp}.png')
    plt.grid(True)
    plt.show()