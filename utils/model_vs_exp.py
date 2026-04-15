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
alpha = 0.001     # новый параметр alpha, А/м
R_coil = 0.01225 + 0.00045  # радиус катушки, м
# --- Электрическая часть ---
N_turns = 56     # число витков катушки
dc = h / N_turns  # шаг между витками катушки
z_coil = np.linspace(0, h, N_turns)  # координаты колец катушки
R_resistance = 0.001 # сопротивление R

# --- Начальные условия ---
z0 = 0.025
v0 = 0.0
I0 = 0.0        # начальный ток
t_span = (0, 20)
t_eval = np.linspace(0, 20, 10000)
S_coil = np.pi * R_coil**2  # площадь поперечного сечения катушки
Bc = (np.pi * Br**2 * R**4) / (4 * mu0)
L_inductance = (mu0 * h * S_coil) / (R_coil*2)**2  # индуктивность катушки

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
    dI_dt = (alpha * total_dB_dz(z, z_coil) - R_resistance * I) / L_inductance
    return [v, dv_dt, dI_dt]

def emf_i(z, v):
    total_emf = sum(-S_coil * v * dB_dz(z - zj) for zj in z_coil)
    return total_emf

def main(f):
    print(f"\n=== Симуляция для f = {f} Гц ===")
    w = 2 * np.pi * f
    file_path = f'experiments/harvester_50mm/exp_1/data/{f}.csv'

    # Решаем систему для [z, v, I]
    sol = solve_ivp(lambda t, y: ode_system(t, y, w, z_coil),
                    t_span, [z0, v0, I0],
                    t_eval=t_eval, method='RK45')

    z = sol.y[0]
    v = sol.y[1]
    I = sol.y[2]
    dI_dt = np.array([ode_system(t, [zz, vv, ii], w, z_coil)[2] for t, zz, vv, ii in zip(sol.t, z, v, I)])

    # Вычисляем индукционную ЭДС
    e_i = np.array([emf_i(zz, vv) for zz, vv in zip(z, v)])

    # Чтение данных из CSV
    csv_data = []
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file, delimiter=';')  # Разделитель — ';'
            for row in reader:
                if len(row) >= 4:
                    try:
                        value = row[3].replace(',', '.')  # Замена ',' на '.' для float
                        csv_data.append(float(value))
                    except ValueError as e:
                        print(f"Предупреждение: Не удалось преобразовать значение '{row[3]}' в float: {e}. Пропускаем строку.")
        csv_data = np.array(csv_data)
    except FileNotFoundError:
        print(f"Файл {file_path} не найден.")
        csv_data = None

    # Построение графиков в одном окне на двух подграфиках
    fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)  # Два графика один над другим, общая ось X

    # График для индукционной ЭДС (e_i)
    axs[0].plot(t_eval, e_i, label='Индукционная ЭДС (e_i), В', color='blue')
    axs[0].set_ylabel('ЭДС, В')
    axs[0].set_title(f'Индукционная ЭДС для f = {f} Гц (модель)')
    axs[0].legend()
    axs[0].grid(True)

    if csv_data is not None and len(csv_data) > 0:
        # Генерация времени для CSV независимо от длины
        t_csv = np.linspace(0, 20, len(csv_data))
        
        # График для данных из CSV
        axs[1].plot(t_csv, csv_data, label='Данные из CSV (4-й столбец)', color='red', linestyle='--')
        axs[1].set_xlabel('Время, с')
        axs[1].set_ylabel('ЭДС, В')
        axs[1].set_title(f'Данные из CSV для f = {f} Гц (эксперимент)')
        axs[1].legend()
        axs[1].grid(True)
        
        # Вычисление глобального минимума и максимума из обеих кривых для единого масштаба Y (опционально)
        global_min = min(np.min(e_i), np.min(csv_data))
        global_max = max(np.max(e_i), np.max(csv_data))
    else:
        print("Данные из CSV отсутствуют.")
        # Если CSV нет, скрываем второй subplot
        axs[1].axis('off')

    plt.tight_layout()  # Автоматическая подгонка布局
    plt.show()

if __name__ == "__main__":
    main(20)