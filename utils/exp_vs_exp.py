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
alpha = 0.01     # новый параметр alpha, А/м
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
t_span = (0, 20)
t_eval = np.linspace(0, 20, 20000)
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
    file_path_50mm = f'experiments/harvester_50mm/exp_1/data/{f}.csv'
    file_path_80mm = f'experiments/harvester_80mm/exp_1/data/{f}.csv'
 
    # Чтение данных из CSV
    csv_data_50mm = []
    try:
        with open(file_path_50mm, 'r') as file:
            reader = csv.reader(file, delimiter=';')  # Разделитель — ';'
            for row in reader:
                if len(row) >= 4:
                    try:
                        value = row[3].replace(',', '.')  # Замена ',' на '.' для float
                        csv_data_50mm.append(float(value))
                    except ValueError as e:
                        print(f"Предупреждение: Не удалось преобразовать значение '{row[3]}' в float: {e}. Пропускаем строку.")
        csv_data_50mm = np.array(csv_data_50mm)
    except FileNotFoundError:
        print(f"Файл {file_path_50mm} не найден.")
        csv_data_50mm = None

    csv_data_80mm = []
    try:
        with open(file_path_80mm, 'r') as file:
            reader = csv.reader(file, delimiter=';')  # Разделитель — ';'
            for row in reader:
                if len(row) >= 4:
                    try:
                        value = row[3].replace(',', '.')  # Замена ',' на '.' для float
                        csv_data_80mm.append(float(value))
                    except ValueError as e:
                        print(f"Предупреждение: Не удалось преобразовать значение '{row[3]}' в float: {e}. Пропускаем строку.")
        csv_data_80mm = np.array(csv_data_80mm)
    except FileNotFoundError:
        print(f"Файл {file_path_80mm} не найден.")
        csv_data_80mm = None


    # Построение графиков в одном окне на двух подграфиках
    plt.figure(figsize=(12, 8))
    plt.suptitle(f'Индукционная ЭДС для f = {f} Гц', fontsize=16)
    plt.subplot(3, 1, 1)
    plt.plot(t_eval, csv_data_50mm, label='Индукционная ЭДС (50mm), В', color='blue')
    plt.plot(t_eval, csv_data_80mm, label='Индукционная ЭДС (80mm), В', color='red')
    plt.ylabel('ЭДС, В')
    plt.xlim(0, 1)
    plt.title('Модель')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main(35)