import os
import pandas as pd

# Путь к директории с файлами
directory = r"D:\PROJECTs\magnet\harvester\experiments\harvester_50mm\exp_1\synthesized_data"

# Временной шаг (dt = 0.001 с, исходя из данных)
dt = 0.001

# Функция для вычисления скорости
def calculate_velocity(acc, dt):
    velocity = [0]  # Начальная скорость = 0
    for i in range(1, len(acc)):
        v = velocity[-1] + acc[i] * dt
        velocity.append(v)
    return velocity

# Обработка всех CSV-файлов в директории
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        
        # Чтение CSV-файла
        df = pd.read_csv(filepath, sep=';')
        
        # Вычисление скорости
        df['velocity_mps'] = calculate_velocity(df['acc_mps2'], dt)
        
        # Сохранение обновленного файла
        df.to_csv(filepath, sep=';', index=False)
        print(f"Обработан файл: {filename}")

print("Все файлы обработаны.")