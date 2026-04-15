import os
import pandas as pd

# Путь к директории с файлами
directory = r"D:\PROJECTs\magnet\harvester\experiments\harvester_50mm\exp_1\synthesized_data"

# Создание директории для результатов, если она не существует
output_directory = os.path.join(directory, "min_max_results")
os.makedirs(output_directory, exist_ok=True)

# Обработка всех CSV-файлов в директории
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        
        # Чтение CSV-файла
        df = pd.read_csv(filepath, sep=';')
        
        # Расчет минимума и максимума для каждого столбца
        result = {
            'Column': ['displacement_mm', 'acc_mps2', 'velocity_mps'],
            'Minimum': [
                df['displacement_mm'].min(),
                df['acc_mps2'].min(),
                df['velocity_mps'].min()
            ],
            'Maximum': [
                df['displacement_mm'].max(),
                df['acc_mps2'].max(),
                df['velocity_mps'].max()
            ]
        }
        
        # Создание DataFrame с результатами
        result_df = pd.DataFrame(result)
        
        # Сохранение результатов в новый CSV-файл
        output_filename = f"min_max_{filename}"
        output_filepath = os.path.join(output_directory, output_filename)
        result_df.to_csv(output_filepath, sep=';', index=False)
        print(f"Результаты для {filename} сохранены в {output_filepath}")

print("Обработка всех файлов завершена.")