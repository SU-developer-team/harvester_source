# Magnet Harvester (ID80mm)

Проект для моделирования электромагнитного виброэнергосборщика с двумя магнитами и катушкой, а также для сравнения модели с экспериментом по RMS и ЭДС.

## Установка
1. Установите Python 3.
2. (Опционально) Создайте и активируйте виртуальное окружение.
3. Установите зависимости:
   `pip install -r requirements.txt`

## Запуск
Перед запуском при необходимости правьте параметры в начале файлов (частоты, диапазоны, длительность, сопротивления, число процессов, путь вывода).

- Частотный прогон по max(z):
  `python double_z_magnet_pos.py`
  Результаты: `graphs_center_pos/experiment_<timestamp>/` плюс CSV и график `center_magnet_pos_vs_freq.png`.
- Модель + сравнение с экспериментом (RMS):
  `python main_double_magnets.py`
  Вход: `experiments/harvester_80mm/rms_compact_freq_rms.csv`
  Результаты: `graphs/experiment_<timestamp>/` (NPZ/CSV по частотам, сводные CSV, графики).
- Одна частота с сохранением временных рядов:
  `python main_double_magnets_save_for_freq.py`
  Результаты: `graphs/single_freq_<timestamp>_f<freq>Hz/` (CSV + графики).
- Пакетная сетка частота x базовая скорость:
  `python main_double_magnets_delta_z.py`
  Результаты: `D:/PROJECTs/magnet/harvester/experiments/model_80mm/exp2/` (настройте `OUT_ROOT` под свой путь).

## Файлы
- `model.py` - ядро модели `ElectromagneticHarvesterID80mm`: геометрия, параметры магнитов и катушки, силы (магнитные, гравитация, шейкер, сопротивление), расчет ЭДС, интегрирование ОДУ (`solve_ivp`), методы `set_frequency`, `set_base_velocity`, `set_electrical`, `solve_all`.
- `double_z_magnet_pos.py` - сканирование частоты и поиск `max(center_magnet_pos)`; сохраняет NPZ с временными рядами, CSV с итогами и график зависимости max(z) от частоты.
- `main_double_magnets.py` - расчет RMS по модели и сравнение с экспериментом; объединяет частоты, фильтрует диапазоны, считает RMS клеммного напряжения и самоиндукции, сохраняет сводные CSV и графики (min-max и raw).
- `main_double_magnets_delta_z.py` - пакетный расчет сетки частота x базовая скорость; для каждой точки пишет `timeseries.csv`, `summary_ranges.csv`, `meta.txt`; использует multiprocessing и `tqdm`.
- `main_double_magnets_save_for_freq.py` - одиночный прогон на одной частоте; сохраняет временные ряды, диапазоны (min/max/ptp) и графики по положению, скорости, ЭДС и току.

## Эксперименты 
- `experiments\harvester_50mm` - храниться записи от эксперимента с устройствой размером 50 мм
- `experiments\harvester_80mm` - храниться записи от эксперимента с размером 80 мм
- `experiments\harvester_80mm_v2` - храниться записи от эксперимента с размером 
- `experiments\harvester_80mm_v2\rms_compact_freq_rms.csv` - храниться записи в РМС формате всех частот