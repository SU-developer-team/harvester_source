# magnet_simulation_refactored.py
"""
Симуляция движения магнита в системе с шейкером и катушкой, разбитая на
отдельные функции для удобства сопровождения и тестирования.

Запускайте скрипт напрямую, чтобы получить графики и PNG-файл с результатами.
Все основные параметры собраны в main(), так что их легко менять без рытья в
глубине кода.
"""
from __future__ import annotations

import os
import logging
from datetime import datetime
from pathlib import Path
import math
from typing import Tuple, Dict, Any

import numpy as np
from scipy.integrate import solve_ivp
import csv
import matplotlib.pyplot as plt

from models import Magnet, Shaker, Coil  # локальные модели пользователя



################################################################################
# ЛОГГИРОВАНИЕ
################################################################################


def configure_logger(name: str = "magnet_simulation",
                     log_dir: str | Path = "logs",
                     level_file: int = logging.DEBUG,
                     level_console: int = logging.INFO) -> logging.Logger:
    """Гибкая настройка логгера с выводом в файл и консоль."""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(min(level_file, level_console))

    # Формат лог-сообщений
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Файл
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fh = logging.FileHandler(log_dir / f"{ts}.log")
    fh.setLevel(level_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Консоль
    ch = logging.StreamHandler()
    ch.setLevel(level_console)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


################################################################################
# ФИЗИКА
################################################################################


def get_magnet_position(shaker: Shaker, t: float) -> float:
    """Положение шейкера во времени (синус)."""
    return shaker.X0 * np.sin(shaker.W * t)


def calculate_f_damping(v_m: float, magnet: Magnet) -> float:
    """Сила аэродинамического сопротивления турбулентного потока."""
    Cd = 1.2              # лобовое сопротивление
    rho = 1.225           # плотность воздуха, кг/м³
    area = math.pi * (magnet.diameter * 0.5) ** 2
    return 0.5 * rho * v_m ** 2 * Cd * np.sign(v_m)


def calculate_f_air(v_m: float, magnet: Magnet, gap: float) -> float:
    """Вязкое трение в кольцевом зазоре между магнитом и цилиндром."""
    if gap <= 0:
        raise ValueError("Диаметр магнита должен быть меньше диаметра цилиндра.")
    mu_air = 1.81e-5  # Па·с
    return -6 * math.pi * mu_air * magnet.height * v_m / gap


################################################################################
# ДИФФЕРЕНЦИАЛЬНАЯ СИСТЕМА
################################################################################


def combined_equations(
    t: float,
    y: np.ndarray,
    magnet: Magnet,
    shaker: Shaker,
    z_top: float,
    z_bottom: float,
    coil: Coil,
    resistance: float,
    gap: float,
) -> list[float]:
    """Правая часть ОДУ для solve_ivp."""
    (
        z_m, v_m,
        z_tm, v_tm,
        z_bm, v_bm,
        z_sk, v_sk,
        i,
    ) = y

    # --- Силы ---
    F_gravity = magnet.mass * shaker.G
    F_shaker = shaker.get_force(magnet, t)
    a_sk = F_shaker / magnet.mass

    top_offset = abs(z_m - magnet.height * 0.5 - z_top)
    bot_offset = abs(z_m - magnet.height * 0.5 - z_bottom)
    F_top_mag = magnet.get_force(top_offset + get_magnet_position(shaker, t))
    F_bot_mag = magnet.get_force(bot_offset + get_magnet_position(shaker, t))
    # F_top_mag = magnet.get_cheedket_force(top_offset + get_magnet_position(shaker, t))
    # F_bot_mag = magnet.get_cheedket_force(bot_offset + get_magnet_position(shaker, t))
     
    F_damping = calculate_f_damping(v_m, magnet)
    F_viscous = calculate_f_air(v_m, magnet, gap)

    F_total = -F_top_mag + F_bot_mag - F_gravity - F_damping + F_viscous
    a_m = F_total / magnet.mass

    # Для верхнего и нижнего магнитов берём то же a_sk (упрощение)
    a_tm = a_bm = a_sk

    # --- Электрика ---
    eds_per_turn, total_eds = coil.get_total_emf(shaker, z_m, v_m, t, a_m)
    L = coil.calculate_inductance()
    di_dt = (total_eds - resistance * i) / L

    return [v_m, a_m, v_tm, a_tm, v_bm, a_bm, v_sk, a_sk, di_dt]


################################################################################
# СИМУЛЯЦИЯ
################################################################################


def run_simulation(
    magnet: Magnet,
    shaker: Shaker,
    coil: Coil,
    *,
    z_top: float,
    z_bottom: float,
    magnet_start_z: float,
    gap: float,
    resistance: float = 0.1,
    sim_time: float = 5.0,
    points: int = 5000,
) -> Dict[str, Any]:
    """Запускает solve_ivp и возвращает результаты в словари."""
    y0 = [magnet_start_z, 0, z_top, 0, z_bottom, 0, 0.0, 0, 0]
    t_eval = np.linspace(0, sim_time, points)

    sol = solve_ivp(
        combined_equations,
        (0, sim_time),
        y0,
        args=(magnet, shaker, z_top, z_bottom, coil, resistance, gap),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-6,
    )

    # Пост-обработка: ускорение, ЭДС, самоиндукция
    v_m = sol.y[1]
    a_m = np.gradient(v_m, sol.t)

    total_eds = []
    for t, z_m, v in zip(sol.t, sol.y[0], v_m):
        _, e = coil.get_total_emf(shaker, z_m, v, t, np.nan)  # a_m не нужен для ЭДС в модели пользователя
        total_eds.append(e)
    total_eds = np.asarray(total_eds)

    L = coil.calculate_inductance()
    emf_self = -L * np.gradient(sol.y[8], sol.t)

    return {
        "t": sol.t,
        "z_m": sol.y[0],
        "v_m": v_m,
        "i": sol.y[8],
        "eds_induction": total_eds,
        "eds_self": emf_self,
        "eds_total": total_eds + emf_self,
    }


################################################################################
# ВИЗУАЛИЗАЦИЯ
################################################################################


def plot_results(results: Dict[str, np.ndarray], coil: Coil,
                 z_top: float, mu: float, save_dir: str | Path = ".") -> None:
    """Строит четыре графика и сохраняет PNG."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    t = results["t"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # 1. Положение
    axes[0].plot(t, results["z_m"], label="Магнит (z)")
    axes[0].axhline(coil.position, color="red", linestyle="--", label="Катушка нижняя")
    axes[0].axhline(coil.position + coil.height, color="red", linestyle="--", label="Катушка верхняя")
    axes[0].set_ylabel("Положение (м)")
    axes[0].legend(); axes[0].grid(True)

    # 2. Итоговая ЭДС
    axes[1].plot(t, results["eds_total"], color="red")
    axes[1].set_ylabel("Итоговая ЭДС (В)"); axes[1].grid(True)

    # 3. Внешняя + самоиндукция
    axes[2].plot(t, results["eds_induction"], label="Внешняя ЭДС")
    axes[2].plot(t, results["eds_self"], label="Самоиндукция")
    axes[2].set_ylabel("ЭДС (В)")
    axes[2].legend(); axes[2].grid(True)

    # 4. Ток
    axes[3].plot(t, results["i"], color="orange")
    axes[3].set_xlabel("Время (с)"); axes[3].set_ylabel("Ток (А)"); axes[3].grid(True)

    fig.tight_layout()
    out = save_dir / f"media/sim_{z_top}_{mu}.png"
    fig.savefig(out)
    plt.show()

def plot_emf(res: Dict[str, np.ndarray], *, z_top: float, mu: float, output: Path | str = ".") -> None:
    """Simplified visualization: only external EMF vs. self‑induction on a single panel (English labels)."""
    output = Path(output); output.mkdir(exist_ok=True)
    t = res["t"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), sharex=True)

    # External EMF and self‑induction with different styles / markers
    ax.plot(t, res["eds_induction"], label="External EMF", linestyle="-", marker="o", markevery=10)
    ax.plot(t, res["eds_self"],     label="Self‑induction EMF", linestyle="--", marker="s", markevery=10)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EMF (V)")
    ax.legend(); ax.grid(True)

    fig.tight_layout(); fig.savefig(Path(output) / f"media/emf_{z_top}_{mu}.png"); plt.show()

def get_mean_self_emf(results: Dict[str, np.ndarray]) -> float:
    """Возвращает среднее значение самоиндукции ЭДС."""
    return np.mean(abs(results["eds_self"])) if "eds_self" in results else 0.0

def get_mean_external_emf(results: Dict[str, np.ndarray]) -> float:
    """Возвращает среднее значение внешней ЭДС."""
    return np.mean(abs(results["eds_induction"])) if "eds_induction" in results else 0.0

def get_mean_total_emf(results: Dict[str, np.ndarray]) -> float:
    """Возвращает среднее значение полной ЭДС."""
    return np.mean(abs(results["eds_total"])) if "eds_total" in results else 0.0

################################################################################
# MAIN
################################################################################


def main(mu=5, plot_evrything: bool = False, amplitude: float = 0.015) -> Dict[str, Any]:
    """Точка входа: задаём параметры, запускаем симуляцию и строим графики."""
    logger = configure_logger()
    logger.info("Запуск симуляции…")

    # --- Параметры системы ---
    magnet = Magnet(diameter=0.0195, mass=0.043, height=0.01)
    z_top, z_bottom = 0.09, 0.01
    gap = (0.0205 - magnet.diameter) / 2  # м

    G = 9.8
    X0 = amplitude
    # mu = 5   # частота колебаний, Гц
    magnet_start_z = 0.045
    sim_time = 5.0

    shaker = Shaker(G=G, miew=mu, X0=X0)
    coil = Coil(
        turns_count=208,
        thickness=0.01025,
        radius=0.01025,
        position=0.015,
        magnet=magnet,
        wire_diameter=0.000961538462,
        layer_count=4,
    )

    # --- Считаем ---
    results = run_simulation(
        magnet, shaker, coil,
        z_top=z_top,
        z_bottom=z_bottom,
        magnet_start_z=magnet_start_z,
        gap=gap,
        resistance=0.1,
        sim_time=sim_time,
    )

    # --- Графики ---
    if plot_evrything:
        plot_results(results, coil, z_top, mu)
        plot_emf(results, z_top=z_top, mu=mu)
    else:
        logger.info("Графики отключены. Запустите с параметром plot_evrything=True для их отображения.") 
    logger.info(f"Симуляция завершена для {mu}. Результаты сохранены в словаре.")
    return results

if __name__ == "__main__":
    # mus = range(2, 70+2, 2)
    mus = np.arange(5, 25, 1)
    # [0.02, 0.019, 0.018, 0.001, 0.016, 0.015, 0.014, 0.013, 0.012, 0.011, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]
    amplitudes =  [
        0.006, 
        0.0048,
        0.0017,
        0.002, 
        0.0046, 
        0.007,
        0.0088, 
        0.0122, 
        0.02, 0.0195, 0.019, 0.018, 
        0.017, 0.009, 0.00760, 0.008634093005, 
        0.009727263027, 
        0.0105, 
        0.001, 
        0.0001
    ]
    print("Частоты для симуляции:", mus)
    print("Амплитуды для симуляции:", amplitudes)
    results = []

    for mu, amplitude in zip(mus, amplitudes):
       result = main(mu=mu, plot_evrything=False, amplitude=amplitude)
       results.append(get_mean_total_emf(result))
       print(f"Средняя полная ЭДС для mu={mu}: {results[-1] * 30} В")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_file = Path(f"logs/{ts}_results.csv")
    with output_file.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frequency (Hz)", "Mean Self-Induction EMF (V)"])
        writer.writerows(zip(mus, results))

    print(f"Results saved to {output_file}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(list(mus), results, marker='o')
    plt.xlabel("F (Гц)")
    plt.ylabel("Средняя ЭДС самоиндукции (В)")
    plt.title("Зависимость средней ЭДС самоиндукции от частоты колебаний")
    plt.grid(True)
    plt.tight_layout()
    plt.show()