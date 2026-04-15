# -*- coding: utf-8 -*-
"""
Модель линейного электромагнитного энергохарвестера.

Считает механику движения магнита и электрическую цепь катушки
с нагрузкой, при возбуждении от шейкера (ускорение либо синус).
"""

from typing import Optional, Tuple
import math
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize  # Для калибровки

class ElectromagneticHarvesterID50:
    def __init__(self, folder_path: Optional[str] = None):
        """Инициализация модели с базовыми параметрами."""

        # --- Масса магнита и гравитация ---
        self.m_kg = 0.021
        self.g_mps2 = 9.81

        # --- Магнитные параметры ---
        self.mu0_Hpm = 4 * np.pi * 1e-7
        self.B_T = 0.9399  # 0.9399 характерная индукция для dB/dz
        self.k_accel = 1.0
        self.k_emf = 0.129

        # --- Геометрия магнита ---
        self.magnet_radius_m = 0.0195 / 2
        self.magnet_height_m = 0.010

        # --- Геометрия катушки ---
        self.coil_height_m = 0.050
        self.coil_inner_diam_m = 0.0205
        self.coil_outer_diam_m = 0.0265
        self.wire_diam_m = 0.000892
        self.coil_diam_m = self.coil_outer_diam_m + self.wire_diam_m
        self.turns_N = 56

        # --- Положение элементов по оси z (м) ---

        self.coil_z_bottom_m = self.magnet_height_m + 0.002
        self.coil_z_top_m = self.coil_z_bottom_m + self.coil_height_m

        self.top_magnet_z_bottom_m = self.coil_z_top_m + 0.002
        self.top_magnet_z_top_m = self.top_magnet_z_bottom_m + self.magnet_height_m

        self.bottom_magnet_z_top_m = self.magnet_height_m
        self.bottom_magnet_z_bottom_m = 0

        self.top_magnet_center_m = 0.5 * (self.top_magnet_z_top_m + self.top_magnet_z_bottom_m)
        self.bottom_magnet_center_m = 0.5 * (self.bottom_magnet_z_top_m + self.bottom_magnet_z_bottom_m)

        # --- Начальные условия ---
        self.z0_m = 0.040
        self.v0_mps = 0.0

        # --- Данные базы шейкера ---
        self.folder_path = folder_path
        self.base_time_s = None
        self.base_accel_mps2 = None
        self.base_accel_interp = None

        # --- Геометрия катушки и сопротивление ---
        self._precompute_coil_geometry()

        # --- Электрические параметры ---
        self.load_resistance_ohm = 1e6  # вход осциллографа

        # --- Для хранения разложения сил ---
        self._last_forces = None

        self.use_linear_friction = True
        self.c_damping = 0.05  # Коэффициент линейного демпфирования, Н·с/м — начните с 0.01–0.1 и настройте
        # Параметры для общей функции трения (по умолчанию, подберите по эксперименту)
        self.F_c = 0.5  # Максимальная сила трения, Н
        self.beta = 10  # Коэффициент чувствительности к скорости, с/м

    # ---------- Геометрия/электрика ----------
    def _precompute_coil_geometry(self):
        """Рассчитать площадь витка, индуктивность и R катушки по геометрии."""
        r_mean_m = (self.coil_inner_diam_m + self.coil_outer_diam_m) / 4.0
        self.coil_turn_area_m2 = np.pi * r_mean_m**2
        self.coil_inductance_H = self.mu0_Hpm * self.turns_N**2 * self.coil_turn_area_m2 / self.coil_height_m

        rho_cu_ohm_m = 1.68e-8
        total_wire_len_m = (2.0 * np.pi * r_mean_m) * self.turns_N
        wire_area_m2 = np.pi * (self.wire_diam_m / 2.0) ** 2
        self.coil_resistance_ohm = rho_cu_ohm_m * total_wire_len_m / max(wire_area_m2, 1e-12)

    def set_electrical(self, R_coil_ohm=None, R_load_ohm=None):
        """Установить сопротивления катушки и нагрузки."""
        if R_coil_ohm is not None:
            self.coil_resistance_ohm = float(R_coil_ohm)
        if R_load_ohm is not None:
            self.load_resistance_ohm = float(R_load_ohm)

    # ---------- Загрузка базы шейкера ----------
    def load_base_from_csv(self, file_path, time_col="t", accel_col="acc_mps2",
                           sep=";", decimal=".", normalize_time=True):
        """Загрузить CSV с (t, acc_mps2) и построить интерполяцию ускорения."""
        import os
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        df = pd.read_csv(file_path, sep=sep, decimal=decimal)
        if time_col not in df or accel_col not in df:
            raise ValueError(f"В {file_path} должны быть колонки '{time_col}' и '{accel_col}'")

        df = df.groupby(time_col, as_index=False)[accel_col].mean().sort_values(time_col)
        t_s = df[time_col].to_numpy(dtype=float)
        a_mps2 = df[accel_col].to_numpy(dtype=float)

        if normalize_time:
            t_s -= t_s[0]

        self.base_time_s = t_s
        self.base_accel_mps2 = a_mps2
        self.base_accel_interp = interp1d(t_s, a_mps2, kind="linear",
                                          bounds_error=False, fill_value=(a_mps2[0], a_mps2[-1]))

    # ---------- Силы ----------
    def force_from_top_magnet(self, z_free_magnet_center):
        return -self._empirical_pair_force(self.top_magnet_center_m - z_free_magnet_center)
    def force_from_bottom_magnet(self, z_free_magnet_center): 
        return self._empirical_pair_force(z_free_magnet_center - self.bottom_magnet_center_m)

    def _empirical_pair_force(self, x_m: float) -> float:
        """
        Упрощённая осевая модель силы между магнитами с регуляризацией и мягким насыщением.
        Исключаем сингулярности и нереалистично огромные значения, чтобы интегратор не залипал.
        """
        Br = self.B_T
        R = self.magnet_radius_m
        mu0 = self.mu0_Hpm

        c = 0.8*R
        xc = x_m + c
        L = self.magnet_height_m

        # Регуляризация, чтобы не делить на ~0
        eps = 1e-9
        term1 = 1.0 / (xc*xc + eps)
        term2 = 1.0 / (((2.0 * L + xc) ** 2) + eps)
        term3 = 2.0 / (((L + xc) ** 2) + eps)

        F = (np.pi * Br**2 * R**4 / (4.0 * mu0)) * (term1 + term2 - term3)

        # Мягкое насыщение силы (tanh), чтобы не улетало в десятки/сотни Н
        F_max = 50.0  # подберите по эксперименту: 10..30 Н обычно хватает
        F_sat = F_max * math.tanh(F / max(F_max, 1e-9))

        return float(F_sat)

    # Общая функция трения, зависящая только от скорости
    def force_trenia(self, v_mps: float) -> float:
        if self.use_linear_friction:
            return -self.c_damping * v_mps
        else:
            return -self.F_c * math.tanh(self.beta * v_mps)


    # ---------- Магнитное поле / ЭДС ----------
    def dB_dz(self, z_free_magnet_center):
        r = self.magnet_radius_m
        L = 0.5 * self.magnet_height_m
        z1, z2 = L + z_free_magnet_center, L - z_free_magnet_center
        term1 = r**2 / ((z1**2 + r**2) ** 1.5)
        term2 = r**2 / ((z2**2 + r**2) ** 1.5)
        return (self.B_T / 2.0) * (term1 - term2)

    def compute_emf(self, z_free_magnet_center, v_mps):
        """ЭДС по закону Фарадея для всех витков катушки."""
        z_turns = self.coil_z_bottom_m + np.arange(self.turns_N) * self.wire_diam_m
        dB_dz = self.dB_dz(z_free_magnet_center - z_turns)
        total_emf = np.sum(self.coil_turn_area_m2 * dB_dz * v_mps)
        return -total_emf

    def compute_emf_batch(self, z_arr, v_arr):
        return np.array([self.compute_emf(z, v) for z, v in zip(z_arr, v_arr)])

    def compute_self_emf(self, i_a, t_s):
        return -self.coil_inductance_H * np.gradient(i_a, t_s)

    # ---------- Возбуждение ----------
    def shaker_force(self, t_s):
        if self.base_accel_interp is None:
            raise ValueError("Данные ускорения не загружены. Вызовите load_base_from_csv для загрузки CSV-файла.")
        a_base = float(self.base_accel_interp(t_s))
        a_base *= self.k_accel
        return -self.m_kg * a_base

    # ---------- ОДУ ----------
    def _sum_forces(self, t_s, z_free_magnet_center, v_mps):
        """Суммарная сила и разложение."""
        F_bottom = self.force_from_bottom_magnet(z_free_magnet_center)
        F_top = self.force_from_top_magnet(z_free_magnet_center)
        F_gravity = -self.m_kg * self.g_mps2
        F_shaker = self.shaker_force(t_s)
        F_trenia = self.force_trenia(v_mps)
        
        F_total = (F_bottom + F_top) + F_gravity + F_shaker + F_trenia
        return F_total, (F_total, F_bottom, F_top, F_gravity, F_shaker, F_trenia)

    def _ode_rhs(self, t_s, y):
        """Правая часть ОДУ."""
        z_free_magnet_center, v_mps, i_a = y
        a_mps2 = self._sum_forces(t_s, z_free_magnet_center, v_mps)[0] / self.m_kg
        e_V = self.compute_emf(z_free_magnet_center, v_mps)
        R_series = self.coil_resistance_ohm + (self.load_resistance_ohm if np.isfinite(self.load_resistance_ohm) else 0.0)
        di_dt = (e_V - R_series * i_a) / max(self.coil_inductance_H, 1e-12)
        return [v_mps, a_mps2, di_dt]

    def compute_forces(self, t_s, z, v):
        """Вернуть массив сил по времени."""
        forces = [self._sum_forces(float(tt), float(zz), float(vv))[1] for tt, zz, vv in zip(t_s, z, v)]
        self._last_forces = np.column_stack([t_s, forces])
        return self._last_forces

    def solve_all(self, t_eval_s=None, t_max_s=None, steps=1000,
                  rtol=1e-9, atol=1e-9, max_step_s=None, clamp_to_base=True, dense_output=False):
        """Решить систему ОДУ."""
        if t_eval_s is None:
            if self.base_time_s is not None and clamp_to_base:
                base_dur = self.base_time_s[-1] - self.base_time_s[0]
                t_max_s = min(t_max_s or base_dur, base_dur)
            t_eval_s = np.linspace(0.0, float(t_max_s or 1.0), steps)
        else:
            t_eval_s = np.asarray(t_eval_s, dtype=float)
            if np.any(np.diff(t_eval_s) <= 0):
                raise ValueError("t_eval_s должен быть строго возрастающим.")

        if max_step_s is None:
            max_step_s = float('inf')

        y0 = [self.z0_m, self.v0_mps, 0.0]
        sol = solve_ivp(
            fun=self._ode_rhs,
            t_span=(float(t_eval_s[0]), float(t_eval_s[-1])),
            y0=y0,
            t_eval=t_eval_s,
            rtol=rtol,
            atol=atol,
            max_step=0.0001,
            dense_output=dense_output,
            method="BDF" if self.base_accel_interp is not None else "RK45"
            #method="RK45"
        )

        z_free_magnet_center, v_mps, i_a = sol.y
        emf_open_V = self.compute_emf_batch(z_free_magnet_center, v_mps)
        self_emf_V = self.compute_self_emf(i_a, sol.t)
        forces = self.compute_forces(sol.t, z_free_magnet_center, v_mps)

        if np.isfinite(self.load_resistance_ohm) and self.load_resistance_ohm > 0:
            v_terminal_V = i_a * self.load_resistance_ohm
        else:
            di_dt = np.gradient(i_a, sol.t)
            v_terminal_V = emf_open_V - self.coil_inductance_H * di_dt - self.coil_resistance_ohm * i_a

        return sol.t, z_free_magnet_center, v_mps, i_a, emf_open_V, self_emf_V, forces, v_terminal_V


    # ---------- Визуализация ----------
    def plot_forces(self, t_s, forces, save_dir=None, timestamp=None):
        """Построить график сил с разными стилями для каждой линии."""
        import matplotlib.pyplot as plt, os

        plt.figure(figsize=(11, 6))

        labels = ["F_total", "F_bottom", "F_top", "F_gravity", "F_shaker", "F_trenia"]

        # Наборы стилей: маркеры и линии будут чередоваться
        linestyles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'd', '*', 'v']

        for i, lbl in enumerate(labels, 1):
            style = linestyles[i % len(linestyles)]
            marker = markers[i % len(markers)]
            plt.plot(t_s, forces[:, i], label=lbl,
                     linestyle=style, marker=marker, markersize=4)

        plt.xlabel("Время, с")
        plt.ylabel("Сила, Н")
        plt.title("Силы на магнит")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save_dir and timestamp:
            plt.savefig(os.path.join(save_dir, f"forces_plot_{timestamp}.png"))
        else:
            plt.savefig("forces_plot.png")
        plt.show()