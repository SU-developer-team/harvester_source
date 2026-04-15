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


class ElectromagneticHarvesterID80mm:
    def __init__(self, folder_path: Optional[str] = None):
        """Инициализация модели с базовыми параметрами."""

        # --- Масса магнита и гравитация ---
        self.g_mps2 = 9.81
        self.magnet_count = 2
        self.m_kg = 0.021 * self.magnet_count

        # --- Магнитные параметры ---
        self.mu0_Hpm = 4 * np.pi * 1e-7
        self.B_T = 1.4471
        self.b_m = -0.0109 # характерная индукция для dB/dz

        # --- Геометрия магнита ---
        self.magnet_radius_m = 0.0195 / 2
        self.magnet_height_m = 0.010 * self.magnet_count

        # --- Геометрия катушки ---
        self.coil_height_m = 0.050
        self.coil_inner_diam_m = 0.0205
        self.coil_outer_diam_m = 0.0265
        self.wire_diam_m = 0.001
        self.turns_N = 56

        # --- Положение элементов по оси z (м) ---
        self.coil_z_bottom_m = self.magnet_height_m + 0.002
        self.coil_z_top_m = self.coil_z_bottom_m + self.coil_height_m + 0.03
        self.top_magnet_z_bottom_m = self.coil_z_top_m + 0.002
        self.top_magnet_z_top_m = self.top_magnet_z_bottom_m + self.magnet_height_m
        self.bottom_magnet_z_top_m = self.coil_z_bottom_m - 0.002
        self.bottom_magnet_z_bottom_m = self.bottom_magnet_z_top_m - self.magnet_height_m
        self.top_magnet_center_m = 0.5 * (self.top_magnet_z_top_m + self.top_magnet_z_bottom_m)
        self.bottom_magnet_center_m = 0.5 * (self.bottom_magnet_z_top_m + self.bottom_magnet_z_bottom_m)

        # --- Возбуждение по умолчанию ---
        self.base_amp_x0_m = 0.001
        self.freq_hz = 25.0
        self.omega_rad = 2 * np.pi * self.freq_hz

        # --- Начальные условия ---
        self.z0_m = 0.037
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
        self.emf_scale = 1.0  # масштаб ЭДС

        # --- Для хранения разложения сил ---
        self._last_forces = None

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
    def force_from_top_magnet(self, z_m):  return self._empirical_pair_force(self.top_magnet_center_m - z_m)
    def force_from_bottom_magnet(self, z_m):  return self._empirical_pair_force(z_m - self.bottom_magnet_center_m)

    def _empirical_pair_force(self, x_m: float) -> float:
        """
        Упрощённая осевая модель силы между магнитами с регуляризацией и мягким насыщением.
        Исключаем сингулярности и нереалистично огромные значения, чтобы интегратор не залипал.
        """
        Br = self.B_T
        R = self.magnet_radius_m
        mu0 = self.mu0_Hpm

        c = (4.0 * R) / 5.0
        xc = x_m + c
        L = (self.top_magnet_z_top_m - self.bottom_magnet_z_bottom_m) / 2.0

        # Регуляризация, чтобы не делить на ~0
        eps = 1e-9
        term1 = 1.0 / (xc*xc + eps)
        term2 = 1.0 / (((2.0 * L + xc) ** 2) + eps)
        term3 = 2.0 / (((L + xc) ** 2) + eps)

        F = (np.pi * Br**2 * R**4 / (4.0 * mu0)) * (term1 + term2 - term3) - self.b_m

        # Мягкое насыщение силы (tanh), чтобы не улетало в десятки/сотни Н
        F_max = 15.0  # подберите по эксперименту: 10..30 Н обычно хватает
        F_sat = F_max * math.tanh(F / max(F_max, 1e-9))

        return F


    def _sign(self, x, eps=1e-12): return 0.0 if abs(x) < eps else (1.0 if x > 0 else -1.0)

    def force_quadratic_drag(self, v_mps, rho_air=1.225, cd=1.2, area_m2=1e-4):
        return -0.5 * rho_air * cd * area_m2 * v_mps * abs(v_mps)

    def force_air_shear(self, v_mps, gap_m):
        if gap_m <= 0:
            raise ValueError("Зазор (gap) должен быть > 0.")
        mu_air = 1.81e-5
        return -(2.0 * math.pi * self.magnet_radius_m * self.magnet_height_m * mu_air * v_mps) / gap_m

    def force_wall_coulomb(self, v_mps: float, mu_s: float = 0.25, mu_c: float = 0.20,
                        v_s: float = 0.01, v_eps: float = 1e-3) -> float:
        """
        Сглаженная модель сухого трения:
        - Сtribeck-переход от статического (mu_s) к кинетическому (mu_c) при |v|~v_s
        - Плавный знак скорости через v/(|v|+v_eps), без скачка
        Параметры можно подбирать: mu_s>=mu_c, v_s ~ 0.005..0.02 м/с, v_eps ~ 1e-3..1e-4.
        """
        N = self.m_kg * self.g_mps2
        # плавный знак (избегаем скачка в 0)
        sign_smooth = v_mps / (abs(v_mps) + v_eps)
        # коэффициент трения по Stribeck
        mu = mu_c + (mu_s - mu_c) * math.exp(-(v_mps / max(v_s, 1e-9))**2)
        return -mu * N * sign_smooth


    # ---------- Магнитное поле / ЭДС ----------
    def dB_dz(self, z_m):
        r = self.magnet_radius_m
        z1, z2 = self.magnet_height_m + z_m, self.magnet_height_m - z_m
        term1 = r**2 / ((z1**2 + r**2) ** 1.5)
        term2 = r**2 / ((z2**2 + r**2) ** 1.5)
        return (self.B_T / 2.0) * (term1 - term2)

    def compute_emf(self, z_m, v_mps):
        """ЭДС по закону Фарадея для всех витков катушки."""
        total_emf = sum(self.coil_turn_area_m2 * self.dB_dz(z_m - (self.coil_z_bottom_m + turn * self.wire_diam_m)) * v_mps
                        for turn in range(self.turns_N))
        return -self.emf_scale * total_emf

    def compute_emf_batch(self, z_arr, v_arr):
        return np.array([self.compute_emf(z, v) for z, v in zip(z_arr, v_arr)])

    def compute_self_emf(self, i_a, t_s):
        return -self.coil_inductance_H * np.gradient(i_a, t_s)

    # ---------- Возбуждение ----------
    def shaker_force(self, t_s):
        """Сила от шейкера: по записи или синусу."""
        if self.base_accel_interp is not None:
            a_base = float(self.base_accel_interp(t_s))
        else:
            a_base = -(self.omega_rad ** 2) * self.base_amp_x0_m * math.cos(self.omega_rad * t_s)
        return -self.m_kg * a_base

    # ---------- ОДУ ----------
    def _sum_forces(self, t_s, z_m, v_mps):
        """Суммарная сила и разложение."""
        gap_m = (self.coil_inner_diam_m - 2.0 * self.magnet_radius_m) / 2.0
        F_bottom, F_top = self.force_from_bottom_magnet(z_m), self.force_from_top_magnet(z_m)
        F_gravity, F_shaker = -self.m_kg * self.g_mps2, self.shaker_force(t_s)
        F_quad, F_air = self.force_quadratic_drag(v_mps), self.force_air_shear(v_mps, gap_m)
        F_wall = self.force_wall_coulomb(v_mps)
        F_total = (F_bottom - F_top) + F_gravity + F_shaker + F_quad + F_air + F_wall
        return F_total, (F_total, F_bottom, F_top, F_gravity, F_shaker, F_quad, F_air, F_wall)

    def _ode_rhs(self, t_s, y):
        """Правая часть ОДУ."""
        z_m, v_mps, i_a = y
        a_mps2 = self._sum_forces(t_s, z_m, v_mps)[0] / self.m_kg
        e_V = self.compute_emf(z_m, v_mps)
        R_series = self.coil_resistance_ohm + (self.load_resistance_ohm if np.isfinite(self.load_resistance_ohm) else 0.0)
        di_dt = (e_V - R_series * i_a) / max(self.coil_inductance_H, 1e-12)
        return [v_mps, a_mps2, di_dt]

    def compute_forces(self, t_s, z, v):
        """Вернуть массив сил по времени."""
        forces = [self._sum_forces(float(tt), float(zz), float(vv))[1] for tt, zz, vv in zip(t_s, z, v)]
        self._last_forces = np.column_stack([t_s, forces])
        return self._last_forces

    def solve_all(self, t_eval_s=None, t_max_s=None, steps=1000,
                  rtol=1e-6, atol=1e-9, max_step_s=None, clamp_to_base=True, dense_output=False):
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

        # Автоматический max_step_s по минимальному шагу в сетке
        #if max_step_s is None and len(t_eval_s) > 1:
        #    max_step_s = float(np.min(np.diff(t_eval_s)))

        # --- ЖЁСТКИЙ ФИКС max_step: SciPy не принимает None ---
        if max_step_s is None:
            max_step_s = float('inf')  # бесконечный шаг = без ограничения
        # (опционально для дебага)
        # print("[dbg] max_step_s:", max_step_s, type(max_step_s))

        t_span = (float(t_eval_s[0]), float(t_eval_s[-1]))

        y0 = [self.z0_m, self.v0_mps, 0.0]
        sol = solve_ivp(
            fun=self._ode_rhs,
            t_span=(float(t_eval_s[0]), float(t_eval_s[-1])),
            y0=[self.z0_m, self.v0_mps, 0.0],
            t_eval=t_eval_s,
            rtol=rtol,
            atol=atol,
            max_step=max_step_s,              # пусть будет None
            dense_output=dense_output,
            method="BDF" if self.base_accel_interp is not None else "RK45"  # <<< ключ
        )


        z_m, v_mps, i_a = sol.y
        emf_open_V = self.compute_emf_batch(z_m, v_mps)
        self_emf_V = self.compute_self_emf(i_a, sol.t)
        forces = self.compute_forces(sol.t, z_m, v_mps)

        if np.isfinite(self.load_resistance_ohm) and self.load_resistance_ohm > 0:
            v_terminal_V = i_a * self.load_resistance_ohm
        else:
            di_dt = np.gradient(i_a, sol.t)
            v_terminal_V = emf_open_V - self.coil_inductance_H * di_dt - self.coil_resistance_ohm * i_a

        return sol.t, z_m, v_mps, i_a, emf_open_V, self_emf_V, forces, v_terminal_V

    # ---------- Визуализация ----------
    def plot_forces(self, t_s, forces, save_dir=None, timestamp=None):
        """Построить график сил."""
        import matplotlib.pyplot as plt, os
        plt.figure(figsize=(11, 6))
        labels = ["F_total", "F_bottom", "F_top", "F_gravity", "F_shaker", "F_quad_drag", "F_air_shear", "F_wall_friction"]
        for i, lbl in enumerate(labels, 1):
            plt.plot(t_s, forces[:, i], label=lbl, linestyle="--" if "F_" in lbl else "-")
        plt.xlabel("Время, с"); plt.ylabel("Сила, Н"); plt.title("Силы на магнит")
        plt.grid(True); plt.legend(); plt.tight_layout()
        if save_dir and timestamp:
            plt.savefig(os.path.join(save_dir, f"forces_plot_{timestamp}.png"))
        else:
            plt.savefig("forces_plot.png")
        plt.show()
