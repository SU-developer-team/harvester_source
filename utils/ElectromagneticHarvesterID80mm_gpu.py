# -*- coding: utf-8 -*-
"""
Линейный электромагнитный харвестер: чисто модельный режим.
"""

from typing import Optional, Tuple
import math
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

class ElectromagneticHarvesterID80mm:
    def __init__(self, folder_path: Optional[str] = None):
        # --- Масса и гравитация ---
        self.magnet_count = 2
        self.m_kg = 0.021 *self.magnet_count
        self.g_mps2 = 9.81

        # --- Магнитные параметры ---
        self.mu0_Hpm = 4 * np.pi * 1e-7
        self.B_T = 1.3711
        self.b_m = -0.0306

        # --- Геометрия магнита ---
        self.magnet_radius_m = 0.0195 / 2
        self.magnet_height_m = 0.010 * self.magnet_count

        # --- Геометрия катушки ---
        self.coil_height_m = 0.050
        self.coil_inner_diam_m = 0.0205
        self.coil_outer_diam_m = 0.0265
        self.wire_diam_m = 0.001
        self.turns_N = 55
        # self.alpha = 0.00035
        self.alpha = 0.4

        # --- Положение по оси z ---
        self.coil_z_bottom_m = self.magnet_height_m + 0.002
        self.coil_z_top_m = self.coil_z_bottom_m + self.coil_height_m 
        self.top_magnet_z_bottom_m = self.coil_z_top_m + 0.002 + 0.03
        self.top_magnet_z_top_m = self.top_magnet_z_bottom_m + self.magnet_height_m
        self.bottom_magnet_z_top_m = self.coil_z_bottom_m - 0.002
        self.bottom_magnet_z_bottom_m = self.bottom_magnet_z_top_m - self.magnet_height_m
        self.top_magnet_center_m = 0.5 * (self.top_magnet_z_top_m + self.top_magnet_z_bottom_m)
        self.bottom_magnet_center_m = 0.5 * (self.bottom_magnet_z_top_m + self.bottom_magnet_z_bottom_m)

        # --- Возбуждение по умолчанию (синус) ---
        self.base_amp_x0_m = 0.0025
        self.freq_hz = 25.0
        self.omega_rad = 2 * np.pi * self.freq_hz
        self.USE_EXP_DATA = False   # ### CHANGED: всегда модель

        # --- Начальные условия ---
        self.z0_m = 0.05
        self.v0_mps = 0.0

        # --- База шейкера (не используется в чисто модельном режиме) ---
        self.folder_path = folder_path
        self.base_time_s = None
        self.base_accel_mps2 = None
        self.base_accel_interp = None

        # --- Геометрия катушки и сопротивление ---
        self._precompute_coil_geometry()

        # --- Электрика ---
        self.load_resistance_ohm = 1e6
        self.emf_scale = 1.0

        self._last_forces = None

    # ---------- Сервис ----------
    def set_frequency(self, hz: float):
        """### CHANGED: удобный сеттер частоты шейкера"""
        self.freq_hz = float(hz)
        self.omega_rad = 2 * np.pi * self.freq_hz

    def _precompute_coil_geometry(self):
        r_mean_m = (self.coil_inner_diam_m + self.coil_outer_diam_m) / 4.0
        self.coil_turn_area_m2 = np.pi * r_mean_m**2
        self.coil_inductance_H = self.mu0_Hpm * self.turns_N**2 * self.coil_turn_area_m2 / self.coil_height_m

        rho_cu_ohm_m = 1.68e-8
        total_wire_len_m = (2.0 * np.pi * r_mean_m) * self.turns_N
        wire_area_m2 = np.pi * (self.wire_diam_m / 2.0) ** 2
        self.coil_resistance_ohm = rho_cu_ohm_m * total_wire_len_m / max(wire_area_m2, 1e-12)

    def set_electrical(self, R_coil_ohm=None, R_load_ohm=None):
        if R_coil_ohm is not None:
            self.coil_resistance_ohm = float(R_coil_ohm)
        if R_load_ohm is not None:
            self.load_resistance_ohm = float(R_load_ohm)

    # ---------- База шейкера (оставлено для совместимости, не использовать) ----------
    def load_base_from_csv(self, *args, **kwargs):
        """Оставлено для совместимости — в чисто модельном сценарии НЕ вызывать."""
        raise RuntimeError("В этой конфигурации не используется база шейкера из CSV.")

    # ---------- Силы ----------
    def force_from_top_magnet(self, z_m):  return self._empirical_pair_force(self.top_magnet_center_m - z_m)
    def force_from_bottom_magnet(self, z_m):  return self._empirical_pair_force(z_m - self.bottom_magnet_center_m)

    def _empirical_pair_force(self, x_m: float) -> float:
        Br = self.B_T
        R = self.magnet_radius_m
        mu0 = self.mu0_Hpm

        c = 1.05 * R
        xc = x_m + c
        L = (self.top_magnet_z_top_m - self.bottom_magnet_z_bottom_m) / 2.0

        eps = 1e-9
        term1 = 1.0 / (xc*xc)
        term2 = 1.0 / (((2.0 * L + xc) ** 2))
        term3 = 2.0 / (((L + xc) ** 2))

        F = (np.pi * Br**2 * R**4 / (4.0 * mu0)) * (term1 + term2 - term3)
        return float(F)

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
        N = self.m_kg * self.g_mps2
        sign_smooth = v_mps / (abs(v_mps) + v_eps)
        mu = mu_c + (mu_s - mu_c) * math.exp(-(v_mps / max(v_s, 1e-9))**2)
        return -mu * N * sign_smooth

    # ---------- Магнитное поле / ЭДС ----------
    def dB_dz(self, z_m):
        r = self.magnet_radius_m
        z1, z2 = self.magnet_height_m + z_m, self.magnet_height_m - z_m
        term1 = r**2 / ((z1**2 + r**2) ** 1.5)
        term2 = r**2 / ((z2**2 + r**2) ** 1.5)
        return (self.B_T / 2.0) * (term1 - term2) * self.alpha

    def compute_emf(self, z_m, v_mps):
        total_emf = sum(self.coil_turn_area_m2 * self.dB_dz(0.015 + z_m - (self.coil_z_bottom_m + turn * self.wire_diam_m)) * v_mps for turn in range(self.turns_N)) 
        return total_emf

    def compute_emf_batch(self, z_arr, v_arr):
        return np.array([self.compute_emf(z, v) for z, v in zip(z_arr, v_arr)])

    def compute_self_emf(self, i_a, t_s):
        return -self.coil_inductance_H * np.gradient(i_a, t_s)

    # ---------- Возбуждение ----------
    def shaker_force(self, t_s):
        """Только синус (без экспериментальных записей)."""
        a_base = -(self.omega_rad ** 2) * self.base_amp_x0_m * math.cos(self.omega_rad * t_s)
        return -self.m_kg * a_base
    
    def calculate_f_air(self, v_m: float, gap: float) -> float:
        """Вязкое трение в кольцевом зазоре между магнитом и цилиндром."""
        if gap <= 0:
            raise ValueError("Диаметр магнита должен быть меньше диаметра цилиндра.")
        mu_air = 1.81e-5  # Па·с
        return -6 * math.pi * mu_air * self.magnet_height_m * v_m / gap

    # ---------- ОДУ ----------
    def _sum_forces(self, t_s, z_m, v_mps):
        Cf = 0.86        # коэффициент аэродинамического сопротивления
        rho = 1.2  

        gap_m = (self.coil_inner_diam_m - 2.0 * self.magnet_radius_m) / 2.0
        F_bottom, F_top = self.force_from_bottom_magnet(z_m), self.force_from_top_magnet(z_m)
        F_gravity, F_shaker = -self.m_kg * self.g_mps2, self.shaker_force(t_s)
        F_quad, F_air = self.force_quadratic_drag(v_mps), self.force_air_shear(v_mps, gap_m)
        F_wall = self.force_wall_coulomb(v_mps)
        # f_aero = - ((Cf * rho * self.magnet_radius_m**2 * np.pi * v_mps**2) / 2 -  self.m_kg * self.g_mps2) / self.m_kg
        F_total = (F_bottom - F_top) + F_gravity + F_shaker  + F_quad + F_air + self.calculate_f_air(v_mps,  gap_m)
        return F_total, (F_total, F_bottom, F_top, F_gravity, F_shaker, F_quad, F_air, F_wall)

    def _ode_rhs(self, t_s, y):
        z_m, v_mps, i_a = y
        a_mps2 = self._sum_forces(t_s, z_m, v_mps)[0] / self.m_kg
        e_V = self.compute_emf(z_m, v_mps)
        R_series = self.coil_resistance_ohm + (self.load_resistance_ohm if np.isfinite(self.load_resistance_ohm) else 0.0)
        di_dt = (e_V - R_series * i_a) / max(self.coil_inductance_H, 1e-12)
        return [v_mps, a_mps2, di_dt]

    def compute_forces(self, t_s, z, v):
        forces = [self._sum_forces(float(tt), float(zz), float(vv))[1] for tt, zz, vv in zip(t_s, z, v)]
        self._last_forces = np.column_stack([t_s, forces])
        return self._last_forces

    def solve_all(self, t_eval_s=None, t_max_s=None, steps=1000,
                  rtol=1e-6, atol=1e-9, max_step_s=None, clamp_to_base=False, dense_output=False):
        # сетка времени
        if t_eval_s is None:
            t_eval_s = np.linspace(0.0, float(t_max_s or 1.0), steps)
        else:
            t_eval_s = np.asarray(t_eval_s, dtype=float)
            if np.any(np.diff(t_eval_s) <= 0):
                raise ValueError("t_eval_s должен быть строго возрастающим.")

        # max_step
        if max_step_s is None:
            max_step_s = float('inf')

        y0 = [self.z0_m, self.v0_mps, 0.0]
        sol = solve_ivp(
            fun=self._ode_rhs,
            t_span=(float(t_eval_s[0]), float(t_eval_s[-1])),
            y0=y0,
            t_eval=t_eval_s,
            rtol=rtol, atol=atol,
            max_step=max_step_s,
            dense_output=dense_output,
            method="RK45"               # ### CHANGED: всегда RK45 в модельном режиме
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
