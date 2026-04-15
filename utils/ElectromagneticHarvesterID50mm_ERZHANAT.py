from typing import Optional, Tuple
import math
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize

class ElectromagneticHarvesterAbsolute:
    def __init__(self, folder_path: Optional[str] = None):
        """Инициализация модели с абсолютными координатами."""
        # --- Масса магнита и гравитация ---
        self.m_kg = 0.021  # Масса центрального магнита
        self.g_mps2 = 9.81

        # --- Магнитные параметры ---
        self.mu0_Hpm = 4 * np.pi * 1e-7
        self.B_T = 0.9399
        self.k_accel = 1.0
        self.k_emf = 0.129
        self.alpha = 1
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

        # --- Относительные позиции элементов по оси z (м) относительно шейкера ---
        self.coil_z_bottom_rel = self.magnet_height_m + 0.002
        self.coil_z_top_rel = self.coil_z_bottom_rel + self.coil_height_m + 0.03
        self.top_magnet_z_bottom_rel = self.coil_z_top_rel + 0.002
        self.top_magnet_z_top_rel = self.top_magnet_z_bottom_rel + self.magnet_height_m
        self.bottom_magnet_z_top_rel = self.magnet_height_m
        self.bottom_magnet_z_bottom_rel = 0
        self.top_magnet_center_rel = 0.5 * (self.top_magnet_z_top_rel + self.top_magnet_z_bottom_rel)
        self.bottom_magnet_center_rel = 0.5 * (self.bottom_magnet_z_top_rel + self.bottom_magnet_z_bottom_rel)
        self.z0_m = 0
        # --- Начальные абсолютные условия ---
        self.z_shaker0 = 0.0
        self.z_rel0 = 0.040  # Начальная относительная позиция центрального магнита (м)
        self.v_shaker0 = 0.0
        self.v_rel0 = 0.0

        # --- Данные базы шейкера ---
        self.folder_path = folder_path
        self.base_time_s = None
        self.base_accel_mps2 = None
        self.base_accel_interp = None
        # добавленные поля для позиции и скорости
        self.base_pos_interp = None
        self.base_vel_interp = None

        # --- Геометрия катушки и сопротивление ---
        self._precompute_coil_geometry()

        # --- Электрические параметры ---
        self.load_resistance_ohm = 1e6  # вход осциллографа

        # --- Для хранения разложения сил ---
        self._last_forces = None

        self.use_linear_friction = True
        self.c_damping = 0.05
        self.F_c = 0.5
        self.beta = 10

        self.Cf  = 0.82        # коэффициент аэродинамического сопротивления
        self.rho = 1.2        # плотность воздуха, кг/м³
        self.S = math.pi * self.magnet_radius_m ** 2
        
    def _precompute_coil_geometry(self):
        """Рассчитать площадь витка, индуктивность и R катушки."""
        r_mean_m = (self.coil_inner_diam_m + self.coil_outer_diam_m) / 4.0
        self.coil_turn_area_m2 = np.pi * r_mean_m**2
        self.coil_inductance_H = self.mu0_Hpm * self.turns_N**2 * self.coil_turn_area_m2 / self.coil_height_m
        rho_cu_ohm_m = 1.68e-8
        total_wire_len_m = (2.0 * np.pi * r_mean_m) * self.turns_N
        wire_area_m2 = np.pi * (self.wire_diam_m / 2.0) ** 2
        self.coil_resistance_ohm = rho_cu_ohm_m * total_wire_len_m / max(wire_area_m2, 1e-12)

    def set_electrical(self, R_coil_ohm=None, R_load_ohm=None):
        """Установить сопротивления."""
        if R_coil_ohm is not None:
            self.coil_resistance_ohm = float(R_coil_ohm)
        if R_load_ohm is not None:
            self.load_resistance_ohm = float(R_load_ohm)

    def load_base_from_csv(self, file_path, accel_col="acc_mps2", pos_col="displacement_mm", time_col=None, sep=";", decimal=".", normalize_time=True, pos_scale: float = 1.0):
        """
        Загрузить CSV с данными шейкера.
        Поддерживает:
        - только ускорение (accel_col),
        - или (дополнительно/вместо) положение (pos_col), из которого строится скорость.
        Если time_col не указан/нет в файле — считаем, что шаг 1 мс.

        Аргументы:
            file_path : str
                Путь к CSV-файлу
            accel_col : str
                Имя колонки с ускорением (м/с^2)
            pos_col : str
                Имя колонки с положением (по умолчанию displacement_mm)
            time_col : str
                Имя колонки с временем (если None — время считается равномерным)
            sep : str
                Разделитель в CSV
            decimal : str
                Десятичный разделитель
            normalize_time : bool
                Нормализовать время (начинать с 0)
            pos_scale : float
                Коэффициент пересчёта положения (например, 0.001 если данные в мм → метры)
        """
        import os
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        df = pd.read_csv(file_path, sep=sep, decimal=decimal)

        # --- Время ---
        if time_col is not None and time_col in df:
            t_s = df[time_col].to_numpy(dtype=float)
            # на случай дублей по времени — усредним и отсортируем
            cols = [c for c in [accel_col, pos_col] if c is not None and c in df]
            if cols:
                df = df.groupby(time_col, as_index=False)[cols].mean().sort_values(time_col)
                t_s = df[time_col].to_numpy(dtype=float)
        else:
            # шаг 1 мс
            if accel_col not in df and (pos_col is None or pos_col not in df):
                raise ValueError(f"В {file_path} нет ни '{accel_col}', ни '{pos_col}'.")
            n = len(df)
            t_s = np.arange(n) * 0.001  # 1 мс шаг

        if normalize_time:
            t_s = t_s - t_s[0]

        self.base_time_s = t_s

        # --- Ускорение (если есть) ---
        if accel_col in df:
            a_mps2 = df[accel_col].to_numpy(dtype=float)
            self.base_accel_mps2 = a_mps2
            self.base_accel_interp = interp1d(
                t_s, a_mps2, kind="linear",
                bounds_error=False, fill_value=(a_mps2[0], a_mps2[-1])
            )
        else:
            self.base_accel_mps2 = None
            self.base_accel_interp = None

        # --- Положение и скорость (если есть pos_col) ---
        if pos_col is not None and pos_col in df:
            # применяем коэффициент масштаба
            z_m = df[pos_col].to_numpy(dtype=float) * pos_scale

            self.base_pos_interp = interp1d(
                t_s, z_m, kind="linear",
                bounds_error=False, fill_value=(z_m[0], z_m[-1])
            )

            # скорость как производная от позиции
            v_m = np.gradient(z_m, t_s)
            self.base_vel_interp = interp1d(
                t_s, v_m, kind="linear",
                bounds_error=False, fill_value=(v_m[0], v_m[-1])
            )

            # если ускорения нет в файле — получим как производную скорости
            if self.base_accel_interp is None:
                a_mps2_from_pos = np.gradient(v_m, t_s)
                self.base_accel_mps2 = a_mps2_from_pos
                self.base_accel_interp = interp1d(
                    t_s, a_mps2_from_pos, kind="linear",
                    bounds_error=False, fill_value=(a_mps2_from_pos[0], a_mps2_from_pos[-1])
                )
        else:
            self.base_pos_interp = None
            self.base_vel_interp = None


    def shaker_accel(self, t_s):
        """Ускорение шейкера."""
        if self.base_accel_interp is None:
            raise ValueError("Данные ускорения не загружены. Вызовите load_base_from_csv.")
        a_base = float(self.base_accel_interp(t_s))
        return a_base * self.k_accel

    def _empirical_pair_force(self, x_m: float) -> float:
        """Сила между магнитами."""
        Br = self.B_T
        R = self.magnet_radius_m
        mu0 = self.mu0_Hpm
        c = R * 0.8
        xc = x_m + c
        L = self.magnet_height_m
        eps = 1e-9
        term1 = 1.0 / (xc * xc + eps)
        term2 = 1.0 / (((2.0 * L + xc) ** 2) + eps)
        term3 = 2.0 / (((L + xc) ** 2) + eps)
        F = (np.pi * Br**2 * R**4 / (4.0 * mu0)) * (term1 + term2 - term3)
        # F_max = 50.0
        # F_sat = F_max * math.tanh(F / max(F_max, 1e-9))
        return float(F)

    def force_from_top_magnet(self, z_free, z_top_center):
        return -self._empirical_pair_force(z_top_center - z_free)

    def force_from_bottom_magnet(self, z_free, z_bottom_center):
        return self._empirical_pair_force(z_free - z_bottom_center)

    def force_trenia(self, v_free, v_shaker):
        """Трение зависит от относительной скорости."""
        v_rel = v_free - v_shaker
        if self.use_linear_friction:
            return -self.c_damping * v_rel
        else:
            return -self.F_c * math.tanh(self.beta * v_rel)

    def dB_dz(self, z_rel):
        r = self.magnet_radius_m
        L = 0.5 * self.magnet_height_m
        z1, z2 = L + z_rel, L - z_rel
        term1 = r**2 / ((z1**2 + r**2) ** 1.5)
        term2 = r**2 / ((z2**2 + r**2) ** 1.5)
        return self.alpha * (self.B_T / 2.0) * (term1 - term2) 

    def compute_emf(self, z_free, v_free, z_shaker, v_shaker):
        """ЭДС по относительным координатам."""
        z_rel = z_free - z_shaker
        v_rel = v_free - v_shaker
        z_turns_rel = self.coil_z_bottom_rel + np.arange(self.turns_N) * self.wire_diam_m
        dB_dz = self.dB_dz(z_rel - z_turns_rel)
        total_emf = np.sum(self.coil_turn_area_m2 * dB_dz * v_rel)
        return -total_emf

    def compute_emf_batch(self, z_free_arr, v_free_arr, z_shaker_arr, v_shaker_arr):
        return np.array([self.compute_emf(zf, vf, zs, vs) for zf, vf, zs, vs in zip(z_free_arr, v_free_arr, z_shaker_arr, v_shaker_arr)])

    def compute_self_emf(self, i_a, t_s):
        return -self.coil_inductance_H * np.gradient(i_a, t_s)

    def _sum_forces(self, t_s, z_free, v_free, z_shaker, v_shaker):
        """Суммарная сила и разложение."""
        z_bottom_center = z_shaker + self.bottom_magnet_center_rel
        z_top_center = z_shaker + self.top_magnet_center_rel
        F_bottom = self.force_from_bottom_magnet(z_free, z_bottom_center)
        F_top = self.force_from_top_magnet(z_free, z_top_center)
        F_gravity = -self.m_kg * self.g_mps2
        F_trenia = self.force_trenia(v_free, v_shaker)
        F_aero = (self.Cf * self.rho * self.S * (v_free**2)) / 2
        F_total = F_bottom + F_top + F_gravity + F_trenia + F_aero
        return F_total, (F_total, F_bottom, F_top, F_gravity, F_aero, F_trenia)

    def _ode_rhs(self, t_s, y):
        """Правая часть ОДУ."""
        z_rel, v_rel, i_a, z_shaker_state, v_shaker_state = y

        # 1) Получаем "внешние" (из файла) z/v/a шейкера, если есть pos_col
        if self.base_pos_interp is not None and self.base_vel_interp is not None:
            z_shaker = float(self.base_pos_interp(t_s))
            v_shaker = float(self.base_vel_interp(t_s))
            a_shaker = float(self.shaker_accel(t_s))  # из файла или из производной
        else:
            # старый режим (нет позиции в файле) — шейкер интегрируется как состояние
            z_shaker = z_shaker_state
            v_shaker = v_shaker_state
            a_shaker = self.shaker_accel(t_s)

        # 2) Абсолютные координаты свободного магнита
        z_free = z_shaker + z_rel
        v_free = v_shaker + v_rel

        # 3) Силы и относительное ускорение магнита
        F_total = self._sum_forces(t_s, z_free, v_free, z_shaker, v_shaker)[0]
        a_rel = F_total / self.m_kg - a_shaker

        # 4) Электрическая часть
        e_V = self.compute_emf(z_free, v_free, z_shaker, v_shaker)
        R_series = self.coil_resistance_ohm + (self.load_resistance_ohm if np.isfinite(self.load_resistance_ohm) else 0.0)
        di_dt = (e_V - R_series * i_a) / max(self.coil_inductance_H, 1e-12)

        # 5) Производные состояний шейкера:
        if self.base_pos_interp is not None and self.base_vel_interp is not None:
            dz_shaker_dt = v_shaker
            dv_shaker_dt = a_shaker
        else:
            dz_shaker_dt = v_shaker_state
            dv_shaker_dt = a_shaker

        return [v_rel, a_rel, di_dt, dz_shaker_dt, dv_shaker_dt]

    def compute_forces(self, t_s, z_free, v_free, z_shaker, v_shaker):
        """Вернуть массив сил по времени."""
        forces = [self._sum_forces(float(tt), float(zf), float(vf), float(zs), float(vs))[1]
                  for tt, zf, vf, zs, vs in zip(t_s, z_free, v_free, z_shaker, v_shaker)]
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

        # Начальные условия
        z_sh0 = self.z_shaker0
        v_sh0 = self.v_shaker0
        if self.base_time_s is not None and self.base_pos_interp is not None and self.base_vel_interp is not None:
            t0 = float(t_eval_s[0])
            z_sh0 = float(self.base_pos_interp(t0))
            v_sh0 = float(self.base_vel_interp(t0))

        y0 = [self.z_rel0, self.v_rel0, 0.0, z_sh0, v_sh0]
        sol = solve_ivp(
            fun=self._ode_rhs,
            t_span=(float(t_eval_s[0]), float(t_eval_s[-1])),
            y0=y0,
            t_eval=t_eval_s,
            rtol=rtol,
            atol=atol,
            max_step=max_step_s,
            method="LSODA"
        )
        z_rel, v_rel, i_a, z_shaker, v_shaker = sol.y
        # восстановим абсолютные координаты
        z_free = z_shaker + z_rel
        v_free = v_shaker + v_rel
        emf_open_V = self.compute_emf_batch(z_free, v_free, z_shaker, v_shaker)
        self_emf_V = self.compute_self_emf(i_a, sol.t)
        forces = self.compute_forces(sol.t, z_free, v_free, z_shaker, v_shaker)
        if np.isfinite(self.load_resistance_ohm) and self.load_resistance_ohm > 0:
            v_terminal_V = i_a * self.load_resistance_ohm
        else:
            di_dt = np.gradient(i_a, sol.t)
            v_terminal_V = emf_open_V - self.coil_inductance_H * di_dt - self.coil_resistance_ohm * i_a

        # Вычисляем z_bottom и z_top как функции от z_shaker
        z_bottom = z_shaker + self.bottom_magnet_center_rel
        z_top = z_shaker + self.top_magnet_center_rel
        v_bottom = v_shaker  # Скорость равна скорости шейкера
        v_top = v_shaker     # Скорость равна скорости шейкера

        return sol.t, z_free, v_free, i_a, emf_open_V, self_emf_V, forces, v_terminal_V, z_shaker, v_shaker, z_bottom, v_bottom, z_top, v_top

    def plot_forces(self, t_s, forces, save_dir=None, timestamp=None):
        """Построить график сил."""
        import matplotlib.pyplot as plt, os
        plt.figure(figsize=(11, 6))
        labels = ["F_total", "F_bottom", "F_top", "F_gravity", "F_shaker (0)", "F_trenia"]
        linestyles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'd', '*', 'v']
        for i, lbl in enumerate(labels, 1):
            style = linestyles[i % len(linestyles)]
            marker = markers[i % len(markers)]
            plt.plot(t_s, forces[:, i], label=lbl,
                     linestyle=style, marker=marker, markersize=4)
        plt.xlabel("Время, с")
        plt.ylabel("Сила, Н")
        plt.title("Силы на центральный магнит")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save_dir and timestamp:
            plt.savefig(os.path.join(save_dir, f"forces_plot_{timestamp}.png"))
        else:
            plt.savefig("forces_plot.png")
        plt.show()
