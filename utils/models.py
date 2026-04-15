import math
import numpy as np 


class Magnet:
    def __init__(self, diameter, height, mass):
        self.radius = diameter / 2
        self.height = height
        self.mass = mass
        self.diameter = diameter
        self.square = self.radius ** 2
        # self.br = 0.648371  # Остаточная магнитная индукция магнита (Тл)
        self.br = 1.35

    def get_force(self, x: float) -> float:
        """
        Улучшенная формула силы взаимодействия между магнитами по Фурлани с поправкой.
        Используется выражение:
            F(x) = (π * Br^2 * R^4) / (4 * μ0) * [1 / x_c^2 + 1 / (2L + x_c)^2 - 2 / (L + x_c)^2]
        где x_c = x + 4R/5
        """
        mu_0 = 4 * math.pi * 1e-7  # магнитная постоянная, Гн/м
        R = self.radius            # радиус магнита
        L = self.height            # длина магнита
        Br = self.br               # остаточная индукция
        c = 0.8 * R                # поправка по Zurek
        x_c = x + c

        term1 = 1 / (x_c ** 2)
        term2 = 1 / ((2 * L + x_c) ** 2)
        term3 = 2 / ((L + x_c) ** 2)

        force = (math.pi * Br ** 2 * R ** 4) / (4 * mu_0) * (term1 + term2 - term3)
        return force


    # def get_force(self, x):
    #     D = 0.525  # мм (диаметр витка)
    #     d = 0.075  # мм (диаметр проволоки)
    #     G = 80e6  # мПа (модуль сдвига для стали)
    #     # G = 44e9  # Па (модуль сдвига для титана)
    #     L = 5.8  # мм (длина пружины) 

    #     # Вычисляем F по упрощенной формуле
    #     # F = (x * d**4 * G * np.pi) / (8 * D**2 * L)
    #     # return F
    #     """
    #     Расчет силы взаимодействия между магнитами по оси z. вариант с магнитом """
    #     mu_0 = 4 * np.pi * 1e-7  # Магнитная проницаемость вакуума

    #     term1 = (2 * (self.height + x)) / np.sqrt((self.height + x) ** 2 + self.square)
    #     term2 = (2 * self.height + x) / np.sqrt(
    #         (2 * self.height + x) ** 2 + self.square
    #     )
    #     term3 = x / np.sqrt(x ** 2 + self.square)

    #     force = (
    #         (np.pi * self.br ** 2 * self.square)
    #         / (2 * mu_0)
    #         * (term1 - term2 - term3)
    #     )
    #     return force
        
    def magnetic_induction(self, z):
        """
        Расчет магнитной индукции в точке z по оси магнита.
        z - расстояние от центра магнита вдоль оси.
        """
        return (self.br / 2) * (
            (z + self.height / 2)
            / math.sqrt(self.radius ** 2 + (z + self.height / 2) ** 2)
            - (z - self.height / 2)
            / math.sqrt(self.radius ** 2 + (z - self.height / 2) ** 2)
        )
    
    def magnetic_induction_derivative(self, z):
        """
        Производная  магнитной индукции в точке z по оси магнита
        z - расстояние от центра магнита вдоль оси.
        """
        r2 = self.radius ** 2
        term1 = r2 / (r2 + (z + self.height / 2) ** 2) ** (3 / 2)
        term2 = r2 / (r2 + (z - self.height / 2) ** 2) ** (3 / 2)
        
        dB_dz = (self.br / 2) * (term1 - term2)
        return dB_dz


class Coil:
    def __init__(
        self,
        turns_count,
        thickness,
        radius,
        position,
        magnet,
        wire_diameter=0.0005,
        layer_count=4,
    ):
        self.turns_count = turns_count  # Общее число витков
        self.radius = radius
        self.position = position  # Базовое положение катушки по оси z
        self.magnet = magnet
        self.thickness = thickness  # Толщина катушки (м)
        self.wire_diameter = wire_diameter  # Диаметр провода (м)
        self.layer_count = layer_count  # Количество слоев проводов

        # Расчет числа витков на слой
        self.turns_per_layer = int(self.turns_count / self.layer_count)

        # Проверка и корректировка общего числа витков
        self.turns_count = self.turns_per_layer * self.layer_count

        # Вычисление толщины катушки (с учетом слоев)
        calculated_thickness = self.layer_count * self.wire_diameter
        if calculated_thickness > self.thickness:
            raise ValueError(
                "Заданная толщина меньше, чем требуется для размещения заданного числа слоев."
            )

        # Вычисление высоты катушки (длина вдоль оси z)
        self.height = self.turns_per_layer * self.wire_diameter

        # Создаем список для хранения информации о каждой витке
        self.turns = []
        self.initialize_turns()

    def initialize_turns(self):
        """
        Инициализирует позиции и параметры каждой витки в катушке.
        """
        for layer in range(self.layer_count):
            # Радиус каждого слоя увеличивается на диаметр провода
            layer_radius = self.radius + layer * self.wire_diameter
            for turn in range(self.turns_per_layer):
                # Позиция каждой витки вдоль оси z
                turn_z_position = self.position + turn * self.wire_diameter
                self.turns.append(
                    {
                        'layer': layer,
                        'radius': layer_radius,
                        'z_position': turn_z_position,
                    }
                )

    def coil_position(self, shaker, t):
        """
        Возвращает базовое положение катушки в момент времени t с учетом колебаний шейкера.
        """
        return self.position + shaker.X0 * np.sin(shaker.W * t)

    def get_total_emf(self, shaker, magnet_position, magnet_velocity, t, a_m):
        """
        Рассчитывает ЭДС для каждой витки и возвращает список ЭДС витков и их суммарное значение.
        """
        total_eds = 0
        eds_per_turn = []

        # Базовое положение катушки с учетом движения шейкера
        coil_base_position = self.coil_position(shaker, t)
 
        for turn in self.turns:
            # Расчёт позиции витка
            z_turn = coil_base_position + (turn['z_position'] - self.position)
            z_distance = magnet_position - z_turn

            # Производная магнитной индукции по z
            dB_dz = self.magnet.magnetic_induction_derivative(z_distance)

            # Площадь витка
            area = math.pi * turn['radius'] ** 2

            # ЭДС
            eds = -dB_dz * magnet_velocity * area

            # Сбор результатов
            eds_per_turn.append(eds)
            total_eds += eds


        return eds_per_turn, total_eds

    def calculate_inductance(self):
        """
        Расчет индуктивности катушки.
        """
        mu_0 = 4 * math.pi * 1e-7  # Магнитная проницаемость вакуума
        A = math.pi * self.radius ** 2  # Площадь поперечного сечения
        L = (mu_0 * self.turns_count ** 2 * A) / self.height
        return L

    def get_self_induction_emf(self, delta_current, delta_time):
        """
        Вычисляет ЭДС самоиндукции на основе индуктивности и скорости изменения тока.
        """
        L = self.calculate_inductance()  # Индуктивность катушки
        dI_dt = delta_current / delta_time  # Скорость изменения тока
        emf_self_induction = -L * dI_dt  # ЭДС самоиндукции
        return emf_self_induction

class Shaker:
    def __init__(self, G, miew, X0):
        self.G = G
        self.miew = miew
        self.X0 = X0
        self.W = -2 * math.pi * miew

    def get_force(self, magnet, t):
        a = -self.W ** 2 * self.X0 * math.cos(self.W * t)
        """
        Расчет силы, действующей на магнит при вибрации.
        """
        return -magnet.mass * a