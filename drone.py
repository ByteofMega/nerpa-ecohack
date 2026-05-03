import threading
import time


class DroneController:
    """
    Имитация движения АНПА:
    - погружение на 4 метра,
    - прямолинейное движение к цели (100 шагов).
    """

    def __init__(self, start_lat: float, start_lon: float):
        self.lat = start_lat
        self.lon = start_lon
        self.depth = 0.0
        self.target_lat = None
        self.target_lon = None
        self.is_moving = False

        self.on_position_update = None  # вызывается при каждом шаге
        self.on_arrival = None          # вызывается по прибытию

    def set_target(self, lat: float, lon: float):
        self.target_lat = lat
        self.target_lon = lon

    def start_mission(self):
        if self.target_lat is None or self.is_moving:
            return
        self.is_moving = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        # Погружение
        for i in range(40):
            self.depth = i / 10.0
            if self.on_position_update:
                self.on_position_update()
            time.sleep(0.05)

        # Движение к цели
        start_lat, start_lon = self.lat, self.lon
        steps = 100
        for step in range(steps + 1):
            t = step / steps
            self.lat = start_lat + (self.target_lat - start_lat) * t
            self.lon = start_lon + (self.target_lon - start_lon) * t
            if self.on_position_update:
                self.on_position_update()
            time.sleep(0.05)

        self.is_moving = False
        if self.on_arrival:
            self.on_arrival()
