import time
import numpy as np

class LapTimer:
    """Counts a lap each time the car re-enters the start zone after having left it."""

    def __init__(self, start_radius: float = 3.0, exit_radius: float = 5.0):
        self.lap = 0
        self.time = time.time()
        self.start_radius = start_radius
        self.exit_radius = exit_radius
        self._left_start = False

    def check_is_lap(self, position):
        distance = np.linalg.norm(position)
        if distance > self.exit_radius:
            self._left_start = True
        elif distance <= self.start_radius and self._left_start:
            self._left_start = False
            self.lap += 1
            print(f"Lap {self.lap} completed")
            self.time = time.time()
