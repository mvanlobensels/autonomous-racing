import time
import numpy as np

class LapTimer:
    def __init__(self):
        self.lap = 0
        self.time = time.time()

    def check_is_lap(self, position):
        if np.linalg.norm(np.zeros(2) - position) <= 3:
            now = time.time()
            if now - self.time > 2.0:
                self.lap += 1
                print(f"Lap {self.lap} completed")
                self.time = now
