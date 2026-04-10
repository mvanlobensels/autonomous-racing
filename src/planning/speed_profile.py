import numpy as np
from scipy import signal
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VehicleParameters:
    a_lat_max: float = 8.0       # max lateral acceleration [m/s^2]
    a_forward_max: float = 5.0  # max forward acceleration [m/s^2]
    a_back_max: float = -10.0     # max braking acceleration [m/s^2] (negative)
    v_max: float = 20.0          # max velocity [m/s]
    find_peaks_prominence: float = 0.007
    window_width: int = 5
    shift: int = 2


class SpeedProfile:
    """
    Generates the speed profile of a given reference path.

    Call update(path, current_velocity, laps_completed) each planning cycle.
    """

    def __init__(self, params: Optional[VehicleParameters] = None):
        self.params = params if params is not None else VehicleParameters()
        self._c1 = 1.0 / self.params.a_lat_max**2
        self._exploration_mode = True

    def update(self, path: dict, current_velocity: float = 0.0, laps_completed: int = 0) -> dict:
        """
        Compute the speed profile for a given path.

        Args:
            path: dict with 'x', 'y', 'curvature' arrays (as returned by MidlinePath)
            current_velocity: current vehicle speed [m/s], used for exploration mode boost
            laps_completed: switches to trackdrive mode after the first lap

        Returns:
            dict with:
                'velocity'     (np.ndarray): target speed at each path point [m/s]
                'acceleration' (np.ndarray): target acceleration at each path point [m/s^2]
        """
        if laps_completed > 0:
            self._exploration_mode = False

        x = path['x']
        y = path['y']
        curvature = np.abs(path['curvature'])
        radius_of_curvature = 1.0 / (curvature + 1e-5)
        distances = [
            self.calculate_arc_length(x[i], y[i], x[i + 1], y[i + 1], radius_of_curvature[i])
            for i in range(len(x) - 1)
        ]

        velocity, acceleration = self.calculate_speed_profile(
            x, y, curvature, radius_of_curvature, distances, current_velocity
        )
        return {'velocity': velocity, 'acceleration': acceleration}

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def calculate_arc_length(self, x0, y0, x1, y1, R):
        """Arc length between two points given the local radius of curvature."""
        distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        arg = np.clip(0.5 * distance / abs(R), -1.0, 1.0)
        theta = 2 * np.arcsin(arg)
        arc_length = abs(R * theta)
        return arc_length if not np.isnan(arc_length) else 0.003

    def maximum_corner_speed(self, R):
        """Maximum speed at a curve apex given lateral acceleration limit."""
        velocity = np.sqrt(self.params.a_lat_max * abs(R))
        return min(velocity, self.params.v_max)

    def propagate_forward(self, v0, a0, R, s):
        """
        Propagate velocity one step forward using kinematics + friction ellipse.

        v1² = v0² + 2·a·s,  c1·(v1²/R)² + c2·a² = 1
        """
        c2 = 1.0 / self.params.a_forward_max**2
        v1 = np.sqrt(max(0.0, v0**2 + 2 * a0 * s))

        if v1 >= self.params.v_max:
            return self.params.v_max, 0.0

        numerator = 1.0 - self._c1 * (v1**2 / abs(R))**2
        if numerator < 0:
            return v0, 0.0
        return v1, np.sqrt(numerator / c2)

    def propagate_backwards(self, v1, R, s):
        """
        Propagate velocity one step backward, solving the friction-ellipse quadratic.
        """
        def _abc_formula(a, b, c):
            disc = b**2 - 4 * a * c
            if disc < 0:
                return np.array([0.0, 0.0])
            return np.asarray([
                (-b + np.sqrt(disc)) / (2 * a),
                (-b - np.sqrt(disc)) / (2 * a)
            ])

        if v1 >= self.params.v_max:
            return self.params.v_max, 0.0

        c2 = 1.0 / self.params.a_back_max**2
        R = abs(R)

        a = c2 + 4 * self._c1 * (s / R)**2
        b = -4 * self._c1 * s * (v1 / R)**2
        c = (self._c1 * v1**4) / (R**2) - 1

        if b**2 - 4 * a * c < 0:
            return v1, 0.0

        a0 = _abc_formula(a, b, c)
        v0_sq = v1**2 - 2 * a0 * s
        v0 = np.where(v0_sq >= 0, np.sqrt(v0_sq), 0.0)
        return float(np.nanmax(v0)), float(min(a0))

    def calculate_speed_profile(self, x, y, curvature, radius_of_curvature, distances,
                                 current_velocity: float = 0.0):
        """
        Full forward/backward propagation to build the speed envelope.

        In exploration mode only the single highest-curvature point is used as an
        apex; in trackdrive mode all peaks are detected via a moving-average filter.
        """
        n = len(x)
        velocity_forward = np.zeros(n)
        acceleration_forward = np.zeros(n)
        velocity_backward = np.zeros(n)
        acceleration_backward = np.zeros(n)
        velocity = np.zeros(n)
        acceleration = np.zeros(n)

        # Find curve apexes
        if self._exploration_mode:
            maximum_curvature_indices = [int(curvature.argmax())]
        else:
            w = self.params.window_width
            cumsum = np.cumsum(curvature)
            moving_average = np.roll((cumsum[w:] - cumsum[:-w]) / w, self.params.shift)
            peaks, _ = signal.find_peaks(moving_average, prominence=self.params.find_peaks_prominence)
            maximum_curvature_indices = peaks.tolist() if len(peaks) > 0 else [int(curvature.argmax())]

            # Snap each peak index to the local curvature maximum
            for i, idx in enumerate(maximum_curvature_indices):
                local_max = idx
                for j in range(max(0, idx - 3), min(n, idx + 3)):
                    if curvature[j] > curvature[local_max]:
                        local_max = j
                maximum_curvature_indices[i] = local_max

        maximum_curvature_indices.insert(0, 0)
        maximum_curvature_indices.append(len(distances))

        # Forward and backward propagation between consecutive apexes
        for i, apex_idx in enumerate(maximum_curvature_indices):
            v_apex = self.maximum_corner_speed(radius_of_curvature[apex_idx])
            velocity[apex_idx] = v_apex
            velocity_forward[apex_idx] = v_apex
            velocity_backward[apex_idx] = v_apex

            if i != len(maximum_curvature_indices) - 1:
                for j in range(apex_idx, maximum_curvature_indices[i + 1]):
                    v1, a1 = self.propagate_forward(
                        velocity_forward[j], acceleration_forward[j],
                        radius_of_curvature[j], distances[j]
                    )
                    velocity_forward[j + 1] = v1
                    acceleration_forward[j + 1] = a1

            if i != 0:
                for j in range(apex_idx, maximum_curvature_indices[i - 1], -1):
                    v0, a0 = self.propagate_backwards(
                        velocity_backward[j],
                        radius_of_curvature[j - 1],
                        distances[j - 1]
                    )
                    velocity_backward[j - 1] = v0
                    acceleration_backward[j - 1] = a0

        # Take the minimum of forward and backward envelopes
        for i in range(n):
            if velocity_forward[i] < velocity_backward[i]:
                velocity[i] = velocity_forward[i]
                acceleration[i] = acceleration_forward[i]
            else:
                velocity[i] = velocity_backward[i]
                acceleration[i] = acceleration_backward[i]

        # Exploration mode: cap profile to what is achievable from current vehicle speed
        if self._exploration_mode:
            v_cur, a_cur = current_velocity, 0.0
            for i in range(len(distances)):
                if v_cur < velocity[i]:
                    velocity[i] = v_cur
                    acceleration[i] = a_cur
                    v_cur, a_cur = self.propagate_forward(v_cur, a_cur, radius_of_curvature[i], distances[i])
                else:
                    break

        return velocity, acceleration
