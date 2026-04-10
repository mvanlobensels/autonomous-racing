import numpy as np


class VelocityController:
    """
    Proportional velocity controller.

    Converts a target speed from the speed profile into a throttle/brake command
    in the range [-1, 1] that is compatible with the bicycle model's control input.
    """

    def __init__(self, kp_throttle: float = 0.001, kp_brake: float = 0.8,
                 max_throttle: float = 1.0):
        """
        Args:
            kp_throttle: proportional gain when accelerating
            kp_brake:    proportional gain when braking (applied to negative error)
            max_throttle: clip throttle to this value (same for braking magnitude)
        """
        self.kp_throttle = kp_throttle
        self.kp_brake = kp_brake
        self.max_throttle = max_throttle

    def update(self, current_velocity: float, target_velocity: float) -> float:
        """
        Compute throttle/brake command.

        Args:
            current_velocity: vehicle longitudinal speed [m/s]
            target_velocity:  desired speed from the speed profile [m/s]

        Returns:
            throttle in [-1, 1]:  positive = accelerate, negative = brake
        """
        error = target_velocity - current_velocity
        if error >= 0:
            return float(np.clip(self.kp_throttle * error, 0.0, self.max_throttle))
        else:
            return float(np.clip(self.kp_brake * error, -self.max_throttle, 0.0))
