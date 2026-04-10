import numpy as np


class VelocityController:
    """
    Proportional velocity controller.

    Converts a target speed from the speed profile into a throttle/brake command
    in the range [-max_brake, max_throttle] that is compatible with the bicycle
    model's control input.
    """

    def __init__(self, kp_throttle: float = 0.2, kp_brake: float = 0.8,
                 max_throttle: float = 1.0, max_brake: float = 1.0):
        """
        Args:
            kp_throttle:  proportional gain when accelerating
            kp_brake:     proportional gain when braking
            max_throttle: upper clip on the throttle command [0, 1]
            max_brake:    magnitude clip on the brake command [0, 1]
        """
        self.kp_throttle = kp_throttle
        self.kp_brake = kp_brake
        self.max_throttle = max_throttle
        self.max_brake = max_brake

    def update(self, current_velocity: float, target_velocity: float) -> float:
        """
        Compute throttle/brake command.

        Args:
            current_velocity: vehicle longitudinal speed [m/s]
            target_velocity:  desired speed from the speed profile [m/s]

        Returns:
            throttle in [-max_brake, max_throttle]:  positive = accelerate, negative = brake
        """
        error = target_velocity - current_velocity
        if error >= 0:
            return float(np.clip(self.kp_throttle * error, 0.0, self.max_throttle))
        else:
            return float(np.clip(self.kp_brake * error, -self.max_brake, 0.0))
