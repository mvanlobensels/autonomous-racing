"""
Kinematic bicycle model in curvilinear (Frenet) coordinates, with arc length s
as the independent variable. Shared by the MPC and the global trajectory
optimiser so both plan with identical dynamics and limits.

    state   x = [n, mu, v]     lateral offset, heading error, speed
    control u = [delta, a]     steering angle, longitudinal acceleration

    beta   = atan(l_r / (l_f + l_r) * tan(delta))
    s_dot  = v cos(mu + beta) / (1 - n kappa)
    dn/ds  = v sin(mu + beta) / s_dot
    dmu/ds = (v sin(beta) / l_r) / s_dot - kappa
    dv/ds  = a / s_dot

The functions take CasADi Opti expressions (or plain numbers); `ca.*` is only
used for math functions, no symbolic MX/SX variables are created here.
"""

from dataclasses import dataclass
from typing import Optional

import casadi as ca
import numpy as np

from src.simulation.bicycle_model import VehicleParameters


@dataclass
class VehicleLimits:
    max_steer: float = np.deg2rad(30.0)  # [rad]
    max_heading_error: float = np.pi / 2.5  # [rad] keeps the Frenet frame well-posed
    v_min: float = 1.0                   # [m/s] keeps s_dot away from zero
    v_max: float = 20.0                  # [m/s]
    grip_factor: float = 0.85            # fraction of mu*g usable in the friction ellipse
    car_half_width: float = 0.75         # [m] margin to the cones


class SpatialKinematicModel:
    g = 9.81

    def __init__(self, vehicle: Optional[VehicleParameters] = None,
                 limits: Optional[VehicleLimits] = None):
        self.veh = vehicle if vehicle is not None else VehicleParameters()
        self.lim = limits if limits is not None else VehicleLimits()
        self.a_grip = self.lim.grip_factor * self.veh.mu * self.g

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def beta(self, delta):
        return ca.atan(self.veh.l_r / self.veh.l * ca.tan(delta))

    def derivatives(self, x, u, kappa):
        """Spatial derivative dx/ds."""
        n, mu, v = x[0], x[1], x[2]
        delta, a = u[0], u[1]
        beta = self.beta(delta)
        s_dot = v * ca.cos(mu + beta) / (1 - n * kappa)
        return ca.vertcat(
            v * ca.sin(mu + beta) / s_dot,
            (v * ca.sin(beta) / self.veh.l_r) / s_dot - kappa,
            a / s_dot,
        )

    def rk4(self, x, u, kappa, ds):
        k1 = self.derivatives(x, u, kappa)
        k2 = self.derivatives(x + ds / 2 * k1, u, kappa)
        k3 = self.derivatives(x + ds / 2 * k2, u, kappa)
        k4 = self.derivatives(x + ds * k3, u, kappa)
        return x + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def stage_time(self, x, u, kappa, ds):
        """Time to traverse one spatial stage: ds / s_dot."""
        n, mu, v = x[0], x[1], x[2]
        return ds * (1 - n * kappa) / (v * ca.cos(mu + self.beta(u[0])))

    # ------------------------------------------------------------------
    # Limits
    # ------------------------------------------------------------------

    def max_drive_acceleration(self, v):
        """Acceleration at full throttle: motor curve minus drag and rolling resistance."""
        veh = self.veh
        f_motor = veh.C_m1 - veh.C_m2 * v
        f_drag = 0.5 * veh.rho * veh.C_d * veh.A_f * v ** 2
        f_roll = veh.m * self.g * (veh.C_r0 + veh.C_r2 * v ** 2)
        return (f_motor - f_drag - f_roll) / veh.m

    def add_constraints(self, opti, X, U):
        """State, input, drivetrain and friction-ellipse constraints (vectorised over stages)."""
        lim = self.lim
        n_u = U.shape[1]
        mu, v = X[1, :], X[2, :]
        delta, a = U[0, :], U[1, :]
        v_u = X[2, :n_u]

        opti.subject_to(opti.bounded(-lim.max_heading_error, mu, lim.max_heading_error))
        opti.subject_to(opti.bounded(lim.v_min, v, lim.v_max))
        opti.subject_to(opti.bounded(-lim.max_steer, delta, lim.max_steer))
        opti.subject_to(a >= -self.a_grip)
        opti.subject_to(a <= self.max_drive_acceleration(v_u))

        a_lat = v_u ** 2 * ca.sin(self.beta(delta)) / self.veh.l_r
        opti.subject_to((a / self.a_grip) ** 2 + (a_lat / self.a_grip) ** 2 <= 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def steady_state_steer(self, kappa):
        """Steering angle that follows curvature kappa (kinematic, small slip)."""
        return np.clip(np.arctan(self.veh.l * np.asarray(kappa)), -self.lim.max_steer, self.lim.max_steer)

    def corner_speed(self, kappa):
        """Steady-state speed limit from lateral grip."""
        k = np.maximum(np.abs(np.asarray(kappa, dtype=float)), 1e-6)
        return np.clip(np.sqrt(self.a_grip / k), self.lim.v_min, self.lim.v_max)

    def acceleration_to_throttle(self, a: float, v_x: float) -> float:
        """Invert the plant's longitudinal model: acceleration -> throttle/brake in [-1, 1]."""
        veh = self.veh
        f_drag = 0.5 * veh.rho * veh.C_d * veh.A_f * abs(v_x) * v_x
        f_roll = veh.m * self.g * (veh.C_r0 + veh.C_r2 * v_x ** 2) if abs(v_x) > 0.01 else 0.0
        force = veh.m * a + f_drag + f_roll
        if force >= 0:
            throttle = force / max(veh.C_m1 - veh.C_m2 * abs(v_x), 1.0)
        else:
            throttle = force / (veh.mu * veh.m * self.g)
        return float(np.clip(throttle, -1.0, 1.0))
