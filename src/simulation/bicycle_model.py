"""
Nonlinear Dynamic Bicycle Model for Vehicle Dynamics Simulation

This module implements a nonlinear dynamic bicycle model that captures
the vehicle's longitudinal and lateral dynamics, including:
- Nonlinear tire forces (Pacejka Magic Formula)
- Weight transfer effects
- Aerodynamic drag
- Lateral and longitudinal load transfer
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class VehicleParameters:
    """Physical parameters of the vehicle"""
    # Mass and inertia
    m: float = 1500.0          # Vehicle mass [kg]
    I_z: float = 2500.0        # Yaw moment of inertia [kg*m^2]

    # Dimensions
    l_f: float = 1.2           # Distance from CG to front axle [m]
    l_r: float = 1.6           # Distance from CG to rear axle [m]
    h_cg: float = 0.5          # Height of center of gravity [m]
    track: float = 1.5         # Track width [m]

    # Tire parameters (Pacejka Magic Formula simplified)
    B: float = 10.0            # Stiffness factor
    C: float = 1.9             # Shape factor
    D: float = 1.0             # Peak factor
    E: float = 0.97            # Curvature factor

    # Tire-road friction
    mu: float = 0.9            # Coefficient of friction

    # Aerodynamics
    C_d: float = 0.3           # Drag coefficient
    A_f: float = 2.2           # Frontal area [m^2]
    rho: float = 1.225         # Air density [kg/m^3]

    # Drivetrain
    C_m1: float = 4000.0       # Motor constant 1 [N]
    C_m2: float = 100.0        # Motor constant 2 [N·s/m]
    C_r0: float = 0.01         # Rolling resistance coefficient
    C_r2: float = 0.00035      # Rolling resistance speed-dependent term

    @property
    def l(self) -> float:
        """Total wheelbase"""
        return self.l_f + self.l_r


class NonlinearBicycleModel:
    """
    Nonlinear dynamic bicycle model for vehicle simulation.

    State vector: [x, y, psi, v_x, v_y, omega]
    - x, y: Global position [m]
    - psi: Yaw angle [rad]
    - v_x: Longitudinal velocity [m/s]
    - v_y: Lateral velocity [m/s]
    - omega: Yaw rate [rad/s]

    Control inputs: [delta, throttle]
    - delta: Steering angle [rad]
    - throttle: Throttle/brake input [-1, 1]
      * throttle > 0: Motor force (forward acceleration)
      * throttle < 0: Braking force (no reverse)
    """

    def __init__(self, params: Optional[VehicleParameters] = None):
        """
        Initialize the nonlinear bicycle model.

        Args:
            params: Vehicle parameters. If None, default parameters are used.
        """
        self.params = params if params is not None else VehicleParameters()
        self.g = 9.81  # Gravitational acceleration [m/s^2]

    def pacejka_tire_model(self, alpha: float, F_z: float) -> float:
        """
        Compute lateral tire force using simplified Pacejka Magic Formula.

        Args:
            alpha: Tire slip angle [rad]
            F_z: Normal force on tire [N]

        Returns:
            Lateral tire force [N]
        """
        p = self.params

        # Normalize vertical load
        F_z_nom = (p.m * self.g) / 2  # Nominal load per axle
        d_F_z = (F_z - F_z_nom) / F_z_nom

        # Adjust peak factor based on normal load
        D = p.mu * F_z

        # Magic formula
        F_y = D * np.sin(p.C * np.arctan(p.B * alpha - p.E * (p.B * alpha - np.arctan(p.B * alpha))))

        return F_y

    def compute_normal_forces(self, a_x: float, a_y: float) -> Tuple[float, float]:
        """
        Compute normal forces on front and rear axles including load transfer.

        Args:
            a_x: Longitudinal acceleration [m/s^2]
            a_y: Lateral acceleration [m/s^2]

        Returns:
            Tuple of (F_z_f, F_z_r): Normal forces on front and rear axles [N]
        """
        p = self.params

        # Static normal forces
        F_z_f_static = p.m * self.g * p.l_r / p.l
        F_z_r_static = p.m * self.g * p.l_f / p.l

        # Longitudinal load transfer
        delta_F_z_long = p.m * a_x * p.h_cg / p.l

        # Total normal forces
        F_z_f = F_z_f_static - delta_F_z_long
        F_z_r = F_z_r_static + delta_F_z_long

        # Ensure positive normal forces
        F_z_f = max(F_z_f, 0.1)
        F_z_r = max(F_z_r, 0.1)

        return F_z_f, F_z_r

    def compute_tire_slips(self, v_x: float, v_y: float, omega: float,
                          delta: float) -> Tuple[float, float]:
        """
        Compute tire slip angles at front and rear axles.

        Args:
            v_x: Longitudinal velocity [m/s]
            v_y: Lateral velocity [m/s]
            omega: Yaw rate [rad/s]
            delta: Steering angle [rad]

        Returns:
            Tuple of (alpha_f, alpha_r): Front and rear slip angles [rad]
        """
        p = self.params

        # Avoid division by zero
        v_x_safe = max(abs(v_x), 0.1) * np.sign(v_x) if v_x != 0 else 0.1

        # Slip angles
        alpha_f = delta - np.arctan2(v_y + p.l_f * omega, v_x_safe)
        alpha_r = -np.arctan2(v_y - p.l_r * omega, v_x_safe)

        return alpha_f, alpha_r

    def compute_drag_force(self, v_x: float) -> float:
        """
        Compute aerodynamic drag force magnitude.
        This returns the magnitude only - the sign is handled in dynamics.

        Args:
            v_x: Longitudinal velocity [m/s]

        Returns:
            Drag force magnitude [N] (always positive)
        """
        p = self.params
        # Use abs(v_x) to ensure drag magnitude is always positive
        # The direction (opposing motion) is handled by sign(v_x) in dynamics
        return 0.5 * p.rho * p.C_d * p.A_f * abs(v_x) * v_x

    def compute_rolling_resistance(self, v_x: float) -> float:
        """
        Compute rolling resistance force (opposes motion).

        Args:
            v_x: Longitudinal velocity [m/s]

        Returns:
            Rolling resistance force [N] (signed to oppose motion)
        """
        p = self.params
        # Rolling resistance opposes motion, proportional to normal force
        # Sign ensures it opposes velocity direction
        magnitude = p.m * self.g * (p.C_r0 + p.C_r2 * v_x**2)
        return magnitude * np.sign(v_x) if abs(v_x) > 0.01 else 0.0

    def compute_motor_force(self, v_x: float, throttle: float) -> float:
        """
        Compute motor force from throttle input.

        Args:
            v_x: Longitudinal velocity [m/s]
            throttle: Throttle input [0, 1]

        Returns:
            Motor force [N]
        """
        p = self.params
        # Motor force model: F = (C_m1 - C_m2 * |v_x|) * throttle
        # Force decreases with speed (represents motor torque curve)
        # Use absolute value to ensure positive throttle always gives forward force
        return (p.C_m1 - p.C_m2 * abs(v_x)) * throttle

    def compute_brake_force(self, v_x: float, brake: float) -> float:
        """
        Compute braking force.

        Args:
            v_x: Longitudinal velocity [m/s]
            brake: Brake input [0, 1]

        Returns:
            Braking force [N] (negative value)
        """
        p = self.params

        # Only apply braking if moving forward
        # if v_x <= 0.01:
        #     # Already stopped or moving backward, no braking force
        #     return 0.0

        # Maximum brake force based on friction and normal load
        max_brake_force = p.mu * p.m * self.g

        # Apply brake force proportional to input
        # Add velocity-dependent component to prevent instabilities at low speed
        if v_x > 0.1:
            return -max_brake_force * brake
        else:
            # Reduce brake force at very low speeds to avoid numerical issues
            return -max_brake_force * brake * (v_x / 0.1)

    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Compute the state derivatives for the nonlinear bicycle model.

        Args:
            state: Current state [x, y, psi, v_x, v_y, omega]
            control: Control inputs [delta, throttle] where throttle is in [-1, 1]

        Returns:
            State derivatives [dx, dy, dpsi, dv_x, dv_y, domega]
        """
        # Unpack state
        x, y, psi, v_x, v_y, omega = state
        delta, throttle = control

        p = self.params

        # Compute slip angles
        alpha_f, alpha_r = self.compute_tire_slips(v_x, v_y, omega, delta)

        # Current accelerations (for load transfer estimation)
        # Use previous step approximation or zero for initial estimate
        a_y = (v_x * omega) if abs(v_x) > 0.1 else 0
        a_x_est = 0  # Initial estimate

        # Compute normal forces with load transfer
        F_z_f, F_z_r = self.compute_normal_forces(a_x_est, a_y)

        # Compute tire forces using Pacejka model
        F_y_f = self.pacejka_tire_model(alpha_f, F_z_f)
        F_y_r = self.pacejka_tire_model(alpha_r, F_z_r)

        # Longitudinal forces - handle throttle/brake based on sign
        if throttle >= 0:
            # Positive throttle: motor force (positive = forward acceleration)
            F_longitudinal = self.compute_motor_force(v_x, throttle)
        else:
            # Negative throttle: braking (negative = deceleration)
            # Brake force already returns negative value
            F_longitudinal = self.compute_brake_force(v_x, throttle)

        F_drag = self.compute_drag_force(v_x)
        F_roll = self.compute_rolling_resistance(v_x)

        # Total longitudinal force (applied at rear axle, assuming RWD)
        F_x_total = F_longitudinal - F_drag - F_roll

        # State derivatives
        # Global position derivatives
        dx = v_x * np.cos(psi) - v_y * np.sin(psi)
        dy = v_x * np.sin(psi) + v_y * np.cos(psi)
        dpsi = omega

        # Body frame accelerations
        dv_x = (F_x_total - F_y_f * np.sin(delta)) / p.m + v_y * omega
        dv_y = (F_y_f * np.cos(delta) + F_y_r) / p.m - v_x * omega
        domega = (F_y_f * p.l_f * np.cos(delta) - F_y_r * p.l_r) / p.I_z

        # Prevent reverse motion: if nearly stopped and decelerating, set acceleration to zero
        # if v_x <= 0.01 and dv_x < 0:
        #     dv_x = 0.0

        return np.array([dx, dy, dpsi, dv_x, dv_y, domega])

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """
        Integrate the dynamics forward by one time step using RK4.

        Args:
            state: Current state [x, y, psi, v_x, v_y, omega]
            control: Control inputs [delta, throttle]
            dt: Time step [s]

        Returns:
            New state after dt seconds
        """
        # RK4 integration
        k1 = self.dynamics(state, control)
        k2 = self.dynamics(state + 0.5 * dt * k1, control)
        k3 = self.dynamics(state + 0.5 * dt * k2, control)
        k4 = self.dynamics(state + dt * k3, control)

        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Normalize yaw angle to [-pi, pi]
        new_state[2] = np.arctan2(np.sin(new_state[2]), np.cos(new_state[2]))

        return new_state

    def get_slip_angles(self, state: np.ndarray, delta: float) -> Tuple[float, float]:
        """
        Get current slip angles for monitoring.

        Args:
            state: Current state [x, y, psi, v_x, v_y, omega]
            delta: Steering angle [rad]

        Returns:
            Tuple of (alpha_f, alpha_r): Front and rear slip angles [rad]
        """
        v_x, v_y, omega = state[3], state[4], state[5]
        return self.compute_tire_slips(v_x, v_y, omega, delta)

    def get_lateral_acceleration(self, state: np.ndarray) -> float:
        """
        Get current lateral acceleration.

        Args:
            state: Current state [x, y, psi, v_x, v_y, omega]

        Returns:
            Lateral acceleration [m/s^2]
        """
        v_x, omega = state[3], state[5]
        return v_x * omega

    def get_beta(self, state: np.ndarray) -> float:
        """
        Get vehicle sideslip angle.

        Args:
            state: Current state [x, y, psi, v_x, v_y, omega]

        Returns:
            Sideslip angle beta [rad]
        """
        v_x, v_y = state[3], state[4]
        return np.arctan2(v_y, v_x) if abs(v_x) > 0.1 else 0.0
