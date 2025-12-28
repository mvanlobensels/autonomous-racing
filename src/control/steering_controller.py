"""
Stanley Controller for Path Tracking

The Stanley controller is a path tracking algorithm that uses both heading error
and cross-track error to compute steering commands. It was developed for the
Stanford autonomous vehicle "Stanley" that won the DARPA Grand Challenge.

The control law combines:
1. Heading error: difference between vehicle heading and path heading
2. Cross-track error: lateral distance from vehicle to path

Steering angle: delta = theta_e + arctan(k * e / v_x)
where:
- theta_e: heading error
- k: gain parameter
- e: cross-track error
- v_x: longitudinal velocity
"""

import numpy as np
from typing import Tuple, Optional


class StanleyController:
    """
    Stanley controller for lateral path tracking.

    The controller computes steering commands to minimize both heading error
    and cross-track error relative to a reference path.
    """

    def __init__(self,
                 k: float = 1.0,
                 k_soft: float = 1.0,
                 max_steer: float = np.deg2rad(30.0)):
        """
        Initialize the Stanley controller.

        Args:
            k: Cross-track error gain (higher = more aggressive correction)
            k_soft: Softening constant for low-speed stability (m/s)
            max_steer: Maximum steering angle [rad]
        """
        self.k = k
        self.k_soft = k_soft
        self.max_steer = max_steer

    def find_closest_point(self,
                          vehicle_pos: np.ndarray,
                          path: dict) -> Tuple[int, float]:
        """
        Find the closest point on the path to the vehicle.

        Args:
            vehicle_pos: Vehicle position [x, y]
            path_x: Array of path x-coordinates
            path_y: Array of path y-coordinates

        Returns:
            Tuple of (closest_index, cross_track_error)
        """
        # Compute distances to all path points
        dx = path['x'] - vehicle_pos[0]
        dy = path['y'] - vehicle_pos[1]
        distances = np.sqrt(dx**2 + dy**2)

        # Find closest point
        closest_idx = np.argmin(distances)
        cross_track_error = distances[closest_idx]

        return closest_idx, cross_track_error

    def compute_cross_track_error(self,
                                  vehicle_state: np.ndarray,
                                  path: dict,
                                  closest_idx: int) -> float:
        """
        Compute signed cross-track error.

        The sign indicates which side of the path the vehicle is on:
        - Positive: vehicle is to the left of the path
        - Negative: vehicle is to the right of the path

        Args:
            vehicle_pos: Vehicle position [x, y]
            vehicle_heading: Vehicle heading angle [rad]
            path_x: Array of path x-coordinates
            path_y: Array of path y-coordinates
            closest_idx: Index of closest path point

        Returns:
            Signed cross-track error [m]
        """
        vehicle_pos = vehicle_state[0:2]
        vehicle_heading = vehicle_state[2]

        # Vector from vehicle to closest path point
        closest_point = np.array([path['x'][closest_idx], path['y'][closest_idx]])
        error_vector = closest_point - vehicle_pos

        # Vehicle heading unit vector
        heading_vector = np.array([np.cos(vehicle_heading),
                                  np.sin(vehicle_heading)])

        # Cross product to determine sign (positive = left, negative = right)
        # In 2D: cross(a, b) = a_x * b_y - a_y * b_x
        cross_product = (heading_vector[0] * error_vector[1] -
                        heading_vector[1] * error_vector[0])

        # Magnitude of cross-track error
        error_magnitude = np.linalg.norm(error_vector)

        # Apply sign based on which side of the path the vehicle is on
        cross_track_error = np.sign(cross_product) * error_magnitude

        return cross_track_error

    def compute_heading_error(self,
                             vehicle_heading: float,
                             path_heading: float) -> float:
        """
        Compute heading error (normalized to [-pi, pi]).

        Args:
            vehicle_heading: Current vehicle heading [rad]
            path_heading: Desired path heading [rad]

        Returns:
            Heading error [rad]
        """
        heading_error = path_heading - vehicle_heading

        # Normalize to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error),
                                   np.cos(heading_error))

        return heading_error

    def update(self,
                vehicle_state: np.ndarray,
                path: dict) -> Tuple[float, dict]:
        """
        Compute steering angle using Stanley controller.

        Args:
            vehicle_pos: Vehicle position [x, y] in global frame
            vehicle_heading: Vehicle heading angle [rad]
            vehicle_velocity: Vehicle longitudinal velocity [m/s]
            path_x: Array of path x-coordinates
            path_y: Array of path y-coordinates
            path_heading: Array of path heading angles [rad]

        Returns:
            Tuple of (steering_angle, debug_info)
            - steering_angle: Commanded steering angle [rad]
            - debug_info: Dictionary with intermediate values for debugging
        """
        # Handle empty path
        if len(path['x']) == 0 or len(path['y']) == 0:
            return 0.0, {'error': 'empty_path'}
        
        vehicle_heading = vehicle_state[2]
        vehicle_velocity = vehicle_state[3]

        # Find closest point on path
        closest_idx, _ = self.find_closest_point(vehicle_state, path)
        # Compute cross-track error (signed)
        cross_track_error = self.compute_cross_track_error(
            vehicle_state, path, closest_idx
        )

        # Compute heading error
        path_heading_at_closest = path['theta'][closest_idx]
        heading_error = self.compute_heading_error(
            vehicle_heading, path_heading_at_closest
        )

        # Stanley control law
        # delta = theta_e + arctan(k * e / (k_soft + v_x))
        # The k_soft term prevents division by zero and reduces gain at low speeds
        velocity_term = self.k_soft + abs(vehicle_velocity)
        cross_track_term = np.arctan2(self.k * cross_track_error, velocity_term)

        steering_angle = heading_error + cross_track_term

        # Clamp steering angle to limits
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)

        # Debug information
        debug_info = {
            'closest_idx': closest_idx,
            'cross_track_error': cross_track_error,
            'heading_error': heading_error,
            'heading_error_deg': np.rad2deg(heading_error),
            'cross_track_term': cross_track_term,
            'steering_angle': steering_angle,
            'steering_angle_deg': np.rad2deg(steering_angle),
        }

        return steering_angle, debug_info

    def set_gains(self, k: Optional[float] = None,
                  k_soft: Optional[float] = None,
                  max_steer: Optional[float] = None):
        """
        Update controller gains.

        Args:
            k: Cross-track error gain (if provided)
            k_soft: Softening constant (if provided)
            max_steer: Maximum steering angle [rad] (if provided)
        """
        if k is not None:
            self.k = k
        if k_soft is not None:
            self.k_soft = k_soft
        if max_steer is not None:
            self.max_steer = max_steer
