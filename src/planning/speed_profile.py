import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import Optional

@dataclass
class VehicleParameters:
    a_lat_max = 5  # [m/s^2]
    a_forward_max = 10
    a_back_max = -5
    v_max = 20
    find_peaks_prominence = 0.007
    window_width = 5
    shift = 2

class SpeedProfile:
    """
    Generates the speed profile of a given reference path.
    """
    
    def __init__(self, params: Optional[VehicleParameters] = None):

        self.params = params if params is not None else VehicleParameters()

        self._c1 = 1. / self.params.a_lat_max**2
        self._distances = [self.calculate_arc_length(self._x[i], self._y[i], self._x[i+1], self._y[i+1], self._radius_of_curvature[i]) for i in range(len(self._x) - 1)]
    
    def calculate_arc_length(self, x0, y0, x1, y1, R):
        """
        Calculates the arc length between to points based on the radius of curvature of the path segment.
        
        Args:
            x0 (float): X-coordinate of first point.
            y0 (float): Y-coordinate of first point.
            x1 (float): X-coordinate of second point.
            y1 (float): Y-coordinate of second point.
            R (float): Radius_of_curvature of path segment in meters.
        Returns:
            (float): Arc length in meters.
        """
        distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        theta = 2 * np.arcsin(0.5 * distance / abs(R))
        arc_length = R * theta
        # if np.isnan(arc_length):
        #     return 0.003
        return arc_length
        
    def maximum_corner_speed(self, R):
        """
        Calculates the maximum corner based on the curvature of the path.
        
        Args:
            R (float): Radius of curvature of the path.
        
        Returns:
            float: Velocity in meters per second.
        """
        velocity = np.sqrt(self._maximum_lateral_accel * R)
        return velocity if velocity <= self._maximum_velocity else self._maximum_velocity
        
    def propagate_forward(self, v0, a0, R, s):
        """
        Calculate maximum velocity and acceleration at next point whilst taking
        into account the limits of the car and the curvature of the path.
        v1^2 = v0^2 + 2 * a * s 
        c1 * a_y + c2 * a_x = 1
        a_y = v^2 / R
        
        Args:
            v0 (float): Velocity at current point in meters per second.
            a0 (float): Acceleration at current point in meters per second squared.
            R (float): Radius of curvature at current point in meters.
            s (float): Distance between current and next point in meters.
        
        Returns:
            (float, float): (velocity at next point, acceleration at next point)
        """
        c2 = 1. / self._maximum_forward_accel**2
        v1 = np.sqrt(v0**2 + 2 * a0 * s)
        
        if v1 >= self._maximum_velocity:
            return self._maximum_velocity, 0
        else:
            numerator = 1 - self._c1 * (v1**2/R)**2
            if numerator < 0:
                return v0, 0
            a1 = [
                np.sqrt(numerator / c2), 
                -np.sqrt(numerator / c2)
            ]
            return v1, max(a1)
            
    def propagate_backwards(self, v1, R, s):
        """
        Calculate maximum velocity and acceleration at previous point whilst taking 
        into account the limits of the car and the curvature of the path.
        
        v1^2 = v0^2 + 2 * a * s 
        c1 * a_y + c2 * a_x = 1
        a_y = v^2 / R
        
        Substitute equations into each other and solve for acceleration.
        
        Args:
            v1 (float): Velocity at current point in meters per second.
            R (float): Radius of curvature at current point in meters.
            s (float): Distance between current and next point in meters.
        
        Returns:
            float, float: velocity at previous point, acceleration at previous point.
        """
               
        def _abc_formula(a, b, c):
            return np.asarray([
                (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a), 
                (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            ])
        
        c2 = 1. / self._maximum_backward_accel**2
        
        a = c2 + 4 * self._c1 * (s/R)**2
        b = -4 * self._c1 * s * (v1/R)**2
        c = (self._c1 * v1**4)/(R**2) - 1

        if b**2 - 4 * a * c < 0:
            return v1, 0
        
        if v1 >= self._maximum_velocity:
            return self._maximum_velocity, 0
        else:
            a0 = _abc_formula(a, b, c)
            v0 = np.sqrt(v1**2 - 2 * a0 * s)
            return np.nanmax(v0), min(a0)
        
    def calculate_speed_profile(self):
        """
        Calculates the maximum speed profile.
        First find the indices of the curves in the received reference path.
        Then propagates backward and forward from these indices until the intersection points.
        """
        # Create velocity and acceleration arrays
        velocity = np.zeros(len(self._x))
        velocity_forward = np.zeros(len(self._x))
        velocity_backward = np.zeros(len(self._x))

        acceleration = np.zeros(len(self._x))
        acceleration_forward = np.zeros(len(self._x))
        acceleration_backward = np.zeros(len(self._x))
        
        # Smooth curvature using moving average
        cumulative_curvature = np.cumsum(self._curvature) 
        moving_average = np.roll((cumulative_curvature[self._window_width:] - cumulative_curvature[:-self._window_width]) / self._window_width, self._shift)

        # Detect curves in reference path and determine intersection points between forward and backward propagation
        maximum_curvature_indices, _ = signal.find_peaks(moving_average, prominence=self._find_peaks_prominence)
        maximum_curvature_indices = maximum_curvature_indices.tolist() if len(maximum_curvature_indices) != 0 else [self._curvature.argmax()]   

        for i in range(len(maximum_curvature_indices)):
            local_max = index = maximum_curvature_indices[i]
            for j in range(max(0, index-3), min(len(self._curvature), index+3)):
                if (self._curvature[j] > self._curvature[local_max]):
                    local_max = j
            maximum_curvature_indices[i] = local_max

        maximum_curvature_indices.insert(0, 0)
        maximum_curvature_indices.append(len(self._distances))

        # Loop between curves
        for i in range(len(maximum_curvature_indices)):
            velocity_forward[maximum_curvature_indices[i]] = self.maximum_corner_speed(self._radius_of_curvature[maximum_curvature_indices[i]])
            velocity_backward[maximum_curvature_indices[i]] = self.maximum_corner_speed(self._radius_of_curvature[maximum_curvature_indices[i]])
            
            # Determine maximum velocity at apex curve
            velocity[maximum_curvature_indices[i]] = self.maximum_corner_speed(self._radius_of_curvature[maximum_curvature_indices[i]])
            
            # Forward propagation from one turning point to the next
            if (i != len(maximum_curvature_indices) - 1):
                for j in range(maximum_curvature_indices[i], maximum_curvature_indices[i+1]):
                    v1, a1 = self.propagate_forward(velocity_forward[j], acceleration_forward[j], self._radius_of_curvature[j], self._distances[j])
                    velocity_forward[j+1] = v1
                    acceleration_forward[j+1] = a1

            # Backward propagation from the next turning point back
            if (i != 0):
                for j in range(maximum_curvature_indices[i], maximum_curvature_indices[i-1], -1):
                    v0, a0 = self.propagate_backwards(velocity_backward[j], self._radius_of_curvature[j-1], self._distances[j-1])
                    velocity_backward[j-1] = v0
                    acceleration_backward[j-1] = a0
        
        # Pick lowest velocities of forward and backward velocity
        for i in range(len(self._x)):
            if (velocity_forward[i] < velocity_backward[i]):
                velocity[i] = velocity_forward[i]
                acceleration[i] = acceleration_forward[i]
            else:
                velocity[i] = velocity_backward[i]
                acceleration[i] = acceleration_backward[i]
                
        return velocity, acceleration