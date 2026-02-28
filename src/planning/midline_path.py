#!/usr/bin/env python
import numpy as np
from scipy import interpolate
from scipy.spatial import Delaunay

class MidlinePath:

    def __init__(self,
                 smoothing_factor_autocross=1.0,
                 smoothing_factor_trackdrive=1.0,
                 cone_epsilon=0.5,
                 max_vertice_length=10.0):
        """
        Initialize the MidlinePath planner.

        Args:
            smoothing_factor_autocross (float): Smoothing factor for exploration mode spline
            smoothing_factor_trackdrive (float): Smoothing factor for full track mode spline
            cone_epsilon (float): Distance threshold for filtering duplicate cones
            max_vertice_length (float): Maximum valid distance between cones for triangulation
        """
        # Store parameters
        self.smoothing_factor_autocross = smoothing_factor_autocross
        self.smoothing_factor_trackdrive = smoothing_factor_trackdrive
        self.cone_epsilon = cone_epsilon
        self.max_vertice_length = max_vertice_length

        # Internal state
        self.exploration_mode = True
        self.midpoints = []
        self.cone_id_to_cone = {}  

    def heading(self, dx_dt, dy_dt):
        """
        Calculates the heading along a line.
        
        Args:
            dx_dt (numpy.ndarray): First derivative of x.
            dy_dt (numpy.ndarray): First derivative of y.
        
        Returns:
            np.ndarray: Heading along line in radians with respect to the world frame.
        """
        return np.arctan2(dy_dt, dx_dt)

    def curvature(self, dx_dt, d2x_dt2, dy_dt, d2y_dt2):
        """
        Calculates the curvature along a line.
        
        Args:
            dx_dt (numpy.ndarray): First derivative of x.
            d2x_dt2 (numpy.ndarray): Second derivative of x.
            dy_dt (numpy.ndarray): First derivative of y.
            d2y_dt2 (numpy.ndarray): Second derivative of y.
        
        Returns:
            np.ndarray: Curvature along line.
        """
        return (dx_dt**2 + dy_dt**2)**-1.5 * (dx_dt * d2y_dt2 - dy_dt * d2x_dt2)

    def filter_duplicate_cones(self, cones):
        """
        This method compares each cone in the cones array to
        the current list of reported cones. It snaps cones that have
        shifted by less than self._cone_epsilon to their original position.

        Args:
            cones (np.ndarray): Nx2 array of cone positions [[x1, y1], [x2, y2], ...]

        Returns:
            np.ndarray: Filtered cone positions
        """
        if len(cones) == 0:
            return cones

        filtered_cones = cones.copy()

        for i, current_cone in enumerate(cones):
            # Create a simple ID based on initial position (rounded to avoid floating point issues)
            cone_id = (round(current_cone[0], 1), round(current_cone[1], 1))

            if cone_id in self.cone_id_to_cone:
                previous_cone = self.cone_id_to_cone[cone_id]
                distance = np.linalg.norm(current_cone - previous_cone)

                if distance < self.cone_epsilon:
                    filtered_cones[i] = previous_cone
                else:
                    self.cone_id_to_cone[cone_id] = current_cone
            else:
                self.cone_id_to_cone[cone_id] = current_cone

        return filtered_cones

    def delauney_midpoints(self, x1, y1, x2, y2):
        """
        Find midpoints using Delauney triangulation.
        Midpoints are the centerpoints of vertices where the 
        source and target of the vertice are not on the same boundary.
        
        Args:
            x1 (list): x-positions of cones in boundary1.
            y1 (list): y-positions of cones in boundary1.
            x2 (list): x-positions of cones in boundary2.
            y2 (list): y-positions of cones in boundary2.
        
        Returns:
            list, list: x-positions of midpoints, y-positions of midpoints.
        """

        def _closest_node(node, nodes):
            nodes = np.asarray(nodes)
            deltas = nodes - node
            distance = np.einsum('ij,ij->i', deltas, deltas)
            return np.argmin(distance)

        def _valid_vertice(cone_A, cone_B):
            if (cone_A in boundary1 and cone_B in boundary2) or (cone_B in boundary1 and cone_A in boundary2):
                if ((cone_A, cone_B)) not in vertices and ((cone_B, cone_A)) not in vertices and np.sqrt(sum((np.asarray(cone_A) - np.asarray(cone_B))**2)) < self.max_vertice_length:
                    vertices.append((cone_A, cone_B))
                    x, y = (np.asarray(cone_A) + np.asarray(cone_B)) / 2
                    midpoints.append((x, y)) 

        boundary1 = list(zip(x1, y1))
        boundary2 = list(zip(x2, y2))
        cones = boundary1 + boundary2

        vertices = []
        midpoints = []
            
        # Find Delaunay triangulation of cones
        triangulation = Delaunay(cones)
        simplices = triangulation.points[triangulation.simplices]

        # Find the vertices whose source and target do not lie in the same boundary
        # Add midpoints of valid vertices to midpoints list
        for simplice in simplices:
            cone1, cone2, cone3 = map(tuple, simplice)
            _valid_vertice(cone1, cone2)
            _valid_vertice(cone1, cone3)
            _valid_vertice(cone2, cone3)
        
        # Sort midpoints by finding the closest midpoint to the previous midpoint
        sorted_midpoints_x = []
        sorted_midpoints_y = []
        first_midpoint = tuple((np.array([x1[0], y1[0]]) + np.array([x2[0], y2[0]])) / 2)
        if first_midpoint in midpoints:
            midpoints.remove(first_midpoint)

        for i in range(-1, len(midpoints) - 1):
            node = first_midpoint if i == -1 else (sorted_midpoints_x[i], sorted_midpoints_y[i])
            index = _closest_node(node, midpoints)
            sorted_midpoints_x.append(midpoints[index][0])
            sorted_midpoints_y.append(midpoints[index][1])
            midpoints.remove(midpoints[index])

        return sorted_midpoints_x, sorted_midpoints_y, vertices
            
    def compute_midline(self, left_cones, right_cones, vehicle_state, laps_completed):
        """
        Interpolates through Delauney midpoints with
        a cubic smoothing spline to generate a midline.

        Args:
            left_cones (np.ndarray): Nx2 array of left cone positions
            right_cones (np.ndarray): Nx2 array of right cone positions
            vehicle_pos (tuple): (x, y) position of vehicle
            vehicle_heading (float): Heading angle in radians
            laps_completed (int): Number of laps completed

        Returns:
            dict: Path dictionary with keys 'x', 'y', 'theta', 'curvature'
        """
        vehicle_pos = (vehicle_state[0], vehicle_state[1])
        vehicle_heading = vehicle_state[2]

        # Split boundaries into x and y positions
        x_pos_left_cones = left_cones[:, 0].tolist()
        y_pos_left_cones = left_cones[:, 1].tolist()

        x_pos_right_cones = right_cones[:, 0].tolist()
        y_pos_right_cones = right_cones[:, 1].tolist()  

        # Find midpoints using Delaunay triangulation
        midpoints_x, midpoints_y, vertices = self.delauney_midpoints(x_pos_left_cones, y_pos_left_cones, x_pos_right_cones, y_pos_right_cones)

        # Save first midpoint of every iteration, if not saved already
        # and only if midpoint is behind the car
        midpoint = (midpoints_x[0], midpoints_y[0])
        angle = np.arctan2(vehicle_pos[1] - midpoint[1], vehicle_pos[0] - midpoint[0])
        if midpoint not in self.midpoints and laps_completed == 0 and abs(vehicle_heading - angle) < np.pi / 2:
            self.midpoints.append(midpoint)

        # When number of midpoints is smaller or equal to the degree of interpolator, i.e 3, use linear interpolation to draw a line from the position of the car to the last midpoint
        if len(midpoints_x) <= 3:
            linear_x = [vehicle_pos[0], midpoints_x[-1]]
            linear_y = [vehicle_pos[1], midpoints_y[-1]]
            linear = interpolate.interp1d(linear_x, linear_y, kind='linear')

            n = 100
            x = np.linspace(linear_x[0], linear_x[1], n)
            y = linear(x)
            theta = np.full(n, vehicle_heading)
            curvature = np.full(n, 0)

        else:
            # Cubic interpolation through generated midpoints to generate midline
            tck, _ = interpolate.splprep([midpoints_x, midpoints_y], s=self.smoothing_factor_autocross)
            t = np.linspace(0, 1, 100)
            x, y = interpolate.splev(t, tck, der=0)
            dx_dt, dy_dt = interpolate.splev(t, tck, der=1)
            d2x_dt2, d2y_dt2 = interpolate.splev(t, tck, der=2)

            theta = self.heading(dx_dt, dy_dt)
            curvature = self.curvature(dx_dt, d2x_dt2, dy_dt, d2y_dt2)

            # Cut off midline behind the car
            coordinates = np.column_stack((x, y))
            closest_point = np.sum((coordinates - vehicle_pos)**2, axis=1)
            index = np.argmin(closest_point)

            x = x[index::]
            y = y[index::]
            theta = theta[index::]
            curvature = curvature[index::]

        path = {
            'x': x,
            'y': y,
            'theta': theta,
            'curvature': curvature
        }
        return path, vertices

    def full_track_midline(self):
        """
        Cubic smoothing spline interpolation between saved
        midpoints to generate a full track midline.

        Returns:
            dict: Path dictionary with keys 'x', 'y', 'theta', 'curvature'
        """
        # Add first midpoint to end of lists for periodic interpolation
        midpoints_x, midpoints_y = map(list, zip(*self.midpoints))
        midpoints_x.append(midpoints_x[0])
        midpoints_y.append(midpoints_y[0])

        tck, _ = interpolate.splprep([midpoints_x, midpoints_y], s=self.smoothing_factor_trackdrive, per=True)
        t = np.linspace(0, 1, 1000)
        x, y = interpolate.splev(t, tck, der=0)
        dx_dt, dy_dt = interpolate.splev(t, tck, der=1)
        d2x_dt2, d2y_dt2 = interpolate.splev(t, tck, der=2)

        theta = self.heading(dx_dt, dy_dt)
        curvature = self.curvature(dx_dt, d2x_dt2, dy_dt, d2y_dt2)

        # Return reference path as dictionary
        return {
            'x': x,
            'y': y,
            'theta': theta,
            'curvature': curvature
        }, []

    def update(self, left_cones, right_cones, vehicle_state, laps_completed=0):
        """
        Main update method to be called each simulation step.

        Args:
            left_cones (np.ndarray): Nx2 array of left cone positions [[x1, y1], [x2, y2], ...]
            right_cones (np.ndarray): Nx2 array of right cone positions [[x1, y1], [x2, y2], ...]
            vehicle_pos (tuple): (x, y) position of vehicle
            vehicle_heading (float): Heading angle in radians
            laps_completed (int): Number of laps completed (default: 0)

        Returns:
            dict: Path dictionary with keys 'x', 'y', 'theta', 'curvature'
                  Returns None if insufficient cones
        """
        # Filter duplicate cones
        left_cones_filtered = self.filter_duplicate_cones(left_cones)
        right_cones_filtered = self.filter_duplicate_cones(right_cones)

        min_cones = min(len(left_cones_filtered), len(right_cones_filtered))

        # Check if we have enough cones
        if min_cones <= 1:
            return None

        # If first lap completed, switch to full track mode
        if self.exploration_mode and laps_completed > 0:
            self.exploration_mode = False
            return self.full_track_midline()

        # Exploration mode: compute midline from current cone observations
        if self.exploration_mode:
            return self.compute_midline(left_cones_filtered, right_cones_filtered,
                                       vehicle_state, laps_completed)
        else:
            # Full track mode: use saved midpoints
            return self.full_track_midline()