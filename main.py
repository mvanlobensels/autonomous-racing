import rerun as rr
import numpy as np
import time

from third_party.random_track_generator import generate_track, load_track
from src.simulation.bicycle_model import NonlinearBicycleModel
from src.planning.midline_path import MidlinePath
from src.control.steering_controller import StanleyController

# Initialize rerun
rr.init("autonomous_racing", spawn=True)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

# Generate track
track = generate_track(n_points=60, n_regions=20, min_bound=0., max_bound=150., mode="extend")
# track = load_track("FSG")

# Initalize bicycle model
vehicle = NonlinearBicycleModel()
x = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])  # Initial state: [x, y, psi, v_x, v_y, omega]
dt = 0.1                    # Time step

# Initialize midline planner
planner = MidlinePath(
    smoothing_factor_autocross=1.0,
    smoothing_factor_trackdrive=1.0,
    cone_epsilon=0.5,
    max_vertice_length=5.0
)

controller = StanleyController(
    k=1.0,
    k_soft=1.0,
    max_steer=np.deg2rad(30.0)
)

# Plot track
cones_left, cones_right = track.as_tuple()
cones_left_3d = np.c_[cones_left, np.zeros(len(cones_left))]
cones_right_3d = np.c_[cones_right, np.zeros(len(cones_right))]

rr.log("track/cones_left", rr.Points3D(cones_left_3d, colors=[0, 0, 255], radii=0.15), static=True)
rr.log("track/cones_right", rr.Points3D(cones_right_3d, colors=[255, 255, 0], radii=0.15), static=True)

lap_counter = 0
lap_timer = time.time()
t = 0
while lap_counter < 1:
    # Set the time for this frame
    rr.set_time("step", sequence=t)

    vehicle_pos = x[0:2]
    if np.linalg.norm(np.zeros(2) - vehicle_pos) <= 3:
        now = time.time()
        if now - lap_timer > 2.0:
            lap_counter += 1
            print(f"Lap {lap_counter} completed in {now - lap_timer:.2f} seconds")
            lap_timer = time.time()

    # Get cones in 15 meter radius of the vehicle
    cones_left_nearby = cones_left[np.linalg.norm(cones_left - vehicle_pos, axis=1) <= 10.0]
    cones_right_nearby = cones_right[np.linalg.norm(cones_right - vehicle_pos, axis=1) <= 10.0]

    # Compute midline trajectory
    path, vertices = planner.update(
        left_cones=cones_left_nearby,
        right_cones=cones_right_nearby,
        vehicle_state=x,
        laps_completed=lap_counter
    )

    steer_angle, _ = controller.update(
        vehicle_state=x,
        path=path
    )

    x = vehicle.step(x, np.array([steer_angle, 0.1]), dt)
    print(f"lap: {lap_counter}, x: {x[0]:.2f}, y: {x[1]:.2f}, psi: {np.rad2deg(x[2]):.2f} deg, v_x: {x[3]:.2f} m/s, v_y: {x[4]:.2f} m/s, omega: {np.rad2deg(x[5]):.2f} deg/s")

    # Log the car pose (position and orientation)
    yaw = x[2]
    quat = [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]

    rr.log("car/pose", rr.Boxes3D(
        centers=[[x[0], x[1], 0.0]],
        half_sizes=[[1.0, 0.5, 0.2]],  # Half-sizes: length (x), width (y), height (z)
        rotations=[quat],  # Quaternion rotation
        colors=[255, 0, 0]
    ))

    # Also log an arrow for clarity
    arrow_length = 1.5
    rr.log("car/heading", rr.Arrows3D(
        origins=[[x[0], x[1], 0.0]],
        vectors=[[arrow_length * np.cos(yaw), arrow_length * np.sin(yaw), 0.0]],
        colors=[255, 255, 0]
    ))

    # Log the path if available
    if path is not None:
        path_3d = np.c_[path['x'], path['y'], np.zeros(len(path['x']))]
        rr.log("planning/midline", rr.LineStrips3D([path_3d], colors=[0, 255, 0]))

    # Visualize the Delaunay vertices
    line_segments = []
    for cone_a, cone_b in vertices:
        segment = np.array([
            [cone_a[0], cone_a[1], 0.0],
            [cone_b[0], cone_b[1], 0.0]
        ])
        line_segments.append(segment)
    rr.log("planning/delaunay_vertices", rr.LineStrips3D(line_segments, colors=[255, 0, 255]))

    # Visualize the 15m detection radius around the vehicle
    radius = 10.0
    num_points = 64
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_x = x[0] + radius * np.cos(theta)
    circle_y = x[1] + radius * np.sin(theta)
    circle_3d = np.c_[circle_x, circle_y, np.zeros(num_points)]
    rr.log("car/detection_radius", rr.LineStrips3D([circle_3d], colors=[128, 128, 128]))

    t += 1

