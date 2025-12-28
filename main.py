import rerun as rr
import numpy as np
import time

from third_party.random_track_generator import TrackGenerator, Mode
from src.simulation.bicycle_model import NonlinearBicycleModel
from src.planning.midline_path import MidlinePath
from src.control.steering_controller import StanleyController

# Initialize rerun
rr.init("autonomous_racing", spawn=True)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

# Generate track
track_gen = TrackGenerator(
    n_points=60,
    n_regions=20,
    min_bound=0.,
    max_bound=150.,
    mode=Mode.EXTEND,
    plot_track=False
)
while True:
    try:
        track = track_gen.create_track()
        cones_left, cones_right = track
        break
    except:
        continue

# Initalize bicycle model
vehicle = NonlinearBicycleModel()
x = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])  # Initial state: [x, y, psi, v_x, v_y, omega]
                                               # Start with 5 m/s forward velocity
u = np.array([0, 0])    # Initial control: [delta, throttle]
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

# Plot track with rerun
cones_left_3d = np.c_[cones_left, np.zeros(len(cones_left))]
cones_right_3d = np.c_[cones_right, np.zeros(len(cones_right))]

# Log the cones as static (they persist across all timesteps)
rr.log("track/cones_left", rr.Points3D(cones_left_3d, colors=[0, 0, 255], radii=0.15), static=True)
rr.log("track/cones_right", rr.Points3D(cones_right_3d, colors=[255, 255, 0], radii=0.15), static=True)

for t in range(10000):
    # if t > 80: u = np.array([0.0, -0.5])
    t_now = time.time()
    # x = vehicle.step(x, u, dt)

    print(f"x: {x[0]:.2f}, y: {x[1]:.2f}, psi: {np.rad2deg(x[2]):.2f} deg, v_x: {x[3]:.2f} m/s, v_y: {x[4]:.2f} m/s, omega: {np.rad2deg(x[5]):.2f} deg/s")

    # Set the time for this frame
    rr.set_time("step", sequence=t)

    # Get cones in 15 meter radius of the vehicle
    vehicle_pos = x[0:2]  # [x, y]
    cones_left_nearby = cones_left[np.linalg.norm(cones_left - vehicle_pos, axis=1) <= 12.0]
    cones_right_nearby = cones_right[np.linalg.norm(cones_right - vehicle_pos, axis=1) <= 12.0]

    # Compute midline trajectory
    path = planner.update(
        left_cones=cones_left_nearby,
        right_cones=cones_right_nearby,
        vehicle_state=x,
        laps_completed=0
    )

    steer_angle, _ = controller.update(
        vehicle_state=x,
        path=path
    )

    x = vehicle.step(x, np.array([steer_angle, 0.1]), dt)

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

    # Visualize the 15m detection radius around the vehicle
    radius = 12.0
    num_points = 64
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_x = x[0] + radius * np.cos(theta)
    circle_y = x[1] + radius * np.sin(theta)
    circle_3d = np.c_[circle_x, circle_y, np.zeros(num_points)]
    rr.log("car/detection_radius", rr.LineStrips3D([circle_3d], colors=[128, 128, 128]))

