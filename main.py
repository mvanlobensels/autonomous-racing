import time
import rerun as rr
import numpy as np

from random_track_generator import generate_track, load_track
from src.simulation.car import Car
from src.simulation.lap_timer import LapTimer
from src.simulation.bicycle_model import NonlinearBicycleModel
from src.planning.midline_path import MidlinePath
from src.planning.lane_detector import AldBoundaryEstimator
from src.planning.track_reference import TrackReference
from src.planning.global_trajectory import GlobalTrajectoryOptimizer
from src.planning.speed_profile import SpeedProfile
from src.control.steering_controller import StanleyController
from src.control.velocity_controller import VelocityController
from src.control.spatial_model import SpatialKinematicModel
from src.control.mpc_controller import MPCController, MPCParameters


def _speed_to_color(velocities: np.ndarray, v_max: float) -> np.ndarray:
    """Map velocities to RGB colors (slow=blue, fast=yellow)."""
    norm = np.clip(velocities / v_max, 0.0, 1.0)
    r = (norm * 255).astype(np.uint8)
    g = ((0.5 - np.abs(norm - 0.5)) * 2 * 200).astype(np.uint8)
    b = ((1.0 - norm) * 255).astype(np.uint8)
    return np.column_stack([r, g, b])

# Initialize rerun
rr.init("autonomous_racing", spawn=True)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

# Generate track
# track = generate_track('medium')
# track = generate_track(n_points=60, n_regions=20, min_bound=0., max_bound=150., mode="extend")
track = load_track("FSG")

# Initalize bicycle model
vehicle = Car(
    initial_state=np.zeros(6),          # [x, y, psi, v_x, v_y, omega]
    model=NonlinearBicycleModel())
dt = 0.1
SENSOR_RANGE = 10.0  # [m] cone detection radius

# Initialize midline planner
planner = MidlinePath(
    smoothing_factor_autocross=1.0,
    smoothing_factor_trackdrive=1.0,
    cone_epsilon=0.5,
    max_vertice_length=5.0
)

# Initialize boundary estimator
# Seed heads offset laterally from the vehicle's initial position along the heading
lateral_offset = 2.0  # [m] initial guess for track half-width
perp = np.array([-np.sin(vehicle.heading), np.cos(vehicle.heading)])  # perpendicular to heading
ald = AldBoundaryEstimator(
    inner_head_pos=vehicle.position - lateral_offset * perp,
    inner_head_tangent=np.array([np.cos(vehicle.heading), np.sin(vehicle.heading)]),
    outer_head_pos=vehicle.position + lateral_offset * perp,
    outer_head_tangent=np.array([np.cos(vehicle.heading), np.sin(vehicle.heading)]),
)

# Exploration lap: speed profile on the local midline, tracked by Stanley + P velocity controller
speed_profile = SpeedProfile()
vel_controller = VelocityController(kp_throttle=1.0, kp_brake=0.8, max_throttle=0.5, max_brake=0.8)
steering_controller = StanleyController(k=1.0, k_soft=1.0, max_steer=np.deg2rad(30.0))
steer_angle = 0.0
explore_accel = 0.0  # measured longitudinal acceleration, used to seed the trackdrive MPC

# Planning model shared by the MPC and the global trajectory optimiser
model = SpatialKinematicModel(vehicle.model.params)

# Trackdrive: MPC on the mapped track, fixed 60 m horizon, warm-started from the global trajectory
mpc_track = MPCController(model, MPCParameters(N=30, ds=2.0, dt_control=dt))
global_optimizer = GlobalTrajectoryOptimizer(model)

# Map of cones observed during the exploration lap (rounded position -> position)
cone_map = {}
trackdrive_ref = None

# Ground-truth cones (3D for visualisation)
cones_left, cones_right = track.as_tuple()
cones_left_3d = np.c_[cones_left, np.zeros(len(cones_left))]
cones_right_3d = np.c_[cones_right, np.zeros(len(cones_right))]

# Cones are only visualised once they have entered the sensor range
seen_left = np.zeros(len(cones_left), dtype=bool)
seen_right = np.zeros(len(cones_right), dtype=bool)

lap_timer = LapTimer()
t = 0
lap_start_step = 0
n_laps = 3

while lap_timer.lap < n_laps:
    rr.set_time("step", sequence=t)

    lap_before = lap_timer.lap
    lap_timer.check_is_lap(vehicle.position)
    if lap_timer.lap != lap_before:
        print(f"Lap {lap_timer.lap} sim time: {(t - lap_start_step) * dt:.1f} s")
        lap_start_step = t

    # Detect cones within sensor range
    in_range_left = np.linalg.norm(cones_left - vehicle.position, axis=1) <= SENSOR_RANGE
    in_range_right = np.linalg.norm(cones_right - vehicle.position, axis=1) <= SENSOR_RANGE
    cones_left_nearby = cones_left[in_range_left]
    cones_right_nearby = cones_right[in_range_right]

    # Log the cones seen so far (only when a new cone enters the sensor range)
    if np.any(in_range_left & ~seen_left) or np.any(in_range_right & ~seen_right):
        seen_left |= in_range_left
        seen_right |= in_range_right
        rr.log("track/cones_left", rr.Points3D(cones_left_3d[seen_left], colors=[0, 0, 255], radii=0.15))
        rr.log("track/cones_right", rr.Points3D(cones_right_3d[seen_right], colors=[255, 255, 0], radii=0.15))

    # ALD: estimate ordered track boundaries from combined visible cones
    if lap_timer.lap == 0 and len(cones_left_nearby) + len(cones_right_nearby) >= 4:
        n_left = len(cones_left_nearby)
        all_cones = np.vstack([cones_left_nearby, cones_right_nearby])
        inner_ix, outer_ix, _ = ald.update(all_cones, allow_closure=False)

        for name, ix in [("boundary_inner", inner_ix), ("boundary_outer", outer_ix)]:
            if len(ix) < 2:
                continue
            # Colour matches the cone side that makes up the majority of this boundary
            left_majority = sum(i < n_left for i in ix) >= len(ix) / 2
            color = [0, 0, 255] if left_majority else [255, 255, 0]
            pts_3d = np.c_[all_cones[ix], np.zeros(len(ix))]
            rr.log(f"planning/{name}", rr.LineStrips3D([pts_3d], colors=color))
            rr.log(f"planning/{name}_pts", rr.Points3D(pts_3d, colors=color, radii=0.2))

    visible_cones = np.vstack([cones_left_nearby, cones_right_nearby])

    mpc_debug = None
    if lap_timer.lap == 0:
        # Autocross lap: build the cone map and track the visible midline with
        # speed profile + Stanley steering + P velocity control
        for cone in visible_cones:
            cone_map[(round(cone[0], 1), round(cone[1], 1))] = cone

        result = planner.update(
            left_cones=cones_left_nearby,
            right_cones=cones_right_nearby,
            vehicle_state=vehicle.state,
            laps_completed=lap_timer.lap
        )
        path, vertices = result if result is not None else (None, [])

        throttle = 0.0  # coast if no midline is available; steering holds its last value
        if path is not None and len(path['x']) > 2:
            midline_3d = np.c_[path['x'], path['y'], np.zeros(len(path['x']))]
            rr.log("planning/local_midline", rr.LineStrips3D([midline_3d], colors=[200, 200, 200]))

            velocity_profile = speed_profile.update(path, current_velocity=vehicle.velocity,
                                                    laps_completed=lap_timer.lap)['velocity']
            path_pts = np.column_stack([path['x'], path['y']])
            closest_idx = int(np.argmin(np.linalg.norm(path_pts - vehicle.position, axis=1)))
            lookahead_idx = min(closest_idx + 3, len(velocity_profile) - 1)
            throttle = vel_controller.update(vehicle.velocity, float(velocity_profile[lookahead_idx]))
            steer_angle, _ = steering_controller.update(vehicle_state=vehicle.state, path=path)

            colors = _speed_to_color(velocity_profile, v_max=speed_profile.params.v_max)
            profile_3d = np.c_[path['x'], path['y'], np.zeros(len(path['x']))]
            rr.log("planning/speed_profile", rr.Points3D(profile_3d, colors=colors, radii=0.15))

    else:
        # Trackdrive: global min-time trajectory on the exploration-lap map, tracked by MPC
        vertices = []
        if trackdrive_ref is None:
            midline, _ = planner.full_track_midline()
            mapped_cones = np.array(list(cone_map.values()))
            trackdrive_ref = TrackReference.from_path(midline, mapped_cones, closed=True)
            global_traj = global_optimizer.optimize(trackdrive_ref)
            print(f"Global trajectory: {global_traj['status']}, predicted lap time "
                  f"{global_traj['lap_time']:.2f} s, solved in {global_traj['solve_time']:.1f} s "
                  f"({len(mapped_cones)} mapped cones, track length {trackdrive_ref.length:.0f} m)")
            if global_traj['status'] == 'ok':
                trackdrive_ref.guess = global_traj
                colors = _speed_to_color(global_traj['v'], v_max=model.lim.v_max)
                line_3d = np.c_[global_traj['x'], global_traj['y'], np.zeros(len(global_traj['x']))]
                rr.log("planning/global_trajectory", rr.Points3D(line_3d, colors=colors, radii=0.15))
            # Seed the input-rate penalty with the exploration controllers' last command
            mpc_track.set_last_command([steer_angle, explore_accel])

        steer_angle, accel, mpc_debug = mpc_track.update(vehicle.state, trackdrive_ref)
        throttle = mpc_track.acceleration_to_throttle(accel, vehicle.state[3])

    if mpc_debug is not None and 'pred_xy' in mpc_debug:
        pred = mpc_debug['pred_xy']
        colors = _speed_to_color(mpc_debug['pred_v'], v_max=model.lim.v_max)
        pred_3d = np.c_[pred, np.zeros(len(pred))]
        rr.log("planning/mpc_prediction", rr.LineStrips3D([pred_3d], colors=[0, 255, 0]))
        rr.log("planning/mpc_prediction_pts", rr.Points3D(pred_3d, colors=colors, radii=0.12))

    # Step vehicle dynamics
    u = np.array([steer_angle, throttle])
    v_before = vehicle.velocity
    vehicle.step(u, dt)
    if mpc_debug is None:
        explore_accel = (vehicle.velocity - v_before) / dt
        ctrl_info = f"a: {explore_accel:.2f} m/s²  ctrl: stanley"
    else:
        ctrl_info = (f"a: {accel:.2f} m/s²  mpc: {mpc_debug['status']} "
                     f"{1000 * mpc_debug.get('solve_time', 0.0):.0f} ms")
    print(
        f"lap: {lap_timer.lap}  x: {vehicle.x:.2f}  y: {vehicle.y:.2f}"
        f"  psi: {np.rad2deg(vehicle.heading):.1f}°"
        f"  v: {vehicle.velocity:.2f} m/s"
        f"  throttle: {throttle:.2f}"
        f"  steer: {np.rad2deg(steer_angle):.1f}°  {ctrl_info}"
    )

    # Rerun visualization
    quat = [0.0, 0.0, np.sin(vehicle.heading / 2), np.cos(vehicle.heading / 2)]

    rr.log("car/pose", rr.Boxes3D(
        centers=[[vehicle.x, vehicle.y, 0.0]],
        half_sizes=[[1.0, 0.5, 0.2]],
        rotations=[quat],
        colors=[255, 0, 0]
    ))
    rr.log("car/heading", rr.Arrows3D(
        origins=[[vehicle.x, vehicle.y, 0.0]],
        vectors=[[1.5 * np.cos(vehicle.heading), 1.5 * np.sin(vehicle.heading), 0.0]],
        colors=[255, 255, 0]
    ))



    line_segments = []
    for cone_a, cone_b in vertices:
        line_segments.append(np.array([
            [cone_a[0], cone_a[1], 0.0],
            [cone_b[0], cone_b[1], 0.0]
        ]))
    rr.log("planning/delaunay_vertices", rr.LineStrips3D(line_segments, colors=[255, 0, 255]))

    # Detection radius
    theta_circle = np.linspace(0, 2 * np.pi, 64)
    circle_3d = np.c_[
        vehicle.x + SENSOR_RANGE * np.cos(theta_circle),
        vehicle.y + SENSOR_RANGE * np.sin(theta_circle),
        np.zeros(64)
    ]
    rr.log("car/detection_radius", rr.LineStrips3D([circle_3d], colors=[128, 128, 128]))

    t += 1
