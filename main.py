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

# Planning model shared by the MPCs and the global trajectory optimiser
model = SpatialKinematicModel(vehicle.model.params)

# Exploration lap: MPC on the locally observed midline, horizon = visible path
mpc_explore = MPCController(model, MPCParameters(N=20, ds=None, dt_control=dt))
# Trackdrive: MPC on the mapped track, fixed 60 m horizon, warm-started from the global trajectory
mpc_track = MPCController(model, MPCParameters(N=30, ds=2.0, dt_control=dt))
global_optimizer = GlobalTrajectoryOptimizer(model)

# Map of cones observed during the exploration lap (rounded position -> position)
cone_map = {}
trackdrive_ref = None

# Plot full track cones (static)
cones_left, cones_right = track.as_tuple()
cones_left_3d = np.c_[cones_left, np.zeros(len(cones_left))]
cones_right_3d = np.c_[cones_right, np.zeros(len(cones_right))]

rr.log("track/cones_left", rr.Points3D(cones_left_3d, colors=[0, 0, 255], radii=0.15), static=True)
rr.log("track/cones_right", rr.Points3D(cones_right_3d, colors=[255, 255, 0], radii=0.15), static=True)

lap_timer = LapTimer()
t = 0
lap_start_step = 0

while lap_timer.lap < 2:
    rr.set_time("step", sequence=t)

    lap_before = lap_timer.lap
    lap_timer.check_is_lap(vehicle.position)
    if lap_timer.lap != lap_before:
        print(f"Lap {lap_timer.lap} sim time: {(t - lap_start_step) * dt:.1f} s")
        lap_start_step = t

    # Detect cones within sensor range
    cones_left_nearby = cones_left[np.linalg.norm(cones_left - vehicle.position, axis=1) <= SENSOR_RANGE]
    cones_right_nearby = cones_right[np.linalg.norm(cones_right - vehicle.position, axis=1) <= SENSOR_RANGE]

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

    if lap_timer.lap == 0:
        # Exploration lap: build the cone map and drive with MPC on what is currently visible
        for cone in visible_cones:
            cone_map[(round(cone[0], 1), round(cone[1], 1))] = cone

        result = planner.update(
            left_cones=cones_left_nearby,
            right_cones=cones_right_nearby,
            vehicle_state=vehicle.state,
            laps_completed=lap_timer.lap
        )
        path, vertices = result if result is not None else (None, [])
        local_ref = None
        if path is not None and len(path['x']) > 2:
            local_ref = TrackReference.from_path(path, visible_cones, closed=False)
            midline_3d = np.c_[path['x'], path['y'], np.zeros(len(path['x']))]
            rr.log("planning/local_midline", rr.LineStrips3D([midline_3d], colors=[200, 200, 200]))

        mpc = mpc_explore
        steer_angle, accel, mpc_debug = mpc.update(vehicle.state, local_ref)

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
            mpc_track.set_last_command(mpc_explore.last_command)

        mpc = mpc_track
        steer_angle, accel, mpc_debug = mpc.update(vehicle.state, trackdrive_ref)

    throttle = mpc.acceleration_to_throttle(accel, vehicle.state[3])

    if 'pred_xy' in mpc_debug:
        pred = mpc_debug['pred_xy']
        colors = _speed_to_color(mpc_debug['pred_v'], v_max=model.lim.v_max)
        pred_3d = np.c_[pred, np.zeros(len(pred))]
        rr.log("planning/mpc_prediction", rr.LineStrips3D([pred_3d], colors=[0, 255, 0]))
        rr.log("planning/mpc_prediction_pts", rr.Points3D(pred_3d, colors=colors, radii=0.12))

    # Step vehicle dynamics
    u = np.array([steer_angle, throttle])
    vehicle.step(u, dt)
    print(
        f"lap: {lap_timer.lap}  x: {vehicle.x:.2f}  y: {vehicle.y:.2f}"
        f"  psi: {np.rad2deg(vehicle.heading):.1f}°"
        f"  v: {vehicle.velocity:.2f} m/s"
        f"  throttle: {throttle:.2f}"
        f"  steer: {np.rad2deg(steer_angle):.1f}°  a: {accel:.2f} m/s²"
        f"  mpc: {mpc_debug['status']} {1000 * mpc_debug.get('solve_time', 0.0):.0f} ms"
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
