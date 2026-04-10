import rerun as rr
import numpy as np
import time


from random_track_generator import generate_track, load_track
from src.simulation.bicycle_model import NonlinearBicycleModel
from src.planning.midline_path import MidlinePath
from src.planning.lane_detector import AldBoundaryEstimator
from src.planning.speed_profile import SpeedProfile
from src.control.steering_controller import StanleyController
from src.control.velocity_controller import VelocityController


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
vehicle = NonlinearBicycleModel()
x = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])  # [x, y, psi, v_x, v_y, omega]
dt = 0.1

# Initialize midline planner
planner = MidlinePath(
    smoothing_factor_autocross=1.0,
    smoothing_factor_trackdrive=1.0,
    cone_epsilon=0.5,
    max_vertice_length=5.0
)

# Initialize ALD boundary estimator
# Seed heads offset laterally from the vehicle's initial position along the heading
lateral_offset = 2.0  # [m] initial guess for track half-width
heading = x[2]
perp = np.array([-np.sin(heading), np.cos(heading)])  # perpendicular to heading
ald = AldBoundaryEstimator(
    inner_head_pos=x[:2] - lateral_offset * perp,
    inner_head_tangent=np.array([np.cos(heading), np.sin(heading)]),
    outer_head_pos=x[:2] + lateral_offset * perp,
    outer_head_tangent=np.array([np.cos(heading), np.sin(heading)]),
)

# Initialize speed profile generator and velocity controller
speed_profiler = SpeedProfile()
vel_controller = VelocityController(kp_throttle=0.3, kp_brake=0.8, max_throttle=0.2, max_brake=0.8)

controller = StanleyController(
    k=1.0,
    k_soft=1.0,
    max_steer=np.deg2rad(30.0)
)

# Plot full track cones (static)
cones_left, cones_right = track.as_tuple()
cones_left_3d = np.c_[cones_left, np.zeros(len(cones_left))]
cones_right_3d = np.c_[cones_right, np.zeros(len(cones_right))]

rr.log("track/cones_left", rr.Points3D(cones_left_3d, colors=[0, 0, 255], radii=0.15), static=True)
rr.log("track/cones_right", rr.Points3D(cones_right_3d, colors=[255, 255, 0], radii=0.15), static=True)

lap_counter = 0
lap_timer = time.time()
t = 0

# Full-track speed profile, computed once after the first lap
full_track_path = None
full_track_speed_profile = None

while lap_counter < 2:
    rr.set_time("step", sequence=t)

    vehicle_pos = x[:2]
    if np.linalg.norm(np.zeros(2) - vehicle_pos) <= 3:
        now = time.time()
        if now - lap_timer > 2.0:
            lap_counter += 1
            print(f"Lap {lap_counter} completed")
            lap_timer = time.time()

    # Detect cones within 10 m
    cones_left_nearby = cones_left[np.linalg.norm(cones_left - vehicle_pos, axis=1) <= 10.0]
    cones_right_nearby = cones_right[np.linalg.norm(cones_right - vehicle_pos, axis=1) <= 10.0]

    # ------------------------------------------------------------------
    # ALD: estimate ordered track boundaries from combined visible cones
    # ------------------------------------------------------------------
    if planner.exploration_mode and len(cones_left_nearby) + len(cones_right_nearby) >= 4:
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

    # ------------------------------------------------------------------
    # Midline planner
    # ------------------------------------------------------------------
    path, vertices = planner.update(
        left_cones=cones_left_nearby,
        right_cones=cones_right_nearby,
        vehicle_state=x,
        laps_completed=lap_counter
    )

    # ------------------------------------------------------------------
    # Speed profile
    # ------------------------------------------------------------------
    throttle = 0.1  # fallback constant throttle
    current_speed = float(np.sqrt(x[3]**2 + x[4]**2))

    if not planner.exploration_mode:
        # Trackdrive mode: compute full-track speed profile once, then reuse every step
        if full_track_speed_profile is None and path is not None:
            full_track_path = path
            sp_result = speed_profiler.update(full_track_path, current_velocity=0.0,
                                              laps_completed=lap_counter)
            full_track_speed_profile = sp_result['velocity']
            print(f"Full-track speed profile computed: "
                  f"{full_track_speed_profile.min():.1f} – {full_track_speed_profile.max():.1f} m/s")
            vel_controller.max_throttle = 1.0

            # Log at this timestep so it appears from lap 2 onwards in the timeline
            colors = _speed_to_color(full_track_speed_profile, v_max=speed_profiler.params.v_max)
            profile_3d = np.c_[full_track_path['x'], full_track_path['y'],
                                np.zeros(len(full_track_path['x']))]
            rr.log("planning/speed_profile", rr.Points3D(profile_3d, colors=colors, radii=0.15))

        if full_track_speed_profile is not None:
            path_pts = np.column_stack([full_track_path['x'], full_track_path['y']])
            closest_idx = int(np.argmin(np.linalg.norm(path_pts - vehicle_pos, axis=1)))
            n = len(full_track_speed_profile)
            lookahead_idx = (closest_idx + 20) % n
            target_speed = float(full_track_speed_profile[lookahead_idx])
            throttle = vel_controller.update(current_speed, target_speed)

    elif path is not None:
        # Exploration mode: recompute speed profile each step from the local path
        sp_result = speed_profiler.update(path, current_velocity=current_speed,
                                          laps_completed=lap_counter)
        velocity_profile = sp_result['velocity']

        path_pts = np.column_stack([path['x'], path['y']])
        closest_idx = int(np.argmin(np.linalg.norm(path_pts - vehicle_pos, axis=1)))
        lookahead_idx = min(closest_idx + 3, len(velocity_profile) - 1)
        target_speed = float(velocity_profile[lookahead_idx])
        throttle = vel_controller.update(current_speed, target_speed)

        # Visualize local speed profile
        colors = _speed_to_color(velocity_profile, v_max=speed_profiler.params.v_max)
        profile_3d = np.c_[path['x'], path['y'], np.zeros(len(path['x']))]
        rr.log("planning/speed_profile", rr.Points3D(profile_3d, colors=colors, radii=0.15))

    # ------------------------------------------------------------------
    # Steering controller
    # ------------------------------------------------------------------
    steering_path = full_track_path if (not planner.exploration_mode and full_track_path is not None) else path
    steer_angle, _ = controller.update(vehicle_state=x, path=steering_path)

    # ------------------------------------------------------------------
    # Step vehicle dynamics
    # ------------------------------------------------------------------
    x = vehicle.step(x, np.array([steer_angle, throttle]), dt)
    print(
        f"lap: {lap_counter}  x: {x[0]:.2f}  y: {x[1]:.2f}"
        f"  psi: {np.rad2deg(x[2]):.1f}°"
        f"  v: {np.sqrt(x[3]**2+x[4]**2):.2f} m/s"
        f"  throttle: {throttle:.2f}"
    )

    # ------------------------------------------------------------------
    # Rerun visualization
    # ------------------------------------------------------------------
    yaw = x[2]
    quat = [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]

    rr.log("car/pose", rr.Boxes3D(
        centers=[[x[0], x[1], 0.0]],
        half_sizes=[[1.0, 0.5, 0.2]],
        rotations=[quat],
        colors=[255, 0, 0]
    ))
    rr.log("car/heading", rr.Arrows3D(
        origins=[[x[0], x[1], 0.0]],
        vectors=[[1.5 * np.cos(yaw), 1.5 * np.sin(yaw), 0.0]],
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
        x[0] + 10.0 * np.cos(theta_circle),
        x[1] + 10.0 * np.sin(theta_circle),
        np.zeros(64)
    ]
    rr.log("car/detection_radius", rr.LineStrips3D([circle_3d], colors=[128, 128, 128]))

    t += 1
