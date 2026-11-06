"""
Global minimum-time trajectory optimisation (after the exploration lap).

Follows the idea of TUMFTM's global_racetrajectory_optimization `mintime`
mode: the vehicle model is written in curvilinear coordinates along the
reference line and integrated over arc length, the track is closed
(periodic boundary conditions), and the objective is the lap time
    T = sum_k ds (1 - n_k kappa_k) / (v_k cos(mu_k + beta_k)).
The track bounds come from the cones observed during the exploration lap.
The model is the same kinematic bicycle the MPC uses, so the MPC can be
warm-started from, and constrained by, this solution.
"""

import time
from dataclasses import dataclass
from typing import Optional

import casadi as ca
import numpy as np

from src.control.spatial_model import SpatialKinematicModel
from src.planning.track_reference import TrackReference


@dataclass
class GlobalTrajectoryParameters:
    ds: float = 2.0             # [m] target stage length
    w_ddelta: float = 1.0       # steering smoothness
    w_da: float = 1e-3          # acceleration smoothness
    max_iter: int = 3000


class GlobalTrajectoryOptimizer:

    def __init__(self, model: SpatialKinematicModel,
                 params: Optional[GlobalTrajectoryParameters] = None):
        self.model = model
        self.p = params if params is not None else GlobalTrajectoryParameters()

    def optimize(self, ref: TrackReference) -> dict:
        assert ref.closed, "global optimisation requires a closed reference"
        model, p = self.model, self.p
        N = max(int(round(ref.length / p.ds)), 10)
        ds = ref.length / N
        s = ds * np.arange(N)
        stage = ref.sample(s)
        kappa = ref.sample(s + ds / 2)['kappa']

        margin = model.lim.car_half_width
        n_ub = np.maximum(stage['n_left'] - margin, 0.1)
        n_lb = np.minimum(-stage['n_right'] + margin, -0.1)

        opti = ca.Opti()
        X = opti.variable(3, N)     # [n, mu, v] at s_k, periodic
        U = opti.variable(2, N)     # [delta, a]

        lap_time = 0
        smooth = 0
        for k in range(N):
            kn = (k + 1) % N
            opti.subject_to(X[:, kn] == model.rk4(X[:, k], U[:, k], kappa[k], ds))
            lap_time += model.stage_time(X[:, k], U[:, k], kappa[k], ds)
            smooth += p.w_ddelta * (U[0, kn] - U[0, k]) ** 2 + p.w_da * (U[1, kn] - U[1, k]) ** 2
        opti.minimize(lap_time + smooth)

        opti.subject_to(opti.bounded(n_lb, X[0, :].T, n_ub))
        model.add_constraints(opti, X, U)

        # Initial guess: midline at steady-state cornering speed
        v_guess = 0.8 * model.corner_speed(stage['kappa'])
        opti.set_initial(X[0, :], 0)
        opti.set_initial(X[1, :], 0)
        opti.set_initial(X[2, :], v_guess.reshape(1, -1))
        opti.set_initial(U[0, :], model.steady_state_steer(kappa).reshape(1, -1))
        opti.set_initial(U[1, :], 0)

        opti.solver('ipopt', {'print_time': False},
                    {'print_level': 0, 'sb': 'yes', 'max_iter': p.max_iter})

        t0 = time.perf_counter()
        try:
            sol = opti.solve()
            val, status = sol.value, 'ok'
        except RuntimeError:
            val, status = opti.debug.value, 'failed'
        solve_time = time.perf_counter() - t0

        X_sol, U_sol = val(X), val(U)
        xy = ref.to_world(s, X_sol[0])
        return {
            'status': status,
            'solve_time': solve_time,
            'lap_time': float(val(lap_time)),
            's': s,
            'n': X_sol[0], 'mu': X_sol[1], 'v': X_sol[2],
            'delta': U_sol[0], 'a': U_sol[1],
            'x': xy[:, 0], 'y': xy[:, 1],
        }
