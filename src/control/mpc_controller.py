"""
Minimum-time Model Predictive Controller (steering + acceleration).

Spatial-domain MPC: arc length along a reference line is the independent
variable, so the objective is the time needed to traverse the horizon (see
`SpatialKinematicModel`). Outputs a steering angle and an acceleration.

Two modes, chosen by the TrackReference that is passed in:
- Exploration (open reference): the local midline built from the cones that
  are currently visible. The horizon spans the visible path, the first solve
  is hot-started from the midline, and a terminal speed constraint keeps the
  car able to handle a tight corner just beyond the sensor range.
- Trackdrive (closed reference with a global solution): the full-track
  midline from the exploration-lap map. The horizon has a fixed length, the
  solver is warm-started from the global minimum-time trajectory, and the
  terminal speed is bounded by the global speed profile.
"""

import time
from dataclasses import dataclass
from typing import Optional

import casadi as ca
import numpy as np

from src.control.spatial_model import SpatialKinematicModel
from src.planning.track_reference import TrackReference


@dataclass
class MPCParameters:
    N: int = 20                     # number of spatial stages
    ds: Optional[float] = None      # [m] fixed stage length; None: spread N stages over the remaining path
    min_horizon: float = 3.0        # [m] minimum remaining path length to run the MPC
    max_steer_rate: float = 1.5     # [rad/s]
    r_min_unseen: float = 5.0       # [m] assumed tightest radius beyond the end of an open reference
    dt_control: float = 0.1         # [s] controller sample time (first-stage steering rate limit)
    # Cost weights (time is the main objective)
    w_ddelta: float = 1.0
    w_da: float = 1e-3
    w_n: float = 1e-3
    w_slack: float = 1e3


class MPCController:

    def __init__(self, model: SpatialKinematicModel, params: Optional[MPCParameters] = None):
        self.model = model
        self.p = params if params is not None else MPCParameters()
        self._prev_X = None
        self._prev_U = None
        self._u_prev = np.zeros(2)
        self._fallback_idx = 0
        self._build()

    # ------------------------------------------------------------------
    # Optimisation problem (built once)
    # ------------------------------------------------------------------

    def _build(self):
        p, model, N = self.p, self.model, self.p.N
        opti = ca.Opti()

        X = opti.variable(3, N + 1)          # [n, mu, v]
        U = opti.variable(2, N)              # [delta, a]
        slack_n = opti.variable(N + 1)       # soft track bounds
        slack_v = opti.variable()            # soft terminal speed

        x0 = opti.parameter(3)
        ds = opti.parameter()
        kappa = opti.parameter(N)
        n_ub = opti.parameter(N + 1)
        n_lb = opti.parameter(N + 1)
        u_prev = opti.parameter(2)
        v_term_max = opti.parameter()

        n, v, delta, a = X[0, :], X[2, :], U[0, :], U[1, :]

        # Objective: traversal time of the horizon + regularisation
        cost = 0
        for k in range(N):
            cost += model.stage_time(X[:, k], U[:, k], kappa[k], ds)
            u_last = u_prev if k == 0 else U[:, k - 1]
            cost += p.w_ddelta * (delta[k] - u_last[0]) ** 2
            cost += p.w_da * (a[k] - u_last[1]) ** 2
            cost += p.w_n * n[k] ** 2
        cost += p.w_slack * (ca.sumsqr(slack_n) + slack_v ** 2)
        opti.minimize(cost)

        # Dynamics (multiple shooting)
        opti.subject_to(X[:, 0] == x0)
        for k in range(N):
            opti.subject_to(X[:, k + 1] == model.rk4(X[:, k], U[:, k], kappa[k], ds))

        # Vehicle limits, soft track bounds, soft terminal speed
        model.add_constraints(opti, X, U)
        opti.subject_to(slack_n >= 0)
        opti.subject_to(n.T <= n_ub + slack_n)
        opti.subject_to(n.T >= n_lb - slack_n)
        opti.subject_to(slack_v >= 0)
        opti.subject_to(v[N] <= v_term_max + slack_v)

        # Steering rate: first stage w.r.t. the applied command, then |d delta/dt| with dt ≈ ds / v
        rate_dt = p.max_steer_rate * p.dt_control
        opti.subject_to(opti.bounded(-rate_dt, delta[0] - u_prev[0], rate_dt))
        for k in range(1, N):
            opti.subject_to(opti.bounded(-p.max_steer_rate * ds,
                                         (delta[k] - delta[k - 1]) * v[k],
                                         p.max_steer_rate * ds))

        opti.solver('ipopt', {'print_time': False},
                    {'print_level': 0, 'sb': 'yes', 'max_iter': 150,
                     'tol': 1e-4, 'acceptable_tol': 1e-3})

        self.opti = opti
        self.X, self.U = X, U
        self.slack_n, self.slack_v = slack_n, slack_v
        self.x0, self.ds, self.kappa = x0, ds, kappa
        self.n_ub, self.n_lb = n_ub, n_lb
        self.u_prev, self.v_term_max = u_prev, v_term_max

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def last_command(self):
        return self._u_prev.copy()

    def set_last_command(self, u):
        self._u_prev = np.asarray(u, dtype=float).copy()

    def update(self, vehicle_state, ref: Optional[TrackReference]):
        """
        Solve the MPC for the current state on the given reference.

        Returns:
            (delta, a, debug)
        """
        p, model, N, opti = self.p, self.model, self.p.N, self.opti
        lim = model.lim
        if ref is None or len(ref.s) < 3:
            return (*self.fallback(), {'status': 'no_reference'})

        s0, n0, mu0 = ref.project(vehicle_state[:2], vehicle_state[2])
        v0 = max(np.hypot(vehicle_state[3], vehicle_state[4]), lim.v_min)
        x0 = np.array([n0, mu0, v0])

        if p.ds is not None:
            ds = p.ds
        else:
            remaining = ref.s[-1] - s0
            if remaining < p.min_horizon:
                return (*self.fallback(), {'status': 'short_horizon'})
            ds = remaining / N

        s_stages = s0 + ds * np.arange(N + 1)
        stage = ref.sample(s_stages)
        kappa = ref.sample(s_stages[:-1] + ds / 2)['kappa']
        n_ub = stage['n_left'] - lim.car_half_width
        n_lb = -stage['n_right'] + lim.car_half_width
        guess = ref.sample_guess(s_stages)

        # Terminal speed
        if guess is not None:
            v_term = float(guess['v'][-1])
        else:
            v_term = float(model.corner_speed(np.max(np.abs(kappa[-max(3, N // 5):]))))
            if not ref.closed:
                v_term = min(v_term, np.sqrt(model.a_grip * p.r_min_unseen))
        v_term = float(np.clip(v_term, lim.v_min, lim.v_max))

        opti.set_value(self.x0, x0)
        opti.set_value(self.ds, ds)
        opti.set_value(self.kappa, kappa)
        opti.set_value(self.n_ub, n_ub)
        opti.set_value(self.n_lb, n_lb)
        opti.set_value(self.u_prev, self._u_prev)
        opti.set_value(self.v_term_max, v_term)

        # Initial guess: global trajectory > previous solution > midline (hot start)
        if guess is not None:
            X_init = np.vstack([guess['n'], guess['mu'], guess['v']])
            U_init = np.vstack([guess['delta'][:-1], guess['a'][:-1]])
        elif self._prev_X is not None:
            X_init, U_init = self._prev_X.copy(), self._prev_U.copy()
        else:
            X_init = np.vstack([np.zeros(N + 1), np.zeros(N + 1), np.full(N + 1, v0)])
            U_init = np.vstack([model.steady_state_steer(kappa), np.zeros(N)])
        X_init[:, 0] = x0
        X_init[2, :] = np.clip(X_init[2, :], lim.v_min, lim.v_max)
        opti.set_initial(self.X, X_init)
        opti.set_initial(self.U, U_init)
        opti.set_initial(self.slack_n, 0)
        opti.set_initial(self.slack_v, 0)

        t_start = time.perf_counter()
        try:
            sol = opti.solve()
            X_sol, U_sol, status = sol.value(self.X), sol.value(self.U), 'ok'
        except RuntimeError:
            X_sol, U_sol, status = opti.debug.value(self.X), opti.debug.value(self.U), 'failed'
        solve_time = time.perf_counter() - t_start

        if not (np.all(np.isfinite(X_sol)) and np.all(np.isfinite(U_sol))):
            self._prev_X = self._prev_U = None
            return (*self.fallback(), {'status': 'nan', 'solve_time': solve_time})

        # Keep failed solutions for the fallback plan, but don't warm-start from them
        if status == 'ok':
            self._prev_X, self._prev_U = X_sol, U_sol
        else:
            self._prev_X = self._prev_U = None
        self._fallback_idx = 0
        delta, a = float(U_sol[0, 0]), float(U_sol[1, 0])
        self._u_prev = np.array([delta, a])

        return delta, a, {
            'status': status,
            'solve_time': solve_time,
            'pred_xy': ref.to_world(s_stages, X_sol[0]),
            'pred_v': X_sol[2],
            'horizon': ds * N,
            'v_term': v_term,
        }

    def fallback(self):
        """Command when no valid observation/solution: continue previous plan, else brake gently."""
        self._fallback_idx += 1
        if self._prev_U is not None and self._fallback_idx < self._prev_U.shape[1]:
            delta, a = (float(u) for u in self._prev_U[:, self._fallback_idx])
        else:
            delta, a = float(self._u_prev[0]), -2.0
        self._u_prev = np.array([delta, a])
        return delta, a

    def acceleration_to_throttle(self, a: float, v_x: float) -> float:
        return self.model.acceleration_to_throttle(a, v_x)
