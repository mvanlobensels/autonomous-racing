"""
Reference line in arc-length parameterisation with track bounds derived from
observed cones. Used both for the local (exploration) midline and for the
closed full-track midline built from the exploration-lap map.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def lateral_bounds(x, y, theta, cones, default_half_width=1.5,
                   cone_window=3.0, max_half_width=4.0):
    """
    Distance from each reference point to the nearest cone on the left and on
    the right (colour-agnostic: side is determined by the reference normal).
    Only cones within +-cone_window along the tangent are considered; where no
    cone is found the default half width is used.
    """
    n_pts = len(x)
    n_left = np.full(n_pts, default_half_width)
    n_right = np.full(n_pts, default_half_width)
    cones = np.asarray(cones, dtype=float).reshape(-1, 2)
    if len(cones) == 0:
        return n_left, n_right

    pts = np.column_stack([x, y])
    tangent = np.column_stack([np.cos(theta), np.sin(theta)])
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    rel = cones[None, :, :] - pts[:, None, :]
    lon = np.einsum('pmk,pk->pm', rel, tangent)
    lat = np.einsum('pmk,pk->pm', rel, normal)
    near = (np.abs(lon) <= cone_window) & (np.abs(lat) <= max_half_width)

    left = np.where(near & (lat > 0), lat, np.inf).min(axis=1)
    right = np.where(near & (lat < 0), -lat, np.inf).min(axis=1)
    n_left = np.where(np.isfinite(left), left, default_half_width)
    n_right = np.where(np.isfinite(right), right, default_half_width)
    return n_left, n_right


@dataclass
class TrackReference:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    cos_th: np.ndarray
    sin_th: np.ndarray
    kappa: np.ndarray
    n_left: np.ndarray
    n_right: np.ndarray
    length: float
    closed: bool
    guess: Optional[dict] = None    # optimal trajectory on its own s-grid (keys: s, n, mu, v, delta, a)

    @classmethod
    def from_path(cls, path: dict, cones, closed: bool, **bound_kwargs) -> "TrackReference":
        x = np.asarray(path['x'], dtype=float)
        y = np.asarray(path['y'], dtype=float)
        theta = np.asarray(path['theta'], dtype=float)
        kappa = np.asarray(path['curvature'], dtype=float)
        if closed and np.hypot(x[-1] - x[0], y[-1] - y[0]) < 1e-6:
            x, y, theta, kappa = x[:-1], y[:-1], theta[:-1], kappa[:-1]

        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        length = s[-1] + (np.hypot(x[0] - x[-1], y[0] - y[-1]) if closed else 0.0)
        n_left, n_right = lateral_bounds(x, y, theta, cones, **bound_kwargs)
        return cls(s, x, y, np.cos(theta), np.sin(theta), kappa,
                   n_left, n_right, float(length), closed)

    # ------------------------------------------------------------------

    def _interp(self, s_query, xp, fp):
        if self.closed:
            return np.interp(s_query, xp, fp, period=self.length)
        return np.interp(s_query, xp, fp)

    def sample(self, s_query) -> dict:
        s_query = np.atleast_1d(np.asarray(s_query, dtype=float))
        c = self._interp(s_query, self.s, self.cos_th)
        sn = self._interp(s_query, self.s, self.sin_th)
        return {
            'x': self._interp(s_query, self.s, self.x),
            'y': self._interp(s_query, self.s, self.y),
            'theta': np.arctan2(sn, c),
            'kappa': self._interp(s_query, self.s, self.kappa),
            'n_left': self._interp(s_query, self.s, self.n_left),
            'n_right': self._interp(s_query, self.s, self.n_right),
        }

    def sample_guess(self, s_query) -> Optional[dict]:
        if self.guess is None:
            return None
        s_query = np.atleast_1d(np.asarray(s_query, dtype=float))
        return {k: self._interp(s_query, self.guess['s'], self.guess[k])
                for k in ('n', 'mu', 'v', 'delta', 'a')}

    def project(self, position, heading):
        """Frenet coordinates (s, n, mu) of a pose."""
        position = np.asarray(position, dtype=float)
        i = int(np.argmin(np.hypot(self.x - position[0], self.y - position[1])))
        d = position - np.array([self.x[i], self.y[i]])
        s0 = self.s[i] + d @ np.array([self.cos_th[i], self.sin_th[i]])
        s0 = s0 % self.length if self.closed else float(np.clip(s0, 0.0, self.s[-1]))

        ref = self.sample(s0)
        d = position - np.array([ref['x'][0], ref['y'][0]])
        th = ref['theta'][0]
        n0 = -np.sin(th) * d[0] + np.cos(th) * d[1]
        mu0 = _wrap(heading - th)
        return float(s0), float(n0), float(mu0)

    def to_world(self, s_query, n):
        ref = self.sample(s_query)
        n = np.asarray(n, dtype=float)
        return np.column_stack([ref['x'] - n * np.sin(ref['theta']),
                                ref['y'] + n * np.cos(ref['theta'])])
