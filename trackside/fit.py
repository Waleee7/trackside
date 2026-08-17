"""Direct non-linear fit of the true flight model to the observed pixels.

The first version of this library fitted a drag-free parabola and then applied
a correction for the bias that introduced. That worked, but it left structure in
the residuals and error bars roughly five times tighter than the observed
scatter -- the classic signature of a model that is slightly wrong being fitted
very precisely.

The honest fix is not a better correction, it is a better model. Here we fit the
actual dragged trajectory to the actual pixel observations and take the
covariance from the fit's own Jacobian. There is no correction factor left to
get wrong, and the uncertainty comes from the same object that produced the
estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from . import physics


@dataclass
class DragFit:
    """Fitted release state, with the covariance the fit itself implies."""

    v0: float
    theta: float          # radians
    h0: float
    x0_px: float
    mpp: float
    cov: np.ndarray       # over [v0, theta, h0, x0_px, (mpp)]
    residual_px: float    # RMS residual, pixels
    free_mpp: bool
    n_obs: int

    @property
    def names(self) -> list[str]:
        base = ["v0", "theta", "h0", "x0_px"]
        return base + ["mpp"] if self.free_mpp else base

    def sigma(self, name: str) -> float:
        i = self.names.index(name)
        return float(np.sqrt(max(self.cov[i, i], 0.0)))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw correlated parameter samples for propagating to derived values."""
        mean = [self.v0, self.theta, self.h0, self.x0_px]
        if self.free_mpp:
            mean.append(self.mpp)
        return rng.multivariate_normal(np.asarray(mean), self.cov, size=n)


def predict_pixels(
    v0: float,
    theta: float,
    h0: float,
    x0_px: float,
    mpp: float,
    t: np.ndarray,
    ground_y_px: float,
    implement: str,
    dt: float = 2e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward model: release state -> pixel positions at times t."""
    traj = physics.integrate(v0, theta, h0, implement, dt=dt)
    tc = np.clip(t, 0.0, traj.flight_time)
    x_m, y_m = traj.at(tc)
    return (x0_px + x_m / mpp, ground_y_px - y_m / mpp)


def fit_drag_model(
    t: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
    ground_y_px: float,
    seed: tuple[float, float, float, float],
    implement: str = "shot_mens",
    mpp_fixed: float | None = None,
    release_offset_s: float = 0.0,
    dt: float = 2e-3,
) -> DragFit:
    """Least-squares fit of the dragged flight to observed pixel positions.

    `seed` is (v0, theta, h0, x0_px) from the cheap parabola fit -- non-linear
    optimisation needs a starting point, and the parabola is a good one because
    drag only perturbs the flight by well under a percent.

    Passing `mpp_fixed` pins the scale to a ruler measurement. Leaving it None
    makes scale a free parameter, which the flight can identify on its own
    because the curvature is set by a gravity we already know.

    `release_offset_s` places the modelled release *before* the first
    observation. This is the same shutter problem the finish timing has: a frame
    exposed over a finite window records the implement's average position across
    that window, so the first blob's centroid sits half an exposure after the
    instant it actually left the hand. Reporting that centroid as the release
    state measures a flight already 2-3 ms old -- which quietly shortens every
    predicted distance by a couple of centimetres.
    """
    t = np.asarray(t, dtype=float)
    t0 = float(t[0])
    # Model time zero is the release instant, which precedes the first sample.
    trel = (t - t0) + release_offset_s
    free_mpp = mpp_fixed is None

    def unpack(p: np.ndarray) -> tuple[float, float, float, float, float]:
        v0, theta, h0, x0 = p[0], p[1], p[2], p[3]
        mpp = p[4] if free_mpp else mpp_fixed
        return v0, theta, h0, x0, mpp

    def resid(p: np.ndarray) -> np.ndarray:
        v0, theta, h0, x0, mpp = unpack(p)
        # Keep the optimiser inside physically meaningful territory.
        if v0 <= 0 or h0 <= 0 or mpp <= 0 or not (0.01 < theta < 1.5):
            return np.full(2 * len(t), 1e6)
        px, py = predict_pixels(v0, theta, h0, x0, mpp, trel,
                                ground_y_px, implement, dt=dt)
        return np.concatenate([px - x_px, py - y_px])

    p0 = list(seed) + ([mpp_fixed if mpp_fixed else 0.01] if free_mpp else [])
    lo = [0.1, 0.01, 0.01, -np.inf] + ([1e-5] if free_mpp else [])
    hi = [60.0, 1.5, 10.0, np.inf] + ([1.0] if free_mpp else [])

    sol = least_squares(resid, np.asarray(p0, dtype=float),
                        bounds=(lo, hi), xtol=1e-12, ftol=1e-12, max_nfev=400)

    v0, theta, h0, x0, mpp = unpack(sol.x)
    n_res, n_par = len(sol.fun), len(sol.x)
    dof = max(n_res - n_par, 1)
    s_sq = float(sol.fun @ sol.fun) / dof

    # Covariance from the Jacobian at the solution, scaled by the residual
    # variance. This is where the error bars come from: the data's own
    # disagreement with the model, not an assumption we supplied.
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * s_sq
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate geometry
        cov = np.eye(n_par) * np.inf

    return DragFit(
        v0=float(v0), theta=float(theta), h0=float(h0), x0_px=float(x0),
        mpp=float(mpp), cov=cov,
        residual_px=float(np.sqrt(np.mean(sol.fun ** 2))),
        free_mpp=free_mpp, n_obs=len(t),
    )
