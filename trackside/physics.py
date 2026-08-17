"""Projectile physics for field-event implements.

Why shot put is the right first target: a 7.26 kg sphere at ~13 m/s carries so
much mass per unit frontal area that aerodynamic drag costs it well under 1% of
its range. The flight is very nearly pure ballistics, so when a predicted
distance misses the tape measure, the error is in the *computer vision* and not
in an unfalsifiable aerodynamic model.

Discus and javelin are a different sport, physically. They generate lift, their
flight is not projectile motion, and predicted-vs-actual there tests your
aerodynamics rather than your tracker. They are deliberately out of scope.
"""

from __future__ import annotations

import numpy as np

G = 9.80665  # standard gravity, m/s^2
RHO_AIR = 1.225  # kg/m^3 at sea level, 15 C
CD_SPHERE = 0.47

# mass (kg), diameter (m)
IMPLEMENTS = {
    "shot_mens": (7.260, 0.120),
    "shot_womens": (4.000, 0.103),
    "shot_hs_boys": (5.443, 0.115),
}


def range_no_drag(v0: float, theta_rad: float, h0: float, g: float = G) -> float:
    """Closed-form range for a projectile released at height h0 above the pad.

    This is the model we *fit*, deliberately. Its drag-free bias is a known,
    measurable quantity rather than a hidden one -- the demo reports it.
    """
    vy = v0 * np.sin(theta_rad)
    vx = v0 * np.cos(theta_rad)
    disc = vy * vy + 2.0 * g * h0
    if disc < 0 or vx <= 0:
        return float("nan")
    return float(vx * (vy + np.sqrt(disc)) / g)


def optimal_angle(v0: float, h0: float, g: float = G) -> float:
    """Range-optimal release angle (radians) for a given speed and height.

    Always below 45 degrees when releasing above ground level, which is why
    real throwers release nearer 37 than 45 and why "aim higher" is bad advice.
    """
    return float(np.arcsin(1.0 / np.sqrt(2.0 + 2.0 * g * h0 / (v0 * v0))))


class Trajectory:
    """A drag-aware trajectory, sampleable at arbitrary time."""

    def __init__(self, t: np.ndarray, x: np.ndarray, y: np.ndarray):
        self.t, self.x, self.y = t, x, y

    @property
    def flight_time(self) -> float:
        return float(self.t[-1])

    @property
    def distance(self) -> float:
        """Horizontal distance travelled from release to ground contact."""
        return float(self.x[-1])

    @property
    def apex(self) -> float:
        return float(np.max(self.y))

    def at(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (np.interp(t, self.t, self.x), np.interp(t, self.t, self.y))


def integrate(
    v0: float,
    theta_rad: float,
    h0: float,
    implement: str = "shot_mens",
    cd: float = CD_SPHERE,
    rho: float = RHO_AIR,
    dt: float = 1e-4,
    g: float = G,
) -> Trajectory:
    """Integrate flight with quadratic drag until the implement returns to y=0.

    Uses RK4 on the 4-state system. Ground contact is found by linear
    interpolation on the final step so the landing distance is not quantised to
    the integration step.
    """
    mass, diameter = IMPLEMENTS[implement]
    area = np.pi * (diameter / 2.0) ** 2
    k = 0.5 * rho * cd * area / mass  # acceleration coefficient, 1/m

    def deriv(s: np.ndarray) -> np.ndarray:
        _, _, vx, vy = s
        speed = np.hypot(vx, vy)
        return np.array([vx, vy, -k * speed * vx, -g - k * speed * vy])

    s = np.array([0.0, h0, v0 * np.cos(theta_rad), v0 * np.sin(theta_rad)])
    ts, xs, ys = [0.0], [0.0], [h0]
    t = 0.0

    # Bound the loop so a pathological input cannot spin forever.
    for _ in range(int(60.0 / dt)):
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s_next = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt

        if s_next[1] <= 0.0:
            # Interpolate the exact ground-contact point.
            frac = s[1] / (s[1] - s_next[1])
            ts.append(t - dt + frac * dt)
            xs.append(s[0] + frac * (s_next[0] - s[0]))
            ys.append(0.0)
            break

        s = s_next
        ts.append(t)
        xs.append(s[0])
        ys.append(s[1])
    else:  # pragma: no cover - only on absurd inputs
        raise RuntimeError("projectile never landed; check inputs")

    return Trajectory(np.array(ts), np.array(xs), np.array(ys))


def integrate_many(
    v0: np.ndarray,
    theta: np.ndarray,
    h0: np.ndarray,
    implement: str = "shot_mens",
    cd: float = CD_SPHERE,
    rho: float = RHO_AIR,
    dt: float = 2e-3,
    g: float = G,
    max_time: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate many flights at once. Returns (distance, apex, flight_time).

    Monte Carlo propagation needs thousands of trajectories, and stepping them
    one at a time in Python costs more than the rest of the pipeline combined.
    The dynamics are identical for every sample, so we advance the whole
    ensemble as one array and freeze each flight as it lands.
    """
    mass, diameter = IMPLEMENTS[implement]
    k = 0.5 * rho * cd * (np.pi * (diameter / 2.0) ** 2) / mass

    v0 = np.asarray(v0, dtype=float)
    x = np.zeros_like(v0)
    y = np.asarray(h0, dtype=float).copy()
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)

    n = len(v0)
    dist = np.full(n, np.nan)
    tflight = np.full(n, np.nan)
    apex = y.copy()
    active = np.ones(n, dtype=bool)
    t = 0.0

    def deriv(s: np.ndarray) -> np.ndarray:
        _, _, svx, svy = s
        speed = np.hypot(svx, svy)
        return np.stack([svx, svy, -k * speed * svx, -g - k * speed * svy])

    for _ in range(int(max_time / dt)):
        if not active.any():
            break
        s = np.stack([x, y, vx, vy])
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s_next = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt

        landed = active & (s_next[1] <= 0.0)
        if landed.any():
            # Interpolate the exact crossing so landing is not quantised to dt.
            frac = y[landed] / (y[landed] - s_next[1][landed])
            dist[landed] = x[landed] + frac * (s_next[0][landed] - x[landed])
            tflight[landed] = t - dt + frac * dt
            active = active & ~landed

        x = np.where(active, s_next[0], x)
        y = np.where(active, s_next[1], y)
        vx = np.where(active, s_next[2], vx)
        vy = np.where(active, s_next[3], vy)
        apex = np.maximum(apex, y)

    return dist, apex, tflight


def drag_penalty(v0: float, theta_rad: float, h0: float, implement: str = "shot_mens") -> float:
    """Fraction of range lost to drag. Sanity check on the 'drag is small' claim."""
    with_drag = integrate(v0, theta_rad, h0, implement).distance
    without = range_no_drag(v0, theta_rad, h0)
    return (without - with_drag) / without
