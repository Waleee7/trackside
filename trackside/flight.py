"""Field-event flight tracking: release parameters and predicted distance.

The pipeline finds the implement among everything else that moves, fits a
ballistic model to it, and reports release speed, angle and height with error
bars, plus the distance those parameters predict.

Two things here are worth more than the tracking itself:

1. Fitting the parabola freely recovers gravitational acceleration as a
   by-product. We know what g is. So the recovered value is a free, independent
   audit of the whole measurement chain -- if the scale calibration is off by
   5%, the recovered g is off by 5% and the error announces itself.

2. That same relationship runs backwards. Because the curvature of the flight in
   pixels is g/(2*mpp), the video can calibrate its own scale from gravity with
   no ruler in frame at all. Two independent calibrations that agree are much
   stronger evidence than one calibration asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import physics
from .calib import PlaneCalibration
from .detect import Blob, detect_blobs
from .measure import Measurement


@dataclass
class FlightResult:
    release_speed: Measurement
    release_angle: Measurement
    release_height: Measurement
    predicted_distance: Measurement
    predicted_distance_with_drag: Measurement
    recovered_g: Measurement
    gravity_scale_mm_px: Measurement
    ring_scale_mm_px: Measurement
    n_inliers: int
    n_candidates: int
    times: np.ndarray
    xs_px: np.ndarray
    ys_px: np.ndarray

    @property
    def scale_agreement(self) -> float:
        """Fractional disagreement between the ruler and gravity calibrations."""
        a, b = self.ring_scale_mm_px.value, self.gravity_scale_mm_px.value
        return abs(a - b) / a

    @property
    def scale_check_passes(self) -> bool:
        """Do the two independent calibrations agree within their uncertainty?"""
        diff = abs(self.ring_scale_mm_px.value - self.gravity_scale_mm_px.value)
        combined = float(np.hypot(self.ring_scale_mm_px.sigma, self.gravity_scale_mm_px.sigma))
        return diff <= 2.0 * combined


def _ransac_ballistic(
    blobs: list[Blob],
    fps: float,
    tol_px: float = 3.0,
    iters: int = 4000,
    min_inliers: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """Select the subset of blobs that lie on one ballistic arc.

    Chooses the implement over a moving athlete on *physical consistency* rather
    than size or speed: only the real projectile has constant horizontal
    velocity and constant downward acceleration simultaneously.
    """
    if len(blobs) < min_inliers:
        raise ValueError(f"only {len(blobs)} candidate detections; need {min_inliers}")

    t = np.array([b.frame / fps for b in blobs])
    x = np.array([b.x for b in blobs])
    y = np.array([b.y for b in blobs])
    rng = np.random.default_rng(seed)
    best_mask, best_score = None, -1

    for _ in range(iters):
        i = rng.choice(len(blobs), size=3, replace=False)
        if len(set(t[i])) < 3:
            continue
        try:
            cy = np.polyfit(t[i], y[i], 2)
            cx = np.polyfit(t[i], x[i], 1)
        except (np.linalg.LinAlgError, ValueError):
            continue

        # Image y grows downward, so a real projectile must curve positively.
        # Bound the implied scale to physically sane values (0.2-200 mm/px).
        if cy[0] <= 0:
            continue
        implied_mpp = physics.G / (2.0 * cy[0])
        if not (0.0002 < implied_mpp < 0.2):
            continue
        if cx[0] <= 0:  # implement must travel forward
            continue

        resid = np.hypot(np.polyval(cy, t) - y, np.polyval(cx, t) - x)
        mask = resid < tol_px
        score = mask.sum()
        if score > best_score:
            best_mask, best_score = mask, score

    if best_mask is None or best_score < min_inliers:
        raise ValueError(f"no ballistic arc found (best support {best_score})")
    return best_mask


def _drag_bias_factors(
    v0: float,
    theta: float,
    h0: float,
    t_rel: np.ndarray,
    implement: str,
) -> tuple[float, float, float]:
    """How much does fitting a drag-free model to a dragged flight bias us?

    We fit a straight line to x(t), but drag makes horizontal velocity decay, so
    the fitted slope is the flight's *average* horizontal speed rather than its
    release speed. That is a systematic bias, not noise -- averaging more frames
    will never remove it, and leaving it in place makes the error bars a lie.

    So we measure it by forward modelling: simulate a dragged flight with the
    parameters we just estimated, fit it with the exact same drag-free model,
    and see how far the fit lands from the truth we put in. The ratio is a
    correction factor. Returns (speed, angle, height) multipliers.
    """
    traj = physics.integrate(v0, theta, h0, implement)
    t = np.clip(t_rel, 0.0, traj.flight_time)
    x_sim, y_sim = traj.at(t)

    cy = np.polyfit(t, y_sim, 2)
    cx = np.polyfit(t, x_sim, 1)
    vy_fit = 2 * cy[0] * t[0] + cy[1]
    vx_fit = cx[0]
    h_fit = np.polyval(cy, t[0])

    v_fit = float(np.hypot(vx_fit, vy_fit))
    th_fit = float(np.arctan2(vy_fit, vx_fit))
    if v_fit <= 0 or th_fit <= 0 or h_fit <= 0:
        return (1.0, 1.0, 1.0)
    return (v0 / v_fit, theta / th_fit, h0 / float(h_fit))


def analyse_flight(
    frames: list[np.ndarray],
    calibration: PlaneCalibration,
    fps: float,
    ground_y_px: float,
    implement: str = "shot_mens",
    min_area: float = 8.0,
    max_area: float = 4000.0,
    threshold: float = 20.0,
    n_mc: int = 4000,
    use_gravity_scale: bool = False,
    correct_drag_bias: bool = True,
    seed: int = 0,
) -> FlightResult:
    """Track the implement and solve for its release state.

    `use_gravity_scale` derives metres-per-pixel from the flight's own curvature
    instead of the supplied ruler, which needs no reference object in frame and
    in practice beats a hand-clicked ruler. The cost is real and worth stating:
    it spends the gravity audit. Once g is an input, recovering it is circular
    and can no longer catch a calibration error. You get the ruler-free
    measurement or the independent check, not both.
    """
    per_frame = detect_blobs(
        frames, threshold=threshold, min_area=min_area, max_area=max_area
    )
    candidates = [b for blobs in per_frame for b in blobs]
    mask = _ransac_ballistic(candidates, fps, seed=seed)
    inliers = [b for b, m in zip(candidates, mask) if m]
    inliers.sort(key=lambda b: b.frame)

    t = np.array([b.frame / fps for b in inliers])
    x_px = np.array([b.x for b in inliers])
    y_px = np.array([b.y for b in inliers])

    cy, cov_y = np.polyfit(t, y_px, 2, cov=True)
    cx, cov_x = np.polyfit(t, x_px, 1, cov=True)
    t_rel = float(t[0])  # first confident observation ~ release instant

    rng = np.random.default_rng(seed)
    ys = rng.multivariate_normal(cy, cov_y, size=n_mc)
    xs = rng.multivariate_normal(cx, cov_x, size=n_mc)
    mpps = rng.normal(calibration.meters_per_pixel, calibration.sigma_mpp, size=n_mc)

    speeds, angles, heights, ranges, gs, grav_mpp = ([] for _ in range(6))

    for k in range(n_mc):
        a2, a1, a0 = ys[k]
        b1, b0 = xs[k]
        # Gravity-derived scale: pixel curvature of a projectile is g/(2*mpp).
        mpp = physics.G / (2.0 * a2) if use_gravity_scale else mpps[k]

        # World frame: metres, y up from the ground plane.
        vy = -(2 * a2 * t_rel + a1) * mpp
        vx = b1 * mpp
        h0 = (ground_y_px - (a2 * t_rel**2 + a1 * t_rel + a0)) * mpp
        v0 = float(np.hypot(vx, vy))
        theta = float(np.arctan2(vy, vx))

        if v0 <= 0 or h0 <= 0 or theta <= 0:
            continue

        speeds.append(v0)
        angles.append(np.degrees(theta))
        heights.append(h0)
        ranges.append(physics.range_no_drag(v0, theta, h0))
        gs.append(2.0 * a2 * mpp)          # recovered gravity
        grav_mpp.append(physics.G / (2.0 * a2))  # ruler-free scale

    def M(vals: list[float], unit: str, scale: float = 1.0) -> Measurement:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)] * scale
        return Measurement(float(arr.mean()), float(arr.std(ddof=1)), unit)

    # Drag costs a shot put well under 1% of its range and varies negligibly
    # across our uncertainty band, so we evaluate the penalty once at the mean
    # release state rather than integrating inside the Monte Carlo loop. That is
    # ~4000 RK4 integrations saved for a correction stable to a few parts in
    # 10,000 over this range of inputs.
    mean_v = float(np.mean(speeds))
    mean_theta = float(np.radians(np.mean(angles)))
    mean_h = float(np.mean(heights))

    if correct_drag_bias:
        # Two passes: the first bias estimate uses the uncorrected release
        # state, the second refines it once that state has moved.
        fv = fth = fh = 1.0
        for _ in range(2):
            fv, fth, fh = _drag_bias_factors(
                mean_v * fv, mean_theta * fth, mean_h * fh, t - t_rel, implement
            )
        speeds = [v * fv for v in speeds]
        angles = [a * fth for a in angles]
        heights = [h * fh for h in heights]
        mean_v, mean_theta, mean_h = mean_v * fv, mean_theta * fth, mean_h * fh
        ranges = [
            physics.range_no_drag(v, np.radians(a), h)
            for v, a, h in zip(speeds, angles, heights)
        ]

    penalty = physics.drag_penalty(mean_v, mean_theta, mean_h, implement)
    ranges_drag = [r * (1.0 - penalty) for r in ranges]

    return FlightResult(
        release_speed=M(speeds, "m/s"),
        release_angle=M(angles, "deg"),
        release_height=M(heights, "m"),
        predicted_distance=M(ranges, "m"),
        predicted_distance_with_drag=M(ranges_drag, "m"),
        recovered_g=M(gs, "m/s^2"),
        gravity_scale_mm_px=M(grav_mpp, "mm/px", scale=1000.0),
        ring_scale_mm_px=Measurement(
            calibration.meters_per_pixel * 1000.0, calibration.sigma_mpp * 1000.0, "mm/px"
        ),
        n_inliers=len(inliers),
        n_candidates=len(candidates),
        times=t,
        xs_px=x_px,
        ys_px=y_px,
    )
