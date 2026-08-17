"""Field-event flight tracking: release parameters and predicted distance.

The pipeline finds the implement among everything else that moves, fits the
true dragged flight model to it, and reports release speed, angle and height
with error bars, plus the distance those parameters predict.

Two things here are worth more than the tracking itself:

1. The curvature of a projectile in pixels is g/(2*mpp). Because we already know
   g, the flight can identify the image scale on its own, with no reference
   object anywhere in frame.

2. That gives two independent estimates of one quantity -- the regulation circle
   cast into the concrete, and gravity. We fuse them rather than picking a
   winner, and their disagreement is itself a diagnostic: if a ruler and gravity
   cannot be reconciled, something is wrong, and the Birge ratio says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import physics
from .calib import PlaneCalibration
from .detect import Blob, detect_blobs
from .fit import DragFit, fit_drag_model
from .fuse import Fusion, fuse
from .measure import Measurement


@dataclass
class FlightResult:
    release_speed: Measurement
    release_angle: Measurement
    release_height: Measurement
    predicted_distance: Measurement
    apex_height: Measurement
    flight_time: Measurement
    recovered_g: Measurement
    gravity_scale_mm_px: Measurement
    ring_scale_mm_px: Measurement
    scale: Fusion
    fit: DragFit
    n_inliers: int
    n_candidates: int
    coverage: float
    times: np.ndarray
    xs_px: np.ndarray
    ys_px: np.ndarray

    @property
    def predicted_distance_with_drag(self) -> Measurement:
        """Kept for compatibility: the model is dragged, so this is the same."""
        return self.predicted_distance

    @property
    def scale_agreement(self) -> float:
        """Fractional disagreement between the ruler and gravity calibrations."""
        a, b = self.ring_scale_mm_px.value, self.gravity_scale_mm_px.value
        return abs(a - b) / a

    @property
    def scale_check_passes(self) -> bool:
        return self.scale.consistent

    @property
    def residual_px(self) -> float:
        return self.fit.residual_px


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
    velocity and constant downward acceleration simultaneously. A drag-free
    parabola is plenty for this stage -- drag perturbs the path by well under a
    percent, far inside the inlier tolerance.
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
        if not (0.0002 < physics.G / (2.0 * cy[0]) < 0.2):
            continue
        if cx[0] <= 0:  # implement must travel forward
            continue

        resid = np.hypot(np.polyval(cy, t) - y, np.polyval(cx, t) - x)
        mask = resid < tol_px
        if mask.sum() > best_score:
            best_mask, best_score = mask, mask.sum()

    if best_mask is None or best_score < min_inliers:
        raise ValueError(f"no ballistic arc found (best support {best_score})")
    return best_mask


def _parabola_seed(
    t: np.ndarray, x_px: np.ndarray, y_px: np.ndarray, ground_y_px: float, mpp: float
) -> tuple[float, float, float, float]:
    """Cheap closed-form starting point for the non-linear fit."""
    cy = np.polyfit(t, y_px, 2)
    cx = np.polyfit(t, x_px, 1)
    t0 = float(t[0])
    vy = -(2 * cy[0] * t0 + cy[1]) * mpp
    vx = cx[0] * mpp
    h0 = (ground_y_px - np.polyval(cy, t0)) * mpp
    return (float(np.hypot(vx, vy)), float(np.arctan2(vy, vx)),
            float(h0), float(np.polyval(cx, t0)))


def analyse_flight(
    frames: list[np.ndarray],
    calibration: PlaneCalibration,
    fps: float,
    ground_y_px: float,
    implement: str = "shot_mens",
    min_area: float = 8.0,
    max_area: float = 4000.0,
    threshold: float = 20.0,
    n_mc: int = 800,
    scale_mode: str = "fuse",
    ensemble: bool = False,
    exposure_s: float | None = None,
    rolling_shutter_s: float | None = None,
    max_per_frame: int = 12,
    seed: int = 0,
) -> FlightResult:
    """Track the implement and solve for its release state.

    `scale_mode` selects how metres-per-pixel is determined:
        "ruler"   - trust the reference object only
        "gravity" - trust the flight's own curvature only, no reference needed
        "fuse"    - inverse-variance combination of both (default)

    Fusing is the default because the two estimates are genuinely independent
    and disagreement between them is diagnostic. A ruler that cannot be
    reconciled with gravity is a ruler worth re-measuring.
    """
    if scale_mode not in {"ruler", "gravity", "fuse"}:
        raise ValueError(f"unknown scale_mode {scale_mode!r}")

    per_frame = detect_blobs(
        frames, threshold=threshold, min_area=min_area, max_area=max_area,
        ensemble=ensemble, max_per_frame=max_per_frame,
    )
    candidates = [b for blobs in per_frame for b in blobs]
    mask = _ransac_ballistic(candidates, fps, seed=seed)
    inliers = sorted((b for b, m in zip(candidates, mask) if m), key=lambda b: b.frame)

    t = np.array([b.frame / fps for b in inliers])
    x_px = np.array([b.x for b in inliers])
    y_px = np.array([b.y for b in inliers])
    span = int(inliers[-1].frame - inliers[0].frame) + 1
    coverage = len({b.frame for b in inliers}) / max(span, 1)

    if rolling_shutter_s:
        # A rolling shutter reads the sensor top to bottom, so a detection's
        # true instant depends on which row it landed on. We know that row, so
        # the correction is direct rather than iterative: shift each
        # observation's timestamp by its own readout delay.
        #
        # This one matters out of proportion to its size. Left uncorrected it
        # costs ~8 cm, and unlike a bad ruler it does *not* break the physics --
        # the fit stays clean and the gravity audit stays happy, so it is a
        # wrong answer that arrives looking entirely healthy.
        height_px = frames[0].shape[0]
        t = t + (y_px / height_px) * rolling_shutter_s

    seed_params = _parabola_seed(t, x_px, y_px, ground_y_px,
                                 calibration.meters_per_pixel)

    # A finite exposure puts the first blob's centroid half a shutter after the
    # true release instant. Same systematic as the finish timing, same fix.
    offset = 0.5 * exposure_s if exposure_s else 0.0

    # Two fits: one letting gravity set the scale, one pinned to the ruler.
    free = fit_drag_model(t, x_px, y_px, ground_y_px, seed_params,
                          implement=implement, mpp_fixed=None,
                          release_offset_s=offset)
    grav_scale = Measurement(free.mpp, max(free.sigma("mpp"), 1e-9), "m/px")
    ruler_scale = Measurement(calibration.meters_per_pixel,
                              calibration.sigma_mpp, "m/px")

    scale = fuse([ruler_scale, grav_scale], ["ruler (reference object)",
                                             "gravity (flight curvature)"])
    chosen = {"ruler": ruler_scale, "gravity": grav_scale,
              "fuse": scale.combined}[scale_mode]

    final = fit_drag_model(t, x_px, y_px, ground_y_px, seed_params,
                           implement=implement, mpp_fixed=chosen.value,
                           release_offset_s=offset)

    # Propagate the fit covariance plus the scale uncertainty through the
    # non-linear map to landing distance, apex and flight time.
    rng = np.random.default_rng(seed)
    draws = final.sample(n_mc, rng)
    mpp_draws = rng.normal(chosen.value, chosen.sigma, size=n_mc)

    # The fit ran at one fixed scale, and every world quantity is linear in
    # metres-per-pixel, so scale uncertainty folds in as a simple rescaling
    # rather than needing the fit repeated per sample.
    s = mpp_draws / chosen.value
    speeds = draws[:, 0] * s
    thetas = draws[:, 1]
    heights = draws[:, 2] * s

    good = (speeds > 0) & (heights > 0) & (thetas > 0.01) & (thetas < 1.5)
    speeds, thetas, heights = speeds[good], thetas[good], heights[good]
    if len(speeds) < 2:
        raise ValueError("fit did not produce usable samples")

    dists, apexes, times_ = physics.integrate_many(
        speeds, thetas, heights, implement, dt=2e-3
    )
    angles = np.degrees(thetas)

    def M(vals, unit: str, scale_by: float = 1.0) -> Measurement:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)] * scale_by
        if len(arr) < 2:
            raise ValueError("fit did not produce usable samples")
        return Measurement(float(arr.mean()), float(arr.std(ddof=1)), unit)

    # If we force the ruler's scale, what gravity would this flight imply?
    # Curvature goes as g/mpp, so the ratio of scales is the ratio of g. This is
    # the audit: a wrong ruler cannot hide, because it breaks physics.
    g_ratio = ruler_scale.value / grav_scale.value
    g_rel = np.hypot(ruler_scale.relative, grav_scale.relative)
    recovered_g = Measurement(physics.G * g_ratio,
                              physics.G * g_ratio * g_rel, "m/s^2")

    return FlightResult(
        release_speed=M(speeds, "m/s"),
        release_angle=M(angles, "deg"),
        release_height=M(heights, "m"),
        predicted_distance=M(dists, "m"),
        apex_height=M(apexes, "m"),
        flight_time=M(times_, "s"),
        recovered_g=recovered_g,
        gravity_scale_mm_px=Measurement(grav_scale.value * 1000,
                                        grav_scale.sigma * 1000, "mm/px"),
        ring_scale_mm_px=Measurement(ruler_scale.value * 1000,
                                     ruler_scale.sigma * 1000, "mm/px"),
        scale=scale,
        fit=final,
        n_inliers=len(inliers),
        n_candidates=len(candidates),
        coverage=coverage,
        times=t,
        xs_px=x_px,
        ys_px=y_px,
    )
