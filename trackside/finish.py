"""Sprint finish timing from an ordinary camera.

A real photo-finish uses a line-scan sensor sampling one column of pixels
thousands of times a second. An ordinary camera cannot do that, but it does not
have to: the athlete's position is a smooth function of time, so fitting that
function and solving for the instant it crosses the finish plane recovers timing
far finer than the frame interval.

The honest part is knowing when to stop. Two athletes separated by less than the
timing resolution are a tie you cannot call, and this module says so rather than
printing a spurious ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detect import Blob, detect_blobs
from .measure import Measurement


@dataclass
class LaneResult:
    lane: int
    cross_time: Measurement       # includes the shutter systematic
    statistical: Measurement      # fit-only; the part that differs between lanes
    speed: Measurement
    n_points: int
    label: str = ""

    @property
    def name(self) -> str:
        return self.label or f"Lane {self.lane + 1}"


@dataclass
class FinishResult:
    lanes: list[LaneResult]
    fps: float
    strip: np.ndarray

    @property
    def placings(self) -> list[LaneResult]:
        return sorted(self.lanes, key=lambda r: r.cross_time.value)

    def separations(self) -> list[dict]:
        """Gap between consecutive finishers, and whether we can actually call it.

        A margin smaller than the combined uncertainty of the two times is not a
        result, it is noise. Timing systems that never say this produce orderings
        that look authoritative and are not.
        """
        out = []
        ordered = self.placings
        for a, b in zip(ordered, ordered[1:]):
            margin = b.cross_time.value - a.cross_time.value
            # Deliberately the statistical term only. The shutter offset is
            # common to every lane in the frame, so it cancels exactly in a
            # difference: it corrupts absolute times while leaving the race
            # result untouched. Including it here would make us refuse to call
            # races we can in fact call.
            combined = float(np.hypot(a.statistical.sigma, b.statistical.sigma))
            out.append({
                "ahead": a.name,
                "behind": b.name,
                "margin_s": margin,
                "resolution_s": combined,
                # 2-sigma separation before we are willing to call an order.
                "resolvable": margin > 2.0 * combined,
            })
        return out


def photo_finish_strip(frames: list[np.ndarray], finish_x_px: float, width: int = 3) -> np.ndarray:
    """Emulate a line-scan photo finish: stack the finish column over time.

    The horizontal axis of the output is *time*, not space -- which is why real
    photo-finish images look stretched and why judges read them left to right.
    """
    x = int(round(finish_x_px))
    cols = []
    for f in frames:
        col = f[:, max(0, x - width // 2): x + width // 2 + 1]
        cols.append(col.mean(axis=1).astype(np.uint8))
    return np.stack(cols, axis=1)


def _lane_of(blob: Blob, height: int, lanes: int) -> int:
    return min(int(blob.y / (height / lanes)), lanes - 1)


def _solve_crossing(
    t: np.ndarray,
    x: np.ndarray,
    finish_x_px: float,
    fps: float,
    mpp: float,
    degree: int = 1,
    n_mc: int = 3000,
    seed: int = 0,
) -> tuple[Measurement, Measurement, float]:
    """Fit x(t) near the line and solve for the crossing instant.

    Uses the covariance of the polynomial fit, so the reported uncertainty
    reflects how noisy this particular athlete's track actually was rather than
    a constant we assumed.
    """
    if len(t) < degree + 2:
        raise ValueError("not enough points to fit")

    coeffs, cov = np.polyfit(t, x, degree, cov=True)

    def cross_from(c: np.ndarray) -> float:
        roots = np.roots(np.concatenate([c[:-1], [c[-1] - finish_x_px]]))
        real = [float(r.real) for r in roots if abs(r.imag) < 1e-9]
        inside = [r for r in real if t[0] - 0.5 <= r <= t[-1] + 0.5]
        return min(inside, key=lambda r: abs(r - t.mean())) if inside else float("nan")

    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(coeffs, cov, size=n_mc)
    draws = np.array([cross_from(c) for c in samples])
    draws = draws[np.isfinite(draws)]

    t_cross = Measurement(float(np.mean(draws)), float(np.std(draws, ddof=1)), "s")

    # Speed at the line, in m/s, from the derivative of the same fit.
    dcoef = np.polyder(coeffs)
    speed_px = float(np.polyval(dcoef, t_cross.value))
    speed_draws = np.array([np.polyval(np.polyder(c), t_cross.value) for c in samples])
    speed = Measurement(speed_px * mpp, float(np.std(speed_draws, ddof=1)) * mpp, "m/s")

    # The fit covariance alone is optimistic here. It assumes independent
    # residuals, but a runner's blob centroid wobbles *with stride* -- arms and
    # legs swing periodically -- so the residuals are strongly autocorrelated
    # and the effective sample size is nearer the number of stride cycles than
    # the number of frames. Rather than guess that count, we take the
    # conservative limit (fully correlated, n_eff = 1) and treat the residual
    # scatter itself as a timing floor: a centroid uncertain by d pixels on an
    # athlete moving at v px/s is an instant uncertain by d/v seconds.
    resid_px = float(np.std(x - np.polyval(coeffs, t)))
    floor_s = resid_px / abs(speed_px) if speed_px else 0.0
    t_cross = Measurement(
        t_cross.value, float(np.hypot(t_cross.sigma, floor_s)), "s"
    )
    return t_cross, speed, resid_px


def analyse_finish(
    frames: list[np.ndarray],
    finish_x_px: float,
    fps: float,
    meters_per_pixel: float,
    lanes: int,
    window_s: float = 0.45,
    labels: dict[int, str] | None = None,
    min_area: float = 120.0,
    exposure_s: float | None = None,
    ensemble: bool = False,
) -> FinishResult:
    """Full pipeline: detect, group by lane, solve each crossing, rank.

    `exposure_s` is the shutter open time. It matters more than it looks. A
    camera integrating over a finite exposure records the athlete's average
    position across that window, so the frame's true timestamp is the middle of
    the exposure, not its start. Ignoring this biases every absolute time by
    half the exposure -- at 60 fps with a typical shutter that is around 5 ms,
    which is an eternity in a sprint.

    Pass the value if you know it and we correct for it. Pass None and we treat
    the offset as unknown within one frame interval and inflate the absolute
    uncertainty accordingly rather than quietly reporting a biased number.
    """
    height = frames[0].shape[0]
    # No ensemble here, deliberately. Ensembling buys coverage on small, fast,
    # low-contrast objects; a sprinter is none of those. Illumination
    # normalisation is on by default and does matter, because an outdoor track
    # in changing light would otherwise poison the background plate.
    per_frame = detect_blobs(frames, threshold=22.0, min_area=min_area,
                             ensemble=ensemble)

    # Largest blob per lane per frame: legs and arms can fragment a silhouette.
    tracks: dict[int, list[Blob]] = {i: [] for i in range(lanes)}
    for blobs in per_frame:
        best: dict[int, Blob] = {}
        for b in blobs:
            ln = _lane_of(b, height, lanes)
            if ln not in best or b.area > best[ln].area:
                best[ln] = b
        for ln, b in best.items():
            tracks[ln].append(b)

    results: list[LaneResult] = []
    for lane, blobs in tracks.items():
        if len(blobs) < 4:
            continue
        t = np.array([b.frame / fps for b in blobs])
        x = np.array([b.x for b in blobs])

        # Restrict the fit to frames near the line. A sprinter is close enough
        # to constant velocity over a short window, and staying local avoids
        # modelling acceleration far up the track.
        approx = np.interp(finish_x_px, x, t) if x[0] < finish_x_px < x[-1] else t.mean()
        sel = np.abs(t - approx) <= window_s
        if sel.sum() < 4:
            sel = np.ones_like(t, dtype=bool)

        try:
            t_cross, speed, _ = _solve_crossing(
                t[sel], x[sel], finish_x_px, fps, meters_per_pixel
            )
        except (ValueError, np.linalg.LinAlgError):
            continue

        if exposure_s is not None:
            # Shift the timestamp to the centre of the exposure window.
            corrected = t_cross.value + 0.5 * exposure_s
            sys_sigma = 0.0
        else:
            corrected = t_cross.value
            # Offset unknown but bounded by one frame interval. A uniform
            # distribution over [0, 1/fps] has std (1/fps)/sqrt(12).
            sys_sigma = (1.0 / fps) / np.sqrt(12.0)

        results.append(LaneResult(
            lane=lane,
            cross_time=Measurement(
                corrected, float(np.hypot(t_cross.sigma, sys_sigma)), "s"
            ),
            statistical=t_cross,
            speed=speed,
            n_points=int(sel.sum()),
            label=(labels or {}).get(lane, ""),
        ))

    return FinishResult(
        lanes=results,
        fps=fps,
        strip=photo_finish_strip(frames, finish_x_px),
    )
