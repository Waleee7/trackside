"""Synthetic scenes with exact ground truth.

A demo that only shows annotated output proves nothing -- you cannot tell a
correct tracker from a confident one. So the demo renders scenes where every
answer is known to machine precision, runs the real pipeline against them, and
reports the error. The renderer is deliberately unkind: sub-pixel motion,
motion blur from a finite exposure, sensor noise, and a moving distractor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

from . import physics

SHIFT = 4  # fixed-point fractional bits -> 1/16 px drawing precision
_S = 1 << SHIFT


def _pt(x: float, y: float) -> tuple[int, int]:
    """Sub-pixel point for OpenCV's fixed-point drawing path."""
    return (int(round(x * _S)), int(round(y * _S)))


def _r(v: float) -> int:
    return int(round(v * _S))


@dataclass
class Runner:
    name: str
    lane: int
    x0_m: float
    speed_ms: float
    color: tuple[int, int, int]

    def x_at(self, t: float) -> float:
        return self.x0_m + self.speed_ms * t


@dataclass
class SceneTruth:
    """Everything the renderer knows and the analyser must not see."""

    kind: str
    fps: float
    meters_per_pixel: float
    frames: int
    payload: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))


def _finish_background(w: int, h: int, lanes: int, finish_x_px: float) -> np.ndarray:
    """A track surface: warm synthetic rubber, white lane dividers, finish line."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (58, 74, 178)  # BGR brick red
    band = h / lanes
    for i in range(lanes + 1):
        y = int(i * band)
        cv2.line(img, (0, y), (w, y), (235, 235, 235), 2, cv2.LINE_AA)
    # Finish line: a white band, as painted on a real track.
    cv2.rectangle(img, (int(finish_x_px) - 4, 0), (int(finish_x_px) + 4, h), (245, 245, 245), -1)
    return img


def _draw_runner(img: np.ndarray, r: Runner, t: float, mpp: float, band: float) -> None:
    """A schematic athlete. Vertical axis is lane (depth), not body height."""
    x_px = r.x_at(t) / mpp
    cy = (r.lane + 0.5) * band
    # Stride bob keeps the blob centroid from being a trivially clean signal.
    bob = 3.0 * np.sin(2 * np.pi * (r.speed_ms * t / 2.05))
    torso_h = band * 0.34
    torso_w = band * 0.20

    cv2.ellipse(img, _pt(x_px, cy + bob), (_r(torso_w), _r(torso_h)),
                0, 0, 360, r.color, -1, cv2.LINE_AA, SHIFT)
    # Head, slightly ahead of the torso as a runner leans.
    cv2.circle(img, _pt(x_px + torso_w * 0.35, cy + bob - torso_h * 0.95),
               _r(band * 0.10), (120, 158, 196), -1, cv2.LINE_AA, SHIFT)
    # Legs, counter-phase, to add realistic blob shape churn.
    for sign in (1, -1):
        swing = sign * 0.55 * torso_h * np.sin(2 * np.pi * (r.speed_ms * t / 2.05))
        cv2.line(img, _pt(x_px, cy + bob + torso_h * 0.6),
                 _pt(x_px + swing, cy + bob + torso_h * 1.5),
                 r.color, max(2, int(band * 0.05)), cv2.LINE_AA, SHIFT)


def render_sprint(
    out_dir: Path,
    runners: list[Runner],
    fps: float = 60.0,
    size: tuple[int, int] = (1280, 720),
    duration: float = 1.5,
    mpp: float = 0.01,
    finish_x_px: float = 980.0,
    exposure: float = 0.6,
    subframes: int = 6,
    noise_sigma: float = 2.5,
    seed: int = 0,
) -> SceneTruth:
    """Render a sprint finish. Truth = each runner's exact finish-line crossing time.

    The camera looks along the finish line, so the horizontal axis is the
    running direction and each lane sits at a different depth. Only the
    horizontal axis is metrically calibrated, which is all the timing needs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = size
    lanes = max(r.lane for r in runners) + 1
    band = h / lanes
    bg = _finish_background(w, h, lanes, finish_x_px)
    rng = np.random.default_rng(seed)
    n_frames = int(round(duration * fps))
    finish_x_m = finish_x_px * mpp

    for k in range(n_frames):
        acc = np.zeros((h, w, 3), np.float64)
        for s in range(subframes):
            t = (k + (s / subframes) * exposure) / fps
            frame = bg.copy()
            for r in runners:
                _draw_runner(frame, r, t, mpp, band)
            acc += frame
        img = acc / subframes
        img += rng.normal(0, noise_sigma, img.shape)
        cv2.imwrite(str(out_dir / f"{k:05d}.png"), np.clip(img, 0, 255).astype(np.uint8))

    truth = SceneTruth(
        kind="sprint_finish",
        fps=fps,
        meters_per_pixel=mpp,
        frames=n_frames,
        payload={
            "finish_x_px": finish_x_px,
            "finish_x_m": finish_x_m,
            "lanes": lanes,
            "runners": [
                {
                    "name": r.name,
                    "lane": r.lane,
                    "speed_ms": r.speed_ms,
                    # The number the pipeline has to recover, exactly.
                    "true_cross_time_s": (finish_x_m - r.x0_m) / r.speed_ms,
                }
                for r in runners
            ],
        },
    )
    truth.save(out_dir / "truth.json")
    return truth


def _throw_background(w: int, h: int, ground_y: float, ring_cx: float, ring_w_px: float) -> np.ndarray:
    """Side-on view of a throwing circle and sector."""
    img = np.zeros((h, w, 3), np.uint8)
    # Sky gradient, darker at top.
    for y in range(int(ground_y)):
        f = y / max(ground_y, 1)
        img[y, :] = (int(150 + 60 * f), int(120 + 60 * f), int(95 + 55 * f))
    img[int(ground_y):, :] = (86, 122, 96)  # grass sector
    cv2.line(img, (0, int(ground_y)), (w, int(ground_y)), (70, 100, 78), 3, cv2.LINE_AA)

    # The regulation circle, seen edge-on: a flattened ellipse whose horizontal
    # extent is exactly the 2.135 m diameter. This is the calibration ruler.
    cv2.ellipse(img, _pt(ring_cx, ground_y), (_r(ring_w_px / 2), _r(7)),
                0, 0, 360, (205, 205, 208), 3, cv2.LINE_AA, SHIFT)
    # Stop board at the front of the circle.
    cv2.rectangle(img, (int(ring_cx + ring_w_px / 2 - 4), int(ground_y) - 10),
                  (int(ring_cx + ring_w_px / 2 + 4), int(ground_y)), (240, 240, 245), -1)
    return img


def render_throw(
    out_dir: Path,
    v0: float = 13.0,
    angle_deg: float = 37.0,
    h0: float = 2.05,
    implement: str = "shot_mens",
    fps: float = 120.0,
    size: tuple[int, int] = (1920, 1080),
    mpp: float = 22.0 / 1920.0,
    release_x_m: float = 1.5,
    ground_y_px: float = 1000.0,
    exposure: float = 0.55,
    subframes: int = 5,
    noise_sigma: float = 2.5,
    distractor: bool = True,
    lighting_drift: float = 0.0,
    occluders: int = 0,
    implement_contrast: float = 1.0,
    rolling_shutter_s: float = 0.0,
    barrel_k1: float = 0.0,
    foul_step_over: bool = False,
    seed: int = 0,
) -> SceneTruth:
    """Render a shot put flight. Truth = release parameters and landing distance.

    `distractor` adds a swaying athlete silhouette at the circle. It is there to
    break naive "biggest moving blob" tracking -- the pipeline has to reject it
    on ballistic consistency, not on size.

    The remaining knobs exist because a demo that only ever runs on clean
    synthetic frames tells you nothing about real footage. The usual failure in
    transferring from simulation is perception mismatch rather than dynamics, so
    these reproduce the specific ways a phone camera differs from an idealised
    one:

        lighting_drift      slow exposure/cloud variation, which poisons a
                            median background plate while leaving frame
                            differencing untouched
        occluders           static clutter the implement passes behind
        implement_contrast  <1 pulls the implement toward the sky colour, as a
                            distant grey shot on an overcast day really looks
        rolling_shutter_s   sensor readout time; rows are exposed at different
                            instants, which skews fast-moving objects
        barrel_k1           radial lens distortion, which bends a true parabola
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = size
    theta = np.radians(angle_deg)
    traj = physics.integrate(v0, theta, h0, implement)
    _, diameter = physics.IMPLEMENTS[implement]

    ring_w_px = 2.135 / mpp
    ring_cx = release_x_m / mpp - ring_w_px / 2
    bg = _throw_background(w, h, ground_y_px, ring_cx, ring_w_px)
    rng = np.random.default_rng(seed)

    n_frames = int(np.ceil(traj.flight_time * fps)) + 3
    rad_px = (diameter / 2.0) / mpp

    # Static clutter the implement must pass *behind*. The bars go into the
    # background plate so they are correctly treated as scenery, and are also
    # redrawn over the implement each subframe -- otherwise the implement would
    # render on top of them and never actually be occluded, which would make
    # this condition a test of nothing.
    occluder_rects: list[tuple[int, int]] = []
    if occluders:
        for i in range(occluders):
            ox = int(w * (0.30 + 0.55 * i / max(occluders - 1, 1)))
            occluder_rects.append((ox - 14, ox + 14))
            cv2.rectangle(bg, (ox - 14, 0), (ox + 14, int(ground_y_px)),
                          (78, 96, 84), -1)

    # Precompute the lens distortion maps once; remap is cheap per frame.
    dist_maps = None
    if barrel_k1:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx_, cy_ = w / 2.0, h / 2.0
        nx, ny = (xx - cx_) / cx_, (yy - cy_) / cy_
        r2 = nx * nx + ny * ny
        f = 1.0 + barrel_k1 * r2
        dist_maps = (cx_ + nx * f * cx_, cy_ + ny * f * cy_)

    implement_color = tuple(
        int(38 + (150 - 38) * (1.0 - implement_contrast)) for _ in range(3)
    )

    for k in range(n_frames):
        acc = np.zeros((h, w, 3), np.float64)
        for s in range(subframes):
            t = (k + (s / subframes) * exposure) / fps
            frame = bg.copy()

            if distractor:
                # Thrower finishing the reverse: real motion near the circle.
                sway = 26.0 * np.sin(2 * np.pi * t * 0.85)
                if foul_step_over:
                    # Carried forward over the stop board while the implement is
                    # still in the air, which is exactly what makes it a foul.
                    sway += ring_w_px * 0.70 * min(
                        t / max(traj.flight_time * 0.45, 1e-6), 1.0
                    )
                cx = ring_cx + sway
                cv2.ellipse(frame, _pt(cx, ground_y_px - 78), (_r(20), _r(52)),
                            0, 0, 360, (54, 62, 74), -1, cv2.LINE_AA, SHIFT)
                cv2.circle(frame, _pt(cx + 4, ground_y_px - 148), _r(19),
                           (120, 158, 196), -1, cv2.LINE_AA, SHIFT)

            if t <= traj.flight_time:
                x_m, y_m = traj.at(t)
                py = ground_y_px - float(y_m) / mpp
                if rolling_shutter_s:
                    # Each row is exposed at its own instant. The implement is
                    # small, so evaluating the readout delay at its own row is
                    # an excellent approximation to a per-row render.
                    t_row = t + (py / h) * rolling_shutter_s
                    x_m, y_m = traj.at(min(t_row, traj.flight_time))
                    py = ground_y_px - float(y_m) / mpp
                px = release_x_m / mpp + float(x_m) / mpp
                cv2.circle(frame, _pt(px, py), _r(rad_px), implement_color, -1,
                           cv2.LINE_AA, SHIFT)
                # Specular highlight; a real steel shot is not a flat disc.
                cv2.circle(frame, _pt(px - rad_px * 0.3, py - rad_px * 0.3),
                           _r(max(rad_px * 0.28, 1.0)),
                           tuple(min(c + 58, 255) for c in implement_color),
                           -1, cv2.LINE_AA, SHIFT)

            # Foreground clutter goes on last, so it genuinely hides whatever
            # passes behind it.
            for ox0, ox1 in occluder_rects:
                cv2.rectangle(frame, (ox0, 0), (ox1, int(ground_y_px)),
                              (78, 96, 84), -1)

            acc += frame
        img = acc / subframes

        if lighting_drift:
            # Slow illumination change: a median background plate built over
            # the whole clip is wrong in every frame once this is present.
            gain = 1.0 + lighting_drift * np.sin(2 * np.pi * (k / n_frames) * 1.3)
            img *= gain

        if dist_maps is not None:
            img = cv2.remap(img.astype(np.float32), dist_maps[0], dist_maps[1],
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        img = img + rng.normal(0, noise_sigma, img.shape)
        cv2.imwrite(str(out_dir / f"{k:05d}.png"), np.clip(img, 0, 255).astype(np.uint8))

    truth = SceneTruth(
        kind="throw_flight",
        fps=fps,
        meters_per_pixel=mpp,
        frames=n_frames,
        payload={
            "v0_ms": v0,
            "angle_deg": angle_deg,
            "release_height_m": h0,
            "implement": implement,
            "ring_width_px": ring_w_px,
            "ring_center_x_px": ring_cx,
            "ground_y_px": ground_y_px,
            "release_x_m": release_x_m,
            "flight_time_s": traj.flight_time,
            "landing_distance_m": traj.distance,
            "apex_m": traj.apex,
            "drag_penalty_frac": physics.drag_penalty(v0, theta, h0, implement),
            "exposure_s": exposure / fps,
            "conditions": {
                "lighting_drift": lighting_drift,
                "occluders": occluders,
                "implement_contrast": implement_contrast,
                "rolling_shutter_s": rolling_shutter_s,
                "barrel_k1": barrel_k1,
                "noise_sigma": noise_sigma,
                "foul_step_over": foul_step_over,
            },
        },
    )
    truth.save(out_dir / "truth.json")
    return truth


def load_frames(path: Path) -> tuple[list[np.ndarray], dict]:
    """Read a rendered scene directory (or any video file) into memory."""
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.glob("*.png"))
        frames = [cv2.imread(str(p)) for p in files]
        truth_path = path / "truth.json"
        truth = json.loads(truth_path.read_text()) if truth_path.exists() else {}
        return frames, truth

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, {}
