"""Visual outputs: the things you actually look at to believe a result."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .detect import median_background
from .finish import FinishResult
from .flight import FlightResult

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate_strip(
    result: FinishResult,
    out_path: Path,
    lanes: int,
    truth: dict[int, float] | None = None,
    target_width: int = 1200,
    pad_s: float = 0.10,
) -> np.ndarray:
    """Photo-finish image with each lane's solved crossing marked.

    Horizontal axis is time, not space -- which is why the athletes look
    smeared and why the image is read left to right. The white line is the
    sub-frame instant we solved for, and it deliberately falls *between*
    columns: that gap is the resolution the frame rate alone could not give you.
    """
    strip = result.strip
    h, n_frames = strip.shape[:2]

    # Crop to the window that actually contains the race.
    times = [r.cross_time.value for r in result.lanes]
    f0 = max(0, int((min(times) - pad_s) * result.fps))
    f1 = min(n_frames, int((max(times) + pad_s) * result.fps) + 1)
    cropped = strip[:, f0:f1]

    scale = max(4, target_width // max(f1 - f0, 1))
    img = cv2.resize(cropped, ((f1 - f0) * scale, h), interpolation=cv2.INTER_NEAREST)
    band = h / lanes

    def to_x(t: float) -> float:
        return (t * result.fps - f0) * scale

    for r in result.lanes:
        x = to_x(r.cross_time.value)
        y0, y1 = int(r.lane * band), int((r.lane + 1) * band)

        # 2-sigma statistical band around the solved instant.
        dx = max(2.0 * r.statistical.sigma * result.fps * scale, 1.0)
        overlay = img.copy()
        cv2.rectangle(overlay, (int(x - dx), y0), (int(x + dx), y1), (0, 220, 255), -1)
        cv2.addWeighted(overlay, 0.40, img, 0.60, 0, img)
        cv2.line(img, (int(x), y0), (int(x), y1), (255, 255, 255), 2, cv2.LINE_AA)

        if truth and r.lane in truth:
            cv2.line(img, (int(to_x(truth[r.lane])), y0),
                     (int(to_x(truth[r.lane])), y1), (90, 255, 90), 1, cv2.LINE_AA)

        label = f"{r.name}  {r.cross_time.value:.4f}s"
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.52, 1)
        # Flip the label to the left of the line when it would run off the edge.
        lx = int(x) + 10 if x + tw + 16 < img.shape[1] else int(x) - tw - 10
        ly = y0 + int(band * 0.58)
        box = img.copy()
        cv2.rectangle(box, (lx - 6, ly - th - 6), (lx + tw + 6, ly + 8),
                      (18, 18, 20), -1)
        cv2.addWeighted(box, 0.55, img, 0.45, 0, img)
        cv2.putText(img, label, (lx, ly), _FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    _text_panel(
        img,
        ["PHOTO FINISH   horizontal axis = TIME",
         "white = solved   green = truth   band = 2 sigma"],
        origin=(12, 16), scale=0.5, line_h=24,
    )
    cv2.imwrite(str(out_path), img)
    return img


def trajectory_overlay(
    frames: list[np.ndarray],
    result: FlightResult,
    out_path: Path,
    threshold: float = 20.0,
) -> np.ndarray:
    """Strobe composite of the whole flight with the fitted arc drawn over it.

    Every position the implement occupied, in one frame, plus the ballistic
    model we fitted. If the curve does not sit on the streak, you can see it.
    """
    bg = median_background(frames)
    comp = bg.copy()
    for f in frames:
        diff = np.abs(f.astype(np.float32) - bg).max(axis=2)
        mask = diff > threshold
        comp[mask] = f[mask]
    comp = np.clip(comp, 0, 255).astype(np.uint8)

    # Fitted model, re-evaluated densely across the tracked interval.
    cy = np.polyfit(result.times, result.ys_px, 2)
    cx = np.polyfit(result.times, result.xs_px, 1)
    ts = np.linspace(result.times[0], result.times[-1], 400)
    pts = np.stack([np.polyval(cx, ts), np.polyval(cy, ts)], axis=1).astype(np.int32)
    cv2.polylines(comp, [pts], False, (60, 240, 255), 2, cv2.LINE_AA)

    for x, y in zip(result.xs_px, result.ys_px):
        cv2.circle(comp, (int(round(x)), int(round(y))), 7, (0, 140, 255), 1, cv2.LINE_AA)

    lines = [
        f"release {result.release_speed.format(3)} @ {result.release_angle.format(2)}",
        f"height {result.release_height.format(3)}",
        f"predicted distance {result.predicted_distance_with_drag.format(3)}",
        f"recovered g {result.recovered_g.format(4)}",
        f"{result.n_inliers} of {result.n_candidates} detections on the arc",
    ]
    _text_panel(comp, lines, origin=(20, 22), scale=0.72)
    cv2.imwrite(str(out_path), comp)
    return comp


def _text_panel(
    img: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (20, 22),
    scale: float = 0.7,
    pad: int = 12,
    line_h: int = 30,
) -> None:
    """Draw text over a translucent dark panel.

    Not an outline. OpenCV widens glyphs as stroke thickness grows, so a thick
    dark pass under a thin light pass renders a *longer* string underneath and
    its tail shows past the end of every line.
    """
    x, y = origin
    widths = [cv2.getTextSize(s, _FONT, scale, 1)[0][0] for s in lines]
    w = max(widths) + 2 * pad
    h = line_h * len(lines) + pad

    panel = img.copy()
    cv2.rectangle(panel, (x - pad, y - pad), (x + w, y + h), (18, 18, 20), -1)
    cv2.addWeighted(panel, 0.55, img, 0.45, 0, img)

    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + line_h * (i + 1) - 8), _FONT, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)


def validation_plot(records: list[dict], out_path: Path) -> None:
    """Predicted vs tape-measured distance across many throws, with residuals.

    This is the plot that decides whether the repo is worth anything. Everything
    else is a demo; this is evidence.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth = np.array([r["truth_m"] for r in records])
    pred = np.array([r["pred_m"] for r in records])
    err = np.array([r["sigma_m"] for r in records])
    resid = (pred - truth) * 100.0  # centimetres

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 8.2), gridspec_kw={"height_ratios": [3, 1.5]}
    )
    lo, hi = truth.min() - 1.0, truth.max() + 1.0
    ax1.plot([lo, hi], [lo, hi], "--", color="#888", lw=1, label="perfect agreement")
    ax1.errorbar(truth, pred, yerr=err, fmt="o", color="#1f6feb", ecolor="#1f6feb",
                 elinewidth=1.4, capsize=3, ms=6, label="trackside estimate")
    ax1.set_xlabel("tape-measured distance (m)")
    ax1.set_ylabel("predicted distance (m)")
    ax1.set_title("Predicted vs measured shot put distance")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25)

    rmse = float(np.sqrt(np.mean(resid ** 2)))
    bias = float(np.mean(resid))
    ax1.text(0.03, 0.95,
             f"n = {len(records)}\nRMSE = {rmse:.1f} cm\nbias = {bias:+.1f} cm",
             transform=ax1.transAxes, va="top", fontsize=10,
             bbox=dict(fc="white", ec="#ccc"))

    ax2.axhline(0, color="#888", ls="--", lw=1)
    ax2.errorbar(truth, resid, yerr=err * 100.0, fmt="o", color="#d1242f",
                 ecolor="#d1242f", elinewidth=1.2, capsize=3, ms=5)
    ax2.set_xlabel("tape-measured distance (m)")
    ax2.set_ylabel("residual (cm)")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
