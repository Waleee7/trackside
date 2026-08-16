"""Classical, training-free blob detection.

No model, no weights, no dataset. A static camera plus a median background is
enough to find a moving athlete or implement, and it has two properties a
detector does not: it runs anywhere with no GPU, and it is fully reproducible
by anyone who clones the repo. Where a learned detector would help is robustness
to clutter, not accuracy -- so the interface below is what a detector would
plug into later.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Blob:
    frame: int
    x: float  # intensity-weighted centroid, pixels
    y: float
    area: float
    w: float
    h: float

    @property
    def aspect(self) -> float:
        return self.w / self.h if self.h else float("inf")


def median_background(frames: list[np.ndarray], samples: int = 48) -> np.ndarray:
    """Per-pixel median over a subsample of frames.

    The median is robust to anything that moves: a subject present in under half
    the sampled frames simply does not survive into the background plate.
    """
    if not frames:
        raise ValueError("no frames")
    idx = np.linspace(0, len(frames) - 1, min(samples, len(frames))).astype(int)
    stack = np.stack([frames[i] for i in idx]).astype(np.float32)
    return np.median(stack, axis=0)


def detect_blobs(
    frames: list[np.ndarray],
    background: np.ndarray | None = None,
    threshold: float = 22.0,
    min_area: float = 6.0,
    max_area: float = 1e9,
    open_ksize: int = 3,
) -> list[list[Blob]]:
    """Find moving regions per frame, with sub-pixel centroids.

    The centroid is weighted by difference magnitude rather than taken from the
    binary mask. That matters more than it sounds: the binary centroid is
    quantised by the thresholding, while the weighted centroid recovers roughly
    a tenth of a pixel, and the finish timing resolution is directly
    proportional to centroid precision.
    """
    bg = median_background(frames) if background is None else background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    out: list[list[Blob]] = []

    for k, frame in enumerate(frames):
        diff = np.abs(frame.astype(np.float32) - bg).max(axis=2)
        mask = (diff > threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs: list[Blob] = []
        for i in range(1, n):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if not (min_area <= area <= max_area):
                continue
            x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

            sub = labels[y0:y0 + bh, x0:x0 + bw] == i
            wsub = diff[y0:y0 + bh, x0:x0 + bw] * sub
            total = wsub.sum()
            if total <= 0:
                continue
            ys, xs = np.mgrid[0:bh, 0:bw]
            cx = x0 + float((wsub * xs).sum() / total)
            cy = y0 + float((wsub * ys).sum() / total)
            blobs.append(Blob(k, cx, cy, area, float(bw), float(bh)))

        out.append(blobs)
    return out
