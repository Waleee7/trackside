"""Classical, training-free blob detection with a cheap-detector ensemble.

No model, no weights, no dataset. A static camera plus a median background is
enough to find a moving athlete or implement, and it has two properties a
learned detector does not: it runs anywhere with no GPU, and it is fully
reproducible by anyone who clones the repo.

The ensemble here is off by default, and the reason is worth recording.

It was built on a genuinely good argument: robustness on hard small-object
tracking often comes from merging several cheap detectors rather than betting
everything on one better one. Measured on `robustness.py`, it did not transfer.
Mean coverage change was *minus* 23 points, and under heavy sensor noise it took
the implement from fully tracked to no track at all.

The reason it fails here is instructive. That argument was made about ensembling
*learned* detectors, which fail in uncorrelated ways because they learned
different things. These three members are variants of one threshold operation on
one difference image, so they share their failure modes almost exactly: when
noise pushes one over threshold it pushes all of them, and the union is then
dominated by false positives that survive the merge and crowd out the real
implement. An ensemble only buys you something when its members are wrong
independently.

It stays in the codebase, switchable, because it does help slightly on
low-contrast frames and because the negative result is worth keeping honest. Do
not turn it on without measuring.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

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
    votes: int = 1  # how many ensemble members found it

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


def _blobs_from_mask(
    mask: np.ndarray, weight: np.ndarray, k: int, min_area: float,
    max_area: float, cap: int = 400,
) -> list[Blob]:
    """Connected components -> sub-pixel, intensity-weighted centroids.

    The centroid is weighted by difference magnitude rather than taken from the
    binary mask. That matters more than it sounds: the binary centroid is
    quantised by the thresholding, while the weighted centroid recovers roughly
    a tenth of a pixel, and finish timing resolution is directly proportional to
    centroid precision.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return []

    # Select components with array operations before touching Python. On a noisy
    # frame the mask can contain thousands of speckles, and a per-component loop
    # over all of them costs far more than everything else in the pipeline.
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    keep = np.nonzero((areas >= min_area) & (areas <= max_area))[0]

    if len(keep) > cap:
        # Still too many: rank by how close each is to a filled circle, which is
        # what we are looking for, and keep the best. Vectorised on the stats
        # table, so the Python loop below stays bounded.
        w_ = stats[1:, cv2.CC_STAT_WIDTH][keep].astype(np.float64)
        h_ = stats[1:, cv2.CC_STAT_HEIGHT][keep].astype(np.float64)
        fill = areas[keep] / np.maximum(w_ * h_, 1.0)
        keep = keep[np.argsort(np.abs(fill - np.pi / 4))[:cap]]

    out: list[Blob] = []
    for j in keep:
        i = int(j) + 1
        area = float(areas[j])
        x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

        sub = labels[y0:y0 + bh, x0:x0 + bw] == i
        wsub = weight[y0:y0 + bh, x0:x0 + bw] * sub
        total = wsub.sum()
        if total <= 0:
            continue
        ys, xs = np.mgrid[0:bh, 0:bw]
        out.append(Blob(
            k,
            x0 + float((wsub * xs).sum() / total),
            y0 + float((wsub * ys).sum() / total),
            area, float(bw), float(bh),
        ))
    return out


def _merge(blobs: list[Blob], radius: float = 6.0) -> list[Blob]:
    """Cluster near-coincident detections from different ensemble members.

    Greedy, by descending area. Agreement is recorded in `votes` so a caller can
    prefer corroborated detections without discarding lone ones -- for a small
    fast implement, the frames where only one member fired are exactly the
    frames worth keeping.
    """
    kept: list[Blob] = []
    for b in sorted(blobs, key=lambda z: -z.area):
        for i, k in enumerate(kept):
            if np.hypot(b.x - k.x, b.y - k.y) <= radius:
                n = k.votes + 1
                # Area-weighted running mean keeps the sub-pixel precision.
                wk, wb = k.area, b.area
                kept[i] = replace(
                    k,
                    x=(k.x * wk + b.x * wb) / (wk + wb),
                    y=(k.y * wk + b.y * wb) / (wk + wb),
                    votes=n,
                )
                break
        else:
            kept.append(b)
    return kept


def detect_blobs(
    frames: list[np.ndarray],
    background: np.ndarray | None = None,
    threshold: float = 22.0,
    min_area: float = 6.0,
    max_area: float = 1e9,
    open_ksize: int = 3,
    ensemble: bool = False,
    merge_radius: float = 6.0,
    normalise_illumination: bool = True,
    max_per_frame: int | None = None,
) -> list[list[Blob]]:
    """Find moving regions per frame, with sub-pixel centroids.

    With `ensemble=True`, three cheap detectors vote:
      1. median-background subtraction at the requested threshold
      2. the same, more sensitive and with lighter morphology
      3. three-frame temporal differencing, which does not depend on the
         background plate at all and so survives lighting drift

    `normalise_illumination` rescales each frame to the background plate's
    overall brightness before differencing. Without it, a cloud crossing the sun
    makes *every* pixel differ from a median plate built over the whole clip,
    which does not merely add noise -- it buries the one real moving object in
    tens of thousands of spurious blobs.

    `max_per_frame` keeps only the most compact candidates. We are looking for a
    round implement, so filling ratio is a cheap and honest prior, and capping
    the count bounds the cost of the RANSAC search downstream.
    """
    bg = median_background(frames) if background is None else background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grey = [f.astype(np.float32).mean(axis=2) for f in frames] if ensemble else None
    bg_level = float(np.median(bg)) if normalise_illumination else 0.0
    out: list[list[Blob]] = []

    for k, frame in enumerate(frames):
        f32 = frame.astype(np.float32)
        if normalise_illumination:
            level = float(np.median(f32))
            if level > 1e-6:
                f32 = f32 * (bg_level / level)
        diff = np.abs(f32 - bg).max(axis=2)

        mask = cv2.morphologyEx((diff > threshold).astype(np.uint8),
                                cv2.MORPH_OPEN, kernel)
        found = _blobs_from_mask(mask, diff, k, min_area, max_area)

        if ensemble:
            # Member 2: more sensitive, lighter cleanup.
            m2 = cv2.morphologyEx((diff > threshold * 0.6).astype(np.uint8),
                                  cv2.MORPH_OPEN, small)
            found += _blobs_from_mask(m2, diff, k, min_area, max_area)

            # Member 3: three-frame differencing. Taking the minimum of the two
            # neighbour differences suppresses the ghost the object leaves at
            # its previous and next positions, so what survives is where it is
            # *now*. Independent of the background plate entirely.
            if 0 < k < len(frames) - 1:
                d_prev = np.abs(grey[k] - grey[k - 1])
                d_next = np.abs(grey[k] - grey[k + 1])
                tdiff = np.minimum(d_prev, d_next)
                m3 = cv2.morphologyEx((tdiff > threshold * 0.5).astype(np.uint8),
                                      cv2.MORPH_OPEN, small)
                found += _blobs_from_mask(m3, tdiff, k, min_area, max_area)

            found = _merge(found, merge_radius)

        if max_per_frame is not None and len(found) > max_per_frame:
            # A circle fills pi/4 of its bounding box. Rank by how close each
            # blob comes to that, breaking ties toward corroborated detections.
            def compactness(b: Blob) -> tuple[float, int]:
                box = max(b.w * b.h, 1.0)
                return (abs(b.area / box - np.pi / 4), -b.votes)
            found = sorted(found, key=compactness)[:max_per_frame]

        out.append(found)
    return out


def coverage(per_frame: list[list[Blob]]) -> float:
    """Fraction of frames in which *anything at all* was detected.

    Read the caveat before using this as a quality metric. It says nothing about
    whether the object you care about was found -- in a throw scene the athlete
    is visible in every frame, so this reads 100% even when the implement is
    completely lost. For the number that actually matters, see
    `FlightResult.coverage`, which counts frames on the reconstructed ballistic
    arc and so is specific to the implement.
    """
    if not per_frame:
        return 0.0
    return sum(1 for b in per_frame if b) / len(per_frame)
