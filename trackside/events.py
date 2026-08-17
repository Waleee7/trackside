"""Typed events, not dashboards.

A measurement pipeline that ends in a wall of numbers has pushed the hard part
onto the reader. The useful output of a perception system is a small set of
things that *happened* -- "the thrower's foot left the circle before the
implement landed", "these two athletes cannot be separated by this camera" --
each carrying the evidence that produced it.

Two consequences fall out of taking that seriously. Events about our own
measurement quality (a calibration that disagrees with gravity, coverage too low
to trust a track) belong in the same stream as events about the athlete, because
a consumer needs to know when not to believe us. And every event carries the
numbers behind it, so nothing here is a claim you have to take on faith.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .detect import Blob
from .finish import FinishResult
from .flight import FlightResult

INFO, WARNING, CRITICAL = "info", "warning", "critical"


@dataclass
class Event:
    kind: str
    severity: str
    message: str
    data: dict = field(default_factory=dict)

    def format(self) -> str:
        tag = {INFO: "·", WARNING: "!", CRITICAL: "✗"}.get(self.severity, "·")
        return f"[{tag}] {self.kind:<22} {self.message}"


@dataclass
class Zone:
    """A polygon on the image plane, tested against a point.

    Used here for the throwing circle, where the rule that matters is a
    containment test on the athlete's ground-contact point over time.
    """

    name: str
    polygon: np.ndarray  # (N, 2) pixel coordinates

    @classmethod
    def from_circle_side_on(
        cls, center_x_px: float, width_px: float, ground_y_px: float,
        depth_px: float = 26.0, name: str = "circle",
    ) -> "Zone":
        """The throwing circle seen edge-on: a band on the ground plane.

        Side-on, a circle projects to a horizontal extent, so legality reduces
        to whether the contact point stays within that span and behind the stop
        board -- which is the part of the rule a single camera can actually see.
        """
        x0, x1 = center_x_px - width_px / 2, center_x_px + width_px / 2
        y0, y1 = ground_y_px - depth_px, ground_y_px + depth_px
        return cls(name, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                                  dtype=np.float32))

    def contains(self, x: float, y: float) -> bool:
        import cv2
        return cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False) >= 0

    @property
    def x_range(self) -> tuple[float, float]:
        return (float(self.polygon[:, 0].min()), float(self.polygon[:, 0].max()))


def detect_foul(
    per_frame: Sequence[Sequence[Blob]],
    zone: Zone,
    implement_frames: Iterable[int],
    ground_y_px: float,
    min_area: float = 400.0,
    grace_px: float = 4.0,
) -> Event | None:
    """Did the thrower leave the circle before the implement landed?

    We track the largest blob near the ground (the athlete, not the implement)
    and test its ground-contact point against the circle zone. Only frames up to
    landing count: stepping out afterwards is legal, and a rule check that
    ignores that would flag every legal throw.
    """
    frames = list(implement_frames)
    if not frames:
        return None
    last = max(frames)
    _, x_max = zone.x_range

    for k, blobs in enumerate(per_frame):
        if k > last:
            break
        ground_blobs = [b for b in blobs
                        if b.area >= min_area and b.y + b.h / 2 > ground_y_px - 160]
        if not ground_blobs:
            continue
        athlete = max(ground_blobs, key=lambda b: b.area)
        contact_x = athlete.x + athlete.w / 2  # leading edge at the stop board
        if contact_x > x_max + grace_px:
            return Event(
                "foul.left_circle", CRITICAL,
                f"contact point crossed the stop board at frame {k}, "
                f"{contact_x - x_max:.1f} px beyond the circle, before landing",
                {"frame": k, "overshoot_px": float(contact_x - x_max),
                 "landing_frame": last},
            )
    return Event("foul.none", INFO, "thrower stayed in the circle until landing",
                 {"checked_frames": last + 1})


def events_from_finish(result: FinishResult, shutter_known: bool = True) -> list[Event]:
    """Race events: the result, and every reason to distrust it."""
    events: list[Event] = []

    for i, r in enumerate(result.placings, 1):
        events.append(Event(
            "finish.placing", INFO,
            f"{i}. {r.name} at {r.cross_time.format(4)} ({r.speed.format(2)})",
            {"place": i, "lane": r.lane, "time_s": r.cross_time.value,
             "sigma_s": r.cross_time.sigma},
        ))

    for s in result.separations():
        if not s["resolvable"]:
            events.append(Event(
                "finish.too_close_to_call", CRITICAL,
                f"{s['ahead']} and {s['behind']} differ by "
                f"{s['margin_s'] * 1000:.1f} ms, inside this camera's "
                f"{s['resolution_s'] * 1000:.1f} ms resolution - do not order them",
                dict(s),
            ))

    if not shutter_known:
        events.append(Event(
            "finish.shutter_unknown", WARNING,
            "absolute times carry an unmodelled offset of up to one frame; "
            "margins between lanes are unaffected because it is common mode",
            {"fps": result.fps},
        ))

    thin = [r for r in result.lanes if r.n_points < 8]
    for r in thin:
        events.append(Event(
            "finish.sparse_track", WARNING,
            f"{r.name} solved from only {r.n_points} frames near the line",
            {"lane": r.lane, "n_points": r.n_points},
        ))
    return events


def events_from_flight(
    result: FlightResult,
    min_coverage: float = 0.85,
    max_residual_px: float = 1.5,
) -> list[Event]:
    """Throw events, including honest doubt about our own measurement."""
    events: list[Event] = [Event(
        "flight.measured", INFO,
        f"{result.predicted_distance.format(2)} from "
        f"{result.release_speed.format(2)} at {result.release_angle.format(1)}",
        {"distance_m": result.predicted_distance.value,
         "sigma_m": result.predicted_distance.sigma,
         "speed_ms": result.release_speed.value,
         "angle_deg": result.release_angle.value,
         "height_m": result.release_height.value},
    )]

    if not result.scale.consistent:
        events.append(Event(
            "flight.calibration_conflict", CRITICAL,
            f"the reference object and gravity disagree "
            f"(Birge ratio {result.scale.birge:.2f}); re-measure the circle "
            f"before trusting this distance",
            {"birge": result.scale.birge,
             "ruler_mm_px": result.ring_scale_mm_px.value,
             "gravity_mm_px": result.gravity_scale_mm_px.value},
        ))

    if result.coverage < min_coverage:
        events.append(Event(
            "flight.sparse_track", WARNING,
            f"implement found in only {result.coverage * 100:.0f}% of frames "
            f"across its flight; occlusion or contrast is hurting detection",
            {"coverage": result.coverage, "inliers": result.n_inliers},
        ))

    if result.residual_px > max_residual_px:
        events.append(Event(
            "flight.model_misfit", WARNING,
            f"ballistic model leaves {result.residual_px:.2f} px residual; "
            f"suspect lens distortion or out-of-plane motion",
            {"residual_px": result.residual_px},
        ))

    # A release angle far from optimal is the single most actionable thing a
    # thrower can hear, and it costs nothing to compute once we have the state.
    from .physics import optimal_angle
    opt = np.degrees(optimal_angle(result.release_speed.value,
                                   result.release_height.value))
    delta = result.release_angle.value - opt
    if abs(delta) > 3.0:
        events.append(Event(
            "flight.angle_off_optimal", INFO,
            f"released at {result.release_angle.value:.1f}deg, "
            f"{abs(delta):.1f}deg {'above' if delta > 0 else 'below'} the "
            f"{opt:.1f}deg optimum for this speed and height",
            {"angle_deg": result.release_angle.value, "optimal_deg": float(opt),
             "delta_deg": float(delta)},
        ))
    return events


def summarise(events: Iterable[Event]) -> str:
    events = list(events)
    order = {CRITICAL: 0, WARNING: 1, INFO: 2}
    lines = [e.format() for e in sorted(events, key=lambda e: order.get(e.severity, 3))]
    return "\n".join(lines)
