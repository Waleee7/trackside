"""Turning pixels into metres, and being honest about how well we know the rate.

Field events are unusually friendly to calibrate: World Athletics fixes the
dimensions of the throwing circle and the stop board, so every legal venue on
earth ships with a known-length ruler cast into the concrete. You do not need a
checkerboard, you need to know which circle you are standing in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# World Athletics regulation dimensions (metres). These are the free rulers.
REGULATION = {
    "shot": {"circle_diameter": 2.135, "stopboard_length": 1.21},
    "hammer": {"circle_diameter": 2.135},
    "discus": {"circle_diameter": 2.500},
    "javelin": {"runway_width": 4.000},
}

# Implement diameters (metres) - a second, independent scale reference.
IMPLEMENT_DIAMETER = {
    "shot_mens": 0.120,  # 7.26 kg, spec allows 0.110-0.130
    "shot_womens": 0.103,  # 4.00 kg, spec allows 0.095-0.110
    "discus_mens": 0.220,
    "discus_womens": 0.180,
}


@dataclass(frozen=True)
class PlaneCalibration:
    """A single scale factor for one plane, with its uncertainty budget.

    Valid only for motion *in* the calibrated plane. Anything nearer to or
    further from the camera than that plane is measured with a systematic
    scale error, which is exactly the "perspective error from movement outside
    the plane of performance" the biomechanics literature warns about and then
    never quantifies. We quantify it: pass `depth_m` and `sigma_depth_m` and it
    enters the budget as a first-class term.
    """

    meters_per_pixel: float
    sigma_mpp: float
    reference_name: str = "unknown"

    @classmethod
    def from_reference(
        cls,
        pixel_length: float,
        world_length_m: float,
        sigma_pixel: float = 2.0,
        sigma_world_m: float = 0.005,
        depth_m: float | None = None,
        sigma_depth_m: float = 0.0,
        reference_name: str = "reference",
    ) -> "PlaneCalibration":
        """Derive scale from one reference of known world length.

        sigma_pixel covers how precisely you can place the two endpoints in the
        image (a couple of pixels if you are clicking by hand).
        sigma_world_m covers how well you know the real length (near zero for a
        regulation circle, larger for a tape you stretched yourself).
        """
        if pixel_length <= 0:
            raise ValueError("pixel_length must be positive")

        mpp = world_length_m / pixel_length

        # Relative uncertainties add in quadrature for a ratio.
        rel_px = sigma_pixel / pixel_length
        rel_world = sigma_world_m / world_length_m
        rel_terms = [rel_px, rel_world]

        # Out-of-plane depth error scales the apparent size by (1 + dz/Z).
        if depth_m is not None and depth_m > 0 and sigma_depth_m > 0:
            rel_terms.append(sigma_depth_m / depth_m)

        rel_total = float(np.sqrt(sum(r**2 for r in rel_terms)))
        return cls(mpp, mpp * rel_total, reference_name)

    @classmethod
    def from_circle(
        cls,
        pixel_diameter: float,
        event: str = "shot",
        **kwargs,
    ) -> "PlaneCalibration":
        """Calibrate off the regulation throwing circle for `event`."""
        if event not in REGULATION or "circle_diameter" not in REGULATION[event]:
            raise ValueError(f"no regulation circle diameter known for {event!r}")
        world = REGULATION[event]["circle_diameter"]
        kwargs.setdefault("sigma_world_m", 0.005)  # construction tolerance
        kwargs.setdefault("reference_name", f"{event} circle ({world} m)")
        return cls.from_reference(pixel_diameter, world, **kwargs)

    def to_meters(self, pixels: float | np.ndarray) -> float | np.ndarray:
        return np.asarray(pixels) * self.meters_per_pixel

    def describe(self) -> str:
        return (
            f"{self.meters_per_pixel * 1000:.3f} ± "
            f"{self.sigma_mpp * 1000:.3f} mm/px  (ref: {self.reference_name}, "
            f"{self.relative_uncertainty * 100:.2f}% scale uncertainty)"
        )

    @property
    def relative_uncertainty(self) -> float:
        return self.sigma_mpp / self.meters_per_pixel
