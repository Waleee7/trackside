"""Combining independent estimates of the same quantity.

A recurring argument in perception work is that no single sensor gives you
everything -- vision carries semantics, radar carries range and velocity, and
the interesting engineering is the probabilistic fusion that turns several
partial views into one consistent world model.

The same argument applies inside a single camera. This library ends up with two
genuinely independent estimates of the image scale: one from a ruler cast into
the concrete, one from the gravitational curvature of the flight. Picking a
winner throws away real information. Fusing them does not, and the disagreement
between them is itself a measurement -- of whether anything is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .measure import Measurement


@dataclass
class Fusion:
    """Inverse-variance combination of independent estimates."""

    combined: Measurement
    chi2: float
    dof: int
    birge: float
    inputs: list[Measurement]
    labels: list[str]

    @property
    def consistent(self) -> bool:
        """Do the inputs agree within their stated uncertainties?

        A Birge ratio near 1 means the spread between estimates is what their
        error bars predicted. Much above 1 means at least one input is biased or
        has understated its uncertainty -- which is a finding, not a nuisance.
        """
        return self.birge <= 1.5

    def describe(self) -> str:
        lines = [f"  {l:<28} {m.format(4)}" for l, m in zip(self.labels, self.inputs)]
        lines.append(f"  {'fused':<28} {self.combined.format(4)}")
        lines.append(
            f"  chi2/dof = {self.chi2 / max(self.dof, 1):.2f}   "
            f"Birge ratio = {self.birge:.2f}   "
            f"{'consistent' if self.consistent else 'INCONSISTENT - suspect a bias'}"
        )
        return "\n".join(lines)


def fuse(
    measurements: list[Measurement],
    labels: list[str] | None = None,
    inflate_when_inconsistent: bool = True,
) -> Fusion:
    """Best linear unbiased estimate from independent measurements.

    Weights go as 1/sigma^2, so a sharp estimate dominates a vague one without
    the vague one being discarded. When the inputs disagree by more than their
    uncertainties allow, we scale the combined uncertainty by the Birge ratio
    rather than reporting a confident average of two things that cannot both be
    right. That is standard practice for reconciling discrepant measurements of
    physical constants, and the reasoning carries over unchanged.
    """
    if not measurements:
        raise ValueError("nothing to fuse")
    if any(m.sigma <= 0 for m in measurements):
        raise ValueError("every input needs a positive uncertainty to be weighted")

    labels = labels or [f"input {i + 1}" for i in range(len(measurements))]
    if len(labels) != len(measurements):
        raise ValueError("labels and measurements must be the same length")

    x = np.array([m.value for m in measurements], dtype=float)
    w = np.array([1.0 / m.sigma**2 for m in measurements], dtype=float)

    value = float((w * x).sum() / w.sum())
    sigma = float(1.0 / np.sqrt(w.sum()))

    dof = len(measurements) - 1
    chi2 = float((w * (x - value) ** 2).sum())
    birge = float(np.sqrt(chi2 / dof)) if dof > 0 else 1.0

    if inflate_when_inconsistent and birge > 1.0:
        sigma *= birge

    unit = measurements[0].unit
    return Fusion(
        combined=Measurement(value, sigma, unit),
        chi2=chi2, dof=dof, birge=birge,
        inputs=list(measurements), labels=list(labels),
    )
