"""Scalar measurements that carry their own uncertainty.

The whole point of this library is that a number without an error bar is not a
measurement, it is a guess with good posture. Every quantity we report to a
user comes back as a `Measurement` so the uncertainty is impossible to drop by
accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Measurement:
    """A value, its 1-sigma standard uncertainty, and a unit."""

    value: float
    sigma: float
    unit: str = ""

    def interval(self, k: float = 1.96) -> tuple[float, float]:
        """Coverage interval. k=1.96 is ~95% for a normal error distribution."""
        return (self.value - k * self.sigma, self.value + k * self.sigma)

    @property
    def relative(self) -> float:
        """Fractional uncertainty. Infinite for a zero-valued measurement."""
        if self.value == 0:
            return float("inf")
        return abs(self.sigma / self.value)

    def format(self, places: int = 2) -> str:
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.value:.{places}f} ± {self.sigma:.{places}f}{unit}"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.format()


def monte_carlo(
    fn: Callable[..., float | Sequence[float]],
    samplers: Sequence[Callable[[np.random.Generator], float]],
    n: int = 4000,
    seed: int = 0,
) -> list[Measurement]:
    """Propagate input uncertainty through an arbitrary (possibly non-linear) fn.

    Analytic error propagation assumes local linearity. Several quantities here
    are badly non-linear -- projectile range in particular goes as roughly v^2
    and is very sensitive near low release angles -- so we sample instead of
    linearising. Returns one Measurement per output component of `fn`.
    """
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        args = [s(rng) for s in samplers]
        out = fn(*args)
        draws.append(np.atleast_1d(np.asarray(out, dtype=float)))

    stacked = np.vstack(draws)
    # Guard against a pathological sample (e.g. a physically impossible draw)
    # poisoning the statistics with NaN.
    finite = stacked[np.all(np.isfinite(stacked), axis=1)]
    if len(finite) == 0:
        raise ValueError("monte_carlo produced no finite samples")

    return [
        Measurement(float(np.mean(finite[:, i])), float(np.std(finite[:, i], ddof=1)))
        for i in range(finite.shape[1])
    ]


def gaussian(mu: float, sigma: float) -> Callable[[np.random.Generator], float]:
    """Sampler factory for a normally distributed input."""
    return lambda rng: float(rng.normal(mu, sigma))


def fixed(value: float) -> Callable[[np.random.Generator], float]:
    """Sampler factory for an input we treat as exactly known."""
    return lambda rng: float(value)
