"""Tests for the fitting, fusion, ensemble and event layers."""

from __future__ import annotations

import numpy as np
import pytest

from trackside import Measurement, Zone, fuse, physics
from trackside.detect import coverage, detect_blobs
from trackside.events import (
    CRITICAL,
    detect_foul,
    events_from_finish,
    events_from_flight,
)
from trackside.fit import fit_drag_model, predict_pixels
from trackside.synth import load_frames, render_throw


# ------------------------------------------------------------------ fusion ---

def test_fusion_lands_between_its_inputs_and_beats_both():
    """Compatible inputs: the combination is sharper than either alone."""
    a = Measurement(10.0, 0.20, "m")
    b = Measurement(10.1, 0.10, "m")
    f = fuse([a, b], ["a", "b"])
    assert 10.0 < f.combined.value < 10.1
    # Weighted toward the sharper input.
    assert f.combined.value > 10.05
    assert f.consistent
    assert f.combined.sigma < min(a.sigma, b.sigma)


def test_discrepant_inputs_do_not_get_a_sharper_answer():
    """The failure mode fusion must never have: confident nonsense.

    Two estimates further apart than their error bars allow cannot be averaged
    into something more certain than either. Inflating by the Birge ratio is
    what stops that, so this asserts the combination is *worse* than the sharper
    input rather than better.
    """
    a = Measurement(10.0, 0.20, "m")
    b = Measurement(10.4, 0.10, "m")   # 0.4 apart, ~0.22 of combined sigma
    f = fuse([a, b], ["a", "b"])
    assert not f.consistent or f.birge > 1.0
    assert f.combined.sigma > min(a.sigma, b.sigma)


def test_agreeing_inputs_are_flagged_consistent():
    f = fuse([Measurement(10.00, 0.10), Measurement(10.05, 0.10)])
    assert f.consistent
    assert f.birge < 1.5


def test_irreconcilable_inputs_are_flagged_and_inflated():
    """Two confident estimates that cannot both be right must not average quietly."""
    tight = fuse([Measurement(10.0, 0.01), Measurement(10.0, 0.01)])
    clash = fuse([Measurement(10.0, 0.01), Measurement(11.0, 0.01)])
    assert not clash.consistent
    assert clash.birge > 1.5
    assert clash.combined.sigma > tight.combined.sigma


def test_fusion_requires_positive_uncertainties():
    with pytest.raises(ValueError):
        fuse([Measurement(1.0, 0.0), Measurement(2.0, 0.1)])


def test_fusion_rejects_mismatched_labels():
    with pytest.raises(ValueError):
        fuse([Measurement(1.0, 0.1)], ["a", "b"])


# --------------------------------------------------------------- drag fit ---

def test_drag_fit_recovers_the_release_state_it_was_generated_from():
    """Round trip through the forward model, with no imaging involved."""
    v0, theta, h0, x0, mpp = 12.4, np.radians(36.0), 2.1, 140.0, 0.0115
    ground_y = 1000.0
    t = np.linspace(0.0, 1.6, 180)
    x_px, y_px = predict_pixels(v0, theta, h0, x0, mpp, t, ground_y, "shot_mens")

    fit = fit_drag_model(t, x_px, y_px, ground_y,
                         seed=(11.0, np.radians(40.0), 1.8, 130.0),
                         mpp_fixed=mpp)
    assert fit.v0 == pytest.approx(v0, rel=1e-3)
    assert fit.theta == pytest.approx(theta, rel=1e-3)
    assert fit.h0 == pytest.approx(h0, rel=1e-2)
    assert fit.residual_px < 0.05


def test_scale_is_identifiable_from_curvature_alone():
    """No ruler in frame: gravity fixes metres-per-pixel by itself."""
    v0, theta, h0, x0, mpp = 12.4, np.radians(36.0), 2.1, 140.0, 0.0115
    ground_y = 1000.0
    t = np.linspace(0.0, 1.6, 180)
    x_px, y_px = predict_pixels(v0, theta, h0, x0, mpp, t, ground_y, "shot_mens")

    fit = fit_drag_model(t, x_px, y_px, ground_y,
                         seed=(11.0, np.radians(40.0), 1.8, 130.0),
                         mpp_fixed=None)
    assert fit.free_mpp
    assert fit.mpp == pytest.approx(mpp, rel=5e-3)


def test_noise_widens_the_error_bars():
    """Error bars must respond to data quality, not be decoration."""
    v0, theta, h0, x0, mpp = 12.4, np.radians(36.0), 2.1, 140.0, 0.0115
    ground_y = 1000.0
    t = np.linspace(0.0, 1.6, 180)
    x_px, y_px = predict_pixels(v0, theta, h0, x0, mpp, t, ground_y, "shot_mens")
    rng = np.random.default_rng(0)
    seed = (11.0, np.radians(40.0), 1.8, 130.0)

    clean = fit_drag_model(t, x_px, y_px, ground_y, seed, mpp_fixed=mpp)
    noisy = fit_drag_model(t, x_px + rng.normal(0, 1.5, len(t)),
                           y_px + rng.normal(0, 1.5, len(t)),
                           ground_y, seed, mpp_fixed=mpp)
    assert noisy.sigma("v0") > clean.sigma("v0")


# ------------------------------------------------------------------ zones ---

def test_zone_containment():
    z = Zone.from_circle_side_on(center_x_px=200.0, width_px=100.0, ground_y_px=500.0)
    assert z.contains(200.0, 500.0)
    assert not z.contains(300.0, 500.0)
    assert z.x_range == (150.0, 250.0)


# ----------------------------------------------------------------- events ---

class _FakeLane:
    def __init__(self, name, t, sigma):
        self.name, self.lane, self.n_points = name, 0, 20
        self.cross_time = Measurement(t, sigma, "s")
        self.statistical = self.cross_time
        self.speed = Measurement(10.0, 0.1, "m/s")


class _FakeFinish:
    fps = 60.0

    def __init__(self, lanes, seps):
        self.lanes, self._seps = lanes, seps

    @property
    def placings(self):
        return self.lanes

    def separations(self):
        return self._seps


def test_unresolvable_margin_raises_a_critical_event():
    lanes = [_FakeLane("A", 1.20, 0.002), _FakeLane("B", 1.2005, 0.002)]
    seps = [{"ahead": "A", "behind": "B", "margin_s": 0.0005,
             "resolution_s": 0.003, "resolvable": False}]
    events = events_from_finish(_FakeFinish(lanes, seps))
    close = [e for e in events if e.kind == "finish.too_close_to_call"]
    assert len(close) == 1
    assert close[0].severity == CRITICAL


def test_unknown_shutter_produces_a_warning():
    lanes = [_FakeLane("A", 1.20, 0.002)]
    events = events_from_finish(_FakeFinish(lanes, []), shutter_known=False)
    assert any(e.kind == "finish.shutter_unknown" for e in events)


# ------------------------------------------------------- ensemble + foul ---

@pytest.fixture(scope="module")
def hard_scene(tmp_path_factory):
    """Deliberately hostile: drifting light and clutter to hide behind."""
    d = tmp_path_factory.mktemp("hard")
    truth = render_throw(
        d, v0=10.5, angle_deg=37.0, h0=2.0, fps=60.0, size=(960, 540),
        mpp=16.0 / 960.0, release_x_m=1.0, ground_y_px=480.0, subframes=2,
        lighting_drift=0.20, occluders=3, implement_contrast=0.45,
    )
    return d, truth


def test_ensemble_never_loses_coverage(hard_scene):
    d, _ = hard_scene
    frames, _ = load_frames(d)
    kw = dict(threshold=20.0, min_area=4.0, max_area=4000.0)
    single = coverage(detect_blobs(frames, ensemble=False, **kw))
    ens = coverage(detect_blobs(frames, ensemble=True, **kw))
    assert ens >= single


@pytest.fixture(scope="module")
def foul_scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("foul")
    truth = render_throw(
        d, v0=10.5, angle_deg=37.0, h0=2.0, fps=60.0, size=(960, 540),
        mpp=16.0 / 960.0, release_x_m=1.0, ground_y_px=480.0, subframes=2,
        foul_step_over=True,
    )
    return d, truth


def test_foul_is_detected_when_the_thrower_steps_over(foul_scene):
    d, meta = foul_scene
    frames, m = load_frames(d)
    p = m["payload"]
    zone = Zone.from_circle_side_on(p["ring_center_x_px"], p["ring_width_px"],
                                    p["ground_y_px"])
    per_frame = detect_blobs(frames, threshold=20.0, min_area=4.0)
    ev = detect_foul(per_frame, zone, range(len(frames)), p["ground_y_px"])
    assert ev is not None and ev.kind == "foul.left_circle"
    assert ev.severity == CRITICAL


def test_legal_throw_is_not_flagged(throw_scene_legal):
    frames, m = throw_scene_legal
    p = m["payload"]
    zone = Zone.from_circle_side_on(p["ring_center_x_px"], p["ring_width_px"],
                                    p["ground_y_px"])
    per_frame = detect_blobs(frames, threshold=20.0, min_area=4.0)
    ev = detect_foul(per_frame, zone, range(len(frames)), p["ground_y_px"])
    assert ev is not None and ev.kind == "foul.none"


def test_rolling_shutter_correction_improves_the_fit(tmp_path_factory):
    """The systematic the gravity audit cannot see, so it must be modelled.

    Rolling shutter does not break the physics -- the fit stays plausible and
    the calibration cross-check stays happy -- which is exactly why leaving it
    uncorrected is dangerous. The residual is the only thing that notices.
    """
    from trackside import PlaneCalibration
    from trackside.flight import analyse_flight

    d = tmp_path_factory.mktemp("rs")
    readout = 1 / 60.0
    truth = render_throw(
        d, v0=10.5, angle_deg=37.0, h0=2.0, fps=60.0, size=(960, 540),
        mpp=16.0 / 960.0, release_x_m=1.0, ground_y_px=480.0, subframes=2,
        rolling_shutter_s=readout,
    )
    frames, meta = load_frames(d)
    p = meta["payload"]
    cal = PlaneCalibration.from_circle(p["ring_width_px"], "shot", sigma_pixel=2.0)

    def run(rs):
        return analyse_flight(frames, cal, meta["fps"], p["ground_y_px"],
                              min_area=4.0, exposure_s=p["exposure_s"],
                              rolling_shutter_s=rs)

    uncorrected, corrected = run(None), run(readout)
    assert corrected.residual_px < uncorrected.residual_px
    err_u = abs(uncorrected.predicted_distance.value - p["landing_distance_m"])
    err_c = abs(corrected.predicted_distance.value - p["landing_distance_m"])
    assert err_c < err_u


@pytest.fixture(scope="module")
def throw_scene_legal(tmp_path_factory):
    d = tmp_path_factory.mktemp("legal")
    render_throw(d, v0=10.5, angle_deg=37.0, h0=2.0, fps=60.0, size=(960, 540),
                 mpp=16.0 / 960.0, release_x_m=1.0, ground_y_px=480.0,
                 subframes=2, foul_step_over=False)
    return load_frames(d)
