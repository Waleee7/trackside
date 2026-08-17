"""Tests. The end-to-end ones assert against exact ground truth, not eyeballs."""

from __future__ import annotations

import numpy as np
import pytest

from trackside import (
    Measurement,
    PlaneCalibration,
    analyse_finish,
    analyse_flight,
    monte_carlo,
    physics,
)
from trackside.measure import gaussian
from trackside.synth import Runner, load_frames, render_sprint, render_throw


# ---------------------------------------------------------------- physics ---

def test_range_matches_integration_when_drag_is_switched_off():
    v0, theta, h0 = 13.0, np.radians(37.0), 2.0
    closed = physics.range_no_drag(v0, theta, h0)
    integrated = physics.integrate(v0, theta, h0, cd=0.0).distance
    assert closed == pytest.approx(integrated, rel=1e-4)


def test_drag_costs_a_shot_put_under_one_percent():
    """The premise the whole flight pipeline rests on."""
    penalty = physics.drag_penalty(13.0, np.radians(37.0), 2.0)
    assert 0.0 < penalty < 0.01


def test_optimal_release_angle_is_below_45_degrees():
    """Releasing above ground level always favours a flatter angle than 45."""
    for h0 in (1.5, 2.0, 2.5):
        assert physics.optimal_angle(13.0, h0) < np.radians(45.0)


def test_higher_release_lowers_the_optimal_angle():
    assert physics.optimal_angle(13.0, 2.5) < physics.optimal_angle(13.0, 1.5)


# ------------------------------------------------------------ measurement ---

def test_interval_covers_the_value():
    m = Measurement(10.0, 0.5, "m")
    lo, hi = m.interval(1.96)
    assert lo < 10.0 < hi
    assert hi - lo == pytest.approx(2 * 1.96 * 0.5)


def test_monte_carlo_recovers_a_known_product():
    """z = x*y with independent inputs -> relative sigmas add in quadrature."""
    res = monte_carlo(lambda x, y: x * y,
                      [gaussian(3.0, 0.03), gaussian(4.0, 0.08)], n=20000)
    expected_rel = np.hypot(0.03 / 3.0, 0.08 / 4.0)
    assert res[0].value == pytest.approx(12.0, rel=0.01)
    assert res[0].relative == pytest.approx(expected_rel, rel=0.06)


def test_monte_carlo_rejects_all_nan_output():
    with pytest.raises(ValueError):
        monte_carlo(lambda x: float("nan"), [gaussian(1.0, 0.1)], n=10)


# ----------------------------------------------------------- calibration ---

def test_regulation_circle_sets_the_scale():
    cal = PlaneCalibration.from_circle(200.0, "shot")
    assert cal.meters_per_pixel == pytest.approx(2.135 / 200.0)


def test_sloppier_clicking_widens_the_uncertainty():
    tight = PlaneCalibration.from_circle(200.0, "shot", sigma_pixel=1.0)
    loose = PlaneCalibration.from_circle(200.0, "shot", sigma_pixel=6.0)
    assert loose.sigma_mpp > tight.sigma_mpp


def test_out_of_plane_depth_enters_the_budget():
    """The error the literature warns about must actually cost us something."""
    flat = PlaneCalibration.from_circle(200.0, "shot", sigma_pixel=2.0)
    deep = PlaneCalibration.from_circle(200.0, "shot", sigma_pixel=2.0,
                                        depth_m=12.0, sigma_depth_m=1.0)
    assert deep.sigma_mpp > flat.sigma_mpp


def test_unknown_event_is_refused():
    with pytest.raises(ValueError):
        PlaneCalibration.from_circle(200.0, "caber_toss")


# ------------------------------------------------------------ end-to-end ---

@pytest.fixture(scope="module")
def sprint_scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("sprint")
    speeds = [10.2, 10.0, 9.9]
    times = [1.180, 1.200, 1.230]
    runners = [
        Runner(f"R{i}", i, 9.8 - speeds[i] * times[i], speeds[i], (60, 180, 240))
        for i in range(3)
    ]
    truth = render_sprint(d, runners, fps=60.0, duration=1.45, exposure=0.6)
    return d, truth


def test_finish_order_and_timing_beat_the_frame_interval(sprint_scene):
    d, _ = sprint_scene
    frames, meta = load_frames(d)
    p = meta["payload"]
    true_t = {r["lane"]: r["true_cross_time_s"] for r in p["runners"]}

    res = analyse_finish(frames, p["finish_x_px"], meta["fps"],
                         meta["meters_per_pixel"], p["lanes"],
                         exposure_s=0.6 / meta["fps"])

    assert [r.lane for r in res.placings] == sorted(true_t, key=lambda k: true_t[k])
    worst = max(abs(r.cross_time.value - true_t[r.lane]) for r in res.lanes)
    # Comfortably finer than one frame; the point of solving sub-frame at all.
    assert worst < (1.0 / meta["fps"]) / 5.0


def test_unknown_shutter_is_reported_not_hidden(sprint_scene):
    """With the shutter unknown the error bar must actually cover the bias."""
    d, _ = sprint_scene
    frames, meta = load_frames(d)
    p = meta["payload"]
    true_t = {r["lane"]: r["true_cross_time_s"] for r in p["runners"]}

    res = analyse_finish(frames, p["finish_x_px"], meta["fps"],
                         meta["meters_per_pixel"], p["lanes"], exposure_s=None)
    for r in res.lanes:
        bias = abs(r.cross_time.value - true_t[r.lane])
        assert bias <= 2.0 * r.cross_time.sigma


def test_common_mode_shutter_offset_does_not_change_the_result(sprint_scene):
    d, _ = sprint_scene
    frames, meta = load_frames(d)
    p = meta["payload"]
    known = analyse_finish(frames, p["finish_x_px"], meta["fps"],
                           meta["meters_per_pixel"], p["lanes"],
                           exposure_s=0.6 / meta["fps"])
    blind = analyse_finish(frames, p["finish_x_px"], meta["fps"],
                           meta["meters_per_pixel"], p["lanes"], exposure_s=None)
    assert [r.lane for r in known.placings] == [r.lane for r in blind.placings]


@pytest.fixture(scope="module")
def throw_scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("throw")
    truth = render_throw(d, v0=9.5, angle_deg=37.0, h0=2.0, fps=60.0,
                         size=(960, 540), mpp=14.0 / 960.0, release_x_m=0.8,
                         ground_y_px=480.0, subframes=2)
    return d, truth


def test_flight_recovers_release_state_and_distance(throw_scene):
    d, _ = throw_scene
    frames, meta = load_frames(d)
    p = meta["payload"]
    cal = PlaneCalibration.from_circle(p["ring_width_px"], "shot", sigma_pixel=2.0)
    res = analyse_flight(frames, cal, meta["fps"], p["ground_y_px"],
                         scale_mode="fuse", exposure_s=p["exposure_s"])

    assert res.release_speed.value == pytest.approx(p["v0_ms"], rel=0.02)
    assert res.release_angle.value == pytest.approx(p["angle_deg"], rel=0.02)
    assert res.predicted_distance.value == pytest.approx(
        p["landing_distance_m"], rel=0.02
    )
    # Fitting the true model should leave sub-pixel residuals.
    assert res.residual_px < 1.0


def test_ballistic_gating_rejects_the_moving_thrower(throw_scene):
    """The distractor is large and moving; only physics separates it out."""
    d, _ = throw_scene
    frames, meta = load_frames(d)
    p = meta["payload"]
    cal = PlaneCalibration.from_circle(p["ring_width_px"], "shot")
    res = analyse_flight(frames, cal, meta["fps"], p["ground_y_px"])
    assert res.n_inliers < res.n_candidates
    assert res.n_inliers >= 0.5 * meta["frames"]


def test_gravity_is_recovered_as_an_independent_audit(throw_scene):
    """A correct scale must reproduce g. This is the calibration check."""
    d, _ = throw_scene
    frames, meta = load_frames(d)
    p = meta["payload"]
    cal = PlaneCalibration.from_circle(p["ring_width_px"], "shot", sigma_pixel=2.0)
    res = analyse_flight(frames, cal, meta["fps"], p["ground_y_px"],
                         scale_mode="ruler", exposure_s=p["exposure_s"])
    assert res.recovered_g.value == pytest.approx(physics.G, rel=0.03)
    assert res.scale.consistent


def test_a_bad_ruler_shows_up_as_bad_gravity(throw_scene):
    """The audit has to fail when it should, or it is decoration."""
    d, _ = throw_scene
    frames, meta = load_frames(d)
    p = meta["payload"]
    # A 10% calibration error, as if the wrong circle diameter were assumed.
    bad = PlaneCalibration.from_circle(p["ring_width_px"] * 1.10, "shot")
    res = analyse_flight(frames, bad, meta["fps"], p["ground_y_px"],
                         scale_mode="ruler", exposure_s=p["exposure_s"])
    assert abs(res.recovered_g.value - physics.G) / physics.G > 0.05
    assert not res.scale_check_passes
    # The fusion layer must call it out, not quietly average it away.
    assert res.scale.birge > 1.5
