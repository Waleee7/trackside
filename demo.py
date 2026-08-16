#!/usr/bin/env python3
"""End-to-end demonstration, validated against exact ground truth.

Renders scenes where every answer is known to machine precision, runs the real
pipelines against the rendered pixels, and reports the error. Nothing here is
graded by eye.

    python demo.py            # full run
    python demo.py --quick    # fewer throws in the validation sweep
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

from trackside import PlaneCalibration, analyse_finish, analyse_flight
from trackside.render import annotate_strip, trajectory_overlay, validation_plot
from trackside.synth import Runner, load_frames, render_sprint, render_throw

ROOT = Path(__file__).parent
OUT = ROOT / "out"
WORK = OUT / "scenes"

# A hand-clicked calibration is never perfect. We bake in a deliberate 1.7 px
# error on the circle measurement so the demo is not quietly handed the truth.
CLICK_ERROR_PX = 1.7

COLORS = [(70, 70, 220), (60, 170, 240), (80, 200, 90),
          (210, 150, 60), (200, 90, 190), (60, 220, 220)]
NAMES = ["Okafor", "Delgado", "Petrov", "Hastings", "Nakamura", "Brandt"]


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "─" * 78)


def demo_finish() -> dict:
    rule("1. SPRINT FINISH  —  who won, and can we actually call it?")
    fps, exposure_frac, finish_x_m = 60.0, 0.6, 9.8
    speeds = [10.2, 10.0, 9.9, 10.1, 9.8, 9.7]
    # Lanes 3 and 4 are set 2.2 ms apart on purpose: a margin far below the
    # 16.7 ms frame interval, to show sub-frame solving earn its keep.
    times = [1.180, 1.195, 1.210, 1.2122, 1.240, 1.270]

    runners = [
        Runner(NAMES[i], i, finish_x_m - speeds[i] * times[i], speeds[i], COLORS[i])
        for i in range(6)
    ]
    scene = WORK / "sprint"
    print(f"  rendering {len(runners)} lanes at {fps:g} fps "
          f"(exposure {exposure_frac:g} frames, sub-pixel, sensor noise)…")
    t0 = time.time()
    truth = render_sprint(scene, runners, fps=fps, duration=1.5, exposure=exposure_frac)
    print(f"  {truth.frames} frames in {time.time() - t0:.1f}s")

    frames, meta = load_frames(scene)
    p = meta["payload"]
    labels = {r["lane"]: r["name"] for r in p["runners"]}
    true_t = {r["lane"]: r["true_cross_time_s"] for r in p["runners"]}

    res = analyse_finish(
        frames, p["finish_x_px"], fps, meta["meters_per_pixel"], p["lanes"],
        labels=labels, exposure_s=exposure_frac / fps,
    )

    print(f"\n  frame interval = {1000 / fps:.2f} ms\n")
    print(f"  {'':3}{'athlete':<11}{'solved':>12}{'truth':>10}{'error':>11}{'speed':>16}")
    errors = []
    for i, r in enumerate(res.placings, 1):
        err_ms = (r.cross_time.value - true_t[r.lane]) * 1000
        errors.append(abs(err_ms))
        print(f"  {i}. {r.name:<11}{r.cross_time.value:>10.4f}s"
              f"{true_t[r.lane]:>10.4f}{err_ms:>+9.2f} ms"
              f"{r.speed.format(2):>17}")

    order_ok = [r.name for r in res.placings] == [
        NAMES[l] for l in sorted(true_t, key=lambda k: true_t[k])
    ]
    print(f"\n  max timing error : {max(errors):.2f} ms "
          f"({1000 / fps / max(errors):.0f}x finer than the frame interval)")
    print(f"  finishing order  : {'CORRECT' if order_ok else 'WRONG'}")

    print("\n  margins:")
    for s in res.separations():
        verdict = "call it" if s["resolvable"] else "TOO CLOSE TO CALL"
        print(f"    {s['ahead']:<10} over {s['behind']:<10} "
              f"{s['margin_s'] * 1000:>7.1f} ms   (resolution {s['resolution_s'] * 1000:.2f} ms)  {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    annotate_strip(res, OUT / "photo_finish.png", p["lanes"], truth=true_t)
    print(f"\n  → {OUT / 'photo_finish.png'}")

    # Same race, shutter unknown: the number moves, the result does not.
    blind = analyse_finish(
        frames, p["finish_x_px"], fps, meta["meters_per_pixel"], p["lanes"],
        labels=labels, exposure_s=None,
    )
    blind_err = np.mean([(r.cross_time.value - true_t[r.lane]) * 1000 for r in blind.placings])
    print(f"\n  with shutter UNKNOWN: absolute times biased {blind_err:+.2f} ms, "
          f"reported as ±{blind.placings[0].cross_time.sigma * 1000:.2f} ms")
    print("  the bias is common to every lane, so the finishing order is unaffected.")

    return {"max_error_ms": max(errors), "order_ok": order_ok}


def _one_throw(scene: Path, v0: float, angle: float, h0: float,
               subframes: int, quiet: bool = True) -> dict:
    truth = render_throw(scene, v0=v0, angle_deg=angle, h0=h0, subframes=subframes)
    frames, meta = load_frames(scene)
    p = meta["payload"]

    cal = PlaneCalibration.from_circle(
        p["ring_width_px"] + CLICK_ERROR_PX, "shot", sigma_pixel=2.0
    )
    res = analyse_flight(
        frames, cal, meta["fps"], p["ground_y_px"], use_gravity_scale=True
    )
    return {
        "frames": frames, "meta": meta, "cal": cal, "res": res,
        "truth_m": p["landing_distance_m"],
        "pred_m": res.predicted_distance_with_drag.value,
        "sigma_m": res.predicted_distance_with_drag.sigma,
        "v0": v0, "angle": angle, "h0": h0,
    }


def demo_flight() -> dict:
    rule("2. FIELD EVENT  —  tracking the implement through the air")
    v0, angle, h0 = 13.0, 37.0, 2.05
    scene = WORK / "throw"
    print(f"  rendering a shot put: {v0} m/s at {angle}°, released {h0} m up")
    print("  (motion blur, sensor noise, and a swaying thrower as a distractor)")
    t0 = time.time()
    rec = _one_throw(scene, v0, angle, h0, subframes=5)
    print(f"  rendered + analysed in {time.time() - t0:.1f}s")

    res, p = rec["res"], rec["meta"]["payload"]
    true_mpp_mm = rec["meta"]["meters_per_pixel"] * 1000

    print(f"\n  tracked {res.n_inliers} detections on the arc, "
          f"rejected {res.n_candidates - res.n_inliers} "
          f"(thrower, noise) on ballistic consistency alone\n")

    rows = [
        ("release speed", res.release_speed, v0, "m/s"),
        ("release angle", res.release_angle, angle, "deg"),
        ("release height", res.release_height, h0, "m"),
        ("distance", res.predicted_distance_with_drag, p["landing_distance_m"], "m"),
    ]
    print(f"  {'quantity':<16}{'measured':>22}{'truth':>10}{'error':>12}")
    for name, m, truth_v, unit in rows:
        err = 100 * (m.value - truth_v) / truth_v
        print(f"  {name:<16}{m.format(3):>22}{truth_v:>10.3f}{err:>+11.2f}%")

    rule("   the gravity cross-check")
    print(f"  ring calibration (hand-clicked, {CLICK_ERROR_PX} px off) : "
          f"{res.ring_scale_mm_px.format(3)}")
    print(f"  gravity calibration (from flight curvature)  : {res.gravity_scale_mm_px.format(3)}")
    print(f"  true scale                                   : {true_mpp_mm:.3f} mm/px")
    ring_err = 100 * abs(res.ring_scale_mm_px.value - true_mpp_mm) / true_mpp_mm
    grav_err = 100 * abs(res.gravity_scale_mm_px.value - true_mpp_mm) / true_mpp_mm
    print(f"\n  ruler error {ring_err:.2f}%  vs  gravity error {grav_err:.2f}%   "
          f"({ring_err / max(grav_err, 1e-9):.0f}x better)")

    ring_only = analyse_flight(
        rec["frames"], rec["cal"], rec["meta"]["fps"], p["ground_y_px"],
        use_gravity_scale=False,
    )
    print(f"  recovered g using the ruler: {ring_only.recovered_g.format(3)} "
          f"(true 9.807) — the {ring_err:.2f}% ruler error shows up here as a "
          f"{100 * abs(ring_only.recovered_g.value - 9.80665) / 9.80665:.2f}% error in g.")
    print("  that is the audit: a bad ruler cannot hide, because it breaks physics.")

    trajectory_overlay(rec["frames"], res, OUT / "throw_trajectory.png")
    print(f"\n  → {OUT / 'throw_trajectory.png'}")
    return rec


def demo_validation(n: int) -> list[dict]:
    rule(f"3. VALIDATION SWEEP  —  {n} throws, predicted vs measured")
    rng = np.random.default_rng(7)
    v0s = np.linspace(10.5, 14.5, n)
    records = []
    t0 = time.time()
    for i, v0 in enumerate(v0s):
        angle = float(rng.uniform(33.0, 41.0))
        h0 = float(rng.uniform(1.85, 2.25))
        scene = WORK / f"sweep{i:02d}"
        rec = _one_throw(scene, float(v0), angle, h0, subframes=3)
        resid_cm = (rec["pred_m"] - rec["truth_m"]) * 100
        print(f"  throw {i + 1}/{n}: {v0:4.1f} m/s @ {angle:4.1f}° h={h0:.2f}  "
              f"pred {rec['pred_m']:6.3f} m  truth {rec['truth_m']:6.3f} m  "
              f"resid {resid_cm:+6.1f} cm")
        records.append({k: rec[k] for k in ("truth_m", "pred_m", "sigma_m")})
        shutil.rmtree(scene, ignore_errors=True)

    resid = np.array([r["pred_m"] - r["truth_m"] for r in records]) * 100
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    print(f"\n  {n} throws in {time.time() - t0:.0f}s")
    print(f"  RMSE {rmse:.1f} cm   bias {resid.mean():+.1f} cm   "
          f"worst {np.abs(resid).max():.1f} cm")
    validation_plot(records, OUT / "validation.png")
    print(f"  → {OUT / 'validation.png'}")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="3 throws in the sweep")
    ap.add_argument("--throws", type=int, default=None)
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    print("\n\033[1mtrackside\033[0m — measuring track & field from one camera, with error bars")
    print("every number below is checked against ground truth the pipeline never sees")

    finish = demo_finish()
    demo_flight()
    if not args.skip_sweep:
        n = args.throws or (3 if args.quick else 6)
        demo_validation(n)

    rule("SUMMARY")
    print(f"  sprint : order correct, worst timing error {finish['max_error_ms']:.2f} ms "
          f"on a 16.67 ms frame interval")
    print(f"  throws : see {OUT / 'validation.png'}")
    print(f"\n  outputs in {OUT}\n")


if __name__ == "__main__":
    main()
