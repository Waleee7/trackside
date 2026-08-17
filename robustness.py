#!/usr/bin/env python3
"""How far does this degrade when the camera stops being ideal?

The usual reason a system trained or tuned in simulation fails on real hardware
is perception mismatch, not dynamics -- the physics transfers fine, the pixels
do not. A validation suite that only ever runs on clean synthetic frames cannot
detect that, and will report excellent numbers right up until someone points a
real phone at a real throw.

So this sweeps the specific ways a phone camera differs from an idealised one,
one at a time and then all together, and reports what each costs. Every
condition is a known failure mode of some part of the pipeline, chosen to break
something specific rather than to be survived.

    python robustness.py            # full sweep
    python robustness.py --quick    # skip the combined worst case
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

from trackside import PlaneCalibration
from trackside.flight import analyse_flight
from trackside.synth import load_frames, render_throw

ROOT = Path(__file__).parent
OUT = ROOT / "out"
WORK = OUT / "scenes" / "robust"

# A smaller frame than the main demo, deliberately: at this scale the implement
# is about 4 px across, which is the regime where small-object detection
# actually gets hard.
BASE = dict(
    v0=11.5, angle_deg=37.0, h0=2.05, fps=120.0,
    size=(1280, 720), mpp=18.0 / 1280.0, release_x_m=1.2,
    ground_y_px=660.0, subframes=3,
)

CONDITIONS: list[tuple[str, dict, str]] = [
    ("clean", {}, "idealised baseline"),
    ("lighting drift", {"lighting_drift": 0.18},
     "poisons the median background plate"),
    ("occluders", {"occluders": 3},
     "implement disappears behind clutter"),
    ("low contrast", {"implement_contrast": 0.35},
     "grey shot against overcast sky"),
    ("heavy noise", {"noise_sigma": 9.0},
     "high ISO / small sensor"),
    ("rolling shutter", {"rolling_shutter_s": 1 / 120.0},
     "rows exposed at different instants"),
    ("rolling, corrected", {"rolling_shutter_s": 1 / 120.0},
     "same clip, readout time supplied to the fit"),
    ("lens distortion", {"barrel_k1": 0.08},
     "bends a true parabola"),
]

COMBINED = ("all combined", {
    "lighting_drift": 0.15, "occluders": 2, "implement_contrast": 0.45,
    "noise_sigma": 6.0, "rolling_shutter_s": 1 / 200.0, "barrel_k1": 0.04,
}, "a realistic bad phone clip")


def run_condition(name: str, kw: dict, note: str, idx: int) -> dict:
    scene = WORK / f"c{idx:02d}"
    truth = render_throw(scene, seed=idx, **{**BASE, **kw})
    frames, meta = load_frames(scene)
    p = meta["payload"]

    cal = PlaneCalibration.from_circle(p["ring_width_px"] + 1.7, "shot",
                                       sigma_pixel=2.0)
    row = {"name": name, "note": note, "truth_m": p["landing_distance_m"]}

    # Only the "corrected" row is told the readout time, so the pair of rolling
    # shutter rows is a like-for-like comparison on identical footage.
    rs = kw.get("rolling_shutter_s") if name.endswith("corrected") else None

    def attempt(use_ensemble: bool):
        return analyse_flight(
            frames, cal, meta["fps"], p["ground_y_px"],
            min_area=4.0, scale_mode="fuse", ensemble=use_ensemble,
            exposure_s=p["exposure_s"], rolling_shutter_s=rs,
        )

    # The single detector is the shipping default; the ensemble is measured
    # alongside it precisely because it turned out not to help.
    try:
        res = attempt(False)
        row.update({
            "coverage_single": res.coverage,
            "pred_m": res.predicted_distance.value,
            "sigma_m": res.predicted_distance.sigma,
            "err_cm": (res.predicted_distance.value - p["landing_distance_m"]) * 100,
            "resid_px": res.residual_px,
            "inliers": res.n_inliers,
            "birge": res.scale.birge,
            # Does our own audit notice that this answer is wrong?
            "flagged": not res.scale.consistent,
            "failed": False,
        })
    except (ValueError, np.linalg.LinAlgError) as exc:
        row.update({"coverage_single": 0.0, "failed": True, "error": str(exc)})

    try:
        row["coverage_ensemble"] = attempt(True).coverage
    except (ValueError, np.linalg.LinAlgError):
        row["coverage_ensemble"] = 0.0

    shutil.rmtree(scene, ignore_errors=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    conditions = list(CONDITIONS) + ([] if args.quick else [COMBINED])

    print("\n\033[1mtrackside robustness sweep\033[0m")
    print("what each way a real camera differs from an ideal one actually costs\n")
    print(f"  {'condition':<17}{'coverage':>19}{'distance err':>15}"
          f"{'resid':>8}{'audit':>13}")
    print(f"  {'':<17}{'single -> ensemble':>19}{'':>15}{'(px)':>8}{'':>13}")
    print("  " + "─" * 72)

    rows = []
    t0 = time.time()
    for i, (name, kw, note) in enumerate(conditions):
        row = run_condition(name, kw, note, i)
        rows.append(row)
        cov = f"{row['coverage_single'] * 100:5.1f}% -> {row['coverage_ensemble'] * 100:5.1f}%"
        if row["failed"]:
            print(f"  {name:<17}{cov:>19}{'NO TRACK':>15}{'-':>8}{'-':>13}")
        else:
            # An error we can detect is survivable; one we cannot is the
            # dangerous kind, so the audit column matters more than the error.
            audit = "FLAGGED" if row["flagged"] else f"ok ({row['birge']:.1f})"
            bad = abs(row["err_cm"]) > 5.0
            mark = "\033[1m" if bad and not row["flagged"] else ""
            reset = "\033[0m" if mark else ""
            print(f"  {name:<17}{cov:>19}{mark}{row['err_cm']:>+13.1f} cm{reset}"
                  f"{row['resid_px']:>8.2f}{audit:>13}")
        print(f"  {'':<17}\033[2m{note}\033[0m")

    print(f"\n  swept {len(rows)} conditions in {time.time() - t0:.0f}s\n")

    ok = [r for r in rows if not r.get("failed", True)]
    gains = [(r["coverage_ensemble"] - r["coverage_single"]) * 100 for r in rows]
    worst = min(rows, key=lambda r: r["coverage_single"])

    print("\033[1mwhat this tells us\033[0m")
    print("─" * 62)
    print(f"  ensemble coverage gain : mean {np.mean(gains):+.1f} pts, "
          f"best {max(gains):+.1f} pts")
    print(f"  hardest for one detector: '{worst['name']}' at "
          f"{worst['coverage_single'] * 100:.1f}% -> ensemble "
          f"{worst['coverage_ensemble'] * 100:.1f}%")
    if ok:
        errs = np.abs([r["err_cm"] for r in ok])
        print(f"  distance error         : median {np.median(errs):.1f} cm, "
              f"worst {errs.max():.1f} cm across {len(ok)} surviving conditions")
        clean = next((r for r in ok if r["name"] == "clean"), None)
        if clean:
            degraded = [r for r in ok if r["name"] != "clean"]
            if degraded:
                factor = np.median([abs(r["err_cm"]) for r in degraded]) / max(
                    abs(clean["err_cm"]), 1e-6)
                print(f"  degradation vs clean   : {factor:.1f}x median error")
    failed = [r for r in rows if r.get("failed", True)]
    if failed:
        print(f"  outright failures      : {', '.join(r['name'] for r in failed)}")
    else:
        print("  outright failures      : none")

    # The distinction that actually matters for trusting this in the field.
    silent = [r for r in ok if abs(r["err_cm"]) > 5.0 and not r["flagged"]]
    caught = [r for r in ok if abs(r["err_cm"]) > 5.0 and r["flagged"]]
    print(f"  large errors caught    : {len(caught)} "
          f"({', '.join(r['name'] for r in caught) or 'none'})")
    print(f"  large errors MISSED    : {len(silent)} "
          f"({', '.join(r['name'] for r in silent) or 'none'})")
    if silent:
        print("\n  a wrong answer the audit does not flag is the dangerous kind:")
        print("  it arrives with tight error bars and a clean residual.")
    print()


if __name__ == "__main__":
    main()
