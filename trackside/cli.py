"""Command line entry point: point it at a video, get measurements back."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .calib import PlaneCalibration
from .finish import analyse_finish
from .flight import analyse_flight
from .render import annotate_strip, trajectory_overlay
from .synth import load_frames


def _cmd_finish(args: argparse.Namespace) -> int:
    frames, _ = load_frames(Path(args.video))
    if not frames:
        print(f"no frames read from {args.video}", file=sys.stderr)
        return 1

    res = analyse_finish(
        frames, args.finish_x, args.fps, args.meters_per_pixel, args.lanes,
        exposure_s=args.exposure,
    )
    if args.exposure is None:
        print("note: shutter unknown, so absolute times carry an unmodelled "
              "offset of up to one frame. Margins are unaffected.\n")

    for i, r in enumerate(res.placings, 1):
        print(f"{i}. {r.name:<12} {r.cross_time.format(4)}   {r.speed.format(2)}")

    print()
    for s in res.separations():
        verdict = "" if s["resolvable"] else "   <- TOO CLOSE TO CALL"
        print(f"{s['ahead']} over {s['behind']}: "
              f"{s['margin_s'] * 1000:.1f} ms "
              f"(resolution {s['resolution_s'] * 1000:.2f} ms){verdict}")

    if args.out:
        annotate_strip(res, Path(args.out), args.lanes)
        print(f"\n-> {args.out}")
    return 0


def _cmd_flight(args: argparse.Namespace) -> int:
    frames, _ = load_frames(Path(args.video))
    if not frames:
        print(f"no frames read from {args.video}", file=sys.stderr)
        return 1

    cal = PlaneCalibration.from_circle(
        args.circle_px, args.event, sigma_pixel=args.click_sigma,
        depth_m=args.depth, sigma_depth_m=args.depth_sigma,
    )
    print(f"calibration: {cal.describe()}\n")

    res = analyse_flight(
        frames, cal, args.fps, args.ground_y,
        implement=args.implement, use_gravity_scale=args.gravity_scale,
    )

    print(f"release speed   {res.release_speed.format(3)}")
    print(f"release angle   {res.release_angle.format(2)}")
    print(f"release height  {res.release_height.format(3)}")
    print(f"distance        {res.predicted_distance_with_drag.format(3)}")
    print(f"\ntracked {res.n_inliers}/{res.n_candidates} detections on the arc")

    if args.gravity_scale:
        print("\nscale taken from gravity: the g audit is unavailable by "
              "construction (it would be circular).")
    else:
        print(f"\nrecovered g     {res.recovered_g.format(4)}  (true 9.8067)")
        print(f"ruler scale     {res.ring_scale_mm_px.format(3)}")
        print(f"gravity scale   {res.gravity_scale_mm_px.format(3)}")
        status = "AGREE" if res.scale_check_passes else "DISAGREE - check calibration"
        print(f"cross-check     {status} ({res.scale_agreement * 100:.2f}% apart)")

    if args.out:
        trajectory_overlay(frames, res, Path(args.out))
        print(f"\n-> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="trackside",
        description="Measure track & field events from one camera, with error bars.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("finish", help="sprint finish timing and placings")
    f.add_argument("video", help="video file or directory of frames")
    f.add_argument("--fps", type=float, required=True)
    f.add_argument("--finish-x", type=float, required=True,
                   dest="finish_x", help="finish line, pixels from left")
    f.add_argument("--lanes", type=int, required=True)
    f.add_argument("--meters-per-pixel", type=float, required=True,
                   dest="meters_per_pixel")
    f.add_argument("--exposure", type=float, default=None,
                   help="shutter open time in seconds; omit if unknown")
    f.add_argument("--out", default=None, help="write annotated photo finish here")
    f.set_defaults(func=_cmd_finish)

    t = sub.add_parser("flight", help="field-event implement tracking")
    t.add_argument("video", help="video file or directory of frames")
    t.add_argument("--fps", type=float, required=True)
    t.add_argument("--circle-px", type=float, required=True, dest="circle_px",
                   help="throwing circle diameter, in pixels")
    t.add_argument("--ground-y", type=float, required=True, dest="ground_y",
                   help="ground plane, pixels from top")
    t.add_argument("--event", default="shot", choices=["shot", "hammer", "discus"])
    t.add_argument("--implement", default="shot_mens")
    t.add_argument("--click-sigma", type=float, default=2.0, dest="click_sigma")
    t.add_argument("--depth", type=float, default=None,
                   help="camera-to-throw-plane distance, metres")
    t.add_argument("--depth-sigma", type=float, default=0.0, dest="depth_sigma",
                   help="uncertainty in that distance; enters the error budget")
    t.add_argument("--gravity-scale", action="store_true", dest="gravity_scale",
                   help="derive scale from flight curvature instead of the ruler")
    t.add_argument("--out", default=None, help="write trajectory overlay here")
    t.set_defaults(func=_cmd_flight)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
