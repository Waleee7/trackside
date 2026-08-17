"""Command line entry point: point it at a video, get measurements back."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .calib import PlaneCalibration
from .events import events_from_finish, events_from_flight, summarise
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

    print("\nevents:")
    print(summarise(events_from_finish(res, shutter_known=args.exposure is not None)))

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
        implement=args.implement, scale_mode=args.scale_mode,
        exposure_s=args.exposure, rolling_shutter_s=args.rolling_shutter,
    )

    print(f"release speed   {res.release_speed.format(3)}")
    print(f"release angle   {res.release_angle.format(2)}")
    print(f"release height  {res.release_height.format(3)}")
    print(f"flight time     {res.flight_time.format(3)}")
    print(f"apex            {res.apex_height.format(3)}")
    print(f"distance        {res.predicted_distance.format(3)}")
    print(f"\ntracked {res.n_inliers}/{res.n_candidates} detections, "
          f"{res.coverage * 100:.0f}% frame coverage, "
          f"{res.residual_px:.2f} px model residual")

    print("\nscale:")
    print(res.scale.describe())
    if args.scale_mode != "gravity":
        print(f"\n  implied g if the ruler is right: {res.recovered_g.format(4)} "
              f"(true 9.8067)")

    if args.exposure is None:
        print("\nnote: shutter unknown. The first blob's centroid sits half an "
              "exposure after release, which shortens the reported distance by "
              "a couple of centimetres. Pass --exposure to correct it.")

    events = events_from_flight(res)
    if events:
        print("\nevents:")
        print(summarise(events))

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
    t.add_argument("--scale-mode", default="fuse", dest="scale_mode",
                   choices=["fuse", "ruler", "gravity"],
                   help="fuse (default) combines the reference object with the "
                        "flight's own gravitational curvature")
    t.add_argument("--exposure", type=float, default=None,
                   help="shutter open time in seconds; omit if unknown")
    t.add_argument("--rolling-shutter", type=float, default=None,
                   dest="rolling_shutter",
                   help="sensor readout time in seconds (a fixed property of "
                        "your camera). Uncorrected it costs several cm and, "
                        "unlike a bad ruler, it does not trip the gravity audit")
    t.add_argument("--out", default=None, help="write trajectory overlay here")
    t.set_defaults(func=_cmd_flight)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
