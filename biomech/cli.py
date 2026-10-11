"""Command line interface.

  python -m biomech track   VIDEO  --out results/run1 --joint-set leg
  python -m biomech analyze POINTS.csv --fps 30 --out results/run1
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from . import JOINT_SETS, __version__
from .kinematics import clean_trajectories, find_cycles, joint_angles, limb_lengths, summarise
from .markers import track_video
from .report import write_outputs
from .video import annotate_video, render_stick_animation


def _analyse(df, fps, joint_set, outdir, facing, glitch_px, smooth_s, jump_px=45.0, video=None, slowdown=1.0,
             animation=True, extra=None):
    names = JOINT_SETS[joint_set]
    clean, qa = clean_trajectories(df, names, fps, glitch_px=glitch_px, smooth_s=smooth_s, jump_px=jump_px)
    angles = joint_angles(clean, fps, joint_set, facing, smooth_s)
    summary = summarise(angles, joint_set)
    cycles = find_cycles(angles, fps, joint_set)
    lengths = limb_lengths(clean, joint_set)
    meta = write_outputs(outdir, clean, qa, angles, summary, cycles, lengths, names, joint_set, fps, extra)
    if video:
        annotate_video(video, clean, angles, names, joint_set, os.path.join(outdir, "annotated.mp4"), slowdown)
    if animation:
        render_stick_animation(clean, angles, names, joint_set, fps, os.path.join(outdir, "stick_figure.mp4"),
                               slowdown=slowdown)
    print(summary.to_string(index=False))
    print(f"\nglitch frames repaired: {meta['glitch_frames_per_marker']}")
    print(f"jumps (possible identity swaps, not repairable): {meta['jump_frames_per_marker']}")
    print(f"repetitions found: {len(cycles)} (peaks rejected at tracking jumps: {cycles.attrs.get('peaks_rejected_at_jumps', 0)})")
    print(f"outputs in: {outdir}")


def _preview(args):
    import cv2
    from .markers import detect_markers
    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read {args.video}")
    det = detect_markers(frame, args.mask, args.close_radius, args.min_area)
    for i, (x, y, a) in enumerate(det):
        cv2.circle(frame, (int(x), int(y)), 14, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{i}", (int(x) + 16, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        print(f"blob {i}: x={x:.1f} y={y:.1f} area={a:.0f}")
    cv2.imwrite(args.out, frame)
    print(f"{len(det)} blobs found. Saved {args.out}. "
          "If there are more blobs than joints, pass the right ones with --init-points \"x1,y1 x2,y2 x3,y3\".")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="biomech", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="results/run", help="output folder")
    common.add_argument("--joint-set", choices=list(JOINT_SETS), default="leg")
    common.add_argument("--facing", choices=["auto", "right", "left"], default="auto")
    common.add_argument("--glitch-px", type=float, default=25.0, help="distance from running median that counts as a tracking glitch")
    common.add_argument("--jump-px", type=float, default=45.0, help="frame-to-frame step that counts as a tracking jump")
    common.add_argument("--smooth-s", type=float, default=0.25, help="Savitzky-Golay window in seconds")
    common.add_argument("--slowdown", type=float, default=1.0, help="output video slow-motion factor")
    common.add_argument("--no-animation", action="store_true")

    t = sub.add_parser("track", parents=[common], help="track red markers in a video and analyse")
    t.add_argument("video")
    t.add_argument("--mask", choices=["rgb", "hsv"], default="rgb")
    t.add_argument("--close-radius", type=int, default=12)
    t.add_argument("--min-area", type=int, default=120)
    t.add_argument("--gate-px", type=float, default=70.0)
    t.add_argument("--init-points", default=None,
                   help='manual start points "x1,y1 x2,y2 x3,y3" (proximal to distal); use when the first frame has extra red blobs')
    t.add_argument("--init-order", default="top-to-bottom",
                   choices=["top-to-bottom", "bottom-to-top", "left-to-right", "right-to-left"])

    pv = sub.add_parser("preview", help="save the first-frame detections with numbered blobs (to choose init points)")
    pv.add_argument("video"); pv.add_argument("--out", default="preview.png")
    pv.add_argument("--mask", choices=["rgb", "hsv"], default="rgb")
    pv.add_argument("--close-radius", type=int, default=12); pv.add_argument("--min-area", type=int, default=120)

    a = sub.add_parser("analyze", parents=[common], help="analyse a CSV of marker positions")
    a.add_argument("csv", help="columns: <name>_x, <name>_y for each joint of the set (frame/time optional)")
    a.add_argument("--fps", type=float, required=True)

    args = ap.parse_args(argv)
    if args.cmd == "preview":
        return _preview(args)
    names = JOINT_SETS[args.joint_set]
    if args.cmd == "track":
        init_pts = None
        if args.init_points:
            init_pts = np.array([[float(v) for v in p.split(",")] for p in args.init_points.split()])
            if len(init_pts) != len(names):
                ap.error(f"--init-points needs {len(names)} points")
        df, fps = track_video(args.video, names, args.mask, args.close_radius, args.min_area,
                              args.gate_px, init_order=args.init_order, init_points=init_pts,
                              progress=lambda k, n: print(f"\rtracking {k}/{n}", end="", file=sys.stderr))
        print(file=sys.stderr)
        os.makedirs(args.out, exist_ok=True)
        df.to_csv(os.path.join(args.out, "trajectories_raw.csv"), index=False)
        _analyse(df, fps, args.joint_set, args.out, args.facing, args.glitch_px, args.smooth_s, args.jump_px,
                 args.video, args.slowdown, not args.no_animation, {"source": os.path.basename(args.video)})
    else:
        df = pd.read_csv(args.csv)
        if "frame" not in df:
            df.insert(0, "frame", np.arange(len(df)))
        if "time_s" not in df:
            df.insert(1, "time_s", df["frame"] / args.fps)
        _analyse(df, args.fps, args.joint_set, args.out, args.facing, args.glitch_px, args.smooth_s, args.jump_px,
                 None, args.slowdown, not args.no_animation, {"source": os.path.basename(args.csv)})


if __name__ == "__main__":
    main()
