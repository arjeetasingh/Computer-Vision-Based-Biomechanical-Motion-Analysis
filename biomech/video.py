"""Annotated video and stick-figure animation."""
from __future__ import annotations

import os
import shutil
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import JOINT_SETS

BGR = {"prox": (235, 120, 30), "mid": (60, 60, 230), "dist": (60, 180, 60)}
CYAN, YELLOW, WHITE = (255, 255, 0), (0, 255, 255), (255, 255, 255)


def _h264(src: str) -> None:
    """Re-encode to H.264 (plays in browsers / GitHub) when ffmpeg is available."""
    if shutil.which("ffmpeg") is None:
        return
    tmp = src + ".tmp.mp4"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", "-movflags", "+faststart", tmp])
    if r.returncode == 0:
        os.replace(tmp, src)
    elif os.path.exists(tmp):
        os.remove(tmp)


def _arc(img, centre, a, b, radius, colour, thick=3):
    """Arc at `centre` from direction a to direction b (image vectors), the short way round."""
    ang_a = np.degrees(np.arctan2(a[1], a[0]))
    ang_b = np.degrees(np.arctan2(b[1], b[0]))
    d = (ang_b - ang_a + 180) % 360 - 180
    cv2.ellipse(img, tuple(int(v) for v in centre), (radius, radius), 0, ang_a, ang_a + d, colour, thick, cv2.LINE_AA)


def annotate_video(video_in: str, clean, angles, names, joint_set: str, out_path: str,
                   slowdown: float = 1.0, trail: int = 25) -> str:
    p, m, d = JOINT_SETS[joint_set]
    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / slowdown, (W, H))
    pts = {nm: clean[[f"{nm}_x", f"{nm}_y"]].to_numpy() for nm in names}
    k = 0
    while True:
        ok, fr = cap.read()
        if not ok or k >= len(clean):
            break
        for nm, col in zip(names, BGR.values()):                       # fading trails
            for j in range(max(1, k - trail), k):
                a = pts[nm][j - 1]; b = pts[nm][j]
                cv2.line(fr, tuple(map(int, a)), tuple(map(int, b)), col, 2, cv2.LINE_AA)
        P, M, D = (pts[nm][k] for nm in names)
        cv2.line(fr, tuple(map(int, P)), tuple(map(int, M)), CYAN, 6, cv2.LINE_AA)
        cv2.line(fr, tuple(map(int, M)), tuple(map(int, D)), CYAN, 6, cv2.LINE_AA)
        _arc(fr, M, P - M, D - M, 40, YELLOW)
        for nm, pt, col in zip(names, (P, M, D), BGR.values()):
            cv2.circle(fr, tuple(map(int, pt)), 9, YELLOW, -1, cv2.LINE_AA)
            cv2.circle(fr, tuple(map(int, pt)), 9, col, 2, cv2.LINE_AA)
        labels = [(P, f"{p} {angles[f'{p}_angle'].iloc[k]:.1f} deg", BGR["prox"]),
                  (M, f"{m} flex {angles[f'{m}_flexion'].iloc[k]:.1f} deg", BGR["mid"]),
                  (D, f"{d} seg {angles[f'{d}_segment_angle'].iloc[k]:.1f} deg", BGR["dist"])]
        for pt, txt, col in labels:
            x, y = int(pt[0]) + 14, int(pt[1]) - 8
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            x = min(x, W - tw - 4); y = min(max(y, th + 4), H - 4)
            cv2.rectangle(fr, (x - 3, y - th - 3), (x + tw + 3, y + 4), col, -1)
            cv2.putText(fr, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        cv2.putText(fr, f"t = {clean['time_s'].iloc[k]:.2f} s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        out.write(fr)
        k += 1
    cap.release(); out.release()
    _h264(out_path)
    return out_path


def render_stick_animation(clean, angles, names, joint_set: str, fps: float, out_path: str,
                           step: int = 1, size=(12.8, 5.4), dpi=80, slowdown: float = 1.0) -> str:
    """Stick figure with trails on the left and live angle curves on the right. No source video needed."""
    p, m, d = JOINT_SETS[joint_set]
    pts = {nm: clean[[f"{nm}_x", f"{nm}_y"]].to_numpy() for nm in names}
    allx = np.concatenate([pts[n][:, 0] for n in names]); ally = np.concatenate([pts[n][:, 1] for n in names])
    padx, pady = 0.08 * np.ptp(allx), 0.08 * np.ptp(ally)
    series = [(f"{p}_angle", f"{p} segment vs vertical", "#1f77b4"),
              (f"{m}_flexion", f"{m} flexion", "#d62728"),
              (f"{d}_segment_angle", f"{d} segment vs vertical", "#2ca02c")]
    t = angles["time_s"].to_numpy()
    fig = plt.figure(figsize=size, dpi=dpi)
    ax0 = fig.add_axes([0.04, 0.08, 0.38, 0.84]); ax1 = fig.add_axes([0.52, 0.12, 0.45, 0.80])
    ax0.set_xlim(allx.min() - padx, allx.max() + padx); ax0.set_ylim(ally.max() + pady, ally.min() - pady)
    ax0.set_aspect("equal"); ax0.axis("off")
    trails = [ax0.plot([], [], color=c, lw=1, alpha=.5)[0] for _, _, c in series]
    bones = ax0.plot([], [], "-", color="k", lw=4)[0]
    joints = ax0.plot([], [], "o", color="gold", mec="k", ms=10)[0]
    txt = ax0.text(0.02, 0.97, "", transform=ax0.transAxes, va="top", fontsize=11)
    lines = []
    for col, lab, c in series:
        ax1.plot(t, angles[col], color=c, alpha=.25, lw=1)
        lines.append(ax1.plot([], [], color=c, lw=2, label=lab)[0])
    cur = ax1.axvline(0, color="k", lw=1)
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("degrees"); ax1.grid(alpha=.3); ax1.legend(loc="upper right", fontsize=8)
    writer = None; tmp = out_path
    for k in range(0, len(clean), step):
        for tr, nm in zip(trails, names):
            lo = max(0, k - 40)
            tr.set_data(pts[nm][lo:k + 1, 0], pts[nm][lo:k + 1, 1])
        X = [pts[nm][k, 0] for nm in names]; Y = [pts[nm][k, 1] for nm in names]
        bones.set_data(X, Y); joints.set_data(X, Y)
        for ln, (col, _, _) in zip(lines, series):
            ln.set_data(t[:k + 1], angles[col].to_numpy()[:k + 1])
        cur.set_xdata([t[k], t[k]])
        txt.set_text(f"t = {t[k]:.2f} s\n{m} flexion {angles[f'{m}_flexion'].iloc[k]:.1f} deg")
        fig.canvas.draw()
        img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        if writer is None:
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps / slowdown / step,
                                     (img.shape[1], img.shape[0]))
        writer.write(img)
    writer.release(); plt.close(fig)
    _h264(out_path)
    return out_path
