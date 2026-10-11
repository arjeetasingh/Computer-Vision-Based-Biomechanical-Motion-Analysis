"""Recover marker positions from a video that already has the MATLAB overlay burnt in.

The assignment programs draw a yellow filled circle (radius 14 px) on every tracked point. This
script finds those circles again, so the trajectory of the original MATLAB tracker can be analysed
without the raw video (PRI.mp4).  The overlay's cyan skeleton splits the circles, so the yellow mask
is closed before blob detection; the yellow label boxes are rejected by size and by temporal
consistency (best 3 of the candidates, closest to the previous frame).

Usage:  python tools/extract_points_from_annotated_video.py VIDEO OUT.csv [--names hip knee ankle]
"""
import argparse
import itertools

import cv2
import numpy as np
import pandas as pd


def candidates(frame):
    b, g, r = (frame[:, :, i].astype(int) for i in range(3))
    m = ((r > 215) & (g > 215) & (b < 110)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    out = [cen[i] for i in range(1, n)
           if 250 <= st[i, 4] <= 1100 and 18 <= st[i, 2] <= 40 and 18 <= st[i, 3] <= 40]
    return np.array(out, float).reshape(-1, 2)


def best_triple(prev, pts, k):
    best, bd = None, 1e18
    for comb in itertools.combinations(range(len(pts)), k):
        q = pts[list(comb)]
        q = q[np.argsort(q[:, 1])]
        d = np.linalg.norm(q - prev, axis=1).sum()
        if d < bd:
            best, bd = q, d
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out")
    ap.add_argument("--names", nargs="+", default=["hip", "knee", "ankle"])
    a = ap.parse_args()
    k = len(a.names)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cands = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        cands.append(candidates(fr))
    n = len(cands)
    seed = next(i for i in range(n) if len(cands[i]) == k)
    sel = [None] * n
    sel[seed] = cands[seed][np.argsort(cands[seed][:, 1])]
    for i in range(seed + 1, n):
        sel[i] = best_triple(sel[i - 1], cands[i], k)
    for i in range(seed - 1, -1, -1):
        sel[i] = best_triple(sel[i + 1], cands[i], k)
    S = np.array(sel)
    df = pd.DataFrame({"frame": np.arange(n), "time_s": np.arange(n) / fps})
    for j, nm in enumerate(a.names):
        df[f"{nm}_x"], df[f"{nm}_y"] = S[:, j, 0].round(2), S[:, j, 1].round(2)
    df.to_csv(a.out, index=False)
    print(f"{n} frames at {fps:.2f} fps -> {a.out}")


if __name__ == "__main__":
    main()
