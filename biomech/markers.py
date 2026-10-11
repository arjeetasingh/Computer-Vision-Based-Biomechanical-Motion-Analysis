"""Colour-marker detection and identity-preserving multi-marker tracking.

The original MATLAB assignments sorted the detected blobs by their y-coordinate every frame
(top = hip, middle = knee, bottom = ankle). That breaks as soon as the leg moves so that the
ordering changes, or a second red object (sticker on the other leg, clothing) is visible: the
labels jump between blobs. Here each marker keeps its identity through a constant-velocity
Kalman filter and Hungarian assignment with a distance gate; extra blobs are ignored and short
occlusions are bridged by prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

DETECTED, PREDICTED, LOST = 0, 1, 2


# --------------------------------------------------------------------------- detection
def colour_mask(frame_bgr: np.ndarray, mode: str = "rgb") -> np.ndarray:
    """Binary mask of red marker pixels.

    mode "rgb": rule used in the course MATLAB code (R > 1.3 G, R > 1.3 B, R > 30).
    mode "hsv": hue-wrap aware red (H < 10 or H > 170), S > 110, V > 60; more robust to shading.
    """
    if mode == "rgb":
        b, g, r = (frame_bgr[:, :, i].astype(np.float32) for i in range(3))
        return ((r > 1.3 * g) & (r > 1.3 * b) & (r > 30)).astype(np.uint8)
    if mode == "hsv":
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        return (((h < 10) | (h > 170)) & (s > 110) & (v > 60)).astype(np.uint8)
    raise ValueError("mode must be 'rgb' or 'hsv'")


def detect_markers(frame_bgr: np.ndarray, mode: str = "rgb", close_radius: int = 12,
                   min_area: int = 120) -> np.ndarray:
    """Return blob centroids as an (n, 3) array of [x, y, area]."""
    mask = colour_mask(frame_bgr, mode)
    if close_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, _, stats, cen = cv2.connectedComponentsWithStats(mask)
    out = [[cen[i][0], cen[i][1], stats[i, cv2.CC_STAT_AREA]]
           for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return np.array(out, dtype=float).reshape(-1, 3)


# --------------------------------------------------------------------------- tracking
class _CVKalman:
    """Constant-velocity Kalman filter for one 2-D point (state x, y, vx, vy; dt = 1 frame)."""

    def __init__(self, xy, q=1.0, r=2.0):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        self.P = np.diag([r, r, 50.0, 50.0])
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        self.Q = np.diag([q * 0.25, q * 0.25, q, q])
        self.R = np.eye(2) * r

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, z):
        y = np.asarray(z, float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


@dataclass
class MarkerTracker:
    """Track `n_markers` identical-looking markers with persistent identities.

    Marker 0 .. n-1 follow the order given at initialisation (default: top to bottom in the first
    frame, i.e. proximal to distal for an upright limb).
    """
    n_markers: int = 3
    gate_px: float = 70.0          # max distance between prediction and a detection to match
    max_gap: int = 15              # frames of pure prediction before a marker is flagged LOST
    init_order: str = "top-to-bottom"   # "top-to-bottom", "bottom-to-top", "left-to-right", "right-to-left"
    init_points: Optional[np.ndarray] = None  # optional manual (n,2) start points
    filters: list = field(default_factory=list, init=False)
    misses: np.ndarray = field(default=None, init=False)

    def _order(self, pts: np.ndarray) -> np.ndarray:
        key = {"top-to-bottom": (1, 1), "bottom-to-top": (1, -1),
               "left-to-right": (0, 1), "right-to-left": (0, -1)}[self.init_order]
        return pts[np.argsort(key[1] * pts[:, key[0]])]

    def initialise(self, det: np.ndarray) -> bool:
        if self.init_points is not None:
            start = np.asarray(self.init_points, float)[:, :2]
        else:
            if len(det) < self.n_markers:
                return False
            keep = det[np.argsort(-det[:, 2])[: self.n_markers]]      # largest blobs
            start = self._order(keep[:, :2])
        self.filters = [_CVKalman(p) for p in start]
        self.misses = np.zeros(self.n_markers, int)
        return True

    def update(self, det: np.ndarray):
        """Advance one frame. Returns (positions (n,2), status (n,))."""
        n = self.n_markers
        pred = np.array([f.predict() for f in self.filters])
        status = np.full(n, PREDICTED)
        matched = {}
        if len(det):
            cost = np.linalg.norm(pred[:, None, :] - det[None, :, :2], axis=2)
            big = 1e6
            r, c = linear_sum_assignment(np.where(cost > self.gate_px, big, cost))
            for i, j in zip(r, c):
                if cost[i, j] <= self.gate_px:
                    matched[i] = j
        for i, f in enumerate(self.filters):
            if i in matched:
                f.update(det[matched[i], :2])
                self.misses[i] = 0
                status[i] = DETECTED
            else:
                self.misses[i] += 1
                if self.misses[i] > self.max_gap:
                    status[i] = LOST
        pos = np.array([f.x[:2] for f in self.filters])
        return pos, status


def track_video(path: str, names: Iterable[str], mode: str = "rgb", close_radius: int = 12,
                min_area: int = 120, gate_px: float = 70.0, max_gap: int = 15,
                init_order: str = "top-to-bottom", init_points=None,
                progress: Optional[Callable[[int, int], None]] = None) -> tuple[pd.DataFrame, float]:
    """Track the markers of a video. Returns (DataFrame, fps).

    Columns: frame, time_s, <name>_x, <name>_y, <name>_status (0 detected, 1 predicted, 2 lost).
    """
    names = list(names)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    tr = MarkerTracker(len(names), gate_px, max_gap, init_order, init_points)
    rows, k, started = [], 0, False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det = detect_markers(frame, mode, close_radius, min_area)
        if not started:
            started = tr.initialise(det)
            if not started:
                rows.append([np.nan] * (2 * len(names)) + [LOST] * len(names))
                k += 1
                continue
        pos, st = tr.update(det)
        rows.append(list(pos.ravel()) + list(st))
        k += 1
        if progress and total:
            progress(k, total)
    cap.release()
    if not rows:
        raise RuntimeError("no frames read")
    data = {"frame": np.arange(len(rows)), "time_s": np.arange(len(rows)) / fps}
    arr = np.array(rows, float)
    for i, nm in enumerate(names):
        data[f"{nm}_x"] = arr[:, 2 * i]
        data[f"{nm}_y"] = arr[:, 2 * i + 1]
        data[f"{nm}_status"] = arr[:, 2 * len(names) + i].astype(int)
    return pd.DataFrame(data), fps
