"""Trajectory cleaning and joint kinematics for a 3-marker limb (proximal, middle, distal).

Image coordinates: x to the right, y DOWN. All angles are in degrees.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, medfilt, savgol_filter

from . import JOINT_SETS


# --------------------------------------------------------------------------- cleaning

def _sg_piecewise(y, breaks, window, polyorder, deriv=0, delta=1.0):
    """Savitzky-Golay filter / derivative applied separately to each segment between break indices,
    so a tracking discontinuity is never smeared into its neighbours."""
    y = np.asarray(y, float)
    edges = [0] + [int(b) for b in breaks if 0 < b < len(y)] + [len(y)]
    out = np.empty_like(y)
    for a, b in zip(edges[:-1], edges[1:]):
        seg = y[a:b]
        w = min(window, len(seg) if len(seg) % 2 else len(seg) - 1)
        if w >= polyorder + 2 and w >= 3:
            out[a:b] = savgol_filter(seg, w, polyorder, deriv=deriv, delta=delta)
        elif deriv == 0:
            out[a:b] = seg
        else:
            out[a:b] = np.gradient(seg, delta) if len(seg) > 1 else 0.0
    return out

def _odd(n: int, minimum: int = 5) -> int:
    n = max(int(n), minimum)
    return n if n % 2 else n + 1


def clean_trajectories(df: pd.DataFrame, names, fps: float, glitch_px: float = 25.0,
                       median_window: int = 9, smooth_s: float = 0.25, polyorder: int = 3,
                       jump_px: float = 45.0):
    """Repair tracking glitches and smooth.

    1. frames flagged LOST (status 2) or NaN are treated as missing;
    2. frames whose position is more than `glitch_px` from the running median are glitches
       (identity swaps / false blobs);
    3. missing and glitch frames are filled by PCHIP interpolation;
    4. Savitzky-Golay smoothing over `smooth_s` seconds, applied separately on each side of a
       "jump" (a frame-to-frame step larger than `jump_px`, typically an identity swap between two
       similar blobs). Jumps cannot be repaired from positions alone; they are flagged in `qa_df`
       (`<name>_jump`) and kept as discontinuities, not smoothed over.

    Returns (clean_df, qa_df) where qa_df has the raw positions, `<name>_glitch` and `<name>_jump` flags.
    The list of jump frame indices per marker is stored in `clean_df.attrs["jumps"]`.
    """
    from scipy.interpolate import PchipInterpolator

    out = df[["frame", "time_s"]].copy()
    qa = df[["frame", "time_s"]].copy()
    win = _odd(median_window, 3)
    sg = _odd(round(smooth_s * fps), polyorder + 2)
    t = np.arange(len(df), dtype=float)
    jumps = {}
    for nm in names:
        xy = df[[f"{nm}_x", f"{nm}_y"]].to_numpy(float).copy()
        if f"{nm}_status" in df:
            xy[df[f"{nm}_status"].to_numpy() == 2] = np.nan
        qa[f"{nm}_x_raw"], qa[f"{nm}_y_raw"] = xy[:, 0], xy[:, 1]
        bad = np.isnan(xy).any(axis=1)
        filled = xy.copy()
        for d in range(2):                                   # temporary fill for the median filter
            ok = ~np.isnan(filled[:, d])
            filled[:, d] = np.interp(t, t[ok], filled[ok, d]) if ok.any() else 0.0
        med = np.column_stack([medfilt(filled[:, d], win) for d in range(2)])
        glitch = (np.linalg.norm(filled - med, axis=1) > glitch_px) & ~bad
        qa[f"{nm}_glitch"] = glitch | bad
        keep = ~(glitch | bad)
        res = np.empty_like(xy)
        for d in range(2):
            if keep.sum() >= 4:
                res[:, d] = PchipInterpolator(t[keep], xy[keep, d], extrapolate=True)(t)
            else:
                res[:, d] = np.nanmean(xy[:, d])
        step = np.r_[0.0, np.linalg.norm(np.diff(res, axis=0), axis=1)]
        breaks = np.where(step > jump_px)[0]
        jump = np.zeros(len(res), bool); jump[breaks] = True
        qa[f"{nm}_jump"] = jump
        jumps[nm] = breaks.tolist()
        for d in range(2):
            res[:, d] = _sg_piecewise(res[:, d], breaks, sg, polyorder)
        out[f"{nm}_x"], out[f"{nm}_y"] = res[:, 0], res[:, 1]
    out.attrs["jumps"] = jumps
    return out, qa


# --------------------------------------------------------------------------- geometry
def _unit_angle_deg(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    dot = (v * w).sum(axis=1)
    den = np.linalg.norm(v, axis=1) * np.linalg.norm(w, axis=1)
    return np.degrees(np.arccos(np.clip(dot / np.maximum(den, 1e-9), -1, 1)))


def detect_facing(prox: np.ndarray, mid: np.ndarray) -> int:
    """+1 if the limb swings towards image-right, -1 if towards image-left."""
    dx = mid[:, 0] - prox[:, 0]
    return 1 if abs(np.nanmax(dx)) >= abs(np.nanmin(dx)) else -1


def segment_angle_vs_vertical(a: np.ndarray, b: np.ndarray, facing: int = 1) -> np.ndarray:
    """Signed angle of the segment a->b from the downward vertical.

    0 for a hanging segment, positive when b is in front of a (in the `facing` direction).
    """
    v = b - a
    return np.degrees(np.arctan2(facing * v[:, 0], v[:, 1]))


def joint_angles(clean: pd.DataFrame, fps: float, joint_set: str = "leg", facing="auto",
                 smooth_s: float = 0.25, polyorder: int = 3) -> pd.DataFrame:
    """Joint angles and their velocity / acceleration for a 3-marker limb."""
    p, m, d = JOINT_SETS[joint_set]
    P = clean[[f"{p}_x", f"{p}_y"]].to_numpy(float)
    M = clean[[f"{m}_x", f"{m}_y"]].to_numpy(float)
    D = clean[[f"{d}_x", f"{d}_y"]].to_numpy(float)
    if facing == "auto":
        facing = detect_facing(P, M)
    elif facing in ("right", "left"):
        facing = 1 if facing == "right" else -1
    out = pd.DataFrame({"frame": clean["frame"], "time_s": clean["time_s"]})
    out[f"{p}_angle"] = segment_angle_vs_vertical(P, M, facing)           # proximal segment vs vertical
    out[f"{m}_included"] = _unit_angle_deg(P - M, D - M)                  # angle between the two segments
    out[f"{m}_flexion"] = 180.0 - out[f"{m}_included"]
    out[f"{d}_segment_angle"] = segment_angle_vs_vertical(M, D, facing)   # distal segment vs vertical
    # legacy definitions, identical to the MATLAB assignment programs
    out[f"{p}_legacy"] = _unit_angle_deg(M - P, np.tile([0.0, 100.0], (len(P), 1)))
    v = D - M
    out[f"{d}_legacy"] = np.abs(np.degrees(np.arctan2(v[:, 1], v[:, 0])))
    sg = _odd(round(smooth_s * fps), polyorder + 2)
    allj = sorted({j for js in clean.attrs.get("jumps", {}).values() for j in js})
    mask = np.zeros(len(out), bool)                     # derivatives are unreliable right at a jump
    for j in allj:
        mask[max(0, j - 2): j + 2] = True
    for col in [f"{p}_angle", f"{m}_flexion", f"{d}_segment_angle"]:
        vel = _sg_piecewise(out[col].to_numpy(), allj, sg, polyorder, 1, 1 / fps)
        acc = _sg_piecewise(out[col].to_numpy(), allj, sg, polyorder, 2, 1 / fps)
        vel[mask] = np.nan; acc[mask] = np.nan
        out[f"{col}_vel"], out[f"{col}_acc"] = vel, acc
    out.attrs["facing"] = int(facing)
    out.attrs["jump_frames"] = allj
    return out


# --------------------------------------------------------------------------- summary
def summarise(angles: pd.DataFrame, joint_set: str = "leg") -> pd.DataFrame:
    """Range of motion and peak velocity per angle."""
    p, m, d = JOINT_SETS[joint_set]
    rows = []
    for col in [f"{p}_angle", f"{m}_flexion", f"{d}_segment_angle"]:
        a = angles[col].to_numpy()
        row = {"angle": col, "min_deg": a.min(), "max_deg": a.max(), "range_deg": a.max() - a.min(),
               "mean_deg": a.mean(), "sd_deg": a.std(ddof=1),
               "t_at_max_s": angles["time_s"].iloc[int(a.argmax())]}
        if f"{col}_vel" in angles:
            vel = angles[f"{col}_vel"].to_numpy()
            row["peak_velocity_dps"] = np.nanmax(np.abs(vel))
            row["peak_accel_dps2"] = np.nanmax(np.abs(angles[f"{col}_acc"].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows).round(2)


def find_cycles(angles: pd.DataFrame, fps: float, joint_set: str = "leg", min_prominence_deg: float = 15.0):
    """Repetitions found as peaks of the middle-joint flexion. Peaks that coincide with a tracking jump
    (identity swap) are rejected and counted in `df.attrs["peaks_rejected_at_jumps"]`."""
    m = JOINT_SETS[joint_set][1]
    a = angles[f"{m}_flexion"].to_numpy()
    peaks, props = find_peaks(a, prominence=min_prominence_deg, distance=max(1, int(0.3 * fps)))
    jumps = np.array(angles.attrs.get("jump_frames", []), int)
    keep = [i for i, k in enumerate(peaks)                      # a peak sitting on a tracking jump is an artefact
            if not (len(jumps) and np.abs(jumps - k).min() <= max(3, int(0.1 * fps)))]
    rows = [{"repetition": n + 1, "peak_time_s": angles["time_s"].iloc[peaks[i]],
             f"peak_{m}_flexion_deg": a[peaks[i]], "prominence_deg": props["prominences"][i]}
            for n, i in enumerate(keep)]
    df = pd.DataFrame(rows)
    df.attrs["peaks_rejected_at_jumps"] = len(peaks) - len(keep)
    if len(df) > 1:
        df["interval_s"] = df["peak_time_s"].diff()
    return df.round(2)


def limb_lengths(clean: pd.DataFrame, joint_set: str = "leg") -> pd.DataFrame:
    """Segment length in pixels per frame. A constant length is a good tracking QA check."""
    p, m, d = JOINT_SETS[joint_set]
    def dist(a, b):
        return np.hypot(clean[f"{a}_x"] - clean[f"{b}_x"], clean[f"{a}_y"] - clean[f"{b}_y"])
    return pd.DataFrame({"frame": clean["frame"], f"{p}_{m}_px": dist(p, m), f"{m}_{d}_px": dist(m, d)})
