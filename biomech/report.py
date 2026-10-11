"""Figures and tables."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import JOINT_SETS

COL = {"prox": "#1f77b4", "mid": "#d62728", "dist": "#2ca02c"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_qa(qa: pd.DataFrame, clean: pd.DataFrame, names, path):
    fig, axs = plt.subplots(len(names), 1, figsize=(10, 2.6 * len(names)), sharex=True)
    for ax, nm in zip(np.atleast_1d(axs), names):
        t = qa["time_s"]
        ax.plot(t, qa[f"{nm}_y_raw"], ".", ms=3, color="0.6", label="tracked y (raw)")
        ax.plot(t, clean[f"{nm}_y"], "-", color="C0", label="cleaned y")
        g = qa[f"{nm}_glitch"].to_numpy()
        ax.plot(t[g], qa[f"{nm}_y_raw"][g], "rx", label="glitch (repaired)")
        j = qa[f"{nm}_jump"].to_numpy()
        for tj in t[j]:
            ax.axvline(tj, color="orange", lw=1, alpha=.7)
        if j.any():
            ax.plot([], [], color="orange", label="jump (identity swap?)")
        ax.set_ylabel(f"{nm} y (px)"); ax.invert_yaxis(); ax.grid(alpha=.3)
    np.atleast_1d(axs)[0].legend(ncol=3, fontsize=8)
    np.atleast_1d(axs)[-1].set_xlabel("time (s)")
    fig.suptitle("Tracking quality: raw vs cleaned marker height")
    _save(fig, path)


def plot_trajectories(clean, names, path, frame_shape=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    for nm, c in zip(names, COL.values()):
        ax.plot(clean[f"{nm}_x"], clean[f"{nm}_y"], color=c, lw=1.4, label=nm)
        ax.plot(clean[f"{nm}_x"].iloc[0], clean[f"{nm}_y"].iloc[0], "o", color=c)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend()
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)"); ax.set_title("Marker paths (dot = start)")
    _save(fig, path)


def plot_angles(angles: pd.DataFrame, joint_set: str, path):
    p, m, d = JOINT_SETS[joint_set]
    series = [(f"{p}_angle", f"{p} segment vs vertical", COL["prox"]),
              (f"{m}_flexion", f"{m} flexion", COL["mid"]),
              (f"{d}_segment_angle", f"{d} segment vs vertical", COL["dist"])]
    fig, axs = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for i, (col, lab, c) in enumerate(series):
        axs[i, 0].plot(angles["time_s"], angles[col], color=c)
        axs[i, 0].set_ylabel(f"{lab}\n(deg)"); axs[i, 0].grid(alpha=.3)
        if f"{col}_vel" in angles:
            axs[i, 1].plot(angles["time_s"], angles[f"{col}_vel"], color=c)
            axs[i, 1].set_ylabel("deg/s"); axs[i, 1].grid(alpha=.3)
    axs[0, 0].set_title("Joint angle"); axs[0, 1].set_title("Angular velocity")
    axs[-1, 0].set_xlabel("time (s)"); axs[-1, 1].set_xlabel("time (s)")
    _save(fig, path)


def plot_phase(angles: pd.DataFrame, joint_set: str, path):
    p, m, _ = JOINT_SETS[joint_set]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sc = ax.scatter(angles[f"{p}_angle"], angles[f"{m}_flexion"], c=angles["time_s"], s=10, cmap="viridis")
    ax.plot(angles[f"{p}_angle"], angles[f"{m}_flexion"], color="0.7", lw=.8, zorder=0)
    ax.set_xlabel(f"{p} angle (deg)"); ax.set_ylabel(f"{m} flexion (deg)")
    ax.grid(alpha=.3); fig.colorbar(sc, label="time (s)"); ax.set_title(f"{p}-{m} coordination (angle-angle)")
    _save(fig, path)


def plot_rom(summary: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    y = np.arange(len(summary))
    ax.barh(y, summary["range_deg"], color=list(COL.values())[: len(y)])
    ax.set_yticks(y); ax.set_yticklabels(summary["angle"])
    for yi, v in zip(y, summary["range_deg"]):
        ax.text(v + 1, yi, f"{v:.1f}°", va="center")
    ax.set_xlabel("range of motion (deg)"); ax.grid(axis="x", alpha=.3); ax.invert_yaxis()
    _save(fig, path)


def write_outputs(outdir: str, clean, qa, angles, summary, cycles, lengths, names, joint_set, fps, extra=None):
    os.makedirs(outdir, exist_ok=True)
    clean.round(3).to_csv(os.path.join(outdir, "trajectories_clean.csv"), index=False)
    angles.round(3).to_csv(os.path.join(outdir, "joint_angles.csv"), index=False)
    summary.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    cycles.to_csv(os.path.join(outdir, "repetitions.csv"), index=False)
    plot_qa(qa, clean, names, os.path.join(outdir, "fig_tracking_qa.png"))
    plot_trajectories(clean, names, os.path.join(outdir, "fig_trajectories.png"))
    plot_angles(angles, joint_set, os.path.join(outdir, "fig_joint_angles.png"))
    plot_phase(angles, joint_set, os.path.join(outdir, "fig_phase_portrait.png"))
    plot_rom(summary, os.path.join(outdir, "fig_range_of_motion.png"))
    glitch_frames = {nm: int(qa[f"{nm}_glitch"].sum()) for nm in names}
    jump_frames = {nm: int(qa[f"{nm}_jump"].sum()) for nm in names}
    meta = {"fps": fps, "frames": int(len(clean)), "duration_s": float(clean["time_s"].iloc[-1]),
            "joint_set": joint_set, "facing": int(angles.attrs.get("facing", 0)),
            "glitch_frames_per_marker": glitch_frames, "jump_frames_per_marker": jump_frames,
            "segment_length_cv_percent": {c: float(100 * lengths[c].std() / lengths[c].mean())
                                           for c in lengths.columns if c != "frame"}}
    if extra:
        meta.update(extra)
    with open(os.path.join(outdir, "run_info.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta
