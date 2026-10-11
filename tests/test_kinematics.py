import numpy as np
import pandas as pd

from biomech import JOINT_SETS
from biomech.kinematics import (clean_trajectories, find_cycles, joint_angles, segment_angle_vs_vertical,
                                _unit_angle_deg)

FPS = 30.0


def make_df(n=150, noise=0.0, seed=0):
    t = np.arange(n) / FPS
    hip = np.column_stack([np.full(n, 100.0), np.full(n, 100.0)])
    theta = np.radians(40 * np.sin(2 * np.pi * 0.7 * t) ** 2)           # thigh swings forward 0..40 deg
    thigh, shank = 100.0, 100.0
    knee = hip + thigh * np.column_stack([np.sin(theta), np.cos(theta)])
    flex = np.radians(60 * np.sin(2 * np.pi * 0.7 * t) ** 2)           # knee flexion 0..60 deg
    shank_ang = theta - flex                                             # shank angle from vertical
    ankle = knee + shank * np.column_stack([np.sin(shank_ang), np.cos(shank_ang)])
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"frame": np.arange(n), "time_s": t})
    for nm, p in zip(JOINT_SETS["leg"], (hip, knee, ankle)):
        p = p + rng.normal(0, noise, p.shape)
        df[f"{nm}_x"], df[f"{nm}_y"] = p[:, 0], p[:, 1]
    return df, np.degrees(theta), np.degrees(flex)


def test_segment_angle_zero_when_hanging():
    a = np.array([[0.0, 0.0]]); b = np.array([[0.0, 50.0]])
    assert abs(segment_angle_vs_vertical(a, b)[0]) < 1e-9


def test_segment_angle_sign_follows_facing():
    a = np.array([[0.0, 0.0]]); b = np.array([[10.0, 10.0]])
    assert np.isclose(segment_angle_vs_vertical(a, b, 1)[0], 45.0)
    assert np.isclose(segment_angle_vs_vertical(a, b, -1)[0], -45.0)


def test_included_angle_right_angle():
    v = np.array([[1.0, 0.0]]); w = np.array([[0.0, 3.0]])
    assert np.isclose(_unit_angle_deg(v, w)[0], 90.0)


def test_angles_recovered_from_clean_synthetic_motion():
    df, hip_true, knee_true = make_df()
    clean, _ = clean_trajectories(df, JOINT_SETS["leg"], FPS)
    ang = joint_angles(clean, FPS, "leg", facing="right")
    assert np.abs(ang["hip_angle"] - hip_true).max() < 1.0
    assert np.abs(ang["knee_flexion"] - knee_true).max() < 1.5
    assert ang.attrs["facing"] == 1


def test_auto_facing_detects_left():
    df, _, _ = make_df()
    for nm in JOINT_SETS["leg"]:
        df[f"{nm}_x"] = 200 - df[f"{nm}_x"]
    clean, _ = clean_trajectories(df, JOINT_SETS["leg"], FPS)
    ang = joint_angles(clean, FPS, "leg", facing="auto")
    assert ang.attrs["facing"] == -1
    assert ang["hip_angle"].max() > 35


def test_spike_is_repaired():
    df, _, _ = make_df()
    clean0, _ = clean_trajectories(df, JOINT_SETS["leg"], FPS)
    bad = df.copy()
    bad.loc[60, "ankle_x"] += 120
    bad.loc[61, "ankle_y"] -= 90
    clean1, qa = clean_trajectories(bad, JOINT_SETS["leg"], FPS)
    assert qa["ankle_glitch"].sum() >= 2
    assert np.abs(clean1["ankle_x"] - clean0["ankle_x"]).max() < 3.0


def test_lost_frames_are_interpolated():
    df, _, _ = make_df()
    clean0, _ = clean_trajectories(df, JOINT_SETS["leg"], FPS)
    df["knee_status"] = 0
    df.loc[50:53, ["knee_x", "knee_y"]] = np.nan
    df.loc[50:53, "knee_status"] = 2
    clean1, qa = clean_trajectories(df, JOINT_SETS["leg"], FPS)
    assert qa["knee_glitch"].iloc[50:54].all()
    assert np.abs(clean1["knee_x"] - clean0["knee_x"]).max() < 5.0     # 4-frame gap in fast motion


def test_jump_is_kept_and_not_smoothed():
    df, _, _ = make_df()
    df.loc[80:, "ankle_x"] += 150                       # permanent level shift (identity swap)
    clean, qa = clean_trajectories(df, JOINT_SETS["leg"], FPS)
    assert qa["ankle_jump"].sum() == 1
    assert np.abs(clean["ankle_x"].iloc[79] - df["ankle_x"].iloc[79]) < 3
    assert np.abs(clean["ankle_x"].iloc[80] - df["ankle_x"].iloc[80]) < 3
    ang = joint_angles(clean, FPS, "leg", "right")
    assert np.isnan(ang["knee_flexion_vel"].iloc[80])    # derivative masked at the jump


def test_repetitions_counted():
    df, _, _ = make_df(n=300)
    clean, _ = clean_trajectories(df, JOINT_SETS["leg"], FPS)
    ang = joint_angles(clean, FPS, "leg", "right")
    cycles = find_cycles(ang, FPS, "leg")
    expected = int(round(300 / FPS * 1.4))               # sin^2 has twice the frequency
    assert abs(len(cycles) - expected) <= 1


def test_arm_joint_set_runs():
    df, _, _ = make_df()
    df = df.rename(columns={"hip_x": "shoulder_x", "hip_y": "shoulder_y", "knee_x": "elbow_x", "knee_y": "elbow_y",
                            "ankle_x": "wrist_x", "ankle_y": "wrist_y"})
    clean, _ = clean_trajectories(df, JOINT_SETS["arm"], FPS)
    ang = joint_angles(clean, FPS, "arm", "right")
    assert "elbow_flexion" in ang and ang["elbow_flexion"].max() > 50


def test_peak_on_a_jump_is_not_counted_as_repetition():
    df, _, _ = make_df(n=300)
    clean0, _ = clean_trajectories(df, JOINT_SETS["leg"], FPS)
    n_true = len(find_cycles(joint_angles(clean0, FPS, "leg", "right"), FPS, "leg"))
    bad = df.copy()
    bad.loc[100:104, "ankle_x"] += 150          # wrong blob for 5 frames: two jumps, spurious flexion peak
    clean, qa = clean_trajectories(bad, JOINT_SETS["leg"], FPS)
    ang = joint_angles(clean, FPS, "leg", "right")
    cyc = find_cycles(ang, FPS, "leg")
    assert len(cyc) == n_true
