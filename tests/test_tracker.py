import cv2
import numpy as np
import pandas as pd

from biomech.markers import DETECTED, MarkerTracker, detect_markers, track_video

W, H, N = 420, 300, 90


def truth(n=N):
    t = np.arange(n) / 30.0
    hip = np.column_stack([np.full(n, 100.0), 60 + 5 * np.sin(t * 3)])
    knee = np.column_stack([100 + 60 * np.sin(t * 2.2), 150 + 10 * np.cos(t * 2)])
    ankle = np.column_stack([100 + 90 * np.sin(t * 2.2 + 0.3), 235 + 25 * np.cos(t * 2.2)])
    return np.stack([hip, knee, ankle], axis=1)           # (n, 3, 2)


def render(path, tr, distractor=True, occlude=range(40, 46)):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for k in range(tr.shape[0]):
        img = np.full((H, W, 3), 190, np.uint8)
        for j in range(3):
            if j == 2 and k in occlude:
                continue
            cv2.circle(img, tuple(int(v) for v in tr[k, j]), 9, (30, 30, 220), -1)
        if distractor:                                        # second red blob that crosses the ankle path
            cv2.circle(img, (int(440 - 3 * k), 238), 9, (30, 30, 220), -1)
        out.write(img)
    out.release()


def test_detect_finds_blobs():
    img = np.full((100, 100, 3), 180, np.uint8)
    cv2.circle(img, (30, 40), 8, (20, 20, 230), -1)
    cv2.circle(img, (70, 60), 8, (20, 20, 230), -1)
    for mode in ("rgb", "hsv"):
        d = detect_markers(img, mode)
        assert len(d) == 2
        assert np.allclose(sorted(d[:, 0]), [30, 70], atol=1.0)


def test_tracker_keeps_identity_with_distractor_and_occlusion(tmp_path):
    tr = truth()
    p = str(tmp_path / "synthetic.mp4")
    render(p, tr)
    df, fps = track_video(p, ["hip", "knee", "ankle"], gate_px=60)
    assert abs(fps - 30) < 0.5
    for j, nm in enumerate(["hip", "knee", "ankle"]):
        est = df[[f"{nm}_x", f"{nm}_y"]].to_numpy()
        err = np.linalg.norm(est - tr[:, j], axis=1)
        assert np.median(err) < 1.5
        assert err.max() < 25, f"{nm}: identity lost, max err {err.max():.1f}"
    st = df["ankle_status"].to_numpy()
    assert (st[40:46] != DETECTED).all()                      # bridged by prediction
    assert (st[:35] == DETECTED).all()


def test_manual_init_points_override_extra_blob(tmp_path):
    tr = truth()
    p = str(tmp_path / "synthetic2.mp4")
    out = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for k in range(N):                                         # extra blob present from frame 0
        img = np.full((H, W, 3), 190, np.uint8)
        for j in range(3):
            cv2.circle(img, tuple(int(v) for v in tr[k, j]), 9, (30, 30, 220), -1)
        cv2.circle(img, (390, 40), 9, (30, 30, 220), -1)
        out.write(img)
    out.release()
    df, _ = track_video(p, ["hip", "knee", "ankle"], init_points=tr[0], gate_px=60)
    for j, nm in enumerate(["hip", "knee", "ankle"]):
        err = np.linalg.norm(df[[f"{nm}_x", f"{nm}_y"]].to_numpy() - tr[:, j], axis=1)
        assert err.max() < 3


def test_sort_by_y_would_fail_where_tracker_holds(tmp_path):
    """Regression guard: the old rule (take 3 lowest-y blobs, sort by y) mislabels this clip."""
    tr = truth()
    mis = 0
    for k in range(N):
        img = np.full((H, W, 3), 190, np.uint8)
        for j in range(3):
            cv2.circle(img, tuple(int(v) for v in tr[k, j]), 9, (30, 30, 220), -1)
        cv2.circle(img, (int(440 - 3 * k), 238), 9, (30, 30, 220), -1)
        d = detect_markers(img, "rgb")
        pts = d[np.argsort(d[:, 1])][:3]
        if np.linalg.norm(pts[2, :2] - tr[k, 2]) > 20:
            mis += 1
    assert mis > 0
