# Methods

## Pipeline
1. **Detect** red markers: R > 1.3·G, R > 1.3·B, R > 30 (or HSV mask), morphological closing, area filter, connected components.
2. **Track**: constant-velocity Kalman filter per marker, Hungarian assignment with a distance gate (default 70 px); short gaps bridged by prediction, flagged `PREDICTED`/`LOST`.
3. **Clean**: glitch = distance from a running median > 25 px; repaired with PCHIP. Jumps (> 45 px in one frame, possible identity swaps) are flagged and never smoothed across; Savitzky-Golay is applied piecewise.
4. **Angles** (image y points down, facing auto-detected or given):
   - segment angle vs vertical = atan2(facing·Δx, Δy) (signed, degrees)
   - included joint angle = arccos(v1·v2 / |v1||v2|), v1 = proximal − joint, v2 = distal − joint
   - flexion = 180° − included angle
   - "legacy" columns reproduce the course-assignment definitions for comparison
5. **Derivatives**: angular velocity/acceleration from the smoothed angle; samples within ±2 frames of a jump are masked.
6. **Repetitions**: peaks of knee/elbow flexion (scipy `find_peaks` with prominence); peaks within max(3, 0.1·fps) frames of a jump are rejected.
7. **QA**: segment-length coefficient of variation (rigid segments should stay constant). Here hip–knee 1.9 %, knee–ankle 6.05 %.

## Validation
Trajectories were recovered from the burnt-in circles of the original annotated video; angles recomputed from them agree with the angles printed on that video to about 1°:

| Frame | Overlay H/K/A (°) | Recovered H/K/A (°) |
|---|---|---|
| 0 | 1.7 / 171.4 / 83.0 | 1.8 / 171.1 / 82.9 |
| 60 | 49.4 / 153.7 / 66.9 | 49.7 / 153.3 / 67.0 |
| 120 | 7.0 / 163.3 / 113.7 | 7.1 / 163.1 / 114.0 |
| 200 | 59.5 / 116.3 / 94.2 | 60.3 / 115.2 / 94.6 |

The Kalman/Hungarian tracker is unit-tested on synthetic video (distractor blobs, occlusion, manual init) but not yet on the raw course videos. `matlab/track_limb_joint_angles.m` mirrors the logic but was not executed (no MATLAB available when written).

## Limitations
- 2-D image-plane angles: valid only if motion is roughly parallel to the image plane.
- Recovered trajectories inherit the original tracker's identity swaps (flagged, not repairable without raw video).
- Pixel units only; no calibration to metres.
