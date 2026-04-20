"""Visualize DTW sequence alignment between two videos of the same gesture.

Shows:
  1. DTW cost matrix heatmap with the optimal alignment path
  2. Per-frame motion energy curves of both sequences with alignment correspondence lines
  3. A selected landmark trajectory (index fingertip x/y) for both sequences

Usage:
    python visualize_dtw_alignment.py [user_stem] [reference_gesture]

Defaults to 'd_slovo' vs 'slovo'. Paths follow the same conventions as compare_strategies.py.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))

from HandAnnotation import HandAnnotation
from Scoring import Scoring
from LandmarkIndices import LandmarkIndices

REFERENCE_FOLDER = "../videa"
ANNOTATED_FOLDER = "../videa/.annotated"
RECORDED_FOLDER = "../videa/.recorded"


# ── Annotation loading (same logic as compare_strategies.py) ─────────────────


def load_annotation(video_path):
    video_path = os.path.abspath(video_path)
    os.makedirs(ANNOTATED_FOLDER, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    ext = os.path.splitext(video_path)[1] or ".mp4"
    annotated_path = os.path.join(ANNOTATED_FOLDER, f"{stem}_annotated{ext}")
    landmarks_path = os.path.splitext(annotated_path)[0] + "_handLandmarks.json"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    annotation = HandAnnotation(cap)

    cached_ok = False
    if os.path.isfile(landmarks_path):
        loaded = annotation.loadLandmarksFromFile(landmarks_path)
        cached_ok = loaded is not None and len(loaded) > 0

    if not cached_ok:
        annotation.createAnnotatedVideo(video_path, annotated_path)

    annotation.cam.release()
    annotation.out.release()
    return annotation


# ── DTW using Scoring._weightedFrameDistance ─────────────────────────────────


def _backtrack(dp, n, m):
    """Backtrack through the (n+1)×(m+1) DP table to get the 0-indexed optimal path."""
    path = []
    i, j = n, m
    while i > 1 or j > 1:
        path.append((i - 1, j - 1))
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            step = np.argmin([dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1]])
            if step == 0:
                i -= 1
                j -= 1
            elif step == 1:
                i -= 1
            else:
                j -= 1
    path.append((0, 0))
    path.reverse()
    return path


def dtw_with_path(scorer, seq1, seq2, hand_weights):
    """Build DTW DP table using scorer._weightedFrameDistance and return (acc_cost, path).

    Uses the exact same distance function and recurrence as Scoring._dtwDistance so
    the cost matrix reflects what actually drives the score.

    Args:
        scorer: Scoring instance.
        seq1, seq2: Lists of per-frame (hand0, hand1) tuples.
        hand_weights: [w0, w1] from _calculateHandMotionWeights.

    Returns:
        acc_cost: np.ndarray (n, m) accumulated cost (0-indexed).
        path: list of (i, j) index pairs from (0, 0) to (n-1, m-1).
    """
    n, m = len(seq1), len(seq2)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = scorer._weightedFrameDistance(seq1[i - 1], seq2[j - 1], hand_weights)
            dp[i, j] = c + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp[1:, 1:], _backtrack(dp, n, m)


# ── Visualization helpers ─────────────────────────────────────────────────────


def motion_energy(frames):
    """Per-frame motion energy (frame-to-frame mean landmark displacement)."""
    energies = [0.0]
    for i in range(1, len(frames)):
        energies.append(float(np.mean(np.linalg.norm(frames[i] - frames[i - 1], axis=1))))
    return np.array(energies)


def landmark_trajectory(frames, landmark_idx):
    """Extract (x, y) trajectory of a single landmark across all frames."""
    xs = [f[landmark_idx, 0] for f in frames]
    ys = [f[landmark_idx, 1] for f in frames]
    return np.array(xs), np.array(ys)


# ── Main visualization ────────────────────────────────────────────────────────


def visualize(user_stem, ref_gesture):
    for ext in (".mp4", ".avi"):
        candidate = os.path.abspath(os.path.join(RECORDED_FOLDER, f"{user_stem}{ext}"))
        if os.path.isfile(candidate):
            user_path = candidate
            break
    else:
        user_path = os.path.abspath(os.path.join(RECORDED_FOLDER, f"{user_stem}.mp4"))
    ref_path = os.path.abspath(os.path.join(REFERENCE_FOLDER, f"{ref_gesture}.mp4"))

    print(f"Loading user:      {user_path}")
    user_ann = load_annotation(user_path)
    print(f"Loading reference: {ref_path}")
    ref_ann = load_annotation(ref_path)

    # Use the exact same frame extraction + trimming as the scorer
    scorer = Scoring(user_ann, ref_ann, strategy="original")
    ref_seq = scorer._trimReferenceSequenceByMarkers(ref_ann.handLandmarksTimestamped)
    user_frames = scorer._extractPerHandArrays(user_ann.handLandmarksTimestamped)
    ref_frames = scorer._extractPerHandArrays(ref_seq)
    hand_weights = scorer._calculateHandMotionWeights(ref_frames)
    user_frames = scorer._trimLowMotionEdges(user_frames, hand_weights)

    print(f"User frames (after trim): {len(user_frames)}, Reference frames (after trim): {len(ref_frames)}")
    print(f"Hand weights: h0={hand_weights[0]:.3f}, h1={hand_weights[1]:.3f}")

    if len(user_frames) == 0 or len(ref_frames) == 0:
        print("ERROR: No valid hand frames found in one of the videos.")
        return

    score = scorer.calculateScore(
        userLandmarks=user_ann.handLandmarksTimestamped,
        includeWristTrajectory=False,
    )
    score_pct = score * 100
    print(f"Score: {score_pct:.1f}%")

    cost_matrix, path = dtw_with_path(scorer, user_frames, ref_frames, hand_weights)

    # Extract dominant hand as plain (21,3) arrays for energy / trajectory plots
    def dominant(f):
        return f[0] if f[0] is not None else f[1]

    user_plain = [dominant(f) for f in user_frames]
    ref_plain = [dominant(f) for f in ref_frames]

    user_energy = motion_energy(user_plain)
    ref_energy = motion_energy(ref_plain)

    # Index fingertip (landmark 8) x-coordinate trajectory
    LANDMARK = LandmarkIndices.INDEX_TIP
    user_traj_x, user_traj_y = landmark_trajectory(user_plain, LANDMARK)
    ref_traj_x, ref_traj_y = landmark_trajectory(ref_plain, LANDMARK)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"DTW Alignment: '{user_stem}'  vs  '{ref_gesture}' reference    |    Score: {score_pct:.1f}%",
        fontsize=14,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # 1. DTW cost matrix with alignment path
    ax_dtw = fig.add_subplot(gs[:, 0])
    im = ax_dtw.imshow(cost_matrix, origin="lower", aspect="auto", cmap="viridis")
    path_i = [p[0] for p in path]
    path_j = [p[1] for p in path]
    ax_dtw.plot(path_j, path_i, color="red", linewidth=1.5, label="Optimal path")
    ax_dtw.set_xlabel("Reference frame index")
    ax_dtw.set_ylabel("User frame index")
    ax_dtw.set_title("DTW Cost Matrix & Alignment Path")
    ax_dtw.legend(loc="upper left", fontsize=8)
    plt.colorbar(im, ax=ax_dtw, fraction=0.046, pad=0.04)
    ax_dtw.text(
        0.98,
        0.02,
        f"Score: {score_pct:.1f}%",
        transform=ax_dtw.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#333333", alpha=0.8),
    )

    # 2. Motion energy curves with alignment correspondence lines
    ax_energy = fig.add_subplot(gs[0, 1])
    ax_energy.plot(user_energy, label=f"User ({user_stem})", color="steelblue")
    ax_energy.plot(ref_energy, label=f"Reference ({ref_gesture})", color="darkorange")

    # Draw every ~10th alignment correspondence as a faint line
    step = max(1, len(path) // 30)
    for pi, pj in path[::step]:
        ax_energy.annotate(
            "",
            xy=(pj, ref_energy[pj]),
            xytext=(pi, user_energy[pi]),
            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3, lw=0.8),
        )

    ax_energy.set_xlabel("Frame index")
    ax_energy.set_ylabel("Motion energy")
    ax_energy.set_title("Per-Frame Motion Energy + DTW Correspondences")
    ax_energy.legend(fontsize=8)

    # 3. Index fingertip X trajectory
    ax_x = fig.add_subplot(gs[1, 1])
    ax_x.plot(user_traj_x, label=f"User ({user_stem})", color="steelblue")
    ax_x.plot(ref_traj_x, label=f"Reference ({ref_gesture})", color="darkorange")
    ax_x.set_xlabel("Frame index")
    ax_x.set_ylabel("X coordinate (normalized)")
    ax_x.set_title(f"Index Fingertip X Trajectory (landmark {LANDMARK})")
    ax_x.legend(fontsize=8)

    plt.savefig(f"dtw_alignment_{user_stem}_vs_{ref_gesture}.png", dpi=130, bbox_inches="tight")
    print(f"Saved: dtw_alignment_{user_stem}_vs_{ref_gesture}.png")
    plt.close("all")


if __name__ == "__main__":
    user_stem = sys.argv[1] if len(sys.argv) > 1 else "d_slovo"
    ref_gesture = sys.argv[2] if len(sys.argv) > 2 else "slovo"
    visualize(user_stem, ref_gesture)
