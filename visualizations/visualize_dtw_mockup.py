"""Run Scoring._dtwDistance on two synthetic landmark sequences and visualise the result.

Two hand sequences where only the index fingertip (landmark 8) X-coordinate moves
in a simple, readable pattern:

  seq1 (10 frames) — triangle wave: ramp up then ramp down
      X:  0.3  0.4  0.5  0.6  0.7  0.7  0.6  0.5  0.4  0.3

  seq2 (15 frames) — same shape but with a pause in the middle
      X:  0.3  0.4  0.5  0.6  0.6  0.6  0.7  0.7  0.7  0.6  0.6  0.5  0.4  0.3  0.3

DTW should stretch seq1's middle to match seq2's pause.

Usage:
    python visualize_dtw_mockup.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from Scoring import Scoring


N_LANDMARKS = 21


def make_sequence(x_values):
    """Build a sequence from a list of X positions for landmark 8.

    All other landmarks are stationary at (0.5, 0.5 - lm*0.01, 0).
    Returns a list of (hand0, None) tuples.
    """
    base = np.zeros((N_LANDMARKS, 3), dtype=np.float32)
    for lm in range(N_LANDMARKS):
        base[lm] = [0.5, 0.5 - lm * 0.01, 0.0]

    frames = []
    for x in x_values:
        hand = base.copy()
        hand[8, 0] = x
        frames.append((hand, None))
    return frames


# ── Replicate DTW DP + backtrack ─────────────────────────────────────────────


def run_dtw(scorer: Scoring, seq1: list, seq2: list, hand_weights: list):
    """Fill DP table with scorer._weightedFrameDistance and backtrack the path."""
    n, m = len(seq1), len(seq2)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    local = np.zeros((n, m))

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = scorer._weightedFrameDistance(seq1[i - 1], seq2[j - 1], hand_weights)
            local[i - 1, j - 1] = c
            dp[i, j] = c + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    acc = dp[1:, 1:]  # accumulated cost (0-indexed) for heatmap

    # Backtrack
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

    dtw_dist, _ = scorer._dtwWithPath(seq1, seq2, hand_weights)
    return acc, local, path, dtw_dist


# ── Per-frame motion energy ───────────────────────────────────────────────────


def motion_energy(frames):
    energies = [0.0]
    for k in range(1, len(frames)):
        h_cur = frames[k][0] if frames[k][0] is not None else frames[k][1]
        h_prev = frames[k - 1][0] if frames[k - 1][0] is not None else frames[k - 1][1]
        if h_cur is not None and h_prev is not None:
            energies.append(float(np.mean(np.linalg.norm(h_cur - h_prev, axis=1))))
        else:
            energies.append(0.0)
    return np.array(energies)


def landmark_x(frames, lm_idx=8):
    return np.array([(f[0] if f[0] is not None else f[1])[lm_idx, 0] for f in frames])


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    # seq1: slow rise then sharp drop (10 frames)
    x1_vals = [0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.65, 0.5, 0.35, 0.3]
    # seq2: fast rise then slow drop — different shape, so DTW cost > 0 (12 frames)
    x2_vals = [0.3, 0.5, 0.7, 0.68, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3]

    seq1 = make_sequence(x1_vals)
    seq2 = make_sequence(x2_vals)

    print(f"seq1 X: {x1_vals}")
    print(f"seq2 X: {x2_vals}")
    print(f"Lengths — seq1: {len(seq1)}, seq2: {len(seq2)}")

    scorer = Scoring.__new__(Scoring)
    hand_weights = [1.0, 0.0]

    acc, local, path, dtw_dist = run_dtw(scorer, seq1, seq2, hand_weights)
    print(f"DTW distance (normalised): {dtw_dist:.6f}")

    e1 = motion_energy(seq1)
    e2 = motion_energy(seq2)
    x1 = landmark_x(seq1, lm_idx=8)
    x2 = landmark_x(seq2, lm_idx=8)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"DTW on synthetic sequences  |  seq1: slow rise / sharp drop ({len(seq1)} fr)   "
        f"seq2: fast rise / slow drop ({len(seq2)} fr)   |   DTW dist: {dtw_dist:.4f}",
        fontsize=13,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1 — accumulated cost matrix + path
    ax_dtw = fig.add_subplot(gs[:, 0])
    im = ax_dtw.imshow(acc, origin="lower", aspect="auto", cmap="viridis")
    ax_dtw.plot([p[1] for p in path], [p[0] for p in path], color="red", linewidth=1.5, label="Optimal path")
    ax_dtw.set_xlabel(f"seq2 frame index ({len(seq2)} fr — fast rise / slow drop)")
    ax_dtw.set_ylabel(f"seq1 frame index ({len(seq1)} fr — slow rise / sharp drop)")
    ax_dtw.set_title("Accumulated DTW Cost Matrix & Path")
    ax_dtw.legend(loc="upper left", fontsize=8)
    plt.colorbar(im, ax=ax_dtw, fraction=0.046, pad=0.04)
    ax_dtw.text(
        0.98,
        0.02,
        f"DTW dist: {dtw_dist:.4f}",
        transform=ax_dtw.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#333333", alpha=0.8),
    )

    # Panel 2 — motion energy + correspondences
    ax_energy = fig.add_subplot(gs[0, 1])
    ax_energy.plot(e1, label=f"seq1 ({len(seq1)} fr)", color="steelblue")
    ax_energy.plot(e2, label=f"seq2 ({len(seq2)} fr)", color="darkorange")
    step = max(1, len(path) // 20)
    for pi, pj in path[::step]:
        ax_energy.annotate(
            "", xy=(pj, e2[pj]), xytext=(pi, e1[pi]), arrowprops=dict(arrowstyle="-", color="gray", alpha=0.4, lw=0.8)
        )
    ax_energy.set_xlabel("Frame index")
    ax_energy.set_ylabel("Motion energy")
    ax_energy.set_title("Per-Frame Motion Energy + DTW Correspondences")
    ax_energy.legend(fontsize=8)

    # Panel 3 — landmark 8 X trajectory (the only thing that moves)
    ax_traj = fig.add_subplot(gs[1, 1])
    ax_traj.plot(x1, "o-", label=f"seq1 ({len(seq1)} fr)", color="steelblue", markersize=5)
    ax_traj.plot(x2, "s-", label=f"seq2 ({len(seq2)} fr)", color="darkorange", markersize=4)
    ax_traj.set_xlabel("Frame index")
    ax_traj.set_ylabel("X (normalised)")
    ax_traj.set_title("Index Fingertip X (landmark 8) — seq1 slow rise vs seq2 fast rise")
    ax_traj.legend(fontsize=8)

    out = "dtw_mockup.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
