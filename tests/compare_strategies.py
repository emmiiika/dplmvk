"""Compare all recorded user videos against reference gestures using every scoring strategy.

Produces a summary table with score percentages, raw distances, and key metrics.
Includes: matching gestures, the static 'd_nic' video, and cross-gesture (mismatched) comparisons.
"""

import os
import sys
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from HandAnnotation import HandAnnotation
from Scoring import Scoring, SCORING_STRATEGIES

REFERENCE_FOLDER = "../videa"
ANNOTATED_FOLDER = "../videa/.annotated"
RECORDED_FOLDER = "../videa/.recorded"

# Reference gestures available (stem names matching .mp4 files in REFERENCE_FOLDER)
GESTURES = ["oko", "oko_left", "oko_side", "dom", "slovo", "hrad", "hrad_2", "hrad_side", "pes", "pes_side"]

# Map recorded filename stems to the reference gesture they attempted.
# Naming convention: {user}_{gesture}[_variant].avi
MATCHING_PAIRS = [
    # user video stem          -> reference gesture
    ("d_oko", "oko"),
    ("d_oko_left", "oko_left"),
    ("d_oko_side", "oko_side"),
    ("d_dom", "dom"),
    ("d_slovo", "slovo"),
    ("e_oko", "oko"),
    ("e_oko_left", "oko_left"),
    ("e_oko_side", "oko_side"),
    ("e_dom", "dom"),
    ("e_slovo", "slovo"),
    ("e_hrad_1", "hrad"),
    ("e_hrad_2", "hrad"),
    ("e_hrad_side", "hrad_side"),
    ("e_pes", "pes"),
    ("e_pes_side_1", "pes_side"),
    ("e_pes_side_2", "pes_side"),
]

# Static user (does nothing) — compare against every reference
STATIC_VIDEO = "d_nic"

# Cross-gesture comparisons (user video vs WRONG reference)
CROSS_PAIRS = [
    ("d_oko", "slovo"),
    ("d_slovo", "oko"),
    ("e_oko", "dom"),
    ("e_dom", "oko"),
    ("d_dom", "slovo"),
    ("e_slovo", "dom"),
    ("e_hrad_1", "dom"),
    ("e_hrad_1", "pes"),
    ("e_pes", "hrad"),
    ("e_pes", "dom"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

_annotation_cache = {}  # type: ignore


def get_annotation(video_path):
    """Load or create annotation with caching."""
    video_path = os.path.abspath(video_path)
    if video_path in _annotation_cache:
        return _annotation_cache[video_path]

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

    _annotation_cache[video_path] = annotation
    return annotation


def resolve_recorded(stem):
    for ext in (".mp4", ".avi"):
        p = os.path.abspath(os.path.join(RECORDED_FOLDER, f"{stem}{ext}"))
        if os.path.isfile(p):
            return p
    return os.path.abspath(os.path.join(RECORDED_FOLDER, f"{stem}.mp4"))


def resolve_reference(gesture):
    return os.path.abspath(os.path.join(REFERENCE_FOLDER, f"{gesture}.mp4"))


def score_pair(user_path, ref_path, strategy_name):
    """Score a user video against a reference using a given strategy. Returns dict of metrics."""
    user_ann = get_annotation(user_path)
    ref_ann = get_annotation(ref_path)

    scorer = Scoring(user_ann, ref_ann, strategy=strategy_name)
    user_seq = user_ann.handLandmarksTimestamped
    ref_seq = ref_ann.handLandmarksTimestamped

    trimmed_ref = scorer._trimReferenceSequenceByMarkers(ref_seq)
    user_frames = scorer._extractPerHandArrays(user_seq)
    ref_frames = scorer._extractPerHandArrays(trimmed_ref)

    if len(user_frames) == 0 or len(ref_frames) == 0:
        return {
            "score": 0.0,
            "dtw_dist": float("inf"),
            "eucl_dist": float("inf"),
            "cos_sim": 0.0,
            "activity": 0.0,
            "u_frames": len(user_frames),
            "r_frames": len(ref_frames),
        }

    hand_weights = scorer._calculateHandMotionWeights(ref_frames)
    ref_energy = scorer._averageMotionEnergy(ref_frames, hand_weights)
    user_energy = scorer._averageMotionEnergy(user_frames, hand_weights)

    # Trim idle/jitter frames from edges of user sequence (matches _calculateSequenceSimilarity)
    user_frames = scorer._trimLowMotionEdges(user_frames, hand_weights)

    s = scorer.strategy
    dtw_dist, warping_path = scorer._dtwWithPath(user_frames, ref_frames, hand_weights)
    aligned_user = [user_frames[i] for (i, j) in warping_path]
    aligned_ref = [ref_frames[j] for (i, j) in warping_path]

    eucl_dist = scorer._averageEuclideanDistance(aligned_user, aligned_ref, hand_weights)
    cos_sim = scorer._averageCosineSimilarity(aligned_user, aligned_ref, hand_weights)

    eucl_sim = scorer._distanceToSimilarity(eucl_dist, maxPossibleDistance=s["euclideanMaxDistance"])
    eucl_cos_weight = s["euclideanWeight"] + s["cosineWeight"]
    combined = (
        (s["euclideanWeight"] * eucl_sim + s["cosineWeight"] * cos_sim) / eucl_cos_weight
        if eucl_cos_weight > 0
        else 0.0
    )

    activity = 1.0
    if ref_energy > 0.003:
        expected = ref_energy * s["activityThreshold"]
        if expected > 0:
            activity = max(0.0, min(1.0, user_energy / expected))

    # Wrist max-displacement penalty (mirrors _calculateSequenceSimilarity)
    raw_ref = scorer._extractRawWristSequences(trimmed_ref)
    raw_user = scorer._extractRawWristSequences(user_seq)
    ref_max_disp = scorer._wristMaxDispFromRawSeq(raw_ref, hand_weights)
    if ref_max_disp is not None and ref_max_disp > 0.05:
        user_max_disp = scorer._wristMaxDispFromRawSeq(raw_user, hand_weights)
        disp_factor = min(1.0, user_max_disp / ref_max_disp) if user_max_disp is not None else 0.0
        activity *= disp_factor

    final = combined * activity

    return {
        "score": final,
        "dtw_dist": dtw_dist,
        "eucl_dist": eucl_dist,
        "cos_sim": cos_sim,
        "activity": activity,
        "u_frames": len(user_frames),
        "r_frames": len(ref_frames),
    }


# ── Table printing ───────────────────────────────────────────────────────────

STRATEGIES = list(SCORING_STRATEGIES.keys())

SEP = "-" * 130


def print_header(title):
    print(f"\n{'=' * 130}")
    print(f"  {title}")
    print(f"{'=' * 130}")


def print_table_header():
    print(
        f"{'User Video':<20} {'Reference':<12} {'Strategy':<20} {'Score%':>7} {'DTW dist':>9} {'Eucl dist':>10} {'Cos sim':>8} {'Activity':>9} {'Frames':>10}"
    )
    print(SEP)


def print_row(user_stem, ref_gesture, strategy, m):
    frames_str = f"{m['u_frames']}/{m['r_frames']}"
    print(
        f"{user_stem:<20} {ref_gesture:<12} {strategy:<20} {m['score']*100:6.1f}% {m['dtw_dist']:9.4f} {m['eucl_dist']:10.4f} {m['cos_sim']:8.4f} {m['activity']:9.3f} {frames_str:>10}"
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    # Suppress verbose Scoring prints
    import builtins

    _real_print = builtins.print
    scoring_verbose = False

    def quiet_print(*args, **kwargs):
        if not scoring_verbose:
            msg = " ".join(str(a) for a in args)
            # Suppress lines from Scoring internals
            if any(
                tag in msg
                for tag in [
                    "Scoring strategy:",
                    "Calculating score",
                    "Hand weights",
                    "Frames used",
                    "Motion energy",
                    "Activity factor",
                    "DTW similarity",
                    "Euclidean similarity",
                    "Cosine similarity",
                    "Calculated similarity",
                    "Reference trim",
                    "User frames after trim",
                    "Warning:",
                    "Creating annotated video",
                    "Trim range:",
                    "VideoWriter initialized",
                    "Annotated video created",
                    "Landmarks saved",
                    "Loaded ",
                    "No hand movement",
                    "✓",
                ]
            ):
                return
        _real_print(*args, **kwargs)

    builtins.print = quiet_print

    # ── Section 1: Matching gesture pairs ──
    print_header("MATCHING GESTURES  (user video vs correct reference)")
    print_table_header()

    for user_stem, ref_gesture in MATCHING_PAIRS:
        user_path = resolve_recorded(user_stem)
        ref_path = resolve_reference(ref_gesture)
        if not os.path.isfile(user_path) or not os.path.isfile(ref_path):
            _real_print(f"  SKIP: {user_stem} / {ref_gesture} (file missing)")
            continue
        for strat in STRATEGIES:
            m = score_pair(user_path, ref_path, strat)
            print_row(user_stem, ref_gesture, strat, m)
        _real_print(SEP)

    # ── Section 2: Static user (d_nic) vs every reference ──
    print_header("STATIC USER 'd_nic'  (no gesture — should score low)")
    print_table_header()

    static_path = resolve_recorded(STATIC_VIDEO)
    for ref_gesture in GESTURES:
        ref_path = resolve_reference(ref_gesture)
        if not os.path.isfile(static_path) or not os.path.isfile(ref_path):
            _real_print(f"  SKIP: {STATIC_VIDEO} / {ref_gesture} (file missing)")
            continue
        for strat in STRATEGIES:
            m = score_pair(static_path, ref_path, strat)
            print_row(STATIC_VIDEO, ref_gesture, strat, m)
        _real_print(SEP)

    # ── Section 3: Cross-gesture (wrong reference) ──
    print_header("CROSS-GESTURE  (user video vs WRONG reference — should score low)")
    print_table_header()

    for user_stem, ref_gesture in CROSS_PAIRS:
        user_path = resolve_recorded(user_stem)
        ref_path = resolve_reference(ref_gesture)
        if not os.path.isfile(user_path) or not os.path.isfile(ref_path):
            _real_print(f"  SKIP: {user_stem} / {ref_gesture} (file missing)")
            continue
        for strat in STRATEGIES:
            m = score_pair(user_path, ref_path, strat)
            print_row(user_stem, ref_gesture, strat, m)
        _real_print(SEP)

    builtins.print = _real_print


if __name__ == "__main__":
    main()
