# AI GENERATED

import glob
import json
import os
import math
import copy
from types import SimpleNamespace

from Scoring import Scoring


def transform_landmarks_sequence(sequence, rotation_degrees=8.0, zoom_factor=1.08):
    """Return a deep-copied sequence with a small 2D rotation and zoom applied per hand."""
    transformed = copy.deepcopy(sequence)
    theta = math.radians(rotation_degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    for frame in transformed:
        landmarks = frame.get("landmarks", [])
        if not landmarks:
            continue

        # Landmarks are typically flattened; split into 21-point hand chunks.
        hand_count = len(landmarks) // 21
        for hand_idx in range(hand_count):
            start = hand_idx * 21
            end = start + 21
            hand = landmarks[start:end]
            if len(hand) < 21:
                continue

            wrist_x = hand[0]["x"]
            wrist_y = hand[0]["y"]
            wrist_z = hand[0]["z"]

            for lm in hand:
                # Translate to wrist-centered local coordinates.
                x = lm["x"] - wrist_x
                y = lm["y"] - wrist_y
                z = lm["z"] - wrist_z

                # Apply slight zoom.
                x *= zoom_factor
                y *= zoom_factor
                z *= zoom_factor

                # Apply 2D rotation around wrist in the image plane.
                xr = x * cos_t - y * sin_t
                yr = x * sin_t + y * cos_t

                # Translate back.
                lm["x"] = xr + wrist_x
                lm["y"] = yr + wrist_y
                lm["z"] = z + wrist_z

    return transformed


def main():
    files = sorted(glob.glob("../videa/.annotated/*_handLandmarks.json"))

    if not files:
        print("No reference landmark files found in ../videa/.annotated")
        return 1

    print("Self-scoring sanity check: reference sequence compared with itself")
    print("-" * 80)

    for path in files:
        with open(path, "r") as f:
            seq = json.load(f)

        reference_annotation = SimpleNamespace(handLandmarksTimestamped=seq)
        scorer = Scoring(webcamAnnotation=None, referenceAnnotation=reference_annotation)

        score = scorer.calculateScore(seq)
        print(f"SELF SCORE | {os.path.basename(path)} | frames={len(seq)} | score={score:.6f}")

    print("-" * 80)
    print("Perturbation test: one sequence rotated and slightly zoomed")

    # Use one real reference sequence and compare it with a transformed version.
    probe_path = files[0]
    with open(probe_path, "r") as f:
        probe_seq = json.load(f)

    transformed_seq = transform_landmarks_sequence(probe_seq, rotation_degrees=8.0, zoom_factor=1.08)

    reference_annotation = SimpleNamespace(handLandmarksTimestamped=probe_seq)
    scorer = Scoring(webcamAnnotation=None, referenceAnnotation=reference_annotation)
    transformed_score = scorer.calculateScore(transformed_seq)

    print(
        "TRANSFORM SCORE | "
        f"{os.path.basename(probe_path)} | "
        "transform=rotate(8deg)+zoom(1.08x) | "
        f"frames={len(probe_seq)} | score={transformed_score:.6f}"
    )

    print("-" * 80)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
