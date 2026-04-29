"""Batch-generate DTW alignment visualizations for all video pairs."""

import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from visualize_dtw_alignment import visualize_with_options

MATCHING_PAIRS = [
    # user video stem      reference gesture
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

ALL_REFERENCES = [
    "oko",
    "oko_left",
    "oko_side",
    "dom",
    "slovo",
    "hrad",
    "hrad_side",
    "pes",
    "pes_side",
]

STATIC_PAIRS = [("d_nic", ref) for ref in ALL_REFERENCES]

PAIRS = MATCHING_PAIRS + STATIC_PAIRS

# CWD must be the code root so HandAnnotation can find hand_landmarker.task
CODE_ROOT = os.path.dirname(ROOT)
os.chdir(CODE_ROOT)

for user_stem, ref_gesture in PAIRS:
    print(f"\n=== {user_stem} vs {ref_gesture} ===")
    try:
        visualize_with_options(user_stem, ref_gesture, includeWristTrajectory=True, output_suffix=None)
    except Exception as e:
        print(f"ERROR: {e}")
