"""Batch-generate DTW alignment visualizations for all video pairs."""
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from dtwAlignment.visualize_dtw_alignment import visualize

PAIRS = [
    # Matching gestures
    ("d_oko",       "oko"),
    ("d_oko_left",  "oko_left"),
    ("d_oko_side",  "oko_side"),
    ("d_dom",       "dom"),
    ("d_slovo",     "slovo"),
    ("e_oko",       "oko"),
    ("e_oko_left",  "oko_left"),
    ("e_oko_side",  "oko_side"),
    ("e_dom",       "dom"),
    ("e_slovo",     "slovo"),
    # Static user
    ("d_nic",       "oko"),
    ("d_nic",       "oko_left"),
    ("d_nic",       "oko_side"),
    ("d_nic",       "dom"),
    ("d_nic",       "slovo"),
    # Cross-gesture
    ("d_oko",       "slovo"),
    ("d_slovo",     "oko"),
    ("e_oko",       "dom"),
    ("e_dom",       "oko"),
    ("d_dom",       "slovo"),
    ("e_slovo",     "dom"),
]

out_dir = os.path.join(ROOT, "dtw_visualizations")
os.makedirs(out_dir, exist_ok=True)
# Change to the output dir so visualize() saves PNGs there
os.chdir(ROOT)

for user_stem, ref_gesture in PAIRS:
    print(f"\n=== {user_stem} vs {ref_gesture} ===")
    try:
        visualize(user_stem, ref_gesture)
        png = f"dtw_alignment_{user_stem}_vs_{ref_gesture}.png"
        if os.path.isfile(png):
            os.rename(png, os.path.join(out_dir, png))
    except Exception as e:
        print(f"ERROR: {e}")
