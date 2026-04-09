# AI GENERATED

import json
from types import SimpleNamespace

from Scoring import Scoring


def load_sequence(path):
    with open(path, "r") as f:
        return json.load(f)


def score(reference_seq, candidate_seq):
    reference_annotation = SimpleNamespace(handLandmarksTimestamped=reference_seq)
    scorer = Scoring(webcamAnnotation=None, referenceAnnotation=reference_annotation)
    return scorer.calculateScore(candidate_seq)


def main():
    files = {
        "oko_front": "../videa/.annotated/oko - (360p)_annotated_handLandmarks.json",
        "oko_side": "../videa/.annotated/oko - zboku - (360p)_annotated_handLandmarks.json",
        "oko_left": "../videa/.annotated/oko - lava - (360p)_annotated_handLandmarks.json",
    }

    sequences = {name: load_sequence(path) for name, path in files.items()}

    print("Pairwise scoring for Oko gesture variants")
    print("=" * 80)

    # One-way scores: reference <- candidate
    pairs = [
        ("oko_front", "oko_side"),
        ("oko_front", "oko_left"),
        ("oko_side", "oko_left"),
        ("oko_side", "oko_front"),
        ("oko_left", "oko_front"),
        ("oko_left", "oko_side"),
    ]

    for ref_name, cand_name in pairs:
        s = score(sequences[ref_name], sequences[cand_name])
        print(f"{ref_name} <- {cand_name}: {s:.6f}")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
