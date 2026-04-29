import json
from types import SimpleNamespace

from Scoring import Scoring


def main():
    reference_path = "../videos/.annotated/oko - (360p)_annotated_handLandmarks.json"

    with open(reference_path, "r") as f:
        reference_seq = json.load(f)

    if not reference_seq:
        print("Reference sequence is empty.")
        return 1

    # Build a static user sequence: repeat the first frame for full duration.
    first_landmarks = reference_seq[0]["landmarks"]
    static_user_seq = []
    for i in range(len(reference_seq)):
        static_user_seq.append(
            {
                "timestamp": i * 0.05,
                "landmarks": first_landmarks,
            }
        )

    scorer = Scoring(webcamAnnotation=None, referenceAnnotation=SimpleNamespace(handLandmarksTimestamped=reference_seq))

    print("Static-rest penalty test")
    print("=" * 70)

    self_score = scorer.calculateScore(reference_seq)
    print(f"reference <- reference: {self_score:.6f}")

    static_score = scorer.calculateScore(static_user_seq)
    print(f"reference <- static-user: {static_score:.6f}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
