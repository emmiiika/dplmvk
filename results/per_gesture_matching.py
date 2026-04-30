"""Per-gesto priemer matching skóre cez všetky stratégie."""

from collections import defaultdict

from common import parse, stats


def main():
    results = parse()

    gesture_scores = defaultdict(list)
    for user, ref, strat, score in results["MATCHING"]:
        gesture_scores[ref].append(score)

    print("=== Per-gesture MATCHING (priemer cez stratégie) ===")
    print(f"{'Gesture':<15}{'Mean%':>8}{'Median%':>9}{'Min%':>7}{'Max%':>7}{'N':>5}")
    rows = []
    for gesture, scores in gesture_scores.items():
        st = stats(scores)
        rows.append((gesture, st))
    rows.sort(key=lambda r: -r[1]["mean"])
    for gesture, st in rows:
        print(
            f"{gesture:<15}{st['mean']:>8.1f}{st['median']:>9.1f}"
            f"{st['min']:>7.1f}{st['max']:>7.1f}{st['n']:>5}"
        )


if __name__ == "__main__":
    main()
