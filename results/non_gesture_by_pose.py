"""Non-gesture: rozbor podľa typu pasivnej polohy (rest / right-up / both-up)."""

from common import STRATEGIES_ORDER, filter_scores, parse, stats

POSE_LABELS = [
    ("d_nic", "d_nic — ruky pri tele (rest)"),
    ("e_right_up", "e_right_up — pravá ruka zdvihnutá"),
    ("e_both_up", "e_both_up — obe ruky zdvihnuté"),
]


def report(label, video_stem, rows):
    selected = [(u, r, st, s) for (u, r, st, s) in rows if u == video_stem]
    n_pairs = len({(u, r) for (u, r, _, _) in selected})
    print(f"\n  {label}  (N={n_pairs} referencií)")
    print(f"  {'Strategy':<22}{'Mean%':>8}{'Median%':>9}{'Min%':>7}{'Max%':>7}")
    for strat in STRATEGIES_ORDER:
        scores = filter_scores(selected, strat)
        st = stats(scores)
        if st:
            print(
                f"  {strat:<22}{st['mean']:>8.1f}{st['median']:>9.1f}"
                f"{st['min']:>7.1f}{st['max']:>7.1f}"
            )


def main():
    rows = parse()["NON-GESTURE"]
    print("=== NON-GESTURE — rozbor podľa typu pasivnej polohy ===")
    for video_stem, label in POSE_LABELS:
        report(label, video_stem, rows)

    # Side-by-side mean comparison
    print("\n=== Priemer naprieč pózami ===")
    header = f"{'Strategy':<22}" + "".join(f"{label.split('—')[0].strip():>12}" for _, label in POSE_LABELS)
    print(header)
    for strat in STRATEGIES_ORDER:
        cells = [f"{strat:<22}"]
        for video_stem, _ in POSE_LABELS:
            scores = filter_scores(rows, strat, lambda r: r[0] == video_stem)
            cells.append(f"{sum(scores) / len(scores):>12.1f}" if scores else f"{'-':>12}")
        print("".join(cells))


if __name__ == "__main__":
    main()
