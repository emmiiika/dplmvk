"""Cross-gesture: vizuálne podobné páry vs vizuálne odlišné páry."""

from common import STRATEGIES_ORDER, filter_scores, parse, stats

# Pairs (user_video, wrong_reference) where the two gestures share visual cues:
# - similar hand position (e.g. at face), or same hand count and area
SIMILAR = {
    ("d_oko", "slovo"),
    ("d_slovo", "oko"),
    ("e_dom", "oko"),
    ("d_dom", "slovo"),
    ("e_hrad_1", "dom"),
    ("e_hrad_1", "pes"),
    ("e_pes", "hrad"),
    ("e_pes", "dom"),
}

# Pairs where the wrong reference is visually distinct from the user gesture
# (different hand count, different position area).
DIFFERENT = {
    ("e_oko", "dom"),
    ("e_slovo", "dom"),
}


def report(label, pairs, rows):
    selected = [(u, r, st, s) for (u, r, st, s) in rows if (u, r) in pairs]
    n_pairs = len({(u, r) for (u, r, _, _) in selected})
    print(f"\n  {label}  (N={n_pairs} párov)")
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
    rows = parse()["CROSS"]
    print("=== CROSS-GESTURE — vizuálne podobné vs vizuálne odlišné páry ===")
    report("Vizuálne podobné páry (oko↔slovo, hrad↔dom/pes, ...)", SIMILAR, rows)
    report("Vizuálne odlišné páry (e_oko↔dom, e_slovo↔dom)", DIFFERENT, rows)

    # Difference summary
    print("\n=== Rozdiel (podobné mínus odlišné) — vyšší rozdiel = horšia diskriminácia ===")
    print(f"{'Strategy':<22}{'Sim%':>7}{'Diff%':>7}{'Δ':>7}")
    for strat in STRATEGIES_ORDER:
        sim = filter_scores(rows, strat, lambda r: (r[0], r[1]) in SIMILAR)
        dif = filter_scores(rows, strat, lambda r: (r[0], r[1]) in DIFFERENT)
        if sim and dif:
            ms = sum(sim) / len(sim)
            md = sum(dif) / len(dif)
            print(f"{strat:<22}{ms:>7.1f}{md:>7.1f}{ms - md:>+7.1f}")


if __name__ == "__main__":
    main()
