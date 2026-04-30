"""Porovnanie matching skóre podľa počtu rúk v referenčnom geste."""

from common import STRATEGIES_ORDER, filter_scores, parse, stats

ONE_HAND = {"oko", "oko_left", "oko_side", "slovo"}
TWO_HAND = {"dom", "hrad", "hrad_side", "pes", "pes_side"}


def report(label, refs, rows):
    matching = [(u, r, st, s) for (u, r, st, s) in rows if r in refs]
    n_pairs = len({(u, r) for (u, r, _, _) in matching})
    print(f"\n  {label}  (N={n_pairs} videí)")
    print(f"  {'Strategy':<22}{'Mean%':>8}{'Median%':>9}{'Min%':>7}{'Max%':>7}")
    for strat in STRATEGIES_ORDER:
        scores = filter_scores(matching, strat)
        st = stats(scores)
        if st:
            print(
                f"  {strat:<22}{st['mean']:>8.1f}{st['median']:>9.1f}"
                f"{st['min']:>7.1f}{st['max']:>7.1f}"
            )


def main():
    rows = parse()["MATCHING"]
    print("=== MATCHING — One-handed vs Two-handed ===")
    report("ONE-HANDED references (oko, oko_left, oko_side, slovo)", ONE_HAND, rows)
    report("TWO-HANDED references (dom, hrad, hrad_side, pes, pes_side)", TWO_HAND, rows)

    # Difference summary
    print("\n=== Rozdiel (1-ručné mínus 2-ručné) ===")
    print(f"{'Strategy':<22}{'1-hand%':>9}{'2-hand%':>9}{'Δ':>7}")
    for strat in STRATEGIES_ORDER:
        oh = filter_scores(rows, strat, lambda r: r[1] in ONE_HAND)
        th = filter_scores(rows, strat, lambda r: r[1] in TWO_HAND)
        if oh and th:
            mo = sum(oh) / len(oh)
            mt = sum(th) / len(th)
            print(f"{strat:<22}{mo:>9.1f}{mt:>9.1f}{mo - mt:>+7.1f}")


if __name__ == "__main__":
    main()
