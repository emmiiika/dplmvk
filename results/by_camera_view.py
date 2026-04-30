"""Porovnanie matching skóre podľa pohľadu kamery (front vs side)."""

from common import STRATEGIES_ORDER, filter_scores, parse, stats

FRONT_VIEW = {"oko", "oko_left", "dom", "slovo", "hrad", "pes"}
SIDE_VIEW = {"oko_side", "hrad_side", "pes_side"}


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
    print("=== MATCHING — Front view vs Side view ===")
    report("FRONT-VIEW references (oko, oko_left, dom, slovo, hrad, pes)", FRONT_VIEW, rows)
    report("SIDE-VIEW references (oko_side, hrad_side, pes_side)", SIDE_VIEW, rows)

    # Difference summary
    print("\n=== Rozdiel (front mínus side) ===")
    print(f"{'Strategy':<22}{'Front%':>8}{'Side%':>7}{'Δ':>7}")
    for strat in STRATEGIES_ORDER:
        fr = filter_scores(rows, strat, lambda r: r[1] in FRONT_VIEW)
        sd = filter_scores(rows, strat, lambda r: r[1] in SIDE_VIEW)
        if fr and sd:
            mf = sum(fr) / len(fr)
            ms = sum(sd) / len(sd)
            print(f"{strat:<22}{mf:>8.1f}{ms:>7.1f}{mf - ms:>+7.1f}")


if __name__ == "__main__":
    main()
