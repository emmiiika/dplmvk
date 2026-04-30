"""Porovnanie matching skóre podľa používateľa (user d vs user e)."""

from common import STRATEGIES_ORDER, filter_scores, parse, stats


def report(label, prefix, rows):
    matching = [(u, r, st, s) for (u, r, st, s) in rows if u.startswith(prefix)]
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
    print("=== MATCHING — User d vs User e ===")
    report("User d", "d_", rows)
    report("User e", "e_", rows)

    # Difference summary
    print("\n=== Rozdiel (user d mínus user e) ===")
    print(f"{'Strategy':<22}{'d%':>7}{'e%':>7}{'Δ':>7}")
    for strat in STRATEGIES_ORDER:
        d = filter_scores(rows, strat, lambda r: r[0].startswith("d_"))
        e = filter_scores(rows, strat, lambda r: r[0].startswith("e_"))
        if d and e:
            md = sum(d) / len(d)
            me = sum(e) / len(e)
            print(f"{strat:<22}{md:>7.1f}{me:>7.1f}{md - me:>+7.1f}")


if __name__ == "__main__":
    main()
