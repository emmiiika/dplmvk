"""Priemerné skóre podľa kategórie (matching / non-gesture / cross) pre každú stratégiu."""

from common import STRATEGIES_ORDER, filter_scores, parse, stats


def main():
    results = parse()

    for section in ["MATCHING", "NON-GESTURE", "CROSS"]:
        print(f"\n=== {section} ===")
        print(f"{'Strategy':<22}{'Mean%':>8}{'Median%':>9}{'Min%':>7}{'Max%':>7}{'N':>5}")
        for strat in STRATEGIES_ORDER:
            scores = filter_scores(results[section], strat)
            st = stats(scores)
            if st:
                print(
                    f"{strat:<22}{st['mean']:>8.1f}{st['median']:>9.1f}"
                    f"{st['min']:>7.1f}{st['max']:>7.1f}{st['n']:>5}"
                )

    # Discriminative power: gap between matching and non-matching means
    print("\n=== Discriminative power: matching mean - non-matching mean ===")
    print(f"{'Strategy':<22}{'Match%':>8}{'NonGest%':>10}{'Cross%':>8}{'Gap(M-NG)':>11}{'Gap(M-Cr)':>11}")
    for strat in STRATEGIES_ORDER:
        m = filter_scores(results["MATCHING"], strat)
        ng = filter_scores(results["NON-GESTURE"], strat)
        cr = filter_scores(results["CROSS"], strat)
        if m and ng and cr:
            mm = sum(m) / len(m)
            ngm = sum(ng) / len(ng)
            crm = sum(cr) / len(cr)
            print(f"{strat:<22}{mm:>8.1f}{ngm:>10.1f}{crm:>8.1f}{mm - ngm:>11.1f}{mm - crm:>11.1f}")


if __name__ == "__main__":
    main()
