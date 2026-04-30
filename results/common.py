"""Shared helpers for parsing strategy_comparison_all_final.txt."""

import os
import re

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(_SCRIPT_DIR, "..", "strategies", "strategy_comparison_all_final.txt")

STRATEGIES_ORDER = [
    "original",
    "relaxed_distances",
    "lower_activity",
    "cosine_heavy",
    "all_combined",
    "strict_distance",
    "euclid_heavy",
]

SECTION_BOUNDS = {
    "MATCHING": (3, 137),
    "NON-GESTURE": (137, 359),
    "CROSS": (359, None),
}

_ROW_RE = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+([\d.]+)%")


def parse(path=DEFAULT_INPUT):
    """Parse the comparison file into {section: [(user, ref, strategy, score%)]}."""
    with open(path) as f:
        lines = f.readlines()

    results = {sec: [] for sec in SECTION_BOUNDS}  # type : ignore
    for sec, (start, end) in SECTION_BOUNDS.items():
        end = end if end is not None else len(lines) + 1
        for i in range(start, min(end - 1, len(lines))):
            m = _ROW_RE.match(lines[i])
            if m:
                results[sec].append((m.group(1), m.group(2), m.group(3), float(m.group(4))))
    return results


def stats(values):
    if not values:
        return None
    sv = sorted(values)
    n = len(values)
    median = sv[n // 2] if n % 2 == 1 else (sv[n // 2 - 1] + sv[n // 2]) / 2
    return {
        "mean": sum(values) / n,
        "median": median,
        "min": min(values),
        "max": max(values),
        "n": n,
    }


def filter_scores(rows, strategy, predicate=None):
    """Return list of scores from rows matching strategy and optional predicate(row)."""
    return [s for (u, r, st, s) in rows if st == strategy and (predicate is None or predicate((u, r, st, s)))]
