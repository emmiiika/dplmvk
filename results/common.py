"""Shared helpers for parsing strategy_comparison_all_final.txt."""

import os
import re

# Default input file lives in ../strategies relative to this module.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(_SCRIPT_DIR, "..", "strategies", "strategy_comparison_all_final.txt")

# Canonical strategy order for printing tables — keeps every script consistent.
STRATEGIES_ORDER = [
    "original",
    "relaxed_distances",
    "lower_activity",
    "cosine_heavy",
    "all_combined",
    "strict_distance",
    "euclid_heavy",
]

# Line ranges (0-indexed, half-open) for each section in the comparison file.
# Hard-coded because the file format is stable: header line, separator,
# data rows, then the next section header. End=None means "to EOF".
SECTION_BOUNDS = {
    "MATCHING": (3, 137),
    "NON-GESTURE": (137, 359),
    "CROSS": (359, None),
}

# Match a data row: user_video, reference, strategy, score%.
# Other columns (DTW dist, Eucl dist, ...) are ignored — only score% is parsed.
_ROW_RE = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+([\d.]+)%")


def parse(path=DEFAULT_INPUT):
    """Parse the comparison file into {section: [(user, ref, strategy, score%)]}."""
    with open(path) as f:
        lines = f.readlines()

    # One bucket per section; non-data lines (headers, separators) are simply skipped
    # because they don't match _ROW_RE.
    results = {sec: [] for sec in SECTION_BOUNDS}  # type : ignore
    for sec, (start, end) in SECTION_BOUNDS.items():
        end = end if end is not None else len(lines) + 1
        for i in range(start, min(end - 1, len(lines))):
            m = _ROW_RE.match(lines[i])
            if m:
                results[sec].append((m.group(1), m.group(2), m.group(3), float(m.group(4))))
    return results


def stats(values):
    """Return mean / median / min / max / n for a list of numbers, or None if empty."""
    if not values:
        return None
    sv = sorted(values)
    n = len(values)
    # For even-sized lists, median is the average of the two middle values.
    median = sv[n // 2] if n % 2 == 1 else (sv[n // 2 - 1] + sv[n // 2]) / 2
    return {
        "mean": sum(values) / n,
        "median": median,
        "min": min(values),
        "max": max(values),
        "n": n,
    }


def filter_scores(rows, strategy, predicate=None):
    """Return list of scores from rows matching strategy and optional predicate(row).

    `predicate` receives the full (user, ref, strategy, score) tuple so callers
    can filter by user prefix, reference name, specific (user, ref) pair, etc.
    """
    return [s for (u, r, st, s) in rows if st == strategy and (predicate is None or predicate((u, r, st, s)))]  # type: ignore
