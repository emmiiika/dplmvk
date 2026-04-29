import re
import sys

def extract(path):
    rows = {}
    for line in open(path):
        m = re.match(r'\s*(\S+)\s+(\S+)\s+(original)\s+([\d.]+)%', line)
        if m:
            rows[(m.group(1), m.group(2))] = float(m.group(4))
    return rows

before_path = sys.argv[1] if len(sys.argv) > 1 else 'strategies/strategy_comparison_latch_filter.txt'
after_path  = sys.argv[2] if len(sys.argv) > 2 else 'strategies/strategy_comparison_latch_filter_005.txt'
before = extract(before_path)
after  = extract(after_path)

keys = sorted(set(before) | set(after))
changed = [(k, before.get(k), after.get(k)) for k in keys if before.get(k) != after.get(k)]
same    = [k for k in keys if before.get(k) == after.get(k)]

print(f"Total pairs: {len(keys)}")
print(f"Unchanged:   {len(same)}")
print(f"Changed:     {len(changed)}")
print()
if changed:
    print(f"{'User':<20} {'Reference':<15} {'Before':>8} {'After':>8} {'Delta':>8}")
    print("-" * 65)
    for (u, r), b, a in sorted(changed, key=lambda x: abs((x[2] or 0)-(x[1] or 0)), reverse=True):
        delta = (a or 0) - (b or 0)
        print(f"{u:<20} {r:<15} {b:>7.1f}% {a:>7.1f}% {delta:>+7.1f}%")
else:
    print("No changes detected — latch filter had no effect on scores.")
