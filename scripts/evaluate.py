"""Reproduce the reported statistics from the command line.

    python -m scripts.evaluate                      # original synthetic demo
    python -m scripts.evaluate --csv export.csv --control control --treatment new_checkout
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from experimentlab.core import (generate, prepare, results_table, sample_size, segment_analysis, simulate_peeking,
                                simulate_tests)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--control", default="A")
    ap.add_argument("--treatment", default="B")
    ap.add_argument("--alpha", type=float, default=.05)
    a = ap.parse_args()
    d, rep = prepare(pd.read_csv(a.csv) if a.csv else generate())
    if a.csv:
        print(rep.as_frame().to_string(index=False), "\n")
    from experimentlab.core import compare
    r = compare(d, a.control, a.treatment, a.alpha)
    print(results_table(r).to_string(index=False))
    print("\nVerdict:", r["verdict"]["title"], "-", r["verdict"]["text"])
    for seg in rep.segments:
        s = segment_analysis(d, a.control, a.treatment, seg, a.alpha)
        print(f"\nSegments by {seg} (heterogeneity p = {s['q_p']:.3f})")
        print(s["table"].round(4).to_string(index=False))
    n = sample_size(.10, .12)
    print("\nPlanner check (10% baseline, +12% lift):", n, "per arm;",
          json.dumps({"simulated_power": simulate_tests(.10, .12, n, n)["rejection_rate"],
                      "false_positive_rate": simulate_tests(.10, 0, n, n)["rejection_rate"],
                      "peeking_10_looks": simulate_peeking(.10, n)["any_look"]}))


if __name__ == "__main__":
    main()
