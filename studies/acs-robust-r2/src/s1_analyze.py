"""S1 big-n reliability verdict (study acs-robust-r2).

Pre-registered secondary claim S1: at pooled L={250,500}, 500 paired seeds
each, R1-it110 (or the round-2 winner) failure count < k12's with two-sided
Fisher p < 0.05. Reports per-L and pooled failure counts, Wilson 95% CIs,
two-sided Fisher exact tests, plus descriptive J medians.

Inputs (produced by eval_c2.py + run_knn_refs.py in this study):
  data/eval/R1_i110_L{250,500}_s500_summary.csv     (policy arms)
  data/eval/A60_L250_s500_summary.csv               (optional third arm)
  data/knnref/k12_L{250,500}_N20_summary.csv        (fresh k12 seeds 1032+)
  + acs-c2-train frontier_L.csv                     (k12 seeds 1000-1031)

Usage: python s1_analyze.py [--policy-prefix R1_i110] [--with-a60]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

STUDY = "/workspace/studies/acs-robust-r2"
FRONTIER = "/workspace/studies/acs-c2-train/data/frontier_L.csv"


def wilson(fail, n, z=1.96):
    p = fail / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def load_k12(L):
    old = pd.read_csv(FRONTIER)
    old = old[(old.k == 12) & (old.L == L)][["seed", "t_fire", "J", "success"]]
    fresh_path = os.path.join(STUDY, "data", "knnref", f"k12_L{L:g}_N20_summary.csv")
    fresh = pd.read_csv(fresh_path)[["seed", "t_fire", "J", "success"]]
    both = pd.concat([old, fresh]).drop_duplicates("seed").sort_values("seed")
    return both.set_index("seed")


def load_pol(label):
    path = os.path.join(STUDY, "data", "eval", f"{label}_summary.csv")
    return pd.read_csv(path).set_index("seed")


def arm_vs_k12(name, pol, k12):
    common = pol.index.intersection(k12.index)
    pol, k12 = pol.loc[common], k12.loc[common]
    n = len(common)
    pf, kf = int((pol.success == 0).sum()), int((k12.success == 0).sum())
    lo, hi = wilson(pf, n)
    klo, khi = wilson(kf, n)
    odds, p = stats.fisher_exact([[pf, n - pf], [kf, n - kf]], alternative="two-sided")
    both = (pol.success == 1) & (k12.success == 1)
    dj = (pol.J[both] - k12.J[both]).values
    print(f"{name}: n={n}  fail {pf} (Wilson [{lo:.4f},{hi:.4f}]) "
          f"vs k12 fail {kf} ([{klo:.4f},{khi:.4f}])  Fisher p={p:.4g}")
    print(f"   J_med {pol[pol.success == 1].J.median():.1f} vs k12 "
          f"{k12[k12.success == 1].J.median():.1f}; paired dJ {np.mean(dj):+.1f} "
          f"(n={len(dj)})")
    return pf, kf, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-prefix", default="R1_i110")
    ap.add_argument("--with-a60", action="store_true")
    args = ap.parse_args()

    tot_pf = tot_kf = tot_n = 0
    for L in (250, 500):
        k12 = load_k12(L)
        pol = load_pol(f"{args.policy_prefix}_L{L}_s500")
        pf, kf, n = arm_vs_k12(f"{args.policy_prefix} L={L}", pol, k12)
        tot_pf, tot_kf, tot_n = tot_pf + pf, tot_kf + kf, tot_n + n
    lo, hi = wilson(tot_pf, tot_n)
    klo, khi = wilson(tot_kf, tot_n)
    odds, p = stats.fisher_exact([[tot_pf, tot_n - tot_pf],
                                  [tot_kf, tot_n - tot_kf]], alternative="two-sided")
    print(f"\nPOOLED: n={tot_n}  policy fail {tot_pf} ([{lo:.4f},{hi:.4f}]) "
          f"vs k12 fail {tot_kf} ([{klo:.4f},{khi:.4f}])  Fisher p={p:.4g}")
    verdict = tot_pf < tot_kf and p < 0.05
    print(f"S1 (policy fail < k12 fail AND two-sided Fisher p<0.05): "
          f"{'MET' if verdict else 'NOT MET'}")

    if args.with_a60:
        print("\n-- optional third arm --")
        arm_vs_k12("A60 L=250", load_pol("A60_L250_s500"), load_k12(250))


if __name__ == "__main__":
    main()
