"""Grid judgment vs the pre-registered criteria (study acs-robust-train).

Reads this study's eval summary CSVs for one checkpoint evaluated at
L in {125,250,500} and the per-seed k-NN frontier references
(acs-c2-train data/frontier_L.csv), then reports:
  - per-L: success count, J_med, paired dJ vs k12 (common successes),
    paired t (+ FC comparison for context),
  - pooled paired dJ across all L (the pre-registered pooled test),
  - the PRIMARY / STRETCH criteria checklist.

Usage: python grid_judge.py --prefix R1_i110 [--suffix _s32] [--k 12]
Labels expected: {prefix}_L{125,250,500}{suffix} under data/eval/.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

STUDY = "/workspace/studies/acs-robust-train"
FRONTIER = "/workspace/studies/acs-c2-train/data/frontier_L.csv"
LS = [125, 250, 500]
# pre-registered per-L bars (PROBLEM.md): success >= k12's, J_med <= k12's + 5
K12_SUCC = {125: 31, 250: 31, 500: 32}
K12_JMED = {125: 155.0, 250: 160.0, 500: 165.6}


def load_eval(label):
    path = os.path.join(STUDY, "data", "eval", f"{label}_summary.csv")
    df = pd.read_csv(path)
    return df.set_index("seed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--suffix", default="_s32")
    ap.add_argument("--k", type=int, default=12)
    args = ap.parse_args()

    ref = pd.read_csv(FRONTIER)
    pooled_d = []
    per_l = {}
    print(f"=== {args.prefix}{args.suffix} vs k={args.k} (paired seeds) ===")
    for L in LS:
        pol = load_eval(f"{args.prefix}_L{L}{args.suffix}")
        r = ref[(ref.k == args.k) & (ref.L == L)].set_index("seed")
        fc = ref[(ref.k == 19) & (ref.L == L)].set_index("seed")
        common = pol.index.intersection(r.index)
        pol, r, fc = pol.loc[common], r.loc[common], fc.loc[common]
        n = len(common)
        succ = int(pol.success.sum())
        jmed = float(pol[pol.success == 1].J.median())
        both = (pol.success == 1) & (r.success == 1)
        d = (pol.J[both] - r.J[both]).values
        pooled_d.append(d)
        t, p = stats.ttest_1samp(d, 0.0) if len(d) > 2 else (np.nan, np.nan)
        both_fc = (pol.success == 1) & (fc.success == 1)
        dfc = (pol.J[both_fc] - fc.J[both_fc]).values
        tfc, pfc = stats.ttest_1samp(dfc, 0.0) if len(dfc) > 2 else (np.nan, np.nan)
        per_l[L] = dict(succ=succ, n=n, jmed=jmed)
        print(f"L={L}: succ {succ}/{n}  J_med {jmed:.1f} (k12 {K12_JMED[L]})  "
              f"paired dJ {np.mean(d):+.1f} (t={t:.2f}, p={p:.3g}, n={len(d)})  "
              f"| vs FC dJ {np.mean(dfc):+.1f} (p={pfc:.3g})")

    alld = np.concatenate(pooled_d)
    tp, pp = stats.ttest_1samp(alld, 0.0)
    print(f"\nPooled paired dJ vs k{args.k}: {np.mean(alld):+.1f} "
          f"(t={tp:.2f}, p={pp:.3g}, n={len(alld)})")

    prim_ok = all(per_l[L]["succ"] >= K12_SUCC[L] and per_l[L]["jmed"] <= K12_JMED[L] + 5
                  for L in LS) and (np.mean(alld) < 0 and pp < 0.05)
    stretch_ok = all(per_l[L]["jmed"] < K12_JMED[L] for L in LS)
    print("\nPRIMARY  (succ>=k12 & J_med<=k12+5 at every L, pooled dJ<0 p<.05):",
          "MET" if prim_ok else "NOT MET")
    for L in LS:
        print(f"  L={L}: succ {per_l[L]['succ']}>={K12_SUCC[L]} "
              f"{'OK' if per_l[L]['succ'] >= K12_SUCC[L] else 'FAIL'}; "
              f"J {per_l[L]['jmed']:.1f}<={K12_JMED[L] + 5:.1f} "
              f"{'OK' if per_l[L]['jmed'] <= K12_JMED[L] + 5 else 'FAIL'}")
    print("STRETCH  (strict J_med < k12 at every L):",
          "MET" if stretch_ok else "NOT MET")


if __name__ == "__main__":
    main()
