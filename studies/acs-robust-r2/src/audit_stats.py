"""Distribution-level audit of the paired-J claims (study acs-robust-r2).

User critique (2026-08-08): mean-only paired comparisons can hide the real
shape — e.g. a policy that "wins" the mean by fixing k-NN's worst cases while
being slightly worse on typical seeds. For every headline comparison this
script reports, per (policy, reference, condition):

  n_pair             joint-success pairs
  mean / t_p         paired mean dJ (policy - ref) + t-test (the round-1/2 stat)
  med / wilcox_p     paired MEDIAN dJ + Wilcoxon signed-rank
  worse / sign_p     #pairs policy worse + exact two-sided sign test
  q10 / q90          dJ quantiles (tail asymmetry)
  dJ_typ             mean dJ over pairs with ref J <= ref median ("typical set")
  dJ_tail            mean dJ over pairs with ref J > ref q90 ("ref worst set")
  succ_pol/ref, mcnemar_p   paired success counts + exact McNemar

Usage: python audit_stats.py [--csv-out ../data/audit_stats.csv]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

R2 = "/workspace/studies/acs-robust-r2"
E = f"{R2}/data/eval"
K = f"{R2}/data/knnref"
OLD = "/workspace/studies/acs-robust-train/data/eval"
FRONT = "/workspace/studies/acs-c2-train/data/frontier_L.csv"

_front = pd.read_csv(FRONT)


def ref_front(k, L):
    return _front[(_front.k == k) & (_front.L == L)][["seed", "J", "success"]]


def ref_knn(k, L, N=20):
    p = f"{K}/k{k}_L{L:g}_N{N}_summary.csv"
    return pd.read_csv(p)[["seed", "J", "success"]]


def pol(label):
    return pd.read_csv(f"{E}/{label}_summary.csv")[["seed", "J", "success"]]


def pol_old(label):
    return pd.read_csv(f"{OLD}/{label}_summary.csv")[["seed", "J", "success"]]


def audit_one(tag, poldf, refdf, collect):
    p = poldf.set_index("seed")
    r = refdf.set_index("seed")
    c = p.index.intersection(r.index)
    p, r = p.loc[c], r.loc[c]
    # --- success side (paired, McNemar exact) ---
    b = int(((p.success == 0) & (r.success == 1)).sum())  # pol fails, ref ok
    cc = int(((p.success == 1) & (r.success == 0)).sum())  # pol ok, ref fails
    mp = stats.binomtest(min(b, cc), b + cc, 0.5).pvalue * 1.0 if (b + cc) else np.nan
    # binomtest two-sided already; keep exact two-sided of min under 0.5
    mp = stats.binomtest(b, b + cc, 0.5).pvalue if (b + cc) else np.nan
    # --- J side (joint successes) ---
    both = (p.success == 1) & (r.success == 1)
    d = (p.J[both] - r.J[both]).values
    rj = r.J[both].values
    n = len(d)
    if n < 6:
        return
    t_p = stats.ttest_1samp(d, 0.0).pvalue
    w_p = stats.wilcoxon(d).pvalue
    worse = int((d > 0).sum())
    s_p = stats.binomtest(worse, n, 0.5).pvalue
    typ = d[rj <= np.median(rj)]
    tail = d[rj > np.percentile(rj, 90)]
    row = dict(tag=tag, n_pair=n, mean=np.mean(d), t_p=t_p,
               med=np.median(d), wilcox_p=w_p, worse=worse, sign_p=s_p,
               q10=np.percentile(d, 10), q90=np.percentile(d, 90),
               dJ_typ=np.mean(typ), dJ_tail=np.mean(tail) if len(tail) else np.nan,
               succ_pol=int(p.success.sum()), succ_ref=int(r.success.sum()),
               n_all=len(c), mcnemar_p=mp)
    collect.append(row)
    print(f"{tag:28s} n={n:3d} mean{row['mean']:+7.1f}(t_p {t_p:.3g}) "
          f"med{row['med']:+7.1f}(w_p {w_p:.3g}) worse {worse:3d}/{n} (s_p {s_p:.3g}) "
          f"q10/90 {row['q10']:+5.0f}/{row['q90']:+5.0f} "
          f"typ{row['dJ_typ']:+6.1f} tail{row['dJ_tail']:+7.1f} "
          f"succ {row['succ_pol']}/{row['succ_ref']} of {row['n_all']} (mcn_p {mp:.3g})")


def pooled(tag, pairs, collect):
    ds = []
    for poldf, refdf in pairs:
        p = poldf.set_index("seed")
        r = refdf.set_index("seed")
        c = p.index.intersection(r.index)
        p, r = p.loc[c], r.loc[c]
        both = (p.success == 1) & (r.success == 1)
        ds.append((p.J[both] - r.J[both]).values)
    d = np.concatenate(ds)
    n = len(d)
    worse = int((d > 0).sum())
    row = dict(tag=tag, n_pair=n, mean=np.mean(d),
               t_p=stats.ttest_1samp(d, 0.0).pvalue, med=np.median(d),
               wilcox_p=stats.wilcoxon(d).pvalue, worse=worse,
               sign_p=stats.binomtest(worse, n, 0.5).pvalue,
               q10=np.percentile(d, 10), q90=np.percentile(d, 90),
               dJ_typ=np.nan, dJ_tail=np.nan, succ_pol=-1, succ_ref=-1,
               n_all=-1, mcnemar_p=np.nan)
    collect.append(row)
    print(f"{tag:28s} n={n:3d} mean{row['mean']:+7.1f}(t_p {row['t_p']:.3g}) "
          f"med{row['med']:+7.1f}(w_p {row['wilcox_p']:.3g}) "
          f"worse {worse:3d}/{n} (s_p {row['sign_p']:.3g})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-out", default=f"{R2}/data/audit_stats.csv")
    args = ap.parse_args()
    rows = []

    print("=== round-2 grid: C1 it80 / F1 it50 vs k12 & FC(k19) ===")
    c1p, f1p = [], []
    for L in (125, 250, 500):
        c1 = pol(f"C1_i80_L{L}_s32")
        f1 = pol(f"F1_i50_L{L}_s32")
        audit_one(f"C1i80 vs k12  L{L}", c1, ref_front(12, L), rows)
        audit_one(f"C1i80 vs FC   L{L}", c1, ref_front(19, L), rows)
        audit_one(f"F1i50 vs k12  L{L}", f1, ref_front(12, L), rows)
        c1p.append((c1, ref_front(12, L)))
        f1p.append((f1, ref_front(12, L)))
    pooled("C1i80 vs k12  POOLED", c1p, rows)
    pooled("F1i50 vs k12  POOLED", f1p, rows)

    print("\n=== round-1 grid re-audit: R1 it110 vs k12 & FC ===")
    r1p = []
    for L in (125, 250, 500):
        r1 = pol_old(f"R1_i110_L{L}_s32")
        audit_one(f"R1    vs k12  L{L}", r1, ref_front(12, L), rows)
        audit_one(f"R1    vs FC   L{L}", r1, ref_front(19, L), rows)
        r1p.append((r1, ref_front(12, L)))
    pooled("R1    vs k12  POOLED", r1p, rows)

    print("\n=== S1 big-n (500 paired seeds) ===")
    for L in (250, 500):
        audit_one(f"R1    vs k12  L{L} n500", pol(f"R1_i110_L{L}_s500"),
                  ref_knn(12, L), rows)
    audit_one("A60   vs k12  L250 n500", pol("A60_L250_s500"), ref_knn(12, 250), rows)

    print("\n=== L=75 compressed probe ===")
    for lab, name in (("R1_i110_L75_s32", "R1"), ("C1_i80_L75_s32", "C1i80"),
                      ("A60_L75_s32", "A60")):
        audit_one(f"{name:5s} vs k10  L75", pol(lab), ref_knn(10, 75), rows)
        audit_one(f"{name:5s} vs k12  L75", pol(lab), ref_knn(12, 75), rows)

    print("\n=== N-axis: same-k vs ratio-matched vs FC ===")
    for lab, name in (("C1_i80_N10L177_s32", "C1i80"), ("R1_i110_N10L177_s32", "R1")):
        audit_one(f"{name:5s} vs k6(.6N) N10", pol(lab), ref_knn(6, 177, 10), rows)
        audit_one(f"{name:5s} vs k8      N10", pol(lab), ref_knn(8, 177, 10), rows)
        audit_one(f"{name:5s} vs FC(k9)  N10", pol(lab), ref_knn(9, 177, 10), rows)
    n40_ks = [12, 24, 39] + [k for k in (16, 20, 28)
                             if os.path.exists(f"{K}/k{k}_L354_N40_summary.csv")]
    for lab, name in (("C1_i80_N40L354_s32", "C1i80"), ("R1_i110_N40L354_s32", "R1")):
        for k in sorted(n40_ks):
            note = {12: "same-k", 24: ".6N", 39: "FC"}.get(k, "")
            audit_one(f"{name:5s} vs k{k:<2d}{note:>7s} N40", pol(lab),
                      ref_knn(k, 354, 40), rows)

    df = pd.DataFrame(rows)
    df.to_csv(args.csv_out, index=False)
    print(f"\n-> {args.csv_out}")


if __name__ == "__main__":
    main()
