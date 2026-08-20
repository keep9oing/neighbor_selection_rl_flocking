"""Phase 4 statistics: compare trained-variant eval summaries vs the k-NN
frontier (paired seeds 1000..1031, C2 gate, J metric).

Inputs:
  - this study's data/eval/<label>_summary.csv (from eval_c2.py)
  - predecessor per-seed frontier: studies/acs-conv-knn/data/j_metric_preview.csv
    (FC (k=19), knn k=10/12 on the SAME seeds 1000-1031)

Reports per candidate label vs each reference:
  - success a/b + Wilson 95% CI
  - J median / mean on own successes
  - paired t-test on J over common-success seeds (primary, protocol-settled)
  - Welch t-test on J (backup)
  - Fisher exact on success counts

Usage: python compare_frontier.py --labels A_final B_final oldNN32
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

STUDY = "/workspace/studies/acs-c2-train"
PRED_CSV = "/workspace/studies/acs-conv-knn/data/j_metric_preview.csv"

REF_FAMS = {
    "FC(k=19)": "FC (k=19)",
    "knn12": "knn k=12",
    "knn10": "knn k=10",
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (c - h, c + h)


def load_ref():
    df = pd.read_csv(PRED_CSV)
    out = {}
    for short, fam in REF_FAMS.items():
        d = df[df.fam == fam][["seed", "t_fire", "J"]].copy()
        d["success"] = (d.t_fire >= 0).astype(int)
        out[short] = d.set_index("seed")
    return out


def load_label(label):
    p = os.path.join(STUDY, "data", "eval", f"{label}_summary.csv")
    d = pd.read_csv(p)[["seed", "t_fire", "J", "success"]].copy()
    return d.set_index("seed")


def compare(cand, cand_name, ref, ref_name):
    n_c, n_r = len(cand), len(ref)
    k_c, k_r = int(cand.success.sum()), int(ref.success.sum())
    lo, hi = wilson(k_c, n_c)
    both = cand.join(ref, lsuffix="_c", rsuffix="_r")
    both = both[(both.success_c == 1) & (both.success_r == 1)]
    dJ = both.J_c - both.J_r
    line = (f"{cand_name:12s} vs {ref_name:9s}: succ {k_c}/{n_c} "
            f"(CI {lo:.2f}-{hi:.2f}) vs {k_r}/{n_r}; "
            f"J_med {cand[cand.success == 1].J.median():6.1f} vs {ref[ref.success == 1].J.median():6.1f}; ")
    if len(both) >= 3:
        t_p, p_p = stats.ttest_rel(both.J_c, both.J_r)
        t_w, p_w = stats.ttest_ind(cand[cand.success == 1].J.dropna(),
                                   ref[ref.success == 1].J.dropna(), equal_var=False)
        line += (f"paired dJ mean {dJ.mean():+7.1f} (n={len(both)}, t={t_p:.2f}, p={p_p:.2g}); "
                 f"Welch p={p_w:.2g}")
    tbl = np.array([[k_c, n_c - k_c], [k_r, n_r - k_r]])
    _, p_f = stats.fisher_exact(tbl)
    line += f"; Fisher succ p={p_f:.2g}"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    args = ap.parse_args()
    refs = load_ref()
    for label in args.labels:
        cand = load_label(label)
        det = cand[cand.success == 1]
        print(f"\n### {label}: success {int(cand.success.sum())}/{len(cand)}, "
              f"t_conv med {det.t_fire.median():.0f}, J med {det.J.median():.1f}, "
              f"J mean {det.J.mean():.1f}")
        for rname, ref in refs.items():
            compare(cand, label, ref, rname)


if __name__ == "__main__":
    main()
