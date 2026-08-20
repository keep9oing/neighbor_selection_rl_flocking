"""Meeting-material statistics, recomputed from per-seed summary CSVs (read-only).

Outputs figures/meeting/stats.json with:
  - fail table: fixed-k k=8..19 x L{125,250,500}, n=500 (fail%, J_med success)
  - policy rows: C1_i80 / R1_i110 / A60 per L (fail%, J_med, CVaR10)
  - paired C1-vs-FC and R1-vs-FC per L (co-success dJ median, Wilcoxon, worse frac)
  - phase stats: rank_dev/deg early-vs-steady per-seed paired tests
  - straggler candidates: seeds where k12@L250 fails [19+1] and C1 succeeds
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R3 = "/workspace/studies/acs-robust-r3-stress/data"
R2 = "/workspace/studies/acs-robust-r2/data"
OUT = "/workspace/figures/meeting/stats.json"


def knn_csv(k, L):
    # k12 @ L250/L500 full 500-seed lanes live in r2 (r3 dir holds 8 gate seeds only)
    if k == 12 and L in (250, 500):
        return f"{R2}/knnref/k12_L{L}_N20_summary.csv"
    return f"{R3}/knnref/k{k}_L{L}_N20_summary.csv"


def pol_csv(label, L):
    if label == "C1":
        return f"{R3}/eval/C1_i80_L{L}_s500_summary.csv"
    if label == "R1":
        d = R3 if L == 125 else R2
        return f"{d}/eval/R1_i110_L{L}_s500_summary.csv"
    if label == "A60":
        return f"{R2}/eval/A60_L250_s500_summary.csv"
    raise KeyError(label)


def arm_stats(df):
    det = df[df.success == 1]
    n = len(df)
    fails = int(n - df.success.sum())
    js = det.J.dropna().values
    cvar = float(np.mean(np.sort(js)[-max(1, int(round(0.1 * len(js)))):]))
    return dict(n=n, fails=fails, fail_pct=100.0 * fails / n,
                J_med=float(det.J.median()), cvar10=cvar,
                t_med=float(det.t_fire.median()))


def paired(dfa, dfb):
    """dJ = J_a - J_b on co-success seeds; + discordant fail counts (McNemar cells)."""
    m = dfa.merge(dfb, on="seed", suffixes=("_a", "_b"))
    co = m[(m.success_a == 1) & (m.success_b == 1)]
    d = (co.J_a - co.J_b).values
    w = wilcoxon(d)
    a_only_fail = int(((m.success_a == 0) & (m.success_b == 1)).sum())
    b_only_fail = int(((m.success_a == 1) & (m.success_b == 0)).sum())
    return dict(n_co=len(co), dJ_med=float(np.median(d)),
                worse=int((d > 0).sum()), wilcox_p=float(w.pvalue),
                fail_a_only=a_only_fail, fail_b_only=b_only_fail)


res = {"knn": {}, "policy": {}, "paired_vs_fc": {}, "phase": {}, "paired_vs_k12": {}}

for L in (125, 250, 500):
    for k in range(8, 20):
        df = pd.read_csv(knn_csv(k, L))
        assert len(df) == 500, (k, L, len(df))
        res["knn"][f"k{k}_L{L}"] = arm_stats(df)

for lab in ("C1", "R1"):
    for L in (125, 250, 500):
        df = pd.read_csv(pol_csv(lab, L))
        assert len(df) == 500
        res["policy"][f"{lab}_L{L}"] = arm_stats(df)
res["policy"]["A60_L250"] = arm_stats(pd.read_csv(pol_csv("A60", 250)))

# paired policy-vs-FC and policy-vs-k12
for lab in ("C1", "R1"):
    for L in (125, 250, 500):
        p = pd.read_csv(pol_csv(lab, L))
        res["paired_vs_fc"][f"{lab}_L{L}"] = paired(p, pd.read_csv(knn_csv(19, L)))
        res["paired_vs_k12"][f"{lab}_L{L}"] = paired(p, pd.read_csv(knn_csv(12, L)))

# phase statistics from per-seed early/steady columns (policy lanes only)
for lab in ("C1", "R1"):
    for L in (125, 250, 500):
        df = pd.read_csv(pol_csv(lab, L))
        rd = df[["rank_dev_early", "rank_dev_ss"]].dropna()
        dg = df[["deg_early", "deg_ss"]].dropna()
        w_rd = wilcoxon(rd.rank_dev_early - rd.rank_dev_ss)
        w_dg = wilcoxon(dg.deg_early - dg.deg_ss)
        res["phase"][f"{lab}_L{L}"] = dict(
            n=len(rd),
            rd_early_med=float(rd.rank_dev_early.median()),
            rd_ss_med=float(rd.rank_dev_ss.median()),
            rd_frac_higher_early=float((rd.rank_dev_early > rd.rank_dev_ss).mean()),
            rd_wilcox_p=float(w_rd.pvalue),
            deg_early_med=float(dg.deg_early.median()),
            deg_ss_med=float(dg.deg_ss.median()),
            deg_frac_lower_early=float((dg.deg_early < dg.deg_ss).mean()),
            deg_wilcox_p=float(w_dg.pvalue),
        )

# straggler candidates: k12@L250 fails, C1 succeeds, final split is [19+1]
k12 = pd.read_csv(knn_csv(12, 250)).set_index("seed")
c1 = pd.read_csv(pol_csv("C1", 250)).set_index("seed")
cand = [s for s in k12.index if k12.success[s] == 0 and c1.success[s] == 1]


def final_split(seed):
    z = np.load(f"{R2}/knnref/k12_L250_N20/k12_L250_N20_s{seed}.npz", allow_pickle=True)
    pos = z["pos_snaps"][-1]
    d = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    adj = d < 60.0
    n = len(pos)
    lab = -np.ones(n, int)
    c = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack = [i]
        lab[i] = c
        while stack:
            u = stack.pop()
            for v in np.where(adj[u])[0]:
                if lab[v] < 0:
                    lab[v] = c
                    stack.append(v)
        c += 1
    sizes = sorted(np.bincount(lab).tolist(), reverse=True)
    return sizes, lab


strag = []
for s in cand:
    sizes, lab = final_split(s)
    strag.append(dict(seed=int(s), split=sizes))
res["straggler_candidates"] = strag
res["n_k12_fail_c1_succ"] = len(cand)

with open(OUT, "w") as f:
    json.dump(res, f, indent=1)

print("k12/C1 discordant seeds (k12 fail, C1 succ):", len(cand))
print("splits:", [(d["seed"], d["split"]) for d in strag[:12]])
print("C1 vs FC paired:", {k: (round(v["dJ_med"], 1), f'{v["wilcox_p"]:.1e}', v["worse"], v["n_co"])
                           for k, v in res["paired_vs_fc"].items() if k.startswith("C1")})
print("R1 vs FC paired:", {k: (round(v["dJ_med"], 1), f'{v["wilcox_p"]:.2f}')
                           for k, v in res["paired_vs_fc"].items() if k.startswith("R1")})
print("phase C1_L250:", res["phase"]["C1_L250"])
print("saved", OUT)
