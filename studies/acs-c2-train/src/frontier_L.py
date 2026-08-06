"""Compute the k-NN/FC frontier at L in {125, 500} (and 250 as sanity) under
the C2 judge + J metric, from the predecessor's main-batch npz files.

Same judge/metric as j_metric_preview.py (phi>0.98/50, single r0-comp/300,
rel p2p sigma_p/300 < 5%; J = -sum reward to t_fire). Writes
data/frontier_L.csv (per seed) and prints the aggregate table.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

PRED = "/workspace/studies/acs-conv-knn"
STUDY = "/workspace/studies/acs-c2-train"
PHI_GOAL, W_A, W, EPS = 0.98, 50, 300, 0.05
KS = [8, 10, 12, 19]
LS = [125, 250, 500]


def t_fire_c2(phi, s, comp):
    pphi, ps, pcomp = pd.Series(phi), pd.Series(s), pd.Series(comp)
    align = (pphi.rolling(W_A).min() > PHI_GOAL).values
    coh = (pcomp.rolling(W).max() == 1).values
    band = ((ps.rolling(W).max() - ps.rolling(W).min()) / ps.rolling(W).mean()).values
    with np.errstate(invalid="ignore"):
        ok = align & coh & (band < EPS)
    hit = np.flatnonzero(ok)
    return int(hit[0]) if hit.size else -1


def main():
    rows = []
    for k in KS:
        for L in LS:
            for f in sorted(glob.glob(os.path.join(PRED, "data", "main", f"k{k:02d}_L{L}_s*.npz"))):
                z = np.load(f, allow_pickle=True)
                m = json.loads(str(z["meta"]))
                t = t_fire_c2(z["phi"], z["s_ent"], z["n_comp_r0"])
                r = z["reward"]
                J = float(-np.nansum(r[1:t + 1])) if t >= 0 else np.nan
                rows.append(dict(k=k, L=L, seed=m["seed"], t_fire=t, J=J,
                                 success=int(t >= 0)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(STUDY, "data", "frontier_L.csv"), index=False)

    def agg(d):
        det = d[d.success == 1]
        return pd.Series(dict(n=len(d), succ=d.success.sum(),
                              t_med=det.t_fire.median(), J_med=det.J.median()))
    pd.set_option("display.width", 200)
    print(df.groupby(["L", "k"]).apply(agg).round(1).to_string())


if __name__ == "__main__":
    main()
