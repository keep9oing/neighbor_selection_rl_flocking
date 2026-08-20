"""Preview of the proposed headline scalar under C2:

    J = cumulative per-agent cost until convergence
      = -sum_{t=1..t_fire} reward_t   (study rollouts: is_training=False, so
        reward_t = -mean_i(dt*speed*|u_i| + rho*dt) = turn energy + cruise cost)

with C2 = [phi>0.98 hold 50] & [single r0-component hold 300] & [rel p2p(sigma_p)
over 300 < 5%].  Reported per family: success rate, t_fire, J, J_turn (J minus
cruise rho*dt*t_fire), realized mean degree at steady state.

Illustrates the degree-conditioned comparison for the evaluation-metric design
discussion; not a full evaluation pipeline.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

STUDY = "/workspace/studies/acs-conv-knn"
PHI_GOAL, W_A, W, EPS = 0.98, 50, 300, 0.05

FAMS = [
    ("main", "k08_L250_*.npz", "knn k=8"),
    ("main", "k10_L250_*.npz", "knn k=10"),
    ("main", "k12_L250_*.npz", "knn k=12"),
    ("main", "k19_L250_*.npz", "FC (k=19)"),
    ("disc", "th1.00_L250_*.npz", "disc R=125m"),
    ("nn_hardtopk", "nn_L250_*.npz", "NN ckpt (hardtopk10)"),
    ("stress_random", "rnd030_L250_*.npz", "random p=0.30"),
]


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
    for batch, pat, label in FAMS:
        for f in sorted(glob.glob(os.path.join(STUDY, "data", batch, pat))):
            z = np.load(f, allow_pickle=True)
            m = json.loads(str(z["meta"]))
            t = t_fire_c2(z["phi"], z["s_ent"], z["n_comp_r0"])
            r = z["reward"]
            J = float(-np.nansum(r[1:t + 1])) if t >= 0 else np.nan
            cruise = m["rho"] * m["dt"] * t if t >= 0 else np.nan
            deg = float(np.nanmedian(z["deg_mean"][-300:]))
            rows.append(dict(fam=label, seed=m["seed"], t_fire=t, J=J,
                             J_turn=J - cruise if t >= 0 else np.nan, deg_ss=deg))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(STUDY, "data", "j_metric_preview.csv"), index=False)

    def agg(d):
        det = d[d.t_fire >= 0]
        return pd.Series(dict(n=len(d), success=(d.t_fire >= 0).mean(),
                              t_med=det.t_fire.median(), J_med=det.J.median(),
                              J_turn_med=det.J_turn.median(),
                              deg_ss=d.deg_ss.median()))
    pd.set_option("display.width", 200)
    print(df.groupby("fam", sort=False).apply(agg).round(2).to_string())


if __name__ == "__main__":
    main()
