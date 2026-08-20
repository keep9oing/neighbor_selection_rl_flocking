"""Evaluate candidate convergence criteria against the main sweep.

C0 (current env criterion): sigma_p < 0.7*r0 AND sigma_v < 0.1 AND trailing-50
    peak-to-peak(sigma_p) < 0.1 AND peak-to-peak(sigma_v) < 0.2.
    (already computed as t_env in summary CSV; recomputed here for completeness)

C1 (proposed, topology-agnostic "settled flock"):
    a) alignment: phi > PHI_GOAL for the trailing 50 steps         (heading consensus)
       PHI_GOAL default 0.97 = the repo's own Vicsek alignment_goal and a standard
       order-parameter convergence level in the swarm literature (user-approved
       relaxation from sigma_v<0.1 == phi>0.99998; earlier analyses used 0.99).
    b) cohesion:  r0-proximity graph has 1 component, trailing 500 (no fragmentation)
    c) stationarity: |sigma_p(t) - sigma_p(t-500)| / sigma_p(t-500) < 2%
                                                                   (spread settled,
                                                                    level-free)
    t_C1 = first t satisfying a&b&c. No absolute spatial level anywhere.

Outputs data/criteria_main.csv + aggregate table to stdout.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

STUDY = "/workspace/studies/acs-conv-knn"


def eval_run(path, phi_goal=0.97):
    z = np.load(path, allow_pickle=True)
    m = json.loads(str(z["meta"]))
    s, v, phi, comp = z["s_ent"], z["v_ent"], z["phi"], z["n_comp_r0"]
    T = m["steps_done"]
    ps, pv, pphi, pcomp = (pd.Series(x) for x in (s, v, phi, comp))

    # C0 — current env criterion
    p2p_s = (ps.rolling(50).max() - ps.rolling(50).min()).values
    p2p_v = (pv.rolling(50).max() - pv.rolling(50).min()).values
    c0 = (s < m["entropy_p_goal"]) & (v < m["entropy_v_goal"]) \
        & (p2p_s < 0.1) & (p2p_v < 0.2)
    hit0 = np.flatnonzero(c0)

    # C1 — proposed
    align_ok = (pphi.rolling(50).min() > phi_goal).values
    coh_ok = (pcomp.rolling(500).max() == 1).values
    s_prev = np.full_like(s, np.nan)
    s_prev[500:] = s[:-500]
    with np.errstate(invalid="ignore", divide="ignore"):
        flat_ok = np.abs(s - s_prev) / np.maximum(s_prev, 1e-9) < 0.02
    c1 = align_ok & coh_ok & flat_ok
    hit1 = np.flatnonzero(c1)

    single = bool((comp[T + 1 - 500:T + 1] == 1).all())
    return dict(
        k=m.get("k"), L=m["initial_position_bound"], seed=m["seed"], single=single,
        t_C0=int(hit0[0]) if hit0.size else -1,
        t_C1=int(hit1[0]) if hit1.size else -1,
        sp_at_C1=float(s[hit1[0]]) if hit1.size else np.nan,
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="main")
    ap.add_argument("--phi", type=float, default=0.97)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(STUDY, "data", args.batch, "*.npz")))
    df = pd.DataFrame([eval_run(f, phi_goal=args.phi) for f in files])
    df.to_csv(os.path.join(STUDY, "data", f"criteria_{args.batch}.csv"), index=False)
    print(f"[batch={args.batch} phi_goal={args.phi}]")
    pd.set_option("display.width", 200)

    def agg(d):
        return pd.Series(dict(
            n=len(d),
            P_single=d.single.mean(),
            C0_pass=(d.t_C0 >= 0).mean(),
            C1_pass=(d.t_C1 >= 0).mean(),
            C1_pass_coh=(d[d.single].t_C1 >= 0).mean() if d.single.any() else np.nan,
            C1_false_pos=(d[~d.single].t_C1 >= 0).mean() if (~d.single).any() else np.nan,
            t_C0_med=d[d.t_C0 >= 0].t_C0.median(),
            t_C1_med=d[d.t_C1 >= 0].t_C1.median(),
            sp_at_C1=d.sp_at_C1.median(),
        ))

    print(df.groupby(["L", "k"]).apply(agg).round(3).to_string())
    print("\nOverall: C0 pass", round(float((df.t_C0 >= 0).mean()), 3),
          "| C1 pass on cohesive", round(float((df[df.single].t_C1 >= 0).mean()), 3),
          "| C1 false-positive on fragmented",
          round(float((df[~df.single].t_C1 >= 0).mean()), 3))


if __name__ == "__main__":
    main()
