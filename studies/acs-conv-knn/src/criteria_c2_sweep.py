"""Window/form/epsilon sensitivity sweep for the C2 convergence criterion.

C2 (this iteration, user-fixed parts):
    A) alignment:    rolling-min over 50 steps of phi > 0.98        [FIXED]
    B) cohesion:     rolling-max over W of n_comp_r0 == 1           [W swept]
                     (single r0-proximity component containing ALL agents;
                      size-1 "flocks" impossible by construction)
    C) stationarity: spatial spread sigma_p stable over W, form in  [W, form, eps swept]
                       p2  : |s(t) - s(t-W)| / s(t-W)            < eps
                       p2p : (max - min over [t-W+1, t]) / mean  < eps
    t_fire = first t with A & B & C.

Ground truth (long-horizon label, per run):
    gt_conv = all(n_comp_r0[-100:] == 1)  and  median(phi[-100:]) > 0.98
    sigma_ss = median(sigma_p[-300:])

Reported per (form, W, eps):
    - detection rate on gt_conv runs; false-positive rate on non-gt runs
    - premature-fire: ratio = sigma_p(t_fire)/sigma_ss; P(ratio > 1.10), median
    - t_fire median / P(t_fire <= 1000) on the "good policy" set
      (N=20 L=250: knn k>=8, disc thr=1.0, nn checkpoint)
    - steady-state noise floor: worst tail rel-p2p per W (sets the eps floor)

Usage: python criteria_c2_sweep.py [--report-only]
Writes data/c2_sweep_runs.csv (per run x config) + prints aggregate tables.
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

STUDY = "/workspace/studies/acs-conv-knn"
BATCHES = ["main", "disc", "stress_random", "nn_hardtopk", "n10", "n40"]

PHI_GOAL = 0.98
W_ALIGN = 50
WS = [50, 100, 150, 200, 250, 300, 400, 500]
EPS = [0.02, 0.03, 0.05]
FORMS = ["p2", "p2p"]


def family(batch, m):
    if batch in ("main", "n10", "n40"):
        return f"knn_N{m['n_agents']}_k{m['k']:02d}"
    if batch == "disc":
        return f"disc_thr{m['distance_threshold']:.2f}"
    if batch == "stress_random":
        return f"random_p{m['sel_p']:.2f}"
    if batch == "nn_hardtopk":
        return "nn_ckpt"
    return batch


def eval_file(args):
    batch, path = args
    z = np.load(path, allow_pickle=True)
    m = json.loads(str(z["meta"]))
    s = z["s_ent"].astype(np.float64)
    phi = z["phi"].astype(np.float64)
    comp = z["n_comp_r0"].astype(np.float64)

    gt_conv = bool((comp[-100:] == 1).all() and np.median(phi[-100:]) > PHI_GOAL)
    sigma_ss = float(np.median(s[-300:]))

    ps, pphi, pcomp = pd.Series(s), pd.Series(phi), pd.Series(comp)
    align = (pphi.rolling(W_ALIGN).min() > PHI_GOAL).values

    base = dict(batch=batch, fam=family(batch, m), L=m["initial_position_bound"],
                n=m["n_agents"], seed=m["seed"], gt=gt_conv, sigma_ss=sigma_ss)
    rows, nf_rows = [], []
    for W in WS:
        coh = (pcomp.rolling(W).max() == 1).values
        # p2 relative two-point change
        s_prev = np.full_like(s, np.nan)
        s_prev[W:] = s[:-W]
        with np.errstate(invalid="ignore", divide="ignore"):
            rel2 = np.abs(s - s_prev) / s_prev
        # p2p relative band over the window
        rmax = ps.rolling(W).max()
        rmin = ps.rolling(W).min()
        relb = ((rmax - rmin) / ps.rolling(W).mean()).values

        if gt_conv:  # steady-state noise floor: worst tail window (last 900 steps)
            nf_rows.append(dict(batch=batch, fam=base["fam"], seed=m["seed"], W=W,
                                nf=float(np.nanmax(relb[-900:]))))
        for form, rel in (("p2", rel2), ("p2p", relb)):
            for eps in EPS:
                with np.errstate(invalid="ignore"):
                    ok = align & coh & (rel < eps)
                hit = np.flatnonzero(ok)
                t = int(hit[0]) if hit.size else -1
                rows.append(dict(base, form=form, W=W, eps=eps, t_fire=t,
                                 ratio=float(s[t] / sigma_ss) if t >= 0 else np.nan))
    return rows, nf_rows


def compute():
    tasks = [(b, f) for b in BATCHES
             for f in sorted(glob.glob(os.path.join(STUDY, "data", b, "*.npz")))]
    print(f"{len(tasks)} runs x {len(FORMS) * len(WS) * len(EPS)} configs ...")
    with Pool(16) as pool:
        out = pool.map(eval_file, tasks, chunksize=8)
    runs = pd.DataFrame([r for rows, _ in out for r in rows])
    nf = pd.DataFrame([r for _, nfr in out for r in nfr])
    runs.to_csv(os.path.join(STUDY, "data", "c2_sweep_runs.csv"), index=False)
    nf.to_csv(os.path.join(STUDY, "data", "c2_noise_floor.csv"), index=False)
    return runs, nf


def report(runs, nf):
    pd.set_option("display.width", 250)

    print("\n=== steady-state noise floor: worst tail rel-p2p (sets eps floor) ===")
    nf["grp"] = np.where(nf.batch == "stress_random", "random(max churn)", "other")
    print(nf.groupby(["grp", "W"]).nf.quantile([0.5, 0.99]).unstack().round(4).to_string())

    # good-policy set for timing: what a decent trained policy looks like
    good = runs[(runs.n == 20) & (runs.L == 250.0) & runs["gt"] &
                (runs.fam.isin(["knn_N20_k08", "knn_N20_k10", "knn_N20_k12",
                                "knn_N20_k15", "knn_N20_k19", "disc_thr1.00",
                                "nn_ckpt"]))]

    def agg(cfg):
        g = runs[(runs.form == cfg[0]) & (runs.W == cfg[1]) & (runs.eps == cfg[2])]
        conv, nonc = g[g["gt"]], g[~g["gt"]]
        gg = good[(good.form == cfg[0]) & (good.W == cfg[1]) & (good.eps == cfg[2])]
        det = gg[gg.t_fire >= 0]
        return dict(form=cfg[0], W=cfg[1], eps=cfg[2],
                    detect=(conv.t_fire >= 0).mean(),
                    fp=(nonc.t_fire >= 0).mean(),
                    prem10=(conv.ratio > 1.10).mean(),
                    prem05=(conv.ratio > 1.05).mean(),
                    ratio_med=conv.ratio.median(),
                    t_med_good=det.t_fire.median(),
                    le1000_good=(det.t_fire <= 1000).mean() if len(det) else np.nan,
                    le1500_good=(det.t_fire <= 1500).mean() if len(det) else np.nan)

    cfgs = [(f, w, e) for f in FORMS for w in WS for e in EPS]
    summ = pd.DataFrame([agg(c) for c in cfgs])
    summ.to_csv(os.path.join(STUDY, "data", "c2_sweep_summary.csv"), index=False)
    print("\n=== per-config summary (detect/fp overall; premature on gt_conv; timing on good set) ===")
    print(summ.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.report_only:
        runs = pd.read_csv(os.path.join(STUDY, "data", "c2_sweep_runs.csv"))
        nf = pd.read_csv(os.path.join(STUDY, "data", "c2_noise_floor.csv"))
    else:
        runs, nf = compute()
    report(runs, nf)


if __name__ == "__main__":
    main()
