"""N-dependence check for the equilibrium-spread power law.

Batches: main (N=20, use L=250 slice), n10 (N=10, L=176.78), n40 (N=40, L=353.55)
— all at matched initial density N/L^2. Fits sigma_p_ss(k) per N, reports
sigma_p_FC(N), tests the collapse sigma_p = sigma_p_FC * ((N-1)/k)^gamma, checks
selected-edge mean distance constancy, and renders figs/fig6_N_dependence.png.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from analyze import load_batch, INK, INK2, MUTED, GRID, BASE, SURFACE, CAT, CRIT, _style

STUDY = "/workspace/studies/acs-conv-knn"


def sel_edge_mean(batch, k, L):
    vals = []
    for f in glob.glob(os.path.join(STUDY, "data", batch, f"k{k:02d}_L{int(L)}_*.npz")):
        z = np.load(f, allow_pickle=True)
        if not (z["n_comp_r0"][-500:] == 1).all():
            continue
        pos = z["pos_snaps"][-1]
        d = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        e = np.take_along_axis(d, np.argsort(d, axis=1)[:, :k], axis=1)
        vals.append(e.mean())
    return float(np.mean(vals)) if vals else np.nan


def main():
    batches = [("n10", 10, 176.78), ("main", 20, 250.0), ("n40", 40, 353.55)]
    rows = []
    for batch, N, L in batches:
        df = load_batch(batch)
        d = df[(df.L.round(0) == round(L)) & df.single]
        med = d.groupby("k")["sp_ss"].median()
        med = med[med.index >= 3] if N > 10 else med[med.index >= 2]
        b, a = np.polyfit(np.log(med.index), np.log(med.values), 1)
        fc_k = N - 1
        sp_fc = df[(df.k == fc_k) & df.single].sp_ss.median()
        psingle = df.groupby("k")["single"].mean()
        rows.append(dict(batch=batch, N=N, coeff=np.exp(a), exponent=b, sp_fc=sp_fc,
                         med=med, psingle=psingle))
        edge = {k: sel_edge_mean(batch, int(k), L) for k in med.index}
        print(f"N={N}: sigma_p ~ {np.exp(a):.1f} * k^({b:.3f}) | sigma_p_FC(k={fc_k})"
              f" = {sp_fc:.1f} | selected-edge mean over k: "
              f"{np.nanmean(list(edge.values())):.1f} ± {np.nanstd(list(edge.values())):.1f} m")

    # collapse test: sigma_p / sigma_p_FC vs k/(N-1)
    print("\ncollapse sigma_p/sigma_p_FC vs (k/(N-1)):")
    xs, ys = [], []
    for r in rows:
        x = r["med"].index.values / (r["N"] - 1)
        y = r["med"].values / r["sp_fc"]
        xs.append(x); ys.append(y)
        print(f"  N={r['N']}: " + " ".join(f"({xi:.2f},{yi:.2f})" for xi, yi in zip(x, y)))
    b, a = np.polyfit(np.log(np.concatenate(xs)), np.log(np.concatenate(ys)), 1)
    print(f"pooled collapse fit: sigma_p/sigma_p_FC ~ {np.exp(a):.2f} * (k/(N-1))^({b:.3f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), facecolor=SURFACE)
    ax = axes[0]
    for i, r in enumerate(rows):
        ax.plot(r["med"].index, r["med"].values, color=CAT[i], lw=1.6, marker="o",
                ms=5, label=f"N = {r['N']}")
    ax.axhline(42.0, color=CRIT, lw=1.2, ls="--")
    ax.text(2, 43, "goal 42 m", color=CRIT, fontsize=8, va="bottom")
    _style(ax, logx=True, logy=True)
    ax.set_xticks([2, 3, 5, 8, 13, 19, 26, 39])
    ax.set_xlabel("k (neighbors, excl. self)", fontsize=9, color=INK2)
    ax.set_ylabel(r"steady-state $\sigma_p$ [m] (cohesive)", fontsize=9, color=INK2)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    ax = axes[1]
    for i, r in enumerate(rows):
        ax.plot(r["med"].index.values / (r["N"] - 1), r["med"].values / r["sp_fc"],
                color=CAT[i], lw=1.6, marker="o", ms=5)
    xx = np.linspace(np.log(0.05), np.log(1.0), 50)
    ax.plot(np.exp(xx), np.exp(a + b * xx), color=MUTED, lw=1.0, ls=":")
    ax.text(0.3, 2.6, rf"$\propto (k/(N\!-\!1))^{{{b:.2f}}}$", color=MUTED, fontsize=9)
    _style(ax, logx=True, logy=True)
    from matplotlib.ticker import NullFormatter, FixedFormatter
    ax.set_xticks([0.1, 0.2, 0.5, 1.0])
    ax.xaxis.set_major_formatter(FixedFormatter(["0.1", "0.2", "0.5", "1"]))
    ax.set_xlabel("k / (N−1)   (1 = fully connected)", fontsize=9, color=INK2)
    ax.set_ylabel(r"$\sigma_p\, /\, \sigma_p^{FC}$", fontsize=9, color=INK2)
    fig.suptitle("Equilibrium spread vs k across N (matched initial density)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(STUDY, "figs", "fig6_N_dependence.png"), dpi=150)
    print("wrote figs/fig6_N_dependence.png")


if __name__ == "__main__":
    main()
