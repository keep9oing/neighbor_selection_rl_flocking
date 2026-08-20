"""Analysis + figures for the ACS k-NN convergence study.

Usage:
  python analyze.py --batch main            # summarize -> data/summary_main.csv
  python analyze.py --batch main --figs     # also render figs/*.png
  python analyze.py --batch disc --figs

Definitions (documented here, used everywhere):
- steady state (ss): mean over the last SS_WIN=1000 steps of the 6000-step run.
- single-cluster run: n_comp_r0 == 1 for ALL of the last 500 steps (r0-proximity
  graph: edge iff pairwise dist < r0).
- t_phi99 / t_sv / t_sp: first t where the condition (phi>0.99 / sigma_v<v_goal /
  sigma_p<p_goal) holds for 50 CONSECUTIVE steps (window start reported).
- t_env: exact replica of the env criterion (NOTES_env.md): first t>=49 with
  sigma_p<p_goal AND sigma_v<v_goal at t AND trailing-50 peak-to-peak of sigma_p
  < 0.1 AND of sigma_v < 0.2.  -1 = never within horizon.
- slope_sp_late: linear-fit slope of sigma_p over the last 2000 steps, in m per
  1000 steps (stationarity indicator; ~0 for equilibrated, >0 for diverging).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

STUDY = "/workspace/studies/acs-conv-knn"
SS_WIN = 1000
SINGLE_WIN = 500
SUSTAIN = 50
P_RATE_GOAL, V_RATE_GOAL = 0.1, 0.2  # env defaults (entropy_*_rate_goal)

# --- palette (dataviz reference instance, light mode) ---
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3
SEQ4 = ["#86b6ef", "#5598e7", "#256abf", "#0d366b"]  # ordinal blue steps 250/350/500/700
CRIT = "#d03b3b"                                  # status: pass/fail threshold lines


def _first_sustained(cond, win=SUSTAIN):
    c = np.convolve(cond.astype(np.int32), np.ones(win, dtype=np.int32), "valid")
    hit = np.flatnonzero(c == win)
    return int(hit[0]) if hit.size else -1


def _t_env(s, v, p_goal, v_goal):
    ss, vv = pd.Series(s), pd.Series(v)
    p2p_s = (ss.rolling(SUSTAIN).max() - ss.rolling(SUSTAIN).min()).values
    p2p_v = (vv.rolling(SUSTAIN).max() - vv.rolling(SUSTAIN).min()).values
    ok = (s < p_goal) & (v < v_goal) & (p2p_s < P_RATE_GOAL) & (p2p_v < V_RATE_GOAL)
    hit = np.flatnonzero(ok)
    return int(hit[0]) if hit.size else -1


def summarize_run(path):
    z = np.load(path, allow_pickle=True)
    m = json.loads(str(z["meta"]))
    s, v, phi = z["s_ent"], z["v_ent"], z["phi"]
    comp = z["n_comp_r0"]
    p_goal, v_goal = m["entropy_p_goal"], m["entropy_v_goal"]
    T = m["steps_done"]
    ss = slice(T + 1 - SS_WIN, T + 1)
    tail2k = slice(T + 1 - 2000, T + 1)
    x = np.arange(2000)
    row = dict(
        k=m.get("k"), th=m.get("distance_threshold"),
        L=m["initial_position_bound"], seed=m["seed"], n=m["n_agents"],
        single=bool((comp[T + 1 - SINGLE_WIN:T + 1] == 1).all()),
        n_comp_end=int(comp[T]),
        sp_ss=float(np.nanmean(s[ss])), sv_ss=float(np.nanmean(v[ss])),
        phi_ss=float(np.nanmean(phi[ss])),
        nnd_ss=float(np.nanmean(z["nnd_mean"][ss])),
        nnd_max_ss=float(np.nanmean(z["nnd_max"][ss])),
        minpair_ss=float(np.nanmean(z["min_pair"][ss])),
        churn_ss=float(np.nanmean(z["churn"][ss])),
        deg_ss=float(np.nanmean(z["deg_mean"][ss])),
        deg_sd_t=float(np.nanstd(z["deg_mean"][1:T + 1])),  # temporal sd of degree
        slope_sp_late=float(np.polyfit(x, s[tail2k], 1)[0] * 1000.0),
        sp0=float(s[0]),
        t_phi99=_first_sustained(phi > 0.99),
        t_sv=_first_sustained(v < v_goal),
        t_sp=_first_sustained(s < p_goal),
        t_env=_t_env(s, v, p_goal, v_goal),
        p_goal=p_goal, v_goal=v_goal, r0=m["r0"],
    )
    return row


def load_batch(batch, refresh=False):
    csv = os.path.join(STUDY, "data", f"summary_{batch}.csv")
    files = sorted(glob.glob(os.path.join(STUDY, "data", batch, "*.npz")))
    if os.path.exists(csv) and not refresh:
        df = pd.read_csv(csv)
        if len(df) == len(files):
            return df
    df = pd.DataFrame([summarize_run(f) for f in files])
    df.to_csv(csv, index=False)
    return df


# ---------------------------------------------------------------- figures ----

def _style(ax, logx=False, logy=False):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    if logx:
        from matplotlib.ticker import NullFormatter
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter("{x:.0f}")
    if logy:
        ax.set_yscale("log")


def _load_series(batch, k, L):
    out = []
    for f in sorted(glob.glob(os.path.join(STUDY, "data", batch, f"k{k:02d}_L{int(L)}_*.npz"))):
        z = np.load(f, allow_pickle=True)
        out.append((z["s_ent"], z["v_ent"], z["n_comp_r0"]))
    return out


def fig_timeseries(df, batch, L=250.0, ks=(2, 5, 8, 19)):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), facecolor=SURFACE, sharex=True, sharey=True)
    for ax, k in zip(axes.ravel(), ks):
        runs = _load_series(batch, k, L)
        n_single = 0
        for s, v, comp in runs:
            single = (comp[-SINGLE_WIN:] == 1).all()
            n_single += single
            ax.plot(s, color=CAT[0] if single else CAT[1], lw=0.7,
                    alpha=0.45 if single else 0.35)
        ax.axhline(42.0, color=CRIT, lw=1.2, ls="--")
        _style(ax, logy=True)
        ax.set_title(f"k = {k}   (cohesive {n_single}/{len(runs)})",
                     fontsize=10, color=INK)
        ax.set_ylim(8, 2e4)
    for ax in axes[1]:
        ax.set_xlabel("step", fontsize=9, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"spatial entropy $\sigma_p$ [m]", fontsize=9, color=INK2)
    axes[0, 0].text(5900, 47, "current goal 42 m", color=CRIT, fontsize=8,
                    ha="right", va="bottom")
    axes[0, 0].text(0.02, 0.96, "cohesive (1 cluster)", color=CAT[0], fontsize=8,
                    transform=axes[0, 0].transAxes, va="top")
    axes[0, 0].text(0.02, 0.89, "fragmented", color=CAT[1], fontsize=8,
                    transform=axes[0, 0].transAxes, va="top")
    fig.suptitle(f"ACS + k-NN: spatial entropy trajectories, 32 seeds each "
                 f"(N=20, L={int(L)} m)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(STUDY, "figs", "fig1_timeseries.png"), dpi=150)
    plt.close(fig)


def _per_kL(df, col, only_single=True):
    d = df[df.single] if only_single else df
    g = d.groupby(["L", "k"])[col]
    return g.median(), g.quantile(0.25), g.quantile(0.75), g.count()


def fig_equilibrium(df):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), facecolor=SURFACE)
    med, q1, q3, cnt = _per_kL(df, "sp_ss")
    ax = axes[0]
    for i, L in enumerate(sorted(df.L.unique())):
        if L not in med.index.get_level_values(0):
            continue
        mk = med.loc[L]
        ax.errorbar(mk.index, mk.values,
                    yerr=[mk.values - q1.loc[L].values, q3.loc[L].values - mk.values],
                    color=CAT[i], lw=1.6, marker="o", ms=5, capsize=2,
                    label=f"L = {int(L)} m")
    ax.axhline(42.0, color=CRIT, lw=1.2, ls="--")
    ax.text(19, 43, "goal 42 m", color=CRIT, fontsize=8, ha="right", va="bottom")
    ax.axhline(60 / np.sqrt(2), color=MUTED, lw=1.0, ls=":")
    ax.text(3, 60 / np.sqrt(2) * 0.90, r"$r_0/\sqrt{2}$", color=MUTED, fontsize=8)
    ax.text(3, 45, "k ≤ 2: never cohesive", color=MUTED, fontsize=8)
    _style(ax, logx=True, logy=True)
    ax.set_xticks([3, 5, 8, 12, 19])
    ax.set_xticklabels([3, 5, 8, 12, 19])
    ax.set_xlim(2.6, 21)
    ax.set_xlabel("k (neighbors, excl. self)", fontsize=9, color=INK2)
    ax.set_ylabel(r"steady-state $\sigma_p$ [m] (cohesive runs)", fontsize=9, color=INK2)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)

    ax = axes[1]
    rng = np.random.RandomState(0)
    d = df[df.single]
    for i, L in enumerate(sorted(df.L.unique())):
        dl = d[d.L == L]
        jit = dl.k * (1 + 0.035 * (i - 1)) * np.exp(rng.uniform(-0.015, 0.015, len(dl)))
        ax.scatter(jit, dl.sv_ss.clip(lower=1e-7), s=10, color=CAT[i], alpha=0.5,
                   edgecolors="none")
        mk = dl.groupby("k")["sv_ss"].median()
        ax.plot(mk.index, mk.values.clip(1e-7), color=CAT[i], lw=1.4, alpha=0.9)
    ax.axhline(0.1, color=CRIT, lw=1.2, ls="--")
    ax.text(19, 0.13, "goal 0.1 m/s", color=CRIT, fontsize=8, ha="right", va="bottom")
    _style(ax, logx=True, logy=True)
    ax.set_xticks([3, 5, 8, 12, 19])
    ax.set_xticklabels([3, 5, 8, 12, 19])
    ax.set_xlim(2.6, 21)
    ax.set_xlabel("k (neighbors, excl. self)", fontsize=9, color=INK2)
    ax.set_ylabel(r"steady-state $\sigma_v$ [m/s] (dots: runs, line: median)",
                  fontsize=9, color=INK2)
    fig.suptitle("Equilibrium levels vs k — the current goals are tuned to the FC equilibrium",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(STUDY, "figs", "fig2_equilibrium_vs_k.png"), dpi=150)
    plt.close(fig)


def fig_fragmentation(df):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.8), facecolor=SURFACE)
    for i, L in enumerate(sorted(df.L.unique())):
        d = df[df.L == L].groupby("k")["single"].mean()
        ax.plot(d.index, d.values, color=CAT[i], lw=1.8, marker="o", ms=5,
                label=f"L = {int(L)} m")
    _style(ax)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 19])
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("k (neighbors, excl. self)", fontsize=9, color=INK2)
    ax.set_ylabel("P(single cluster at t = 6000)", fontsize=9, color=INK2)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    ax.set_title("Cohesion probability vs k (32 seeds per point)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(STUDY, "figs", "fig3_fragmentation.png"), dpi=150)
    plt.close(fig)


def fig_times(df, L=250.0):
    import matplotlib.pyplot as plt
    d = df[(df.L == L) & df.single]
    specs = [("t_phi99", r"heading: $\phi>0.99$", CAT[0]),
             ("t_sv", r"heading goal: $\sigma_v<0.1$", CAT[1]),
             ("t_env", "full env criterion", CAT[2])]
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 5.6), facecolor=SURFACE, sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.0]})
    ax = axes[0]
    for col, label, c in specs:
        g = d[d[col] >= 0].groupby("k")[col]
        med = g.median()
        ax.plot(med.index, med.values, color=c, lw=1.8, marker="o", ms=5, label=label)
    _style(ax, logy=True)
    ax.set_ylabel("median steps to reach (when reached)", fontsize=9, color=INK2)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    ax.set_title(f"Time scales vs k (cohesive runs, L={int(L)} m)", fontsize=10, color=INK)
    ax = axes[1]
    for col, label, c in specs:
        frac = d.groupby("k").apply(lambda x: (x[col] >= 0).mean())
        ax.plot(frac.index, frac.values, color=c, lw=1.8, marker="o", ms=5)
    _style(ax)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 19])
    ax.set_xlabel("k (neighbors, excl. self)", fontsize=9, color=INK2)
    ax.set_ylabel("fraction reached\nwithin 6000", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(STUDY, "figs", "fig4_times_vs_k.png"), dpi=150)
    plt.close(fig)


def fig_local(df):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), facecolor=SURFACE)
    for j, (col, ylab, goalline) in enumerate([
            ("nnd_ss", "steady-state mean NN distance [m]", 60.0),
            ("churn_ss", "steady-state neighbor churn (1 - Jaccard)", None)]):
        ax = axes[j]
        med, q1, q3, _ = _per_kL(df, col)
        for i, L in enumerate(sorted(df.L.unique())):
            if L not in med.index.get_level_values(0):
                continue
            mk = med.loc[L]
            ax.errorbar(mk.index, mk.values,
                        yerr=[mk.values - q1.loc[L].values, q3.loc[L].values - mk.values],
                        color=CAT[i], lw=1.6, marker="o", ms=5, capsize=2,
                        label=f"L = {int(L)} m" if j == 0 else None)
        if goalline:
            ax.axhline(goalline, color=MUTED, lw=1.0, ls=":")
            ax.text(1, goalline * 1.02, r"$r_0 = 60$ m", color=MUTED, fontsize=8)
        _style(ax)
        ax.set_xticks([1, 2, 3, 5, 8, 12, 19])
        ax.set_xlabel("k (neighbors, excl. self)", fontsize=9, color=INK2)
        ax.set_ylabel(ylab, fontsize=9, color=INK2)
    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK2)
    fig.suptitle("Local structure at equilibrium (cohesive runs)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(STUDY, "figs", "fig5_local_structure.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="main")
    ap.add_argument("--figs", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    df = load_batch(args.batch, refresh=args.refresh)
    pd.set_option("display.width", 200)
    is_knn = df.k.notna().all() if "k" in df else False
    key = "k" if is_knn else "th"
    agg = df.groupby(["L", key]).agg(
        n=("single", "size"), P_single=("single", "mean"),
        sp_ss_med=("sp_ss", lambda x: x[df.loc[x.index, "single"]].median()),
        sv_ss_med=("sv_ss", lambda x: x[df.loc[x.index, "single"]].median()),
        deg_ss=("deg_ss", "median"), deg_sd_t=("deg_sd_t", "median"),
        nnd=("nnd_ss", lambda x: x[df.loc[x.index, "single"]].median()),
        churn=("churn_ss", lambda x: x[df.loc[x.index, "single"]].median()),
        slope=("slope_sp_late", lambda x: x[df.loc[x.index, "single"]].median()),
        t_phi99=("t_phi99", lambda x: x[(x >= 0) & df.loc[x.index, "single"]].median()),
        t_sv=("t_sv", lambda x: x[(x >= 0)].median()),
        P_env_pass=("t_env", lambda x: (x >= 0).mean()),
    ).round(3)
    print(agg.to_string())
    if args.figs and is_knn:
        os.makedirs(os.path.join(STUDY, "figs"), exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        fig_timeseries(df, args.batch)
        fig_equilibrium(df)
        fig_fragmentation(df)
        fig_times(df)
        fig_local(df)
        print("figures written to figs/")


if __name__ == "__main__":
    main()
