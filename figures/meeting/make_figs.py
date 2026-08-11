"""Meeting figures F1-F7. Palette: validated default (blue/orange/aqua + ink)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

M = "/workspace/figures/meeting"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", RAMP)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
})

S = json.load(open(f"{M}/stats.json"))
P = np.load(f"{M}/profiles.npz")
R = np.load(f"{M}/reroll_1014.npz")


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- F1 criterion
def f1():
    ks = [3, 4, 5, 6, 7, 8, 10, 12, 15, 19]
    sig = [112, 94, 84, 76, 70, 66, 60, 55, 49, 39.8]
    fig, ax = plt.subplots(figsize=(8.6, 4.0), constrained_layout=True)
    x = np.arange(len(ks))
    colors = [BLUE if k < 19 else "#104281" for k in ks]
    ax.bar(x, sig, 0.62, color=colors)
    ax.plot(x, 198 * np.array(ks, float) ** -0.53, color=INK2,
            lw=1.1, ls=":", label=r"fit  $\sigma_p \approx 198\,k^{-0.53}$")
    ax.axhline(42, color=RED, lw=1.4, ls="--")
    ax.text(0.02, 44.5, "legacy convergence threshold  $\\sigma_p < 0.7\\,r_0 = 42$ m",
            color=RED, fontsize=9.5)
    ax.annotate("only FC (k=19)\never passes", xy=(len(ks) - 1, 39.8),
                xytext=(len(ks) - 2.6, 22), fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))
    ax.set_xticks(x, [f"k={k}" if k < 19 else "FC" for k in ks])
    ax.set_ylabel("equilibrium spread $\\sigma_p$ (m), median")
    ax.set_title("A level-based convergence criterion picks a topology, not a behavior",
                 fontsize=11.5, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=9)
    despine(ax)
    fig.savefig(f"{M}/f1_criterion_bias.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- F2 fail vs k
def f2():
    ks = list(range(8, 20))
    fig, ax = plt.subplots(figsize=(8.6, 4.2), constrained_layout=True)
    for L, c in ((125, BLUE), (250, ORANGE), (500, AQUA)):
        y = [S["knn"][f"k{k}_L{L}"]["fail_pct"] for k in ks]
        ax.plot(ks, y, "-o", color=c, lw=1.6, ms=5, label=f"L = {L}")
    ax.axhline(1.0, color=MUTED, lw=1.0, ls="--")
    ax.text(8.05, 1.25, "policy-grade bar (1%)", color=MUTED, fontsize=8.5)
    ax.annotate("FC cliff: 0/1500\n(k = N−1: no one\ncan be dropped)",
                xy=(18.9, 0.3), xytext=(16.4, 4.0), fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))
    ax.set_xticks(ks, [str(k) if k < 19 else "19\n(FC)" for k in ks])
    ax.set_xlabel("fixed k (N = 20)")
    ax.set_ylabel("failure rate (%), n = 500 per point")
    ax.set_title("Every fixed k fails 3–15% — non-monotone, no tuning escape",
                 fontsize=11.5, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(-0.4, 16)
    despine(ax)
    fig.savefig(f"{M}/f2_fail_vs_k.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- F3 pareto
def f3():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
    for ax, L in zip(axes, (250, 500)):
        for k in range(8, 19):
            a = S["knn"][f"k{k}_L{L}"]
            ax.scatter(a["J_med"], a["fail_pct"], s=26, color=MUTED, zorder=3)
            if k in (8, 10, 12, 13, 16, 18):
                ax.annotate(f"k{k}", (a["J_med"], a["fail_pct"]),
                            textcoords="offset points", xytext=(5, 3),
                            fontsize=8, color=INK2)
        fc = S["knn"][f"k19_L{L}"]
        ax.scatter(fc["J_med"], fc["fail_pct"], marker="s", s=54, color=INK, zorder=4)
        ax.annotate("FC", (fc["J_med"], fc["fail_pct"]), textcoords="offset points",
                    xytext=(5, -13), fontsize=9, color=INK, fontweight="bold")
        c1 = S["policy"][f"C1_L{L}"]
        r1 = S["policy"][f"R1_L{L}"]
        ax.scatter(c1["J_med"], c1["fail_pct"], marker="*", s=210, color=BLUE, zorder=5)
        ax.annotate("policy-E (C1)", (c1["J_med"], c1["fail_pct"]),
                    textcoords="offset points", xytext=(7, 6), fontsize=9.5,
                    color=BLUE, fontweight="bold")
        ax.scatter(r1["J_med"], r1["fail_pct"], marker="D", s=64, color=ORANGE, zorder=5)
        ax.annotate("policy-R (R1)", (r1["J_med"], r1["fail_pct"]),
                    textcoords="offset points", xytext=(-6, 9), ha="right",
                    fontsize=9.5, color=ORANGE, fontweight="bold")
        if L == 250:
            a60 = S["policy"]["A60_L250"]
            ax.scatter(a60["J_med"], a60["fail_pct"], marker="^", s=46, color=MUTED,
                       zorder=4)
            ax.annotate("specialist A60", (a60["J_med"], a60["fail_pct"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8,
                        color=MUTED)
        ax.set_xlabel("typical cost  (median J on successes)")
        ax.set_title(f"L = {L}", fontsize=10.5, color=INK)
        ax.set_ylim(-0.6, 16)
        despine(ax)
    axes[0].set_ylabel("failure rate (%), n = 500")
    fig.suptitle("Reliability × typical-cost plane: learned policies occupy the empty corner",
                 fontsize=12, color=INK)
    fig.savefig(f"{M}/f3_pareto.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- F4 profiles
def phase_bands(ax, t_merge, t_conv, ymax_frac=1.0):
    ax.axvspan(0, t_merge, color="#2a78d6", alpha=0.06, lw=0)
    ax.axvspan(t_merge, t_conv, color="#eb6834", alpha=0.06, lw=0)
    ax.axvspan(t_conv, 1500, color="#898781", alpha=0.07, lw=0)


def f4():
    t = np.arange(1501)
    ncomp = P["C1_L250_n_comp_r0_med"]
    t_merge = int(np.argmax(ncomp <= 1.0))
    t_conv = int(S["policy"]["C1_L250"]["t_med"])
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.6), sharex=True,
                             constrained_layout=True)
    ax = axes[0]
    phase_bands(ax, t_merge, t_conv)
    ax.fill_between(t, P["C1_L250_deg_mean_q25"], P["C1_L250_deg_mean_q75"],
                    color=BLUE, alpha=0.18, lw=0)
    ax.plot(t, P["C1_L250_deg_mean_med"], color=BLUE, lw=1.7, label="policy-E (C1)")
    ax.plot(t, P["R1_L250_deg_mean_med"], color=ORANGE, lw=1.5, label="policy-R (R1)")
    ax.axhline(12, color=MUTED, lw=1.1, ls="--")
    ax.axhline(19, color=INK, lw=1.1, ls=":")
    ax.text(1495, 12.35, "k-NN (k=12): flat", ha="right", fontsize=8.5, color=MUTED)
    ax.text(1495, 19.25, "FC: flat 19", ha="right", fontsize=8.5, color=INK2)
    ax.set_ylabel("selected neighbors\nper agent (median)")
    ax.set_ylim(8, 20.6)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    despine(ax)
    ax = axes[1]
    phase_bands(ax, t_merge, t_conv)
    ax.fill_between(t, P["C1_L250_rank_dev_q25"], P["C1_L250_rank_dev_q75"],
                    color=BLUE, alpha=0.18, lw=0)
    ax.plot(t, P["C1_L250_rank_dev_med"], color=BLUE, lw=1.7)
    ax.plot(t, P["R1_L250_rank_dev_med"], color=ORANGE, lw=1.5)
    ax.axhline(0, color=MUTED, lw=1.1, ls="--")
    ax.text(1495, 0.012, "any fixed rule (k-NN, FC): 0 by construction", ha="right",
            fontsize=8.5, color=MUTED)
    ax.set_ylabel("non-nearest share of\nselected edges (rank-dev)")
    ax.set_xlabel("time step")
    ax.set_ylim(-0.02, 0.48)
    ax.set_xlim(0, 1500)
    ax.text(1495, 0.44, "in 500/500 episodes: merge-phase rank-dev > hold-phase",
            ha="right", fontsize=8.8, color=INK2)
    for x, lab in ((t_merge / 2, "merging"), ((t_merge + t_conv) / 2, "settling"),
                   ((t_conv + 1500) / 2, "holding (post-convergence)")):
        axes[0].text(x, 20.15, lab, ha="center", fontsize=8.8, color=INK2)
    fig.suptitle("The policy re-wires by phase — median over 500 episodes (L = 250)",
                 fontsize=11.5, color=INK)
    fig.savefig(f"{M}/f4_phase_profiles.png", dpi=160)
    plt.close(fig)


def f4b():
    t = np.arange(1501)
    fig, ax = plt.subplots(figsize=(8.6, 3.6), constrained_layout=True)
    for L, c in ((125, "#86b6ef"), (250, "#2a78d6"), (500, "#104281")):
        ax.plot(t, P[f"C1_L{L}_deg_mean_med"], color=c, lw=1.6, label=f"L = {L}")
    ax.set_xlabel("time step")
    ax.set_ylabel("selected neighbors\nper agent (median)")
    ax.set_xlim(0, 1500)
    ax.legend(frameon=False, fontsize=9, title="same policy-E,\ndifferent scale",
              title_fontsize=8.5)
    ax.set_title("Not a timer: the schedule shifts with the condition (C1, per-L medians)",
                 fontsize=11, color=INK, pad=8)
    despine(ax)
    fig.savefig(f"{M}/f4b_degree_by_L.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- F5 straggler
def f5():
    TW = 2500
    zc = np.load("/workspace/studies/acs-robust-r3-stress/data/eval/C1_i80_L250_s500/"
                 "C1_i80_L250_s500_s1014.npz", allow_pickle=True)
    zk = np.load("/workspace/studies/acs-robust-r2/data/knnref/k12_L250_N20/"
                 "k12_L250_N20_s1014.npz", allow_pickle=True)
    strag = int(R["strag"])
    tf_c1 = int(pd.read_csv("/workspace/studies/acs-robust-r3-stress/data/eval/"
                            "C1_i80_L250_s500_summary.csv").set_index("seed").t_fire[1014])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), constrained_layout=True)
    lims = None
    xs, ys = [], []
    for z in (zk, zc):
        keep = z["snap_ts"] <= TW
        p = z["pos_snaps"][keep]
        xs += [p[:, :, 0].min(), p[:, :, 0].max()]
        ys += [p[:, :, 1].min(), p[:, :, 1].max()]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 * 1.06
    lims = ((cx - half, cx + half), (cy - half, cy + half))
    for ax, z, name, tf in ((axes[0], zk, "k-NN (k=12) — FAILS", -1),
                            (axes[1], zc, f"policy-E (C1) — converges, $t_{{conv}}$={tf_c1}", tf_c1)):
        keep = z["snap_ts"] <= TW
        Ppos, ts = z["pos_snaps"][keep], z["snap_ts"][keep]
        n = Ppos.shape[1]
        for i in range(n):
            if i == strag:
                continue
            pts = Ppos[:, i]
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=CMAP, linewidths=0.8, alpha=0.75)
            lc.set_array(0.5 * (ts[:-1] + ts[1:]))
            lc.norm.vmin, lc.norm.vmax = 0, TW
            ax.add_collection(lc)
        ax.plot(Ppos[:, strag, 0], Ppos[:, strag, 1], color=ORANGE, lw=1.9, zorder=5)
        ax.scatter(Ppos[0, :, 0], Ppos[0, :, 1], s=13, facecolors="none",
                   edgecolors=MUTED, lw=0.8, zorder=3)
        ax.scatter(*Ppos[-1, strag], s=40, color=ORANGE, zorder=6)
        ax.scatter(Ppos[-1, [i for i in range(n) if i != strag], 0],
                   Ppos[-1, [i for i in range(n) if i != strag], 1],
                   s=11, color=RAMP[-1], zorder=4)
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=10.5, color=INK)
        ax.set_xlabel("x (m)")
        despine(ax)
    axes[0].annotate("agent 16: chases forever,\nflock never pulls back\n(in-edges → 0)",
                     xy=R["pos_k12"][2000, strag], xytext=(0.03, 0.72),
                     textcoords="axes fraction", fontsize=9, color=ORANGE,
                     arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0))
    axes[1].annotate("same agent, integrated\nduring the merge",
                     xy=R["pos_c1"][1500, strag], xytext=(0.03, 0.8),
                     textcoords="axes fraction", fontsize=9, color=ORANGE,
                     arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0))
    axes[0].set_ylabel("y (m)")
    fig.suptitle("Straggler abandonment, same initial condition (seed 1014, L=250) — "
                 "shown to t = 2500 (k-NN gap persists to 6000)",
                 fontsize=11.5, color=INK)
    fig.savefig(f"{M}/f5_straggler_1014.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- F6 edges
def f6():
    strag = int(R["strag"])
    pos, act = R["pos_c1"], R["act_c1"]
    times = [30, 200, 1200]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.9), constrained_layout=True)
    for ax, tt in zip(axes, times):
        p, a = pos[tt], act[tt].astype(bool).copy()
        np.fill_diagonal(a, False)
        d = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        order = np.argsort(d, axis=1)
        n_non = n_tot = 0
        for i in range(len(p)):
            deg = int(a[i].sum())
            if deg == 0:
                continue
            nearest = set(order[i, :deg].tolist())
            for j in np.where(a[i])[0]:
                non = j not in nearest
                n_tot += 1
                n_non += non
                ax.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]],
                        color=ORANGE if non else BLUE,
                        lw=0.95 if non else 0.45,
                        alpha=0.8 if non else 0.3, zorder=2 + non)
        ax.scatter(p[:, 0], p[:, 1], s=17, color=INK, zorder=5)
        ax.scatter(*p[strag], s=95, facecolors="none", edgecolors=ORANGE,
                   lw=1.6, zorder=6)
        ax.set_title(f"t = {tt}   ·   non-nearest {100 * n_non / n_tot:.0f}%",
                     fontsize=10, color=INK)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(GRID)
        # 100 m scale bar in a clear strip below the data (panels autoscale,
        # so absolute size must be shown)
        ax.margins(0.06)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        yr = y1 - y0
        ax.set_ylim(y0 - 0.14 * yr, y1)
        by = y0 - 0.085 * yr
        bx = (x0 + x1) / 2 - 50
        ax.plot([bx, bx + 100], [by, by], color=INK2, lw=1.6, zorder=7)
        ax.text(bx + 50, by + 0.022 * yr, "100 m", ha="center",
                fontsize=8, color=INK2, zorder=7)
    h = [Line2D([], [], color=BLUE, lw=1.2, alpha=0.5, label="nearest-set selection"),
         Line2D([], [], color=ORANGE, lw=1.5, label="non-nearest selection (adaptive)"),
         Line2D([], [], marker="o", ls="none", markerfacecolor="none",
                markeredgecolor=ORANGE, markersize=8, label="the straggler-to-be (agent 16)")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("What the policy actually selects (C1, seed 1014): "
                 "long-range picks while merging → near-FC once settled",
                 fontsize=11.5, color=INK)
    fig.savefig(f"{M}/f6_edges_1014.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- F7 in-edges
def f7():
    w = 25
    def smooth(x):
        return pd.Series(x).rolling(w, min_periods=1).mean().values
    t = np.arange(len(R["strag_in_c1"]))
    fig, ax = plt.subplots(figsize=(8.8, 3.7), constrained_layout=True)
    ax.plot(t, smooth(R["strag_in_c1"]), color=BLUE, lw=1.7, label="policy-E (C1)")
    ax.plot(t, smooth(R["strag_in_k12"]), color=MUTED, lw=1.6, ls="--", label="k-NN (k=12)")
    ax.set_xlabel("time step")
    ax.set_ylabel("flock → agent-16 in-edges\n(how many flock-mates listen to it)")
    ax.set_xlim(0, 2000)
    ax.legend(frameon=False, fontsize=9)
    ax.annotate("k-NN: everyone stops listening\n→ no pull → abandoned",
                xy=(700, 0.3), xytext=(760, 5.2), fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))
    ax.annotate("C1: kept in the ear of the flock\n(6.6 of ~12 early in-edges are\nnon-nearest picks)",
                xy=(70, 12.2), xytext=(430, 2.6), fontsize=9, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9))
    ax.set_title("The mechanism on one paired episode (seed 1014): who listens to the straggler",
                 fontsize=11, color=INK, pad=8)
    despine(ax)
    fig.savefig(f"{M}/f7_straggler_inedges.png", dpi=160)
    plt.close(fig)


for f in (f1, f2, f3, f4, f4b, f5, f6, f7):
    f()
    print(f.__name__, "done")
