"""Paired-seed convergence trajectories: ACS(FC) vs RL(C1) vs k-NN(k=12).

Reads existing rollout npz (no re-simulation) from:
  - ACS  : studies/acs-robust-r3-stress/data/knnref/k19_L250_N20   (k=N-1 == fully connected == Pure-ACS)
  - RL   : studies/acs-robust-r3-stress/data/eval/C1_i80_L250_s500 (C1 ft-init, iter 80)
  - k-NN : studies/acs-robust-r2/data/knnref/k12_L250_N20          (frontier reference k)

All three were rolled out with common.rollout under identical config
(N=20, L=250, 6000 steps) and identical env seeds -> initial conditions are
bitwise-paired (verified on pos_snaps[0]).

Per seed: one row of 3 panels (one method each, never overlaid), sharing the
same time window (max t_conv of the row + margin) and the same xy limits.
Time along each agent path is encoded with a single-hue sequential blue ramp
(light = start, dark = end); start = open gray circle, end = dark dot.

Outputs: figures/traj_paired/traj_paired_s<seed>.png  (per seed)
         figures/traj_paired/traj_paired_all.png      (10x3 contact sheet)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

R3 = "/workspace/studies/acs-robust-r3-stress/data"
R2 = "/workspace/studies/acs-robust-r2/data"
OUT = "/workspace/figures/traj_paired"

SEEDS = list(range(1000, 1010))
MARGIN = 100          # steps shown past the slowest method's t_conv
METHODS = [
    ("ACS (fully connected)", f"{R3}/knnref/k19_L250_N20/k19_L250_N20_s{{s}}.npz",
     f"{R3}/knnref/k19_L250_N20_summary.csv"),
    ("RL policy (C1)", f"{R3}/eval/C1_i80_L250_s500/C1_i80_L250_s500_s{{s}}.npz",
     f"{R3}/eval/C1_i80_L250_s500_summary.csv"),
    ("k-NN (k=12)", f"{R2}/knnref/k12_L250_N20/k12_L250_N20_s{{s}}.npz",
     f"{R2}/knnref/k12_L250_N20_summary.csv"),
]

# Sequential blue ramp, steps 100->700 (light -> dark), from the validated palette.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", RAMP)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})

tconv = {name: pd.read_csv(csv).set_index("seed")["t_fire"].to_dict()
         for name, _, csv in METHODS}


def load(seed):
    """-> list of (pos_snaps, snap_ts) per method + row time window T_show."""
    runs = []
    fires = []
    for name, pat, _ in METHODS:
        z = np.load(pat.format(s=seed), allow_pickle=True)
        runs.append((z["pos_snaps"], z["snap_ts"]))
        fires.append(tconv[name].get(seed, -1))
    tmax = max([f for f in fires if f >= 0] or [6000])
    T_show = min(int(tmax) + MARGIN, 6000)
    p0 = [r[0][0] for r in runs]
    assert np.allclose(p0[0], p0[1]) and np.allclose(p0[0], p0[2]), f"unpaired init s{seed}"
    return runs, fires, T_show


def draw_panel(ax, snaps, ts, t_fire, T_show, lims):
    keep = ts <= T_show
    P, t = snaps[keep], ts[keep]          # (S, n, 2), (S,)
    norm = Normalize(0, T_show)
    n = P.shape[1]
    segs = np.concatenate([P[:-1, :, None, :], P[1:, :, None, :]], axis=2)  # (S-1, n, 2, 2)
    segs = segs.transpose(1, 0, 2, 3).reshape(-1, 2, 2)
    seg_t = np.tile(0.5 * (t[:-1] + t[1:]), n)
    lc = LineCollection(segs, cmap=CMAP, norm=norm, linewidths=0.9,
                        capstyle="round", alpha=0.85)
    lc.set_array(seg_t)
    ax.add_collection(lc)
    ax.scatter(P[0, :, 0], P[0, :, 1], s=14, facecolors="none",
               edgecolors=MUTED, linewidths=0.8, zorder=3)
    ax.scatter(P[-1, :, 0], P[-1, :, 1], s=11, color=RAMP[-1], zorder=4)
    if t_fire >= 0:
        i = int(np.searchsorted(t, t_fire))
        i = min(i, len(t) - 1)
        ax.scatter(P[i, :, 0], P[i, :, 1], s=26, facecolors="none",
                   edgecolors="#eb6834", linewidths=0.9, zorder=5)
        label = f"$t_{{conv}}$ = {t_fire}"
    else:
        label = "not converged"
    ax.text(0.03, 0.97, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color=INK2)
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return norm


def row_limits(runs, T_show, pad=0.06):
    xs, ys = [], []
    for snaps, ts in runs:
        k = ts <= T_show
        xs += [snaps[k][:, :, 0].min(), snaps[k][:, :, 0].max()]
        ys += [snaps[k][:, :, 1].min(), snaps[k][:, :, 1].max()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    # equal-aspect square-ish limits
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2 * (1 + pad)
    return (cx - half, cx + half), (cy - half, cy + half)


def legend_handles():
    return [
        Line2D([], [], marker="o", ls="none", markerfacecolor="none",
               markeredgecolor=MUTED, markersize=5, label="start (t = 0)"),
        Line2D([], [], marker="o", ls="none", color=RAMP[-1], markersize=5,
               label="end of window"),
        Line2D([], [], marker="o", ls="none", markerfacecolor="none",
               markeredgecolor="#eb6834", markersize=7, label="at $t_{conv}$ (C2)"),
    ]


def per_seed_figure(seed):
    runs, fires, T_show = load(seed)
    lims = row_limits(runs, T_show)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.1), constrained_layout=True)
    for ax, (name, _, _), (snaps, ts), tf in zip(axes, METHODS, runs, fires):
        norm = draw_panel(ax, snaps, ts, tf, T_show, lims)
        ax.set_title(name, fontsize=10, color=INK, pad=6)
        ax.set_xlabel("x (m)", fontsize=8)
    axes[0].set_ylabel("y (m)", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=Normalize(0, T_show))
    cb = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.025, pad=0.015)
    cb.set_label("time (steps)", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, color=MUTED, labelcolor=MUTED)
    cb.outline.set_visible(False)
    fig.suptitle(f"Paired initial condition — seed {seed}   (N=20, L=250, shown to "
                 f"t = {T_show})", fontsize=10.5, color=INK)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.035))
    path = os.path.join(OUT, f"traj_paired_s{seed}.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def contact_sheet():
    fig, axes = plt.subplots(len(SEEDS), 3, figsize=(10.5, 3.35 * len(SEEDS)),
                             constrained_layout=True)
    for r, seed in enumerate(SEEDS):
        runs, fires, T_show = load(seed)
        lims = row_limits(runs, T_show)
        for c, ((name, _, _), (snaps, ts), tf) in enumerate(zip(METHODS, runs, fires)):
            ax = axes[r, c]
            draw_panel(ax, snaps, ts, tf, T_show, lims)
            if r == 0:
                ax.set_title(name, fontsize=10.5, color=INK, pad=8)
        axes[r, 0].set_ylabel(f"seed {seed}\n(to t = {T_show})", fontsize=8.5,
                              color=INK2)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.006,
                      pad=0.012, aspect=45)
    cb.set_label("normalized time within row window", fontsize=8.5, color=INK2)
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=7, color=MUTED, labelcolor=MUTED)
    cb.outline.set_visible(False)
    fig.suptitle("Convergence trajectories, paired initial conditions — "
                 "ACS(FC) / RL(C1) / k-NN(k=12), N=20, L=250", fontsize=12, color=INK)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.008))
    path = os.path.join(OUT, "traj_paired_all.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for s in SEEDS:
        print(per_seed_figure(s))
    print(contact_sheet())
