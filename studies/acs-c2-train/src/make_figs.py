"""Report figures for study acs-c2-train (Phase 4 forensics).

fig1_profiles.png — median-over-seeds time profiles of (a) mean selected
degree and (b) rank-deviation for A it60 vs the oldNN kNN-mimic, with k=12
as the fixed-k reference. Shows phase-dependent adaptivity vs frozen mimicry.

fig2_case1014.png — seed 1014 (the k=12 failure): r0-proximity component
count over time for k=12 vs A it60/it50; A merges the flock, k=12 never does.

Colors: dataviz reference palette slots 1-3 (documented all-pairs validated,
light mode; node unavailable to re-run the validator — values used unchanged).
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = "/workspace/studies/acs-c2-train"
PRED = "/workspace/studies/acs-conv-knn"

C_A = "#2a78d6"      # slot 1 blue  — A it60 (candidate)
C_OLD = "#eb6834"    # slot 2 orange — oldNN mimic
C_K12 = "#1baf7a"    # slot 3 aqua  — k=12 fixed reference
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"


def load_series(pattern, key, tmax):
    out = []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f, allow_pickle=True)
        if key not in z.files:
            continue
        s = z[key][:tmax + 1].astype(np.float64)
        if len(s) < tmax + 1:
            s = np.concatenate([s, np.full(tmax + 1 - len(s), np.nan)])
        out.append(s)
    return np.vstack(out) if out else None


def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def fig1(tmax=800):
    a_deg = load_series(os.path.join(STUDY, "data/eval/A_i60_s32/*.npz"), "deg_mean", tmax)
    a_rd = load_series(os.path.join(STUDY, "data/eval/A_i60_s32/*.npz"), "rank_dev", tmax)
    o_deg = load_series(os.path.join(STUDY, "data/eval/oldNN32/*.npz"), "deg_mean", tmax)
    o_rd = load_series(os.path.join(STUDY, "data/eval/oldNN32/*.npz"), "rank_dev", tmax)
    k_deg = load_series(os.path.join(PRED, "data/main/k12_L250_s*.npz"), "deg_mean", tmax)

    t = np.arange(tmax + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), facecolor=SURFACE)

    ax = axes[0]
    ax.plot(t, np.nanmedian(a_deg, 0), color=C_A, lw=2, label="A it60 (learned)")
    ax.plot(t, np.nanmedian(o_deg, 0), color=C_OLD, lw=2, label="old NN (kNN mimic)")
    ax.plot(t, np.nanmedian(k_deg, 0), color=C_K12, lw=2, label="k-NN k=12")
    style_ax(ax, "a. Mean selected degree (median over 32 seeds)", "env step", "degree (off-diag)")
    ax.set_ylim(0, 20)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")

    ax = axes[1]
    ax.plot(t, np.nanmedian(a_rd, 0), color=C_A, lw=2, label="A it60 (learned)")
    ax.plot(t, np.nanmedian(o_rd, 0), color=C_OLD, lw=2, label="old NN (kNN mimic)")
    ax.axhline(0.0, color=C_K12, lw=2)
    ax.annotate("k-NN ≡ 0 by construction", xy=(tmax * 0.55, 0.012),
                color=INK2, fontsize=8)
    style_ax(ax, "b. Rank deviation (fraction of selected edges\noutside nearest-deg set)",
             "env step", "rank deviation")
    ax.set_ylim(-0.02, 0.45)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")

    fig.tight_layout()
    out = os.path.join(STUDY, "figs", "fig1_profiles.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


def fig2(tmax=1500):
    seed = 1014
    z_k = np.load(os.path.join(PRED, "data", "main", f"k12_L250_s{seed}.npz"), allow_pickle=True)
    z_a60 = np.load(os.path.join(STUDY, "data/eval/A_i60_s32", f"A_i60_s32_s{seed}.npz"), allow_pickle=True)
    z_a50 = np.load(os.path.join(STUDY, "data/eval/A_i50_s32", f"A_i50_s32_s{seed}.npz"), allow_pickle=True)

    fig, ax = plt.subplots(figsize=(6.4, 3.2), facecolor=SURFACE)
    t = np.arange(tmax + 1)
    for z, c, lab in ((z_a60, C_A, "A it60"), (z_a50, "#7ba7dd", "A it50"), (z_k, C_K12, "k-NN k=12")):
        comp = z["n_comp_r0"][:tmax + 1]
        ax.plot(t[:len(comp)], comp, color=c, lw=2, label=lab)
    style_ax(ax, f"Seed {seed}: r0-proximity components over time — the k=12 failure case",
             "env step", "# components")
    ax.set_ylim(0.5, 6.5)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    out = os.path.join(STUDY, "figs", "fig2_case1014.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)

    # numeric case notes
    for name, z in (("k12", z_k), ("A_i60", z_a60), ("A_i50", z_a50)):
        comp = z["n_comp_r0"]
        phi = z["phi"]
        print(f"  {name}: n_comp end={comp[-1]:.0f}, phi end={phi[-1]:.3f}, "
              f"n_comp@1500={comp[min(1500, len(comp)-1)]:.0f}")


def failure_1016():
    z = np.load(os.path.join(STUDY, "data/eval/A_i60_s32", "A_i60_s32_s1016.npz"), allow_pickle=True)
    comp, phi, s = z["n_comp_r0"], z["phi"], z["s_ent"]
    print("A it60 seed 1016 (its only failure): "
          f"n_comp end={comp[-1]:.0f}, phi end={phi[-1]:.4f}, sigma_p end={s[-1]:.1f}, "
          f"n_comp min={np.nanmin(comp):.0f}, frac steps single={np.nanmean(comp[1:]==1):.2f}")


if __name__ == "__main__":
    os.makedirs(os.path.join(STUDY, "figs"), exist_ok=True)
    fig1()
    fig2()
    failure_1016()
