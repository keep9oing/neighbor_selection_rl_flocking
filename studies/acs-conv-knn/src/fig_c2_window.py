"""fig7: C2 window/form sensitivity (from c2_sweep_summary.csv).

Panels: (1) premature-fire and false-positive rates vs W (p2 vs p2p, eps=0.05);
(2) median t_fire on the good-policy set vs W; (3) fraction of good-set fires
within 1000/1500 steps vs W.  Chosen config W=300 marked.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STUDY = "/workspace/studies/acs-conv-knn"
BLUE, ORANGE, GREEN, CRIT = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
EPS = 0.05
W_PICK = 300


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    df = pd.read_csv(os.path.join(STUDY, "data", "c2_sweep_summary.csv"))
    d = df[df.eps == EPS]
    p2, p2p = d[d.form == "p2"], d[d.form == "p2p"]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    ax = axes[0]
    ax.plot(p2p.W, 100 * p2p.prem10, "o-", color=BLUE, label="premature >10% (p2p)")
    ax.plot(p2.W, 100 * p2.prem10, "o--", color=BLUE, alpha=0.45, label="premature >10% (p2)")
    ax.plot(p2p.W, 100 * p2p.fp, "s-", color=ORANGE, label="false positive (p2p)")
    ax.plot(p2.W, 100 * p2.fp, "s--", color=ORANGE, alpha=0.45, label="false positive (p2)")
    ax.set_xlabel("window W (steps)"); ax.set_ylabel("rate (%)")
    ax.set_title("error rates vs W  (eps=5%)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    ax.plot(p2p.W, p2p.t_med_good, "o-", color=BLUE)
    ax.set_xlabel("window W (steps)"); ax.set_ylabel("median t_fire (steps)")
    ax.set_title("detection time, good-policy set (p2p)", fontsize=10)

    ax = axes[2]
    ax.plot(p2p.W, 100 * p2p.le1000_good, "o-", color=BLUE, label="within 1000 steps")
    ax.plot(p2p.W, 100 * p2p.le1500_good, "o-", color=GREEN, label="within 1500 steps")
    ax.set_xlabel("window W (steps)"); ax.set_ylabel("fires in time (%)")
    ax.set_ylim(70, 102)
    ax.set_title("fit inside training episode cap", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    for ax in axes:
        style(ax)
        ax.axvline(W_PICK, color=CRIT, linestyle=":", linewidth=1.2)
    fig.suptitle("C2 stationarity/cohesion window sensitivity (1,648 runs; phi>0.98 hold 50 fixed)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = os.path.join(STUDY, "figs", "fig7_c2_window.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
