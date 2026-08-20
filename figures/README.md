# figures/

Paper/meeting figure pipelines. Outputs (`*.png`, `*.npz`, `stats.json`) are
gitignored per repo convention — regenerate with the scripts below. Everything
is read-only over `studies/*/data` except one cached 2-episode re-roll.

## traj_paired/ — paired-seed convergence trajectories (ACS-FC / RL / k-NN)

- `make_traj_paired.py` — seeds 1000–1009 × 3 methods from existing rollout npz
  (r3 `k19` = FC anchor, r3 `C1_i80`, r2 `k12`). Same seed ⇒ bitwise-identical
  initial condition (asserted on `pos_snaps[0]`). Outputs per-seed 1×3 panels
  (`traj_paired_s<seed>.png`) + 10×3 contact sheet (`traj_paired_all.png`).

## meeting/ — meeting-brief / paper figures

Run order:

1. `compute_stats.py` → `stats.json` — fixed-k fail table (k8–19 × L, n=500),
   policy rows, paired policy-vs-FC/vs-k12 (Wilcoxon), phase stats
   (rank_dev/deg early-vs-steady per seed), straggler-candidate seeds.
2. `cache_profiles.py` → `profiles.npz` — median/IQR time profiles (phi,
   n_comp_r0, deg_mean, rank_dev) over the 500-seed lanes.
3. `reroll_1014.py` → `reroll_1014.npz` — seed-1014 action-capture re-roll for
   C1 and k12 (reproduces archived rollouts to 3e-05; straggler = agent 16).
4. `make_figs.py` → `f1..f7*.png` — criterion bias, fail-vs-k cliff, reliability
   Pareto, phase profiles, degree-by-L, straggler case, straggler in-edges.

Data sources: `studies/acs-robust-r3-stress/data`, `studies/acs-robust-r2/data`.
NOTE: the k12 @ L250/L500 n=500 lanes live in **r2**, not r3 — r3's
`k12_L250_N20/` holds only the 8 gate seeds. Re-roll checkpoint:
`test_results/c2C1_ft40_lmix_260808/manual/checkpoint_000080` (π_E / C1 it80).
