# PLAN.md

> **Purpose of this file**: Living work plan for this study. Defines HOW we attack
> PROBLEM.md: phases, experiment designs (variables, levels, seeds, metrics), and
> current status per phase (`[ ]` todo / `[~]` in progress / `[x]` done / `[!]`
> blocked-or-revised). A fresh Claude session reads this to know what to do next.
> Update statuses and design details here as they firm up; log actual executions in
> RUNLOG.md.

## Phase 0 — Codebase grounding  [x]

Extracted (no repo code modified) → condensed into `NOTES_env.md`:
ACS equations, metric formulas, termination logic + how to disable via config,
k-NN/disc baseline contracts, standalone rollout pattern, env introspection.

## Phase 1 — Experiment harness  [x]

`src/common.py` (rollout + 15 per-step metric series + npz writer, own metrics
cross-checked against env's — diff 0.0) and `src/run_sweep.py` (multiprocessing
CLI, threads pinned). Smoke + calibration passed; 6000-step horizon validated
(cohesive runs plateau ≤ ~2000; fragmented diverge linearly forever).

## Phase 2 — Main k-NN sweep  [x]

`data/main/`: k ∈ {1..8,10,12,15,19} × L ∈ {125,250,500} × 32 seeds × 6000 steps
(N=20). Plus `data/disc/`: disc radii {25,44,60,87.5,125} m × 16 seeds (L=250).
Optional N ∈ {10,40} extension NOT run (deferred, see Phase 5).

## Phase 3 — Analysis  [x]

`src/analyze.py` → `data/summary_main.csv` + 5 figures in `figs/`.
Headline findings (details in RUNLOG + REPORT_KO):
- Current criterion passes only at k=19 (FC); goal 42 = r0/sqrt2 = FC-equilibrium
  anchor (measured FC sigma_p 39.8).
- Cohesive-equilibrium sigma_p ≈ 198*k^(-0.53), independent of initial density.
- Fragmentation is a separate stochastic failure mode (k≤2 always, k=4-7 partial).
- phi>0.99 fast for every cohesive k; sigma_v<0.1 == phi>0.99998, marginal/slow.
- Selected-edge mean distance ≈ 0.91*r0 constant over k — what ACS actually
  regulates (theoretical basis for level-free criteria).

## Phase 4 — Criterion discussion & report  [x]

`src/criteria_eval.py`: proposed criterion C1 (phi>0.99/50 + single r0-component
/500 + relative sigma_p change <2%/500) → 100% detection on cohesive, 0.6% false
positive, times comparable to current criterion at FC. Candidates A-E compared in
`REPORT_KO.md` (user-facing, Korean) with recommendation = C1 (+ selected-edge
distance as auxiliary), incl. RL reward-shaping implications.

## Phase 5 — Possible follow-ups (not started, awaiting user direction)  [ ]

- [x] N-dependence: DONE (batches n10/n40, `src/analyze_ndep.py`, fig6).
      sigma_p_FC ≈ 40 m independent of N; collapse sigma_p/sigma_p_FC ≈
      1.05·(k/(N-1))^(-0.53); edge-length constancy and C1 behavior hold.
- [x] C1 robustness under stochastic topologies: DONE (batches stress_random /
      nn_hardtopk; see RUNLOG). Detection 100% at phi 0.97.
- [!] Kill the false positives: superseded by C2 (p2p band form); residual 1.9%
      at W=300 accepted as training label noise (1.0% at W=500 for offline eval).
- [ ] Reward-shaping fix experiment: replace std_pos_target=39.5 anchor with
      cohesion/alignment shaping and retrain.

## Phase 6 — C2 finalization + evaluation-metric design  [~]

- [x] User decisions folded in: phi threshold 0.98 hold 50; spatial = maintenance
      (no absolute level); single-flock-only success (delegated choice; rationale
      in RUNLOG 2026-08-06); early termination on success only.
- [x] Window/form/eps sweep (`src/criteria_c2_sweep.py`, 1,648 runs x 48 configs)
      → C2 = phi>0.98/50 + all-agents single r0-component/300 + rel p2p(sigma_p)
      over 300 < 5%. Headline: 100% detection on real strategy families, fp 1.9%,
      premature(>1.1x final) 9.6% (median ratio 1.011), good-set median t_fire
      ~550, 100% <= 1500 steps. `figs/fig7_c2_window.png`.
- [x] J-metric preview (`src/j_metric_preview.py`): return-to-convergence J +
      reliability + degree table across FC / knn / disc / NN ckpt / random.
- [!] Training harness: HANDED OFF to the successor study
      `studies/acs-c2-train/` (2026-08-06). User decisions there: repo edits
      now ALLOWED (additive/flag-gated), variants A (bernoulli) + B (threshold,
      un-rigged) on cuda:1/cuda:3, small success bonus, careful dist_aux
      anneal. Implementation happens in a NEW session via its
      KICKOFF_PROMPT_KO.md. This study is now reference material.
- [ ] Soft-selection sampled checkpoint validation (optional, bracketed).

## Working rules

- Never modify existing repo code; everything under studies/acs-conv-knn/.
- data/, figs/, logs/ are gitignored; only .md/.py/.gitignore may be committed, and
  only with explicit user approval.
- CPU only (≤64 threads used: sweeps ran with 56 workers). No GPU.
