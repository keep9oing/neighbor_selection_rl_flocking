# PLAN.md — acs-robust-r3-stress

## Phase 0 — Grounding + script isolation  [ ]
- [x] Inventory existing coverage, seed sets, script interfaces, runtimes
      (subagent report, 2026-08-08; key facts in RUNLOG entry 1).
- [ ] Copy r2 src/eval_c2.py -> src/eval_c2_r3.py and src/run_knn_refs.py ->
      src/run_knn_refs3.py; change ONLY the STUDY constant to this study dir.
- [ ] Gate G1 (k-NN copy fidelity): k12@L250 seeds 1000-1007, bit-exact vs r2.
- [ ] Gate G2 (policy copy fidelity): C1@L250 seeds 1000-1003, dt=0 dJ=0.

## Phase 1 — T1 k-NN 500-seed lanes + T3 32-seed cells  [ ]
- [ ] T1: k 13,14,16,18 x L 125/250/500 + k12@L125, seeds 1000-1499,
      15 workers (one call per L, comma k-list; ~4-5 min per lane).
      [Amendment A1: k20@N20 impossible -> k18.]
- [ ] T3 (after T1 frees workers): k 12,16,19 @ L 375 and 750; k7 @ N10
      L177; k26 @ N40 L354; seeds 1000-1031, 10 workers.

## Phase 2 — T2 policy wave 1 (concurrent with Phase 1, 30 workers)  [ ]
- [ ] C1_i80 @ L250 x 500 seeds, 15 workers (~65-70 min).
- [ ] C1_i80 @ L500 x 500 seeds, 15 workers (~65-70 min).

## Phase 3 — T4 policy wave 2 (tier-2, after wave 1)  [ ]
- [ ] C1_i80 @ L125 x 500 seeds, 15 workers.
- [ ] R1_i110 @ L125 x 500 seeds, 15 workers.

## Phase 4 — Analysis + pre-registered verdicts  [ ]
- [ ] src/stress_stats.py: reads r3 + r2 (+ frontier_L.csv) read-only;
      emits data/stress_stats.csv + verdict block per PROBLEM definitions
      (H-A per policy x L, H-B b1-b3, H-C; frontier table incl. CVaR10).
- [ ] Sanity: k12@L250/L500 rows re-derived from r2 npz must reproduce the
      published 460/500, 455/500 and audit medians before any new claim.

## Phase 5 — Wrap-up  [ ]
- [ ] REPORT_KO.md (Korean; honest verdicts incl. any claim weakening).
- [ ] RUNLOG wall-clock reconciled against artifact mtimes.
- [ ] Memory project_success_criterion.md item 6; kill monitors/bg tasks;
      offer commit (explicit approval only).

## Design decisions already made (do not relitigate without user input)

| Decision | Rationale |
|---|---|
| Verdicts + arms pre-registered in PROBLEM.md before data | audit lesson: no post-hoc criteria fitting |
| 500-seed set = 1000-1499, 32-seed = 1000-1031 | pairs bit-for-bit with r2/S1 data |
| k13/k14 get full 500-seed lanes (no 32-seed screen step) | k-NN lane = ~4 min; screening saves nothing |
| L125 k-NN lanes promoted into T1 (32-seed rows sliced from them) | supersets; avoids duplicate cells |
| T4 (policy L125) is tier-2 / cuttable with explicit log | H-C judgeable on {250,500} if resources bind |
| FC and A60@L500 not re-run at n=500 | FC success-safe + J-dominated; A60 off-scale; stated as limitations |
| L375/L750 refs at 32 seeds only | extrapolation axis secondary this round; first-ever refs there is already new signal |
| Worker cap: <= 45 concurrent across all pools | machine constraint <= 64 threads, headroom for OS/main |
| New data isolated under r3-stress; r2/r1/predecessor read-only | r2 is a closed, committed study; provenance hygiene |
