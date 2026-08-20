# RUNLOG.md — acs-robust-train

> Append-only log of what was ACTUALLY executed (commands, configs, numbers,
> deviations from PLAN). Newest at the bottom.

## 2026-08-07 ~00:20 — Study created (same session as predecessor close-out)

- Direction + GPUs user-approved (robust single policy; cuda:1/3). Success
  criteria pre-registered in PROBLEM.md BEFORE any training.
- Env recon done pre-edit (site anchors in PLAN Phase 0/1). Key facts:
  r0=60; obs normalized by static L/2 at env.py:1198 (ego) and :1261
  (centralized/aux target); global_stats sigma_p/(L/2) at :1140; reset init
  at :463; C2 machinery + c2_shaping reward are level-free (no edits needed
  for L mixing).
- eval_c2.py (predecessor src) builds its eval env config FRESH via
  acs-conv-knn build_config — new-mode checkpoints will need the obs mode
  read back from params.json (PLAN Phase 4 item).

## 2026-08-07 ~01:25 — Phase 1+2 done (env + gates + script + smokes); Phase 3 LAUNCHED

- envs/env.py additive edits: EnvConfig initial_position_bound_pool /
  obs_position_scale; validation (pool: acs + non-periodic + c2-or-fixed-len;
  r0_log: non-periodic); _sample_episode_bound() in reset (np_random draw
  ONLY when pool set -> default RNG stream untouched); episode_position_bound
  property routed through: reset init, ego obs normalization, centralized/aux
  obs, global_stats sigma term. r0_log transform: d -> unit(d)*log1p(|d|/r0)
  (ego + centralized), sigma_p -> log1p(sigma_p/r0). Periodic-only branches
  untouched (pool forbids periodic).
- Gates ALL GREEN:
  * regression_check --compare ref_pre.json: 3 cases byte-identical;
    test_baselines.py green.
  * src/scale_unit_check.py (new): pool coverage/bounds/seed-determinism;
    legacy obs follows EPISODE L (convention d = p_j - p_i); r0_log exact
    math (max dev 2.4e-7), diag 0, direction preserved, max |obs| 2.15 at
    L=500; aux target + global_stats mappings exact in both modes.
  * C2 parity at L=500 (kNN-12, seeds 1000-1003): env t_conv == offline
    t_fire EXACTLY (559/563/528/589) — C2 machinery is level-free in
    practice, not just by design. r0_log preserves the kNN trajectory
    exactly (same t_conv 559: log1p monotone -> same ordering) — baseline
    behavior is obs-mode-invariant.
  * Side fact: kNN(12) fires ~530-590 at L=500 -> training cap 2000 ample.
- train_robust.py (repo root): single script, --variant legacy|r0log ->
  cuda:1|cuda:3, run c2R1_lmix_legacy_260807 | c2R2_lmix_r0log_260807.
  A-recipe PPO (entropy 1e-3 FLAT — A2 anneal not repeated), L pool
  {125,250,500} in training env only (eval env fixed L=250, 16ep/10it),
  max_time_steps 2000, stop 120 iters, ckpt every 10 + end. Smokes (2-iter
  CPU) both variants: gnorm 1.5-2.8 nonzero, entropy ~262 (=max), sat_p_dev
  0.02-0.06, aux_mse finite — same healthy signature as the A-line smoke.
- Phase-4 prep: src/eval_c2.py copied from predecessor study with obs-mode
  passthrough (reads obs_position_scale from the ckpt's params.json; asserts
  pool None in eval builds); compare_frontier.py copied as-is.
- LAUNCH ~01:25: R1 pid 240505 (cuda:1, 9.5 GB, ~76% util), R2 pid 241927
  (cuda:3, 9.9 GB). NOTE the first R2 launch attempt died instantly on the
  session-cwd trap (&-chaining ran it from src/) — relaunched with the
  absolute script path; R1 unaffected. Monitor: src/monitor_runs.py (polls
  result.json/120 s -> logs/monitor.log, exits on first process exit or
  error signature). ETA ~120 iters x ~250-270 s + evals ~= 8.5-9.5 h ->
  ~10:00-11:00.

## 2026-08-07 ~10:20 — both runs COMPLETE (120/120); 16-seed screens; candidates picked

- Both TERMINATED cleanly ~09:37 (8.2 h each, no crash/restart).
- Online eval trajectories (16ep, fixed L=250, argmax; result.json):
  * R1: it10-30 argmax DEGENERATE (0.00-0.06) — L-mix slows readout
    formation vs the pure-250 A-line (0.875 at it10) — then recovers:
    0.81/386 (it40) -> 1.00 from it70, J 318->247->235->212->205->211.
  * R2: readout works from it20 (0.88/262) but J NEVER descends
    (min 242 at it10; late 325-409). succ 1.00 from it100.
- Offline 16-seed screens (argmax, seeds 1000-1015, both L=250 and L=500):
  * R1 it100: 16/16-214.0 | 16/16-310.3   (L250 | L500)
  * R1 it110: 16/16-212.5 | 16/16-255.3   <- best R1
  * R1 it120: 16/16-223.2 | 16/16-263.4
  * R2 it60:  14/16-320.8 | 16/16-324.2
  * R2 it110: 16/16-325.8 | 16/16-408.2   <- best R2 (success-first)
  * R2 it120: 16/16-348.6 | 16/16-440.7
- Readings:
  * RELIABILITY: perfect nearly everywhere (16/16 x 11 of 12 screens,
    incl. ALL L=500 screens) — L-mix bought success-robustness.
  * EFFICIENCY: far off the k12 frontier at every L for both runs
    (R1 +50..+90 J; R2 ~2x frontier). The pre-registered J bar (k12+5)
    is out of reach — no 32-seed correction closes +50.
  * ABLATION ANSWER (early): r0_log scale-free obs HURTS (R2 uniformly
    worse than R1 by ~100 J); per-episode L/2 normalization (R1) is the
    better representation under L-mix.
  * it110 -> it120 slightly WORSE at both L in both runs -> plateau, not
    still-descending; a length extension is not supported by evidence.
- Decision (pre-registered success-first rule): grid candidates =
  R1 it110 and R2 it110. Launched ~10:20: 32-seed protocol grid
  L={125,250,500} for both + L={375,750} inter-/extrapolation probes for
  R1 it110 (success-robustness is the remaining live claim; probes test it
  at unseen scales).

## 2026-08-07 ~11:05 — GRID FINALS + formal judgment (grid_judge.py)

- R1 it110 (32 paired seeds per L, argmax, offline C2):
  * L=125: 32/32, t_conv 517, J_med 189.9 (rank_dev e/ss 0.354/0.308)
  * L=250: 32/32, t_conv 535, J_med 203.6 (0.299/0.061)
  * L=375: 32/32, t_conv 598, J_med 220.3 (0.321/0.007)  [UNSEEN interp]
  * L=500: 32/32, t_conv 654, J_med 244.8 (0.336/0.000)
  * L=750: 32/32, t_conv 772, J_med 313.8 (0.408/0.000)  [UNSEEN extrap 1.5x]
  -> 160/160 successes across a 6x scale range.
- R2 it110: L125 32/32-313.2 | L250 32/32-322.7 | L500 32/32-411.8.
- Paired tests (grid_judge.py, refs frontier_L.csv):
  * R1 vs k12: dJ +46.1 (p=0.004) / +35.1 (p=0.039) / +89.8 (p=0.399);
    pooled +57.3 (p=0.116, n=94). vs FC: -75.4 (p=0.012) / -44.9 (p=0.042)
    / +55.3 (p=0.539).
  * R2 vs k12 pooled +173.8 (p=7e-13) — uniformly far worse than R1.
- VERDICT vs pre-registered criteria: PRIMARY NOT MET (success half OK at
  every L — 32>=31/31/32 with strict wins at 125/250; J half FAIL at every
  L: +35..+90 over the k12+5 bar). STRETCH NOT MET. Phase-2b (failure
  weighting) NOT triggered per plan: no reliability gap exists.
- MECHANISM of the J gap (grid CSVs vs k12 refs): t_conv is k12-grade
  (+22/+18/+91 steps -> time-term +2..+9 J only) => the gap is almost
  entirely TURN ENERGY during transit, not slowness. Selection structure is
  healthy and scale-adaptive: deg_early 12.5 -> 10.0 as L grows 250->750,
  rank_dev early 0.30-0.41 at all scales, hold-phase densifies to FC
  (J-free). Diagnosis: c2_w_ctrl=0.1 too weak to compress maneuver cost.
- A it60 (specialist) probes at unseen L=375/750 launched (32 seeds) to
  decide how much of the reliability robustness is L-mix-specific vs
  already present in the specialist.

## 2026-08-07 ~11:25 — specialist probes landed; report finalized

- A it60 at UNSEEN scales (32 seeds): L=375 31/32, t_conv 551, J_med 181.3
  (rank_dev e/ss 0.227/0.050); L=750 32/32(!), t_conv 910, J_med 401.0.
- Specialist-vs-generalist across 5 scales: success 158/160 vs 160/160
  (L-mix's reliability gain = +2 episodes — success robustness was nearly
  free already); efficiency SHAPE differs — specialist wins at L<=375
  (-20/-48/-39 J), explodes at 750 (401 vs 314, +87 for R1); crossover
  L~500 (248.9 vs 244.8). L-mix buys out-of-scale efficiency flatness at
  the cost of in-distribution efficiency.
- REPORT_KO.md finalized (table + 4.4 rewritten with the crossover story).
  All Phase-4 outputs in data/eval/ (labels R1_i110_L*, R2_i110_L*,
  A60_L375/L750_s32); judgment via src/grid_judge.py.

## 2026-08-07 (afternoon) — STUDY CLOSED; round-2 decisions; commits approved

- Final report + discussion delivered. User decisions for the NEXT round
  (executed by the successor study `studies/acs-robust-r2/`, new session):
  * F1 = L-mix fine-tune from specialist A it60; C1 = curriculum variant
    implemented as L-mix fine-tune from A it40 (reuses the existing 250-run
    as curriculum stage 1 — zero redundant compute). One GPU each.
  * Training pool UNCHANGED {125,250,500} (single-variable discipline vs R1).
  * Small-L probe (L=75) approved (eval-only this round).
  * Paper-evidence sweeps approved: big-n reliability (500 seeds) and
    N-axis evaluation. Distillation baseline DEFERRED to paper stage.
  * New session stop rule: at context 319K (cat /tmp/ctx) wrap to a clean
    boundary and hand off; kill monitors/background tasks BEFORE ending so
    notifications cannot revive the finished session.
- Commits approved and executed as proposed: code commit (envs/env.py +
  train_robust.py) + this study commit. Successor-study docs + kickoff
  remain UNCOMMITTED by design (session artifacts; same handling as the
  previous round's kickoff files).
