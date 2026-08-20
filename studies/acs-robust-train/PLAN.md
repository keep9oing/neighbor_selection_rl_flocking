# PLAN.md — acs-robust-train

> Phases are ordered; statuses `[ ]` todo / `[~]` in progress / `[x]` done /
> `[!]` blocked-or-revised. Log actual executions in RUNLOG.md. File:line
> anchors verified 2026-08-07 pre-edit; re-grep before patching blindly.

## Phase 0 — Grounding  [x]
- [x] Predecessor REPORT/RUNLOG absorbed (same session as study creation);
      env recon done: obs normalization sites env.py:463 (reset init),
      :1192-1198 (ego obs non-periodic), :1260-1261 (centralized/aux target),
      :1140 (global_stats sigma_p), render :948/:978; r0=60, L default 250
      (ControlConfig); eval_c2.py builds env config FRESH via predecessor
      build_config (needs obs-mode passthrough for new ckpts).

## Phase 1 — Env: L pool + scale-free obs (flag-gated)  [x]
File: `envs/env.py`. New `EnvConfig` fields (defaults = legacy):
```python
initial_position_bound_pool: Optional[List[float]] = None  # per-episode L
obs_position_scale: str = "legacy"   # "legacy" | "r0_log"
```
- Sampling: `_sample_episode_bound()` called from reset() AND custom_reset()
  BEFORE position init: `self._episode_L = np_random.choice(pool)` if pool
  else None. Property `episode_position_bound` returns episode value or the
  static config value. Same episode seed -> same (L, init) pair (np_random
  drawn in fixed order).
- Route through the property: reset init l2 (:463), ego obs normalization
  (:1198), centralized obs normalization (:1261), global_stats sigma_p
  (:1140), render sizes (:948/:978). Periodic branch (:586) untouched
  (pool forbids periodic).
- r0_log transform (ego obs non-periodic branch + centralized obs + the
  global_stats sigma_p term):
  `d -> d * log1p(|d|/r0)/max(|d|,1e-12)` (direction preserved, |.| ->
  log1p(|.|/r0), diag/self rows stay 0); `sigma_p -> log1p(sigma_p/r0)`.
  Range check: |d|max ~ L*sqrt(2) -> log1p(11.8)=2.55 at L=500 (vs legacy
  ~2.83 corner) — same numeric regime, model hyperparams carry over.
- Validation (_validate_config): pool requires task acs + non-periodic +
  (termination_mode=="c2" or use_fixed_episode_length) + all values > 0;
  r0_log requires non-periodic.
- Gates:
  - [x] `src/regression_check.py --compare` (acs-c2-train's ref_pre.json):
        3 cases byte-identical (defaults untouched). test_baselines.py green.
  - [x] New `src/scale_unit_check.py`: (a) pool sampling determinism (same
        seed -> same L sequence; ~30 resets cover all pool values; positions
        bounded by episode L/2); (b) r0_log: directions preserved, diag 0,
        magnitudes == log1p(r/r0), finite, range within +-3 at L=500;
        (c) legacy+pool: obs scale switches with episode L.
  - [x] C2 online/offline parity at L=500 (k-NN 12, 4 seeds, termination c2):
        env t_conv == offline judge t_fire exactly (mirror of predecessor
        Gate 2 at 250).

## Phase 2 — Train script + smoke  [x]
`train_robust.py` (repo root; single script, both variants):
- Pre-parse argv BEFORE torch/ray import: `--variant legacy|r0log` ->
  CUDA_VISIBLE_DEVICES "1"|"3", RUN_NAME c2R1_lmix_legacy_260807 |
  c2R2_lmix_r0log_260807. `--smoke`, `--resume` as before.
- Env config vs train_c2_a.py deltas ONLY:
  `initial_position_bound_pool=[125,250,500]`, `max_time_steps=2000`
  (L=500 headroom; C2 success-only early stop keeps successes short),
  `obs_position_scale` per variant. Absolute default-yaml path (cwd trap).
- PPO block = proven A recipe: entropy_coeff 1e-3 FLAT (NO schedule — the
  A2 anneal was proven a no-op for bernoulli), lr/clip/batch identical,
  `stop=120 iters` (~20% more than A for the 3x-diverse task; ~9 h at
  ~250-270 s/iter), checkpoint_freq 10 + at_end.
- Eval during training: every 10 iters, 16 eps, explore=False, eval env
  FIXED L=250 (`initial_position_bound_pool=None` override in eval
  env_config) for continuity with A-line eval traces; offline grid is the
  arbiter. NOTE eval cap 2000 (vs A's 1500) — minor comparability caveat.
- [x] 2-iter CPU smoke both variants: gnorm nonzero, entropy ~ln2*400,
      obs finite; R2 smoke additionally hits all three L (episode lens vary).

## Phase 3 — Launch + monitor  [x]
> DONE 2026-08-07: both runs 120/120 clean (01:25-09:37, 8.2 h); no restarts.
- [x] Launch both (cuda:1 R1, cuda:3 R2), logs logs/train_r{1,2}.log.
- [x] Background monitor: poll result.json ~120 s; append per-iter line
      (len/succ/J/eval) to logs/monitor.log; EXIT on both-TERMINATED or
      error signature or either process dead -> session notification.
- Playbook (from predecessor): eval 0/16 x3 -> that variant restarts with a
  documented single-knob change; training succ fine but J plateau >> 200 by
  iter ~60 -> note, do NOT touch (offline decides); crash -> --resume once,
  then diagnose.

## Phase 4 — Offline grid evaluation  [x]
> DONE 2026-08-07 ~11:00: R1/R2 it110 grids + probes + judgment complete;
> PRIMARY NOT MET (J), success-half exceeded (160/160 across 5 scales);
> r0_log ablation NEGATIVE. See RUNLOG + REPORT_KO.
- [x] Copy eval_c2.py -> src/ with: STUDY path update; run_one reads the
      ckpt's params.json env_config to apply obs_position_scale (and asserts
      pool is None in eval builds). --bound flows as before.
- [x] Checkpoint screening (16 seeds, L=250 first) on it60..it120 of both
      runs -> pick 2-3 candidates per run by (success, then J).
- [x] Grid protocol for candidates: 32 paired seeds x L={125,250,500};
      single checkpoint per run enters the primary judgment. Per-L paired
      k12 references: extract per-seed J/success at each L from predecessor
      npz (extend frontier_L.py to dump per-seed rows if csv is
      summary-only).
- [x] Probes: L=375 (interpolation), L=750 (extrapolation) 32 seeds,
      descriptive; N-grid optional if time allows.
- [x] Forensics: rank_dev/degree profiles per L (does adaptivity structure
      adapt to scale?); R1-vs-R2 representation comparison.
- [x] Judgment vs pre-registered criteria (PROBLEM.md). If reliability gap
      (success < k12 somewhere): Phase-2b failure-weighted follow-up run
      (single knob, documented) — else skip.

## Phase 5 — Wrap-up  [x]
- [x] REPORT_KO.md here (numbers-first, honest significance); RUNLOG
      complete; pointer line appended to predecessor REPORT_KO.
- [x] Korean report to user; commit only on explicit approval (no AI
      attribution, never push). -> Delivered 2026-08-07; user approved the
      2-commit split (code / study) and round-2 plan -> studies/acs-robust-r2.

## Design decisions already made (do not relitigate without user input)

| Topic | Decision | Source |
|---|---|---|
| Study direction | robust single policy across L (over L=250-deepening / B-line / stop) | user choice 2026-08-07 |
| GPUs | cuda:1 (R1), cuda:3 (R2) | user re-confirmed 2026-08-07 |
| Bar | fixed k=12 across L grid (success >= 31/31/32, J_med <= +5 of 155/160/165.6, pooled paired win) | proposed in options, user selected |
| Head | bernoulli only (A-line recipe, entropy 1e-3 flat) | proven; B out of scope per user choice |
| Mechanism pair | R1 L-mix only vs R2 L-mix + r0_log obs (single-variable ablation) | this plan |
| Criterion/metrics | C2 + J + settled eval scheme, unchanged from predecessor | user-settled, level-free |
| Failure weighting | deferred to conditional Phase-2b | this plan (keeps single-variable discipline) |
