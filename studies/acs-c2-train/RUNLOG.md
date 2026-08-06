# RUNLOG.md — acs-c2-train

> **Purpose of this file**: Append-only log of what was ACTUALLY executed in this
> study (commands, configs, numbers, deviations from PLAN). Newest entries at the
> bottom. Keep entries dense and citable — a fresh session reconstructs state
> from here.

## 2026-08-06 — Study created (planning session, no implementation yet)

- Created by the criterion/metric-design session (predecessor:
  `studies/acs-conv-knn/`, see its RUNLOG 2026-08-06 entries for: C2 window
  sweep, evaluation-metric decisions, architecture saturation diagnosis).
- User decisions recorded in PROBLEM.md/PLAN.md: variants A+B, B un-rigged;
  repo edits allowed (additive/flag-gated); success bonus small (10);
  dist_aux annealed carefully (1.0->0.2/400k, never 0); GPUs cuda:1 (A) and
  cuda:3 (B); implementation + training happen in the NEXT session using
  KICKOFF_PROMPT_KO.md.
- Nothing implemented, nothing trained, nothing committed yet.

## 2026-08-06 — Phase 0 done (grounding + regression baseline)

- Read PROBLEM/PLAN + predecessor RUNLOG 2026-08-06 entries + NOTES_env.md;
  read ppo.py (full), env.py (full), train_hardtopk.py, grad_logging_ppo.py,
  callbacks.py, predecessor src (common.py / criteria_c2_sweep.py /
  j_metric_preview.py).
- **Plain-binary logits path EXISTS** (ppo.py:341-350): when
  continuous_action=False, top_k=None, hard_top_k=False -> logits concat(-s,+s)
  with s = scale_factor*att (in-place scaled at :301) + 1e9 on diag; masked/
  padded att arrives as -1e9 from local_forward/generator -> s=-1e8, never
  selected. Variant A rides this path unchanged (diag forced via 1e9, not +10 —
  legacy exact; deviation from PLAN text, behaviorally identical).
- Ray 2.1.0 facts verified in installed source:
  * `on_learn_on_batch` IS called on the multi-GPU path
    (torch_policy_v2.py:736 inside learn_on_loaded_batch, once per minibatch)
    AND in learn_on_batch (:606) -> callback-driven dist_aux anneal is viable.
  * `model_gpu_towers[0] is self.model` (assert at :715) -> setting attrs on
    policy.model + iterating model_gpu_towers covers all towers.
  * `policy.global_timestep` exists (policy.py:341, updated via global vars).
  * int8 Box(0,1,(N,N)) action space -> TorchMultiCategorical with
    input_lens=[2]*N^2, num_outputs=2N^2 (catalog.py:259-271) — matches
    model's (-s,+s) pair layout.
  * evaluation keys in 2.1.0: evaluation_interval / evaluation_duration /
    evaluation_duration_unit / evaluation_num_workers / evaluation_config.
  * EnvContext has .worker_index / .vector_index (env_context.py:51-52) —
    will use to de-duplicate env seeds across workers in the NEW train
    scripts only (legacy scripts all seed 42 -> 16 identical init streams).
- Offline-judge parity design note: offline series index 0 = initial state and
  pandas rolling(W) includes it, so the env-side C2 buffers must get an
  initial-state append in reset() — then env fire step == offline t_fire
  exactly (both = number of env steps taken; env computes it as time_step+1
  pre-increment).
- `python test_baselines.py` green (pre-change).
- Pre-change byte-level regression reference captured:
  `src/regression_check.py --out data/ref_pre.json` (3 cases: train+aux /
  eval plain / eval early-term legacy; 120-400 steps; md5 per obs key +
  reward + done + entropies per step).
- nvidia-smi: cuda:1 22MiB used, cuda:3 421MiB used (both free-ish, 49GB).
  cuda:0/2 untouched per constraint. CPU: 72 cores, load ~1.5.

## 2026-08-06 — Phase 1 done (env: C2 termination + c2_shaping + global_stats)

- `envs/env.py` edits (all additive, flag-gated, defaults = legacy):
  * EnvConfig: termination_mode/c2_phi_goal/c2_align_window/c2_window/c2_eps/
    reward_mode/c2_w_pos(4.0)/c2_w_vel(0.2)/c2_w_ctrl(0.1)/c2_success_bonus(10)/
    expose_global_stats — grouped after continuous_action.
  * _validate_config: modes validated; c2/global-stats require task acs +
    non-periodic.
  * New methods (before get_extra_info): _c2_stats_needed, _c2_termination_active,
    _c2_reset, _connected_component_labels (DFS), _compute_swarm_stats
    (phi=|mean unit vel|, strict d<r0 proximity components, f_largest, sigma_p),
    _c2_observe_state (buffer append + fire eval; deques ==rolling(W) incl.
    initial-state sample appended in reset/custom_reset).
  * step(): observes post-step state right after next_rel_state (before obs).
  * get_obs: "global_stats" = [phi, f_largest, n_comp/N, sigma_p/(L/2)] float64
    (4,) when expose_global_stats.
  * check_episode_termination acs branch: `if _c2_termination_active(): done on
    fire only` else legacy check untouched; cap at max_time_steps unchanged.
  * compute_custom_reward: reward_mode=="c2_shaping" branch BEFORE legacy:
    -4(1-f_largest)^2 - 0.2*max(0.98-phi,0)^2 + 0.1*(mean raw reward + rho*dt)
    + 10*fire; conn_ratio bookkeeping duplicated from legacy for info logging.
  * get_extra_info: info c2_phi/c2_f_largest/c2_n_comp every step (when c2
    stats on); c2_success + t_conv(=time_step+1, ==offline index) when
    termination active.
- Gates: (1) regression_check --compare ref_pre.json: 3 cases byte-identical;
  test_baselines.py green. (2) c2_crosscheck.py Gate 2: k-NN(10), seeds
  1000-1007, termination_mode=c2: 8/8 SUCCESS with env t_conv == offline
  t_fire EXACTLY (786/489/463/445/471/700/631/520). (3) Gate 3 reward audit:
  0 mismatches (tol 2e-6 vs float32 op-order noise); magnitudes: early(t=5)
  pos -0.04 vel -0.18 ctrl -0.06 (real inits start f_largest~0.9 at L=250, so
  the -1.0 two-cluster calibration point is rarely visited; vel dominates
  transient), settled ~ctrl-only, fire step +9.997 total.

## 2026-08-06 — Phase 2 done (model heads + aux rework) & Phase 3 scripts

- `models/ppo.py` (additive, flag-gated; legacy default = old behavior):
  * New custom_model_config keys: selection_head("legacy"), logit_scale(10.0),
    dist_aux_k(10, replaces hardcoded K), dist_aux_schedule(None),
    dist_aux_coef_current(=dist_aux_coef, mutable), use_global_stats(False).
    bernoulli/threshold assert top_k None + not hard_top_k + not continuous.
  * "bernoulli" = the existing plain path (asserts only, zero new hot code).
  * "threshold" head: threshold_head = Linear(d_ctx,1) ZERO-init weight AND
    bias (PLAN said bias only; zero weight is strictly safer for tau=0 at
    init — documented deviation); tau from per_agent_h_c_N in forward;
    attention_scores_to_logits branch (before hard_top_k):
    s = logit_scale*(scaled_att - tau) + 1e9*diag -> concat(-s,+s).
  * use_global_stats: actor side — NeighborSelectorTorch(global_stats_dim=4)
    -> gs_proj Linear(132->128) applied to h_c_N before decode (flattened
    (N*B,4) replication follows the i-major layout); critic side —
    values_gs_proj Linear(132->128) on the pooled vector in model.forward.
  * custom_loss: K=dist_aux_k; coefficient = dist_aux_coef_current (getattr
    fallback to static); tower_stats stash: aux_mse, dist_aux,
    dist_aux_coef_current, sat_p_dev (mean |p_sel-0.5| off-diag; 0.5 =
    saturated) — route to results via GradLoggingPPO.stats_fn (extended).
- `callbacks.py`: new C2Callbacks(FlockingCallbacks): J accumulation from
  info original_reward (J_episode always, J_success on success), c2_success/
  t_conv/final phi/f_largest/n_comp custom metrics; on_learn_on_batch anneals
  dist_aux_coef_current on policy.model AND model_gpu_towers via np.interp
  over policy.global_timestep (verified: on_learn_on_batch fires on BOTH
  learn paths in 2.1.0; towers[0] is model).
- Gate 1 (src/model_unit_check.py): A |dlogp/datt|=0.95, B att 8.7 / tau 34 /
  head-params 1e4; OLD hardtopk neg-control 3.96e-18 (ratio A/OLD 2.4e17);
  diag p=1.0 forced; entropy A 0.693/edge, B 0.652; |p-0.5|: A 0.012,
  B 0.116, OLD 0.500. custom_loss finite; tower_stats keys present; A coef 0.
- Gate 2 (2-iter CPU smoke via `train_c2_{a,b}.py --smoke`, logs/smoke_*.log):
  A gnorm_actor 1.29->1.80, entropy ~260 (sum over 400 edges), sat_p_dev
  0.049->0.062, aux_mse 0.33->0.16, dist_aux None (off). B gnorm 3.61->3.01,
  sat_p_dev 0.245->0.191, dist_aux 0.31, coef 1.0->0.998 (= 1-0.8*1000/400k
  exactly — anneal wiring correct). A fired C2 at t=875 during iter 2:
  random p~0.5 init policy (deg~9.5) does converge occasionally -> success
  bonus signal exists from the start.
- Phase 3 scripts: `train_c2_a.py` (cuda:1) / `train_c2_b.py` (cuda:3),
  cloned from train_hardtopk.py; changes vs winner run: C2 env block
  (termination_mode=c2, reward_mode=c2_shaping, max_time_steps=1500,
  use_fixed_episode_length=False, expose_global_stats=True), entropy_coeff
  1e-3 (was 0), eval every 10 iters 16 episodes explore=False on 2 eval
  workers with env_config override is_training=False + seed base 900000
  (eval reward == -J up to termination), stop 100 iters, checkpoint_freq 10
  + at_end (keep all), ray.init(num_cpus=16) per run, per-env seed
  de-duplication in make_env (seed + 10007*worker_index + 101*vector_index;
  legacy scripts gave all 16 envs identical episode streams — fixed in NEW
  scripts only), --smoke flag for the Phase-2 gate.

## 2026-08-06 ~02:25 — Phase 3 training LAUNCHED + Phase 4 prep

- Launched `python train_c2_a.py` (cuda:1) and `python train_c2_b.py`
  (cuda:3) concurrently; logs in logs/train_{a,b}.log; runs under
  /workspace/test_results/c2{A_bernoulli,B_threshold}_260806/. GPU usage
  ~9.5GB / 55-66% util each. Persistent monitor: per-10-iter metric line
  (len/rew/succ/t_conv/J/gnorm/entropy/sat_p_dev/dacoef/eval) + error
  signatures, polling progress.csv every 90s.
- iter 1: A len=616 succ=1.00 J_succ=354 gnorm=0.75 ent=263 sat=0.01;
  B len=679 succ=1.00 J_succ=392 gnorm=3.23 ent=252 sat=0.09 dacoef=1.00.
  KEY OBSERVATION: dense stochastic init policies already pass C2 (100% of
  completed episodes) — success is not the bottleneck at L=250; the learning
  problem is EFFICIENCY (J 354-392 vs frontier 160-228) i.e. cutting selection
  down + converging faster. Success-only early termination + ctrl/time costs
  point exactly there.
- Phase 4 prep while training runs:
  * src/eval_c2.py — standalone ckpt loader (adds global_stats input),
    ForensicsWrapper (per-step rank_dev + per-agent degree), offline C2 judge,
    32-seed multiprocessing rollouts, summary CSV; --rank-runs mode ranks
    checkpoints by eval metrics from progress.csv.
  * Smoke on OLD hardtopk10 ckpt (2 seeds, 1200 steps): success 2/2, t_conv
    ~527, deg 10.00, churn 0.0000, rank_dev early/ss = 0.017/0.005 ~ 0 —
    NEGATIVE CONTROL for the adaptivity metric: old winner is a kNN mimic by
    this measure too (metric validated).
  * Launched full oldNN32 eval (32 seeds x 6000 steps, 8 CPU workers,
    background) to extend the old-NN reference from 16 to 32 seeds
    (PLAN Phase 4 paper-table item).
  * src/compare_frontier.py — Wilson CI, paired t on common-success J,
    Welch backup, Fisher on success counts vs FC/knn12/knn10 per-seed rows
    from predecessor j_metric_preview.csv.

## 2026-08-06 ~02:35 — oldNN reference extended to 32 seeds (Phase 4 item)

- oldNN32 (hardtopk10_distaux ckpt10, 32 seeds x 6000 steps, deterministic):
  success 32/32, t_conv med 544, J med 162.5 (mean 186.5), phi_ss 1.0000,
  sigma_p_ss 58.8, min_pair min 0.2 m (!), deg_ss 10.00, churn_ss 0.0000,
  rank_dev early/ss 0.018/0.005.
- vs FC: paired dJ -74.8 (t=-2.86, p=0.0075) — beats FC. vs knn12: +4.7
  (p=0.77) — ON the frontier, indistinguishable. vs knn10: J p=0.85 but
  success 32/32 vs 29/32 (Fisher p=0.24) — the tiny (1.8% early) rank
  deviations of the mimic already rescue knn10's 3 fragmenting seeds; more
  headroom for real adaptivity.
- Updated success bars for the new variants: match/beat J_med 160 (knn12) AND
  the mimic's 162.5, at success >= 31/32, with rank_dev clearly > 0 —
  otherwise the reopened gradient path added nothing over supervision.
- NOTE min_pair 0.2 m transient for oldNN — keep collision margin in the
  quality-margin table for all candidates.

## 2026-08-06 ~03:50 — mid-training reads (iters 10-21)

- Training-side (stochastic) both variants: succ 0.91-0.96, len ~880-930,
  entropy pinned at ~262-263 (=max, p~0.5 per edge), sat_p_dev 0.02-0.04;
  B dacoef anneal on schedule (0.68 @ it11, 0.36 @ it21). Mechanism note:
  entropy bonus 1e-3*263 ~ +0.26/step vs settled shaping ~ -0.003/step ->
  entropy pressure keeps SAMPLING near-uniform; the learning shows up in the
  argmax ORDERING, not in logit magnitudes.
- tune progress.csv quirk (Ray 2.1.0): evaluation/* columns are dropped
  because the CSV header is fixed at iter 1 before the first eval — read
  result.json (JSONL) instead for eval metrics.
- Deterministic (explore=False) eval, 16 episodes/round:
  * A: it10 succ 0.875 J_succ 170; it20 succ 0.75 J_succ 164 (sharpening:
    better J on successes, more fragmentation losses).
  * B: it10 AND it20 succ 0.0 — argmax degenerate-sparse. Offline diag of
    ckpt it10 (2 seeds): deg_ss 3.4, churn 0.10, rank_dev 0.15-0.25 -> tau
    sits above most att; k~3 regime fragments. Stochastic B succeeds (dense
    sampling), so PG pressure on margins is weak and the entropy bonus pushes
    |margin|->0. DECISION POINT set: if B eval still 0/16 at iter 30-40,
    restart B with lower entropy_coeff (e.g. 2e-4) and/or logit_scale bump.
- A ckpt it20 offline probe (8 protocol seeds, 3000 steps, argmax): success
  6/8, t_conv med 570, J med 177; deg/rank_dev TIME PROFILE: [1-100] deg
  12.6-14.2 rd 0.22-0.31 -> [100-300] deg 15.5-18.7 rd ~0.03 -> hold FC
  (deg 17-19, rd 0.00). GENUINE phase-dependent adaptivity (merge: selective
  non-nearest; hold: dense) — the opposite signature of oldNN's frozen
  kNN(10) mimic. Post-fire density is J-free (J integrates to t_fire only) —
  policy exploits exactly the intended margin.

## 2026-08-06 ~04:30 — B restarted as B2 with entropy anneal (playbook action)

- Evidence at decision point: B eval 0/16 at iters 10, 20 AND 30; offline
  diag ckpt it30: argmax deg_ss 2.33 (worse than it10's 3.4), churn 0.07,
  rank_dev 0.24-0.28 -> the degenerate-sparse argmax attractor persists while
  stochastic-side succ is 0.96-0.97. Mechanism: entropy bonus (1e-3 * ~262 =
  +0.26/step) >> settled shaping (-0.003/step) pins per-edge p at ~0.5;
  sampling stays dense (succeeds), so PG pressure on (att - tau) margins is
  weak and cannot beat the pin. A escapes because its argmax needs only an
  att SIGN structure (readout at 0), which aux-shaped representations give;
  B's readout is att vs LEARNED tau — the pin keeps that comparison noise-level.
- A iter-30 eval meanwhile: succ 0.8125, J_succ 149, tconv 499 — J already
  under the k=12 frontier (160) and stretch (150); success rate is the gap
  (needs ~0.97). A left UNTOUCHED (working run; do-not-disturb).
- Action: killed c2B_threshold_260806 at ~iter 31 (checkpoints 10/20/30 kept
  on disk for forensics); relaunched as `c2B2_threshold_entsched_260806`
  with ONE change: `entropy_coeff_schedule=[[0,1e-3],[500000,1e-4]]`
  (verified supported in Ray 2.1.0 ppo.py:97-194). Rationale: keep early
  exploration identical, then anneal the pin 10x by ~iter 31 so margins can
  open; dist_aux schedule/logit_scale unchanged (single-variable change).
  Fresh ray session confirmed; cuda:3 freed then re-occupied (1.7GB ramp).
  Monitor replaced (now reads result.json so eval columns are visible;
  progress.csv drops evaluation/* columns in Ray 2.1.0 — header fixed at
  iter 1).
- Deviation from PLAN Phase 3 monitoring playbook: the listed B knob was
  "entropy collapse -> lower logit_scale"; the OBSERVED failure is the
  opposite (entropy pinned high). Entropy-schedule restart is the analogous
  minimal knob; logged here as the deviation with rationale.

## 2026-08-06 ~08:40 — B2 verdict (failed) -> B3 dense-start; A finished course

- B2 evals: 0/16 at iters 10/20/30/40/50/60 — six consecutive, INCLUDING
  three evals after the entropy anneal reached its 1e-4 floor (iter ~31).
  Stochastic side fine (succ 0.96-1.00, len 907->699, J_succ 533->412).
  REVISED mechanism: at p~0.5 the entropy gradient is ~0 (max is flat), so
  the anneal was not the binding fix — the per-edge PG signal itself is
  advantage-noise dominated under dense sampling (single-edge flips barely
  move the reward), so tau never finds a useful split; argmax readout stays
  degenerate-sparse. Entropy-anneal hypothesis: REJECTED for B.
- B3 launched (killed B2 at iter ~61; ckpts 10-60 kept): ONE change vs B2 —
  `threshold_bias_init=-0.1` (new flag-gated model config; zero-weight
  init unchanged). Init verified on real obs: p mean 0.882 (q10 0.80,
  q90 0.95), argmax degree 19.0 (=FC, known-feasible), entropy 0.347/edge.
  Argmax now STARTS at success (FC J~228) and pruning must be LEARNED —
  the failure mode "argmax infeasible" is structurally excluded at t=0.
  Entropy schedule kept (harmless; sampling should sharpen late).
  Run: c2B3_threshold_biasinit_260806 (cuda:3), log logs/train_b3.log.
- A run: evals it10-90: succ .875/.75/.81/.94/.94/.88/1.0/1.0/1.0,
  J_succ 170/164/149/157/145/157/171/190/187 — clear late-training drift:
  reliability up, J up (denser/safer). J-optimal ckpts mid-run (it40-50),
  success-optimal late (it70+). Offline 32-seed protocol will decide.
- A2 staged (train_c2_a.py now = A2): same single knob as B2
  (entropy_coeff_schedule [[0,1e-3],[500000,1e-4]], run
  c2A2_bernoulli_entsched_260806) to test whether aligning late-training
  sampling with the argmax readout gets success~1.0 AT J~145 simultaneously.
  Launches on cuda:1 when A's 100 iters complete.

## 2026-08-06 ~09:05 — A completed (100 iters, 02:27-09:01, ~235 s/iter); ckpt screening; A2/B3 running

- A finished 100/100 (final online eval@100: 0.75/163 — online n=16 keeps
  oscillating; offline protocol is the arbiter). 10 checkpoints kept.
- Offline 16-seed screens (seeds 1000-1015, 6000 steps, argmax):
  * it40: 15/16, t_conv 467, J_med 139.9 (mean 145.9), rank_dev early 0.140
  * it50: 16/16, t_conv 466, J_med 143.6 (mean 203.4, one expensive outlier),
    rank_dev early 0.168
  * it70: 16/16, t_conv 490, J_med 178.3, rank_dev early 0.247
  -> mid-run it40-50 = J-optimal zone as predicted; it50 fronts (perfect
  success + J 143.6 << frontier 160). Full 32-seed protocol run of it50
  launched; it40/it100 32-seed extensions pending as needed.
- B3 iter 1: len 571, J_succ 223 (~FC's 228, vs B/B2's 362-392 at iter 1) —
  dense start behaves exactly as designed; watch whether J prunes down and
  argmax eval holds ~1.00 from it10.
- A2 launched on cuda:1 after A's completion (same monitor covers A2/B3).

## 2026-08-06 ~09:10 — A checkpoint landscape on the 32-seed protocol

- Full-protocol (32 paired seeds, 6000 steps, argmax, offline C2 judge):
  * it40: 29/32 (fail 1005/1019/1030), t_conv med 499, J_med 144.4 (mean 149.7)
  * it50: 30/32 (fail 1018/1021),      t_conv med 460, J_med 141.3 (mean 173.7;
    outlier seed 1001 fires at t=5789 -> J 912; 1014 J 254)
  * 16-seed screens: it60 16/16 J_med 159.6; it70 16/16 J_med 178.3
  -> J-vs-reliability slope along training confirmed on protocol seeds.
- it50 vs frontier (paired): vs FC dJ -87.3 (p=0.015); vs knn12 J_med 141.3
  vs 160.0, dJ mean -12.6 (p=0.68 — mean washed by the 1001 outlier); succ
  30/32 vs 31/32 (Fisher p=1). PRIMARY criterion (>=31/32 AND J_med<=160):
  it50 misses by ONE success.
- Cross-ckpt failure sets nearly DISJOINT (it40 {1005,1019,1030}, it50
  {1018,1021}; knn12 fails 1014 which it50 SOLVES at J 254, and it50 solves
  all three knn10 failures 1022/1026/1030) — failures are snapshot-specific
  fragilities, not hard inits; the A policy family covers every seed.
- min_pair dips below 1 m on ~all seeds (min 0.04 m) — same for oldNN (0.2);
  point-mass sim, no collision model; report as a shared descriptive margin,
  not a per-policy differentiator.
- it60 32-seed extension launched (16/16 + J 159.6 on screen -> candidate to
  clear the primary bar if it holds at n=32).

## 2026-08-06 ~09:16 — PRIMARY CRITERION MET by A it60 (31/32, J_med 156)

- A it60 @32 protocol seeds: success 31/32 (only 1016 fails), t_conv med 508,
  J_med 156.0 (mean 167.4), rank_dev early/ss 0.213/0.087.
- vs FC: paired dJ -84.6 (t=-4.05, p=0.00033), Welch p=0.00012 — decisive.
- vs knn12: succ 31/32 == 31/32; J_med 156.0 vs 160.0 (dJ mean -13.3,
  p=0.4) — MATCHES the best fixed k, does not significantly beat it on J.
- Pre-registered primary criterion (success>=31/32 AND J_med<=160): MET.
  And unlike the old winner, via GENUINE adaptivity (rank_dev early 0.213 =
  13x the oldNN mimic's 0.017; degree densifies merge->hold).
- Candidate table (A run, 32-seed unless noted): it40 29/32 J 144.4 |
  it50 30/32 J 141.3 | it60 31/32 J 156.0 | it70 (16s) 16/16 J 178.
  Reliability-J slope: pick by use — headline = it60 (criterion),
  J-optimal = it50.
- Pending upside: A2 (entropy-annealed) and B3 (dense-start threshold) still
  training; either could yield success>=31 AT J~140s. Stretch (J<150 @ 32/32)
  not yet achieved.

## 2026-08-06 ~09:25 — L-generalization: frontier recomputed; it60 probed

- Frontier under C2+J from predecessor main npz (src/frontier_L.py,
  data/frontier_L.csv), 32 seeds each:
  L=125: k10 J 131.8 (30/32), k12 155.0 (31/32), FC 224.0 (32/32)
  L=250: (sanity, matches j_metric_preview) k12 160.0 (31/32), FC 228.4
  L=500: k10 172.6 (31/32), k12 165.6 (32/32), FC 234.8 (32/32)
  NOTE: predecessor hypothesis "fixed kNN more fragile at L=500" does NOT
  hold for k>=10 (k12 is 32/32 there); only low k degrades (k8 27/32).
- A it60 generalization probes (32 seeds, argmax):
  L=125: 32/32, t_conv 570, J_med 170.1 — worse than k12 (155), better than
  FC (224); L=500: 31/32, t_conv 695, J_med 248.9 — WORSE than FC (235).
- Verdict: SUCCESS generalizes across L (31-32/32 everywhere); EFFICIENCY is
  L=250-specialized (obs normalized by L/2 -> geometry rescaled; the learned
  merge-phase behavior mistuned off-distribution, while kNN is scale-free).
  Stretch goal "Pareto dominance at harder inits" NOT achieved. Honest
  limitation for the report; fix directions (train across L / scale-invariant
  features) belong to future work.

## 2026-08-06 ~09:50 — Forensics figures + case studies (report inputs)

- figs/fig1_profiles.png (A it60 vs oldNN vs k12, median over 32 seeds):
  degree: A starts ~12.5, dips ~11.7 (t~30), densifies to ~17 by t~400;
  oldNN flat 10.0; k12 flat 12. rank_dev: A 0.36-0.40 EARLY (36-40% of
  selected edges outside the nearest-deg set during the merge phase!) decaying
  to ~0.10 at hold; oldNN 0.02->0.005; kNN==0. The learned policy is
  low-degree + strongly non-nearest early, dense + near-nearest late.
- figs/fig2_case1014.png (k12's failure seed): all three merge 5->1 comps by
  t~70; k12 then SPLITS to 2 at t~160 and never re-merges (phi=1.0 with two
  flocks — legacy criterion would have called parts of this "converged");
  A it60 holds 1 comp from ~160 (fires ~500s); A it50 wobbles 1-3 until ~760
  then holds (fires late, J 254). Adaptivity's value = keeping/merging the
  flock where fixed k loses it.
- A it60's only failure (1016): touches single-comp briefly (1% of steps)
  but settles as a 2-flock split (phi 0.9996, sigma_p 45.8) — same failure
  topology as kNN's, on a different init. B3 eval@10 first working readout:
  0.81/J 213 (dense start pruning down); A2 eval@10: 1.00/J 252.
- Palette note: dataviz reference slots 1-3 used unchanged (documented
  all-pairs validation cited; node unavailable to re-run validator locally).

## 2026-08-06 ~12:40 — timeline correction + run-length reality check

- Corrected the wall-clock markers of earlier entries (my running estimates
  had drifted up to ~3 h fast; times above now anchored to result.json
  timestamps and output-file mtimes). Ground truth: one training iteration =
  ~235 s wall (16k env steps; sampling-bound), eval rounds add ~120-160 s
  each; A ran 02:27-09:01. All ordering/numbers in earlier entries unchanged.
- A2 (started 09:05) and B3 (started 08:51) are at iters ~54-57 at 12:39;
  projected completion ~15:30-15:45. Latest evals: A2 1.00/202 -> 1.00/211 ->
  0.94/205 -> 0.94/217 (slower J-descent than A at the same iters — the
  entropy anneal did NOT sharpen sampling: entropy stays ~263 because the
  entropy gradient vanishes at p=0.5; A2 is effectively a reliability-first
  replicate); B3 0.94/247 -> 0.81/253 -> 0.94/261 -> 0.88/225 (working
  readout, still FC-grade J, slow pruning).
- Plan for the remaining window: draft REPORT_KO around the already-locked
  A-line + frontier + forensics results; fold in A2/B3 final checkpoints
  (offline 16->32-seed selection) when they land ~15:30.

## 2026-08-06 ~12:55 — SESSION HANDOFF (context limit; user-requested)

- Handoff point chosen: A-line results + report LOCKED (primary criterion met
  by A it60); A2 (iter ~57) and B3 (iter ~59) still training to 100, ETA
  ~15:30-15:45 — they are improvement attempts, not blockers.
- Prepared for the next session:
  * `KICKOFF_NEXT_KO.md` — successor kickoff (state, remaining steps,
    commands, pitfalls). REPORT_KO.md carries [PENDING] markers at the two
    spots awaiting A2/B3 finals.
  * `--resume` flag added to train_c2_a.py / train_c2_b.py (tune
    resume="AUTO" from last checkpoint) in case the background training
    processes die with this session; evaluating existing checkpoints
    without resuming is also acceptable.
  * Predecessor pointer appended to studies/acs-conv-knn/REPORT_KO.md
    (Phase 5 item done).
  * Monitor + wakeup loop stopped at session end.
- NOT committed (user approval required; nothing pushed): repo edits
  (envs/env.py, models/ppo.py, callbacks.py, grad_logging_ppo.py,
  train_c2_a.py, train_c2_b.py) + study dir docs/src. Untracked root file
  REPORT_hardtopk_distaux_KO.md predates this study (2026-05-30 session) —
  left untouched.
- Remaining work (next session): A2/B3 finals -> 16->32-seed checkpoint
  selection (success-first-then-J; replace headline only if better than
  it60 31/32-156 / it50 30/32-141.3) -> REPORT finalize -> Korean report to
  user -> commit discussion.

## 2026-08-06 ~15:45 — A2/B3 completed 100/100 (session survived; finishing here)

- Both runs TERMINATED cleanly at 15:37 (B3 total 24600 s = 246 s/iter incl.
  evals; matches projection). Handoff docs from ~12:55 remain valid but the
  finals are being folded in NOW instead of next session.
- A2 online evals (16ep): 1.00/252, 1.00/202, 1.00/211, 0.94/205, 0.94/217,
  1.00/252, 0.94/233, 1.00/247, 1.00/277, 0.88/266 — J NEVER entered A's
  140-160 zone; entropy-anneal hypothesis adds nothing for the bernoulli
  head (entropy gradient vanishes at p~0.5); A2 = reliability-first
  replicate, run-to-run variance dominates trajectory shape.
- B3 online evals: 0.81/213, 0.94/247, 0.81/253, 0.94/261, 0.88/225, then
  FIVE consecutive 1.00: /231, /232, /190 (it80, best), /199, /201 — the
  dense-start threshold head is the only B variant with a working argmax
  readout; it prunes J from FC-grade 253 to ~190 in 100 iters but does not
  reach the 160 frontier.
- Screening was started here but STOPPED on user direction: the offline
  selection + report finalization belong to the NEXT session (that was the
  point of the 12:55 handoff). No B3/A2 eval outputs written (verified
  clean). KICKOFF_NEXT_KO.md updated to the post-completion state with the
  exact screening commands (candidates: B3 it80 / it100, A2 it20).

## 2026-08-06 ~23:05 — Successor session: A2/B3 offline screening -> HEADLINE UNCHANGED

- Session start ~23:00. Re-verified RUNLOG ~15:45 online-eval numbers against
  result.json (exact match, both runs TERMINATED at iter 100 / 1.6M steps).
  Machine idle (load 0.5, all GPUs ~0%); screens are CPU-only, 3x10 workers
  in parallel, ~23:01-23:04 wall.
- 16-seed offline screens (seeds 1000-1015, 6000 steps, argmax, offline C2
  judge — the pre-registered candidates from KICKOFF_NEXT_KO.md):
  * B3 it80:  16/16, t_conv med 614, J_med 218.3 (q25/75 211/231),
    deg early/ss 11.49/11.35, churn_ss 0.014, rank_dev early/ss 0.124/0.131
  * B3 it100: 16/16, t_conv med 553, J_med 198.4 (q25/75 175/220),
    deg early/ss 11.22/10.80, rank_dev early/ss 0.127/0.123
  * A2 it20:  16/16, t_conv med 554, J_med 171.4 (q25/75 159/258),
    deg early/ss 18.81/19.00, churn_ss 0.0000, rank_dev early/ss 0.006/0.000
- Reading, B-line: the offline ordering REVERSES the online impression
  (online min was it80's 190; offline it100 < it80 by 20 J) — online 16ep
  noise, as pre-warned. B3 was still descending at cutoff (218 -> 198 over
  the last 20 iters) with a working readout throughout (16/16 both ckpts):
  dense-start rescued the argmax readout definitively, pruned FC -> deg ~11.3
  with genuinely nonzero, phase-FLAT rank_dev ~0.12 (mild real adaptivity,
  unlike oldNN's 0.005-0.018 mimicry and unlike A's phased 0.36->0.10), but
  efficiency ends ~198, well off the 160 frontier. Verdict: "readout revived,
  frontier not reached within 100 iters (still improving at cutoff)".
- Reading, A2: best-online ckpt (it20) argmax is near-FC dense
  (deg 18.8-19.0, rank_dev ~0.006) — reliability via density, no efficiency,
  no adaptivity. Entropy-anneal hypothesis for the bernoulli head stays
  rejected (mechanism: entropy gradient ~0 at p~0.5; sampling never sharpens).
- Pre-registered decision rule applied (success first, then J_med; replace
  headline only if better than A it60 31/32-156.0 / it50 30/32-141.3):
  NO candidate qualifies (best is A2 it20 at 16/16-171.4, and its rank_dev~0
  would fail the research question even if J improved at n=32; B3 198/218 far
  off). NO 32-seed extensions warranted ("extend only if promising" rule) —
  screening closed. Headline stands: A it60 (31/32, J_med 156.0).
- Outputs: data/eval/{B3_i80_s16,B3_i100_s16,A2_i20_s16}*/; logs
  logs/eval_*_s16.log. REPORT_KO.md [PENDING] x2 resolved next (this entry
  is the source for those numbers).

## 2026-08-06 ~23:15 — STUDY CLOSED (user-approved commits executed)

- Final report delivered to user; user approved commit scope explicitly:
  repo code -> commit d600293 (env/model/callbacks/grad_logging + train
  scripts); study docs (PROBLEM already tracked; PLAN/RUNLOG/REPORT_KO) +
  src/ + predecessor REPORT_KO pointer -> the study commit containing this
  entry. Per user: KICKOFF_NEXT_KO.md deleted (was untracked),
  KICKOFF_PROMPT_KO.md git-rm'ed in the study commit;
  REPORT_hardtopk_distaux_KO.md (pre-study artifact) left untracked,
  NOT committed. Nothing pushed (forbidden). Working tree clean otherwise.
- All training/eval processes terminated; GPUs free. Checkpoints retained
  (not committed) under /workspace/test_results/c2{A_bernoulli,A2_...,
  B_threshold,B2_...,B3_...}_260806/ — headline artifact = A it60
  (c2A_bernoulli_260806 checkpoint_000060).
