# PLAN.md — acs-c2-train

> **Purpose of this file**: The implementation & training plan the next session
> executes. Phases are ordered; each has concrete specs, file anchors, and
> validation gates. Statuses: `[ ]` todo / `[~]` in progress / `[x]` done /
> `[!]` blocked-or-revised. Log actual executions + numbers in RUNLOG.md.
> Design rationale lives in PROBLEM.md and the predecessor study's RUNLOG
> (2026-08-06 entries). File:line anchors below were verified 2026-08-06 and may
> drift a few lines after edits — re-grep before patching blindly.

## Phase 0 — Ground yourself + regression baseline  [x]

- [x] Read PROBLEM.md here, then `studies/acs-conv-knn/RUNLOG.md` (at least the
      three 2026-08-06 entries) and `NOTES_env.md`. Skim `models/ppo.py` forward
      + `attention_scores_to_logits` + `custom_loss`, and `envs/env.py`
      termination (~:1236-1253) + `compute_custom_reward` (~:1309-1367).
- [x] Verify the plain-binary logits path in `attention_scores_to_logits`
      (ppo.py:291-350): what logits are produced when `continuous_action=False`
      and `top_k=None`? (Expected: ±s with s = scale_factor·att, masked entries
      -inf-ish.) Variant A rides this path — confirm it exists; if only
      topk/soft/continuous branches exist, add the plain branch.
      -> EXISTS (ppo.py:341-350); diag forced via +1e9 (legacy), masked -1e8.
- [x] Run `python test_baselines.py` and record it green (pre-change baseline).
- [x] `nvidia-smi` on cuda:1 and cuda:3 (both must be free-ish; they are
      user-assigned to this study). -> 22MiB / 421MiB used, both free.

## Phase 1 — Env: C2 termination + shaped reward + global stats  [x]
File: `envs/env.py`. All new fields ADDITIVE with legacy defaults (Pydantic v1
`EnvConfig`; note `validator`, not Pydantic v2 syntax).

New `EnvConfig` fields (defaults = legacy behavior; group near the existing
acs_* fields ~env.py:60-80):
```python
termination_mode: str = "legacy"      # "legacy" | "c2"; used only when
                                      # use_fixed_episode_length is False
c2_phi_goal: float = 0.98
c2_align_window: int = 50
c2_window: int = 300                  # cohesion hold AND stationarity window
c2_eps: float = 0.05                  # relative p2p band on sigma_p
reward_mode: str = "legacy"           # "legacy" | "c2_shaping"
c2_w_pos: float = 4.0                 # cohesion shaping weight
c2_w_vel: float = 0.2                 # alignment shaping weight
c2_w_ctrl: float = 0.1                # control-cost weight (reuse acs value)
c2_success_bonus: float = 10.0        # one-time, at C2 fire (user: start small;
                                      # sane range 5-20)
expose_global_stats: bool = False     # adds obs key "global_stats" (4,)
```

C2 state machine (new small helper on the env, reset in `reset()`):
- Per step AFTER state update, compute from active agents:
  `phi` = |mean(unit velocity)|; `d` = pairwise distance matrix;
  `single = (n_components(d < r0) == 1)` (union-find, O(N^2), include ALL
  active agents — singletons break it by construction); `sigma_p` = spatial
  entropy (already computed for info — reuse, do not recompute).
- Buffers (collections.deque): `phis` maxlen=c2_align_window, `singles`
  maxlen=c2_window, `sps` maxlen=c2_window.
- Fire condition (exactly the offline judge's pandas `rolling(W)` semantics =
  last W samples including current):
  `len(phis)==50 and min(phis)>0.98` AND `len(singles)==300 and all(singles)`
  AND `len(sps)==300 and (max(sps)-min(sps))/mean(sps) < 0.05`.
- On fire: `done=True`, `info["c2_success"]=True`, `info["t_conv"]=t`; reward
  gets `+c2_success_bonus` this step. Episodes that never fire run to
  `max_time_steps` (failure; no bonus) — success-only early termination
  (blocks the fragment-freeze shortcut).
- Gating: active only when `use_fixed_episode_length=False and
  termination_mode=="c2"`. Legacy termination branch stays untouched.

Reward branch (`compute_custom_reward`, ~env.py:1309-1367): when
`reward_mode=="c2_shaping"` replace pos/vel terms, keep plumbing:
```
r_t = - c2_w_pos * (1 - f_largest)^2          # f_largest = largest r0-component
      - c2_w_vel * max(0.98 - phi, 0)^2       #   fraction of active agents
      + c2_w_ctrl * control_cost_term         # existing negative ctrl term
      + c2_success_bonus * 1[C2 fired this step]
```
Calibration rationale (keep — matches legacy per-step magnitudes so PPO
hyperparams carry over): early two-cluster state f~0.5 -> pos term ~ -1.0/step
(legacy pos at sigma_p=100 was ~ -1.02); phi~0.3 -> vel term ~ -0.09 (legacy at
sigma_v=10 was ~ -0.09). No compactness pressure anywhere (user decision).
f_largest/phi/components are shared with the C2 state machine — compute once.

Global stats obs (when `expose_global_stats`): new obs key `"global_stats"`,
Box shape (4,): `[phi, f_largest, n_comp/N, sigma_p/(initial_position_bound/2)]`.
Follow the `expose_aux_target` pattern for space+value (env.py:235-241 space,
:1077-1082 value). Purpose: phase awareness (merge vs hold) for the policy.

Validation gates for Phase 1:
- [x] Defaults regression: default-config env == old behavior (diff a 100-step
      scripted rollout's obs/reward/done against pre-change git stash or by
      constructing with legacy flags). `test_baselines.py` still green.
      -> src/regression_check.py: 3 cases byte-identical; test_baselines green.
- [x] **Online/offline C2 cross-check**: run 8 episodes of the 'nearest' k=10
      baseline with termination_mode="c2", is_training=False, logging series
      via `studies/acs-conv-knn/src/common.py::rollout`; assert env-reported
      t_conv == offline judge t_fire (same W/eps/phi; watch the off-by-one:
      both count post-step states, buffers==rolling(W) — fix env side if they
      disagree, the offline judge is the reference).
      -> src/c2_crosscheck.py Gate 2: 8/8 EXACT parity (t_conv 445-786).
- [x] Reward magnitude print on one scripted episode (transient vs settled):
      per-term values match the calibration table above (~-1.0 / ~-0.09 early;
      ~0 settled except ctrl).
      -> Gate 3: 0 reconstruction mismatches (tol 2e-6); NOTE real L=250 inits
      start f_largest~0.9, so pos term is small from t=0 and the vel term
      (-0.18 at phi~0.03) dominates the transient; fire step total = +9.997.

## Phase 2 — Model: two selection heads, unsaturated; aux rework  [x]
File: `models/ppo.py` (+ `grad_logging_ppo.py`, `callbacks.py`). Keep old
branches intact; add flag-gated ones. New `custom_model_config` keys:
`selection_head` ("legacy" | "bernoulli" | "threshold"), `logit_scale` (float,
default 10.0; threshold head), `dist_aux_k` (int, default 10 — replaces the
HARDCODED K=10 at ppo.py:441), `dist_aux_schedule` (list [[t,coef],...] or None),
`use_global_stats` (bool).

- **Variant A head ("bernoulli")**: logits directly from scaled att:
  `s = scale_factor*att` -> logits concat(-s, +s); diagonal forced selected
  (+10 like the continuous path); masked entries large-negative. At init
  p~0.5 per edge -> exploration by construction; the policy learns to saturate
  its own logits (annealing is learned, not scheduled). dist_aux OFF (coef 0).
- **Variant B head ("threshold")**: per-agent threshold tau_i from the agent
  context `h_c_N` (B*N,1,128) -> Linear(128,1), zero-init bias;
  `s_ij = logit_scale * (scale_factor*att_ij - tau_i)`; logits concat(-s,+s);
  diag forced; mask forced. Selection = "edges whose score clears MY bar" —
  the differentiable generalization of top-K with learned, per-agent,
  state-dependent K (literal top-K + stochastic K-head would need a composite
  action distribution — painful/risky in Ray 2.1.0; this is the same knob,
  cleanly). Degree is free to vary over agents and time. logit_scale=10:
  margin 0.1 in score units -> p~0.88; margin 0.5 -> p~0.9999 — deterministic
  enough when confident, NEVER ±20-saturated (that bug is what made the old
  winner a fake — do not reintroduce it).
- **dist_aux (variant B only, "carefully" per user)**: keep the rank-regression
  loss but (i) K from `dist_aux_k`, (ii) coefficient from
  `model.dist_aux_coef_current` (mutable attr), scheduled
  1.0 -> 0.2 linearly over the first 400k env timesteps, then hold 0.2 (NEVER
  to 0 — weak prior retained; NEVER dominant after anneal). Drive the schedule
  from `FlockingCallbacks.on_learn_on_batch(policy, ...)`: set
  `policy.model.dist_aux_coef_current = interp(policy.global_timestep)`.
- pair_embedding aux (0.3) + critic aux (0.05): keep for BOTH variants
  (representation scaffold, proven).
- `use_global_stats`: concat the (4,) obs vector to (a) the decoder query
  context (project 128+4 -> 128 with a small Linear) and (b) the critic's
  pooled vector (same treatment on the critic tower). Also give the critic
  f_largest explicitly (it is inside global_stats) — value of cohesion states
  becomes learnable.
- Metrics visibility: model.metrics() does not reach results in 2.1.0
  multi-GPU (known repo issue) — surface `aux_mse`, `dist_aux`,
  `dist_aux_coef_current`, and mean |p-0.5| (saturation monitor) through
  `GradLoggingPPO.stats_fn` (grad_logging_ppo.py:38-46) by stashing scalars on
  the model in `custom_loss`.

Validation gates for Phase 2:
- [x] Unit forward: both heads produce (B, 2*N*N) logits; masked/diag
      constraints hold on random obs; gradient flows: d(logp)/d(att) and
      d(logp)/d(tau) are nonzero at init (assert > 1e-6; the OLD saturated
      path gives ~1e-18 — this is THE regression to prevent).
      -> src/model_unit_check.py: A d(logp)/d(att)=0.95, B att-grad 8.7 +
      tau-grad 34; OLD neg-control 4e-18 reproduced; diag p=1; entropy ~ln2;
      |p-0.5| A 0.012 / B 0.116 / OLD 0.500.
- [x] 2-iteration smoke train per variant (local_mode or 1 worker, CPU ok):
      actor grad norm (GradLoggingPPO logs it) clearly nonzero; entropy > 0;
      dist_aux_coef_current logged and moving (B).
      -> logs/smoke_{a,b}.log: A gnorm 1.3-1.8, entropy ~260, sat_p_dev 0.05;
      B gnorm 3.0-3.6, sat_p_dev 0.19-0.24, dist_aux 0.31, coef 1.0->0.998
      (matches 1-0.8*1000/400k). A even fired C2 at t=875 in iter 2 (random
      p~0.5 policy is dense enough to converge sometimes).

## Phase 3 — Training scripts + launch  [~]
> STATUS 2026-08-06 12:50: A done (100 it, 02:27-09:01); B->B2->B3
> restarts logged in RUNLOG (argmax-degeneracy playbook actions);
> A2/B3 follow-ups training, ETA ~15:30.

Two repo-root scripts, cloned from `train_hardtopk.py` (NOT train.py — the
continuous path stays untouched):
- `train_c2_a.py`: selection_head="bernoulli", aux_enabled=True (pair_embedding
  0.3, critic 0.05), dist_aux off, expose_global_stats=True,
  use_global_stats=True.
- `train_c2_b.py`: selection_head="threshold", logit_scale=10, dist_aux
  schedule [[0,1.0],[400_000,0.2]], dist_aux_k=10, same aux/global-stats.

Shared env config (both): task acs, N pool [20], L=250, ego_centric,
`use_fixed_episode_length=False`, `termination_mode="c2"`,
`reward_mode="c2_shaping"`, `max_time_steps=1500`, expose_aux_target=True,
is_training=True, continuous_action=False.

Shared PPO config: start from train_hardtopk.py's block; changes:
`entropy_coeff=1e-3` (real stochasticity now exists; env churn cost
self-anneals it — random-family J=1079 shows jitter is expensive),
`evaluation_interval=10`, `evaluation_duration=16` episodes,
`evaluation_config={"explore": False}` (verify exact 2.1.0 key names),
keep gamma=0.99, lr schedule, clip 0.15, batch 16000 / minibatch 256 / 10 sgd,
`stop={"training_iteration": 100}`, checkpoint_freq 10 + at_end.
Callbacks: extend FlockingCallbacks -> `C2Callbacks`: on_episode_end log
t_conv (=length if success), c2_success, J_episode (accumulate
info["original_reward"] per step), final phi & f_largest; note the existing
`flocking_success = length < max_steps` becomes meaningful automatically.

Launch (user-assigned GPUs): variant A on cuda:1, variant B on cuda:3 —
`CUDA_VISIBLE_DEVICES=1 python train_c2_a.py` / `=3 python train_c2_b.py`,
each `num_gpus: 1` (or 0.5 if sharing needed later), `num_workers: 4`,
`num_envs_per_worker: 4`. Two runs + drivers stay well under 64 threads (pin
OMP_NUM_THREADS=1 for workers as in the sweep harness). Run names:
`c2A_bernoulli_<yymmdd>`, `c2B_threshold_<yymmdd>` under /workspace/test_results.

Monitoring cadence (log in RUNLOG): every ~10 iters check episode_len_mean
(should FALL below 1500 as success rate rises), flocking_success_mean,
eval-mode success, actor grad norm, entropy, dist_aux coef, mean degree of
sampled actions (from conn_ratio info). Goodhart guard: training return up
while eval success/J flat -> inspect shaping terms before continuing.
Failure-mode playbook:
- success stuck ~0 by iter ~15 but f_largest rising -> cohesion learning ok,
  fire too rare: consider max_time_steps 1500->2000 (documented knob), do NOT
  loosen C2 itself.
- entropy collapse to ~0 in first iters (B): lower logit_scale to 5.
- A stuck in high-churn regime (phi plateau <0.9): raise c2_w_vel to 0.4.

## Phase 4 — Evaluation vs the frontier  [x]
> STATUS 2026-08-06 23:05: COMPLETE. A it40/50/60(/70-16s) protocol evals,
> oldNN32, frontier-L, forensics, case studies done; PRIMARY CRITERION MET
> by A it60 (31/32, J 156). A2/B3 finals screened (16 seeds, pre-registered
> candidates it80/it100/it20): best B3 it100 16/16-198.4, best A2 it20
> 16/16-171.4 (near-FC argmax) -> headline UNCHANGED, no 32-seed extension
> warranted. RUNLOG ~23:05 entry has the numbers.

- [x] Checkpoint selection BY EVAL METRIC (success rate, then J), not training
      return: for each variant take best-eval checkpoint + final.
      -> A: it40-70 screened+protocol; A2: it20 (best online); B3: it80+it100.
- [x] Rollout protocol (reuse predecessor harness `common.py::rollout` +
      `evaluate_checkpoint.RLPolicy` loader): 32 paired seeds (1000..1031),
      N=20, L=250, 6000 steps, fixed length, is_training=False, CPU.
      Offline C2 judge (criteria settings phi 0.98 / W 300 / eps 0.05) ->
      success, t_conv, J per episode. -> src/eval_c2.py; data/eval/.
- [x] Compare against frontier references (same seeds, already computed in
      `studies/acs-conv-knn/data/j_metric_preview.csv`): FC J=228 (32/32),
      k=12 J=160 (31/32), k=10 J=171 (29/32), old NN ckpt J=166 (16/16 on 16
      seeds — extend to 32 for the paper table). Welch/paired-t as usual.
      -> oldNN32 done (32/32, 162.5); A it60 vs FC p=3.3e-4, vs k12 p=0.4.
- [x] Generalization probes: L=125 and L=500 (frontier exists in summary_main).
      -> success transfers (31-32/32), efficiency L=250-specialized (REPORT 4.6).
      N=10/40 skipped (optional; out of remaining budget, noted as future work).
- [x] Adaptivity forensics (the actual research question):
      -> fig1 profiles (A: merge deg~12/rank_dev 0.36-0.40 -> hold deg~17/0.10),
      fig2 case 1014 (k12's failure rescued), knn10's 3 failures all solved;
      oldNN = mimic negative control (rank_dev ~0.01).
- [x] Report to user in Korean (numbers-first, honest re: significance).
      -> delivered end-of-session 2026-08-06 ~23:10 local; REPORT_KO.md final.

## Phase 5 — Wrap-up  [x]
> STATUS 2026-08-06 ~23:15: COMPLETE. Docs final; user approved and commits
> executed (code commit d600293 + this study commit). Study CLOSED.

- [x] RUNLOG complete; REPORT (Korean) in this study dir; predecessor study's
      REPORT_KO gets a one-line pointer here.
- [x] Decide with user: commit (explicit approval only, no AI attribution,
      never push). -> User-approved scope committed 2026-08-06: repo code
      (d600293) + study docs/src + predecessor pointer (this commit).
      Per user: KICKOFF docs deleted (not committed);
      REPORT_hardtopk_distaux_KO.md left untracked. Nothing pushed.

## Design decisions already made (do not relitigate without user input)

| Topic | Decision | Source |
|---|---|---|
| Criterion | C2 = phi>0.98/50 + all-agents single r0-comp/300 + rel p2p sigma_p/300 < 5% | user + sweep 2026-08-06 |
| Success topology | single flock only; multi-flock/singleton = failure | user-delegated choice |
| Termination | success-only early stop; failures run to cap | user approved |
| Eval metrics | success rate; t_conv; J (=energy + rho*dt*t_conv); margins phi_ss/sigma_v_ss/NND_min; degree/churn/sigma_p descriptive only | user settled |
| Degree | NOT a communication cost (global obs); diagnostic only | user correction |
| Compactness | NOT an objective; tripwire only | user correction |
| Action space | A (bernoulli, no prior) AND B (threshold-adaptive + annealed prior); B must not be rigged into kNN | user |
| Success bonus | small: 10 (range 5-20) | user ("작게 시작") |
| dist_aux | keep but careful: anneal 1.0->0.2 over 400k steps, never 0, never dominant | user ("신중히") |
| Repo edits | allowed, additive + flag-gated + regression-checked | user |
| GPUs | A: cuda:1, B: cuda:3; CPU total <= 64 threads | user |
