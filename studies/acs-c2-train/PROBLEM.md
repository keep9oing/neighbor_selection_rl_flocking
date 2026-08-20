# PROBLEM.md — acs-c2-train

> **Purpose of this file**: Defines WHAT this study is trying to achieve and under
> which constraints. Read this first, then PLAN.md (how), then RUNLOG.md (what has
> actually been done). A fresh Claude session must be able to reconstruct the full
> context from these three files plus the referenced predecessor study.

## Goal

Train neighbor-selection policies (PPO, ego-centric transformer) for the ACS
flocking task under the **C2 convergence criterion** and the **settled evaluation
scheme**, such that the learned policy **genuinely beats the k-NN frontier** —
not by mimicking a fixed k-NN (which is what the previous winner did), but by
exploiting adaptivity: initial-condition-, agent-, and phase-dependent selection.

Two action-space variants are trained and compared (user decision 2026-08-06):
- **Variant A ("bernoulli")**: plain per-edge binary policy, NO distance-rank
  prior (dist_aux off). Tests whether RL alone discovers selection structure.
- **Variant B ("threshold")**: adaptive-degree structured selection — per-agent
  learned threshold over pair scores (a differentiable generalization of
  "top-K with learned K"), WITH a distance-rank prior (dist_aux) that is
  annealed carefully. Must NOT be rigged into k-NN mimicry (user directive:
  "억지로 k-NN 비슷하게 만들어버리는 꼼수 부리지 말고").

## Why (background from predecessor study `studies/acs-conv-knn/`)

1. The legacy convergence criterion (sigma_p<42 & sigma_v<0.1) is FC-anchored:
   only fully-connected-like policies can ever "converge". Replaced by **C2**
   (level-free, validated on 1,648 runs):
   - alignment: phi > 0.98 held 50 steps,
   - cohesion: ALL agents form a single r0-proximity component (pair dist < r0)
     held 300 steps (multi-flock and singleton outcomes are failures),
   - stationarity: relative peak-to-peak of sigma_p over last 300 steps < 5%.
2. Evaluation scheme (user-settled): gate = C2 success rate; on successes
   t_conv and **J = turn energy + rho*dt*t_conv** (per-agent cumulative raw cost
   to convergence); quality margins phi_ss, sigma_v_ss, min pair distance;
   degree/churn/sigma_p_ss are descriptive diagnostics only. Degree is NOT a
   communication cost (observation is already global); compactness is NOT an
   objective.
3. Architecture diagnosis (2026-08-06): the previous "winner"
   (hardtopk10_distaux, +15.6% vs FC) had a **saturated action path**
   (logits ±(0.1·att ± 20) → p = sigmoid(0.2·att ± 40)): PPO's policy gradient
   through selection was ~1e-18, i.e. zero. Learning was carried entirely by
   auxiliary supervision (dist_aux rank regression, K=10 hardcoded), so the
   policy IS k-NN(10) by construction (confirmed behaviorally: deg exactly
   10.00, churn ~0, sigma_p ~ kNN10, J=166 ~ kNN10's 171). "Beat FC" == "kNN10
   beats FC". The k-NN frontier itself: FC J=228 (success 32/32), k=12 J=160
   (31/32), k=10 J=171 (29/32) — reference numbers, L=250, N=20, paired seeds.
4. Therefore the exploitable margin for RL is **adaptivity** — e.g. k-NN(10)
   fragments in ~9% of L=250 inits; an adaptive policy can bridge clusters in
   the merge phase and slim down in the hold phase. The NN ckpt's 16/16 success
   (vs 29/32 for frozen kNN10) is suggestive but not significant.

## Success criteria (honest, pre-registered)

Primary (protocol: N=20, L=250, paired seeds 1000..1031, 32 episodes,
deterministic eval, offline C2 judge, Welch/paired tests):
- success rate >= 31/32 AND J_med <= 160 (match-or-beat k=12, the best fixed k).
Stretch:
- J_med < 150 with success 32/32 (clearly above the frontier), and/or clear
  Pareto dominance at harder inits (L=500 generalization, where fixed k-NN is
  more fragile).
Diagnostic value even if not beaten:
- quantify WHERE adaptivity acts (rank-deviation metric: fraction of selected
  edges outside the nearest-deg_i set, per phase; degree-vs-time profiles).

## Constraints (binding, inherited + updated)

- **Repo code MAY be modified for this study** (user decision 2026-08-06),
  but changes must be ADDITIVE and FLAG-GATED (new config fields default to
  legacy behavior); existing entry points (train.py, train_hardtopk.py,
  evaluate_checkpoint.py, test_baselines.py) must keep working unchanged.
- GPU: **cuda:1 and cuda:3** are available for this study (user update
  2026-08-06); cuda:0/2 are off-limits. Plan: variant A on cuda:1, variant B on
  cuda:3 (one full GPU each via CUDA_VISIBLE_DEVICES). CPU <= 64 threads total
  across everything.
- **git push is absolutely forbidden**, even if asked. Commits only with
  explicit user approval; NO AI attribution in commit messages.
- Heavy artifacts (checkpoints, npz, figs, logs) are never committed; this
  study dir has data/, figs/, logs/ gitignored. Checkpoints go to
  /workspace/test_results/<run_name>/ as per repo convention.
- User-facing interaction/reports in Korean; all code and internal docs in
  English.
- Keep PROBLEM/PLAN/RUNLOG in THIS directory current (session-handoff quality).
- Pinned stack: Pydantic v1, gym 0.23.1 (4-tuple step), Ray/RLlib 2.1.0,
  Torch 1.12.1+cu113, NumPy 1.23.4. Do not migrate APIs.

## Key reference material

- `studies/acs-conv-knn/REPORT_KO.md` — C2 definition + validation numbers.
- `studies/acs-conv-knn/RUNLOG.md` — entries of 2026-08-06: C2 sweep, metric
  decisions, architecture diagnosis (file:line cites for ppo.py/env.py paths).
- `studies/acs-conv-knn/NOTES_env.md` — condensed env reference (ACS equations,
  metrics, rollout contract).
- `studies/acs-conv-knn/src/` — offline C2 judge (`criteria_c2_sweep.py`),
  J metric (`j_metric_preview.py`), generic rollout harness (`common.py`).
- Frontier reference data: `studies/acs-conv-knn/data/` (gitignored npz/csv;
  regenerable via run_sweep.py).
