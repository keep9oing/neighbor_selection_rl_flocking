# PROBLEM.md — acs-robust-r2

> Round 2 of the robust-single-policy program. Read this, then PLAN.md, then
> RUNLOG.md. Predecessors: `studies/acs-robust-train/` (round 1: success-robust
> but J gap +35..+90 vs k12; r0_log ablation negative; gap = merge-phase turn
> energy) and `studies/acs-c2-train/` (specialist A it60, C2 criterion).

## Goal (user decisions 2026-08-07)

Close the efficiency gap of the round-1 generalist WITHOUT losing its
success-robustness, by changing the TRAINING INITIALIZATION (not the reward,
not the pool). Two contrasted runs, one GPU each:

- **F1 ("ft60")**: L-mix fine-tune starting from specialist **A it60**
  (the converged, criterion-meeting L=250 policy; 100 iters of prior training).
- **C1 ("ft40")**: the curriculum idea implemented as L-mix fine-tune from
  **A it40** (mid-training specialist, 29/32-J144 offline; equals
  "40 iters fixed-250 then switch to mix", reusing the existing 250-run as
  curriculum stage 1 at zero extra compute).

F1 vs C1 differ ONLY in the init checkpoint; both vs R1 (scratch) isolate
"init strategy". Hypothesis: starting from a policy that already selects
efficiently at L=250 preserves in-distribution efficiency while L-mix
training adds the flat cross-scale behavior.

In parallel (CPU only), assemble paper-grade evidence:
- **Big-n reliability**: 500 paired seeds at L=250 and L=500 for R1 it110 vs
  k12 (+ A it60 at 250) — turn the 32-seed "perfect success" observation into
  a statistically decisive claim (or refute it honestly).
- **N-axis evaluation**: N=10 (L=177) and N=40 (L=354) matching-density
  probes for the round-1/2 winners + k-NN frontier — first evidence beyond
  N=20.
- **Small-L probe**: L=75 (initial spacing ~17 m ~ r0/3.6, compressed
  regime) for R1 it110, A it60, and the k-NN frontier — completes the regime
  map (user hypothesis: compressed starts may reward control frugality).

## Success criteria (pre-registered, unchanged bar + new secondary claims)

Primary (same as round 1; judged on ONE checkpoint per run, grid 32 paired
seeds x L={125,250,500}, argmax, offline C2):
- at EVERY L: success >= k12's (31/31/32) AND J_med <= k12's + 5
  (160.0/165.0/170.6); AND pooled paired dJ vs k12 < 0, p < 0.05.

Secondary (pre-registered here):
- S1 reliability: at pooled L={250,500}, 500 seeds each, R1-it110 (or the
  round-2 winner) failure count < k12's with two-sided Fisher p < 0.05.
- S2 N-transfer: at (N=10,L=177) and (N=40,L=354), 32 seeds: success >= 30/32
  without any retraining (descriptive J vs the locally computed frontier).

Diagnostic regardless: F1/C1/R1 efficiency-vs-scale curves (does init
strategy preserve the specialist's low-L efficiency?); L=75 regime behavior.

## Constraints (inherited + new session rules)

- Additive, flag-gated repo edits only; regression gates before training.
- GPU cuda:1 (F1) and cuda:3 (C1) ONLY; CPU <= 64 threads total.
- git push forbidden; commits only on explicit user approval, no AI
  attribution; heavy artifacts never committed.
- Korean for user-facing reports; English for code/internal docs.
- Pinned stack: Pydantic v1 / gym 0.23.1 / Ray 2.1.0 / Torch 1.12.1 / NumPy
  1.23.4.
- **Session management (user directive 2026-08-07)**: check context size via
  `cat /tmp/ctx` at natural checkpoints; when it reaches **319K**, finish only
  up to a clean handoff boundary, then hand off. BEFORE ending the session,
  kill/stop ALL background monitors and pending background tasks so that no
  late notification revives the finished session for extra work.

## Key reference material

- `studies/acs-robust-train/REPORT_KO.md` — round-1 results incl. grid table,
  turn-energy diagnosis (§4.2), specialist crossover (§4.4).
- `studies/acs-robust-train/src/` — eval_c2.py (obs-mode passthrough),
  grid_judge.py, scale_unit_check.py, monitor_runs.py (adapt run names).
- `studies/acs-c2-train/data/frontier_L.csv` — per-seed k-NN refs at
  L={125,250,500}, k={8,10,12,19}, seeds 1000-1031. New conditions (L=75,
  N-axis, seeds 1032+) need fresh k-NN runs via
  `studies/acs-conv-knn/src/common.py::run_episode`.
- Checkpoints: A it40/it60 under
  /workspace/test_results/c2A_bernoulli_260806/GradLogging*/checkpoint_0000{40,60};
  R1 it110 under /workspace/test_results/c2R1_lmix_legacy_260807/.../checkpoint_000110.
- Paper-stage backlog (NOT this round): distillation baseline (hand-coded
  phase-switching k rule vs learned policy), N-mix training, small-L training
  pool extension, related-work sweep (topological interaction literature).
