# PROBLEM.md — acs-robust-train

> **Purpose of this file**: Defines WHAT this study is trying to achieve and under
> which constraints. Read this first, then PLAN.md (how), then RUNLOG.md (what has
> actually been done). Fresh sessions reconstruct context from these three files
> plus the two predecessor studies.

## Goal

Produce a **single learned neighbor-selection policy that is robust across
initial-condition scale L** — the user's stated objective (2026-08-07) is "a
sufficiently high-performing neighbor-selection method obtained by learning",
i.e. a METHOD, not a per-condition artifact. The predecessor policy
(acs-c2-train A it60) meets the frontier at its training scale but its
efficiency collapses off-distribution (L=500: J 249, worse than FC's 235),
because observations are normalized by the static L/2.

Two mechanisms are trained and compared (single-variable ablation pair):
- **R1 ("lmix_legacy")**: L-mixed training (episode L ~ U{125, 250, 500}) with
  the legacy per-episode L/2 obs normalization. Tests whether scale DIVERSITY
  alone fixes robustness (note: per-episode normalization makes coordinates
  init-relative, but r0/L density still varies 4x across the pool).
- **R2 ("lmix_r0log")**: same L mixing PLUS scale-free observations — relative
  positions mapped d -> unit(d) * log1p(|d|/r0) (r0=60 is the physically
  meaningful cohesion radius; log compresses far-field). Tests whether
  r0-anchored scale-free features are needed on top of diversity.

Head: bernoulli (the proven A-line). B-line threshold head is out of scope
(user chose the robustness direction over B continuation).

## Why (from predecessor `studies/acs-c2-train/`)

1. A it60 (bernoulli, C2-trained at L=250): 31/32, J_med 156 — matches the
   best fixed k (k=12: 31/32, 160), decisively beats FC (p=3.3e-4), with
   GENUINE adaptivity (merge-phase rank_dev 0.36-0.40, rescues k-NN failure
   seeds). But L-generalization probes: L=125 J 170 (vs k12 155), L=500 J 249
   (vs k12 165.6, FC 235) — efficiency is L=250-specialized.
2. Mechanism: ego obs divide relative positions by the STATIC L/2
   (env.py `_get_ego_centric_obs`), so changing L rescales the perceived
   geometry; the learned merge behavior mistunes. k-NN is scale-free.
3. Frontier under C2+J (32 paired seeds, predecessor data): single fixed
   k=12 is success-first optimal at ALL of L={125,250,500}:
   success 31/31/32, J_med 155.0/160.0/165.6. THIS is the bar a "robust
   method" must clear — not a strawman: k12 needs no retuning across L.
   (FC: 32/32 everywhere, J 224/228/235. k10: 30/29/31, J 132/171/173.)
4. C2 criterion + J metric are level-free by construction (phi, r0-proximity
   components, relative sigma_p stationarity; J = turn energy + rho*dt*t_conv)
   — they transfer across L without modification. Verified: predecessor
   frontier_L.py ran them at 125/500 unchanged.

## Success criteria (honest, pre-registered 2026-08-07 before training)

Protocol: ONE checkpoint (no per-L selection), N=20, paired seeds 1000..1031
(32 episodes) at EACH L in {125, 250, 500}, 6000 steps, deterministic
(argmax), offline C2 judge, paired tests vs the k=12 per-seed references.

- **Primary**: at EVERY L in {125,250,500}: success >= k12's (31/31/32) AND
  J_med <= k12's + 5 (i.e. <= 160.0/165.0/170.6); AND pooled paired
  dJ vs k12 < 0 with p < 0.05 (96 paired episodes).
- **Stretch**: strict J_med < k12 at every L; and/or extrapolation probe
  L=750 (outside training pool) beats FC and matches k12.
- Diagnostic value even if missed: where does the single policy pay for
  generality (vs the specialized A it60 at L=250)? Do adaptivity profiles
  (rank_dev, degree-vs-time) transfer/adapt across L? Does R2's scale-free
  representation beat R1's per-episode normalization?

Deferred (Phase-2 within this study, only if a reliability gap persists after
the L-mix runs): failure-weighted seed resampling / checkpoint ensembling
(disjoint failure-set observation from predecessor).

## Constraints (binding, inherited)

- Repo code MAY be modified, ADDITIVE and FLAG-GATED only (new config fields
  default to legacy behavior); existing entry points keep working; byte-level
  regression gate before training.
- GPU: **cuda:1 and cuda:3** (user re-confirmed 2026-08-07); cuda:0/2
  off-limits. CPU <= 64 threads total.
- **git push absolutely forbidden.** Commits only with explicit user approval;
  no AI attribution. Heavy artifacts never committed (data/figs/logs
  gitignored; checkpoints under /workspace/test_results/).
- User-facing reports in Korean; code + internal docs in English.
- Pinned stack: Pydantic v1, gym 0.23.1, Ray/RLlib 2.1.0, Torch 1.12.1+cu113,
  NumPy 1.23.4.
- Keep PROBLEM/PLAN/RUNLOG current (session-handoff quality).

## Key reference material

- `studies/acs-c2-train/REPORT_KO.md` — predecessor results incl. frontier-L
  table (§4.6) and adaptivity forensics; RUNLOG there for run mechanics.
- `studies/acs-c2-train/src/` — eval_c2.py (ckpt rollout + forensics + C2
  judge), frontier_L.py, compare_frontier.py, regression/crosscheck gates.
- `studies/acs-conv-knn/src/common.py` — rollout harness + offline judge
  internals; NOTES_env.md — env equations reference.
- Frontier references: `studies/acs-c2-train/data/frontier_L.csv` (32-seed
  summaries per L) + predecessor npz for per-seed pairing.
