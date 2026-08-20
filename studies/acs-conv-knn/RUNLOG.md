# RUNLOG.md

> **Purpose of this file**: Append-only chronological log of everything actually DONE
> in this study: commands run, experiment batches launched (with exact parameters and
> output paths), key numeric results, decisions made and why, and failures/gotchas.
> A fresh Claude session should read PROBLEM.md → PLAN.md → this file (newest entries
> last) to know the exact current state and resume without repeating work.
> Convention: one `## [YYYY-MM-DD HH:MM] title` block per event, terse bullet style.

## [2026-07-30] Session 1 start

- Created study dir `/workspace/studies/acs-conv-knn/` with `src/ data/ figs/ logs/`,
  plus PROBLEM.md / PLAN.md / RUNLOG.md and .gitignore (ignores data, figs, logs).
- Dispatched two exploration subagents: (a) `envs/env.py` ACS dynamics + metrics +
  termination + config fields; (b) `baselines.py` k-NN baseline + rollout pattern
  from `evaluate_checkpoint.py`.
- Machine: 72 cores visible (user cap: 64 threads), Python 3.9.5. GPU not needed.

## [2026-07-30] Phase 0 done — codebase grounding

- Both exploration reports distilled into `NOTES_env.md` (ACS equations, metric
  formulas, termination logic, rollout contract, parameter card). Read that file
  first in any future session.
- Decision: harness derives all metrics from `env.state['agent_states']` xy/vel
  columns only (no heading-column dependency) and cross-checks own sigma_p/sigma_v
  against info values each run (`env_metric_max_diff_*` in npz meta).

## [2026-07-30] Harness + smoke + calibration

- Wrote `src/common.py` (rollout + per-step metrics + npz writer) and
  `src/run_sweep.py` (multiprocessing sweep CLI, OMP threads pinned to 1).
- Fixed off-by-one: series arrays sized T+1 (t=0 initial sample included).
- Smoke (`data/smoke/`, k=5 L=250 300 steps): OK, own-vs-env metric diff = 0.0.
  Speed: 300 steps ≈ 0.3 s → 6000 steps ≈ 6.7 s/run.
- Calibration (`data/calib/`, k∈{1,2,5,19} × L∈{250,500} × 2 seeds × 6000 steps):
  - k=19 (FC): sigma_p freezes at 38.7-40.9 by ~1000 steps (threshold is 42!);
    t(phi>0.99) ≈ 300-680; t(sigma_v<0.1) ≈ 460-1030. Converges under env criterion.
  - k=5: single-cluster case plateaus at sigma_p≈90 (never < 42); phi>0.99 by ~430
    but sigma_v ≈ 0.2 forever (>0.1 goal). Fragmented cases: sigma_p grows linearly.
  - k=1,2: always fragmented into 6-17 proximity clusters; no global alignment.
  - Conclusion: 6000-step horizon sufficient (plateaus by ~2000 or diverging).

## [2026-07-30] Main sweep launched

- `python run_sweep.py --batch main --ks 1,2,3,4,5,6,7,8,10,12,15,19
  --bounds 125,250,500 --reps 32 --steps 6000 --workers 56`
- 1152 runs → `data/main/`, log `logs/main.log`. Seeds 1000+rep paired across k
  within (L, rep). ETA ~3 min.

## [2026-07-30] Main sweep done (254 s, 1152/1152 ok) — headline numbers

- Summary: `data/summary_main.csv` (via `analyze.py --batch main`).
- **Current env criterion passes ONLY at k=19** (P_env_pass=1.0 there, 0.0 at every
  other (k,L) incl. k=15). FC equilibrium sigma_p = 39.4-40.0 vs goal 42 = 0.7*r0
  (= r0/sqrt2 "all pairs at r0" idealization, identity E[r_ij^2]=2 sigma_p^2).
- Equilibrium sigma_p of cohesive runs is L-INDEPENDENT (densities 16x apart
  overlap) and follows sigma_p ≈ 198*k^(-0.53) ≈ sigma_p_FC*(19/k)^0.53 (≤8% err).
- P(single cluster): k≤2: 0%; k=3: ~3%; k=4-7: 16-81%; k≥10: 88-100%; k=19: 100%.
  Fragments never re-merge (clusters > k are k-NN-self-contained).
- Alignment phi>0.99 fast for all cohesive runs (230-570 steps); sigma_v<0.1 goal
  == phi>0.99998 (speed 15) — reached 2-3x slower on sparse k, equilibrium sigma_v
  bimodal across ~6 decades straddling 0.1.
- Selected-edge mean distance at equilibrium = 54.6±1 m ≈ 0.91*r0, CONSTANT over
  k=3..19 (while sigma_p varies 3x, NND varies 52→17 m) — ACS bonding regulates
  selected-pair distances only; global spread is a topology-diameter byproduct.
- Steady-state churn ≈ 0 (k-NN topology freezes); slope_sp_late ≈ 0 for cohesive.

## [2026-07-30] Disc batch + criteria evaluation + figures + report

- Disc batch (`data/disc/`, 80 runs, thresholds {0.2,0.35,0.48,0.7,1.0} in
  L/2-normalized units = {25,44,60,87.5,125} m, L=250): radius ≤ 87.5 m → 0%
  cohesion (graph empties, final degree 0); 125 m → 87.5% cohesion, converges to
  ~FC state (deg 18.8, sigma_p 42.1, env-criterion pass 0.44). Disc degree drifts
  over time (deg temporal SD up to 0.85) — per-k threshold correction infeasible.
- `src/criteria_eval.py`: candidate criterion C1 = [phi>0.99 sustained 50] AND
  [r0-proximity graph single component sustained 500] AND [|sigma_p(t)/sigma_p
  (t-500)-1| < 2%]. Result over all 1152 runs: 100% detection on cohesive runs at
  every (k,L), 0.6% false-positive on fragmented (5/818, late-fragmentation edge
  cases), t_C1 ≈ 660-2050 (FC ~800, comparable to current criterion's ~700).
  Output: `data/criteria_main.csv`.
- Figures in `figs/` (fig1 timeseries, fig2 equilibrium-vs-k, fig3 fragmentation,
  fig4 time scales, fig5 local structure). fig2 right panel reworked to per-run
  scatter (sigma_v_ss spans decades; error bars misleading). Log-x minor labels
  suppressed via NullFormatter in `_style`.
- User-facing report written: `REPORT_KO.md` (Korean). Study complete for N=20.
- Open follow-ups (see REPORT_KO.md limitations): N-dependence of the power law,
  C1 robustness under a stochastic NN policy checkpoint, false-positive removal
  via longer cohesion window, reward-shaping still anchored to std_pos_target=39.5.

## [2026-07-30] Autonomous tick — N-dependence verified (Phase 5 item 1 done)

- Batches `data/n10/` (N=10, L=176.78, k=1..9, 16 seeds, 6000 steps, 25 s) and
  `data/n40/` (N=40, L=353.55, k∈{2..6,8,10,13,16,20,26,32,39}, 16 seeds, 8000
  steps, 104 s) at initial density matched to N=20/L=250.
- `src/analyze_ndep.py` (fits + collapse + fig6), `criteria_eval.py` gained
  --batch arg. Results:
  - sigma_p_FC(N) = 39.8 / 39.8 / 40.6 for N=10/20/40 → **N-independent FC anchor**
    (r0/sqrt2 idealization), so goal 42 stays "FC-only" at every N.
  - Per-N exponents -0.50..-0.53; universal collapse
    **sigma_p/sigma_p_FC ≈ 1.05·(k/(N-1))^(-0.53)** (pooled fit over 3 N).
  - Selected-edge mean distance 56.3/54.7/53.2 (±1.2) m across k — constancy holds.
  - C1: 100% detection on cohesive at both new N; false positives 2.0% (n10) /
    1.1% (n40), same late-fragmentation type.
- REPORT_KO.md updated (new "N 의존성" section; limitation #1 resolved),
  fig6_N_dependence.png added. Autonomous loop stopped — study wrapped; remaining
  Phase 5 items (NN-checkpoint validation, reward-shaping experiment) await user
  direction.

## [2026-07-30] User challenge → stochastic-topology validation + phi relaxation

- User (correctly) challenged that C1 was claimed for NN without evidence; k-NN and
  disc both END with quasi-frozen topologies, so per-step stochastic switching was
  never stress-tested. Also user approved relaxing the heading criterion to a
  swarm-literature-standard order-parameter level → C1 alignment condition changed
  from phi>0.99 to **phi>0.97** (= the repo's own Vicsek alignment_goal default;
  `criteria_eval.py --phi`, new default 0.97).
- `common.py` refactored: generic `rollout(policy, cfg, seed, ...)` extracted from
  run_episode (any obs->action callable; run_episode is now a thin wrapper).
- Worst-case churn stress (`src/run_stress_random.py` → `data/stress_random/`,
  48 runs): 'random' baseline re-samples neighbors i.i.d. EVERY step,
  p∈{0.15,0.3,0.6} (mean degree 2.9/5.7/11.4), N=20 L=250 6000 steps. Results:
  churn_ss 0.92/0.82/0.57 (vs ~0.000 for k-NN) yet 100% cohesion and sigma_p_ss ≈
  38.4 (≈FC level — annealed random pairs ≈ mean-field). BUT phi_ss plateaus at
  0.983/0.991/0.997 (random far-neighbor kicks) → sigma_v_ss 1.2-2.7 >> 0.1, so
  **C0 passes 0% here despite sigma_p<42** (heading side fails; mirror image of
  the k-NN failure). C1 with phi>0.99 passed only 66.7% of these cohesive runs;
  **with phi>0.97 → 100%**. False-positive rates unchanged on all other batches
  (main 0.6%, n10 2%, n40 1.1%, disc 0%); C1@0.97 detection stays 100% everywhere.
- NN checkpoint rollouts (`src/run_nn_rollouts.py`, CPU-only): Phase-14 winner
  hardtopk10_distaux_260529/checkpoint_000010 via evaluate_checkpoint.RLPolicy
  (deterministic argmax = trained-as-evaluated), 16 seeds × 6000 steps →
  `data/nn_hardtopk/`. Results: 16/16 single cluster, deg exactly 10.00, churn
  ≈0.000-0.008 (quasi-frozen learned-KNN), phi→1.000, sigma_p_ss ≈ 55-65 m
  (matches 'nearest' k=10 equilibrium ~60 m). **C1@phi0.97: 16/16 pass (median
  t=761, sigma_p≈60 at detection). Current criterion C0: 0/16** — the project's
  own FC-beating winner never "converges" under the current criterion.
- Full C1@0.97 sweep over all batches: cohesive detection 100% everywhere
  (main/n10/n40/disc/stress_random/nn_hardtopk); false positives main 0.6%,
  n10 2%, n40 1.1%, disc 0% (phi change did not affect them).
- REPORT_KO.md updated: C1 definition now phi>0.97 (user-directed relaxation to
  swarm-standard order-parameter level; == repo Vicsek alignment_goal), new
  section "추가 검증 2: 확률적 선택과 학습된 NN 정책" with strategy-family table,
  limitation #2 resolved (soft-selection sampled checkpoints still untested but
  bracketed by frozen/max-churn extremes).

## 2026-08-06 — C2 finalization (window sweep) + evaluation-metric groundwork

- User decisions: heading phi>0.98 hold 50 (global OP); spatial = maintenance-only
  (no absolute level); multi-flock question delegated -> CHOSE single-flock-only
  success. Rationale: (i) with success-only early termination, multi-flock success
  legalizes the sacrifice/fragment-freeze reward hack; (ii) the trained policy has
  global observation + free selection, and FC coheres 100% in all our data, so
  single-flock is always feasible => no unfairness; (iii) per-component "settled"
  detection kept only as a diagnostic for heuristic parameter studies (disc), not
  as success. Size-1 components impossible by construction (n_comp==1 includes all).
- `src/criteria_c2_sweep.py`: 1,648 runs (main/disc/stress_random/nn_hardtopk/
  n10/n40) x 48 configs: form in {2-point, windowed rel p2p}, W in {50..500},
  eps in {2,3,5}%; alignment fixed phi>0.98 roll-min 50. Outputs
  data/c2_sweep_runs.csv, c2_noise_floor.csv, c2_sweep_summary.csv, figs/fig7.
- Steady-state noise floor (worst tail rel-p2p): non-random strategies p99
  1.6->3.7% (W 50->500); max-churn random p50 ~5.1-5.7%, p99 13-34%. => eps=2%
  starves stochastic strategies; eps=5% is the inclusive choice (mid-training
  stochastic policies resemble the random family, not frozen k-NN).
- Form: p2p strictly dominates 2-point at equal (W,eps) on fp and premature
  (equal on monotone signals, strict on oscillations) -> p2p chosen.
- W tradeoff (eps=5%, p2p): W=50: premature(>1.1x final) 18.6%, fp 3.0%, good-set
  t_med 312; W=300: premature 9.6%, fp 1.9%, t_med 551, <=1000 96.1%, <=1500 100%;
  W=500: premature 6.4%, fp 1.0%, t_med 756, <=1000 89%. CHOSE W=300 (both cohesion
  hold and stationarity band): premature halved vs 50, fires 100% within 1500.
- **C2 final: phi>0.98 hold 50  AND  all-agents single r0-component hold 300  AND
  rel p2p(sigma_p) over 300 < 5%.** Per-family: detection 100% on every real
  strategy family (all knn N10/20/40, disc R125, NN ckpt 16/16 t_med 537); only
  miss = random p=0.15 83% (phi0.98-hold is the binding constraint; stress case,
  not a real policy). Premature concentrates in mid-k slow contraction (k6-8 N20
  ~17-29%, ratio_med <=1.04; FC and NN ~0-19% with ratio_med 1.00-1.02).
- Known biases (documented): t_fire includes +W offset (common to all strategies,
  cancels in comparisons); premature = quasi-stationary tail drift, not transient
  (median fire at 1.011x final); fp 1.9% = late fragmentation -> training label
  noise, harmless for eval (full-horizon re-check available offline).
- Training wiring (user approved): early-terminate ONLY on C2 fire (success);
  failed-but-settled states run to cap (blocks fragment-freeze shortcut).
  Recommend max_time_steps 1500 (100% fire coverage; good-phase episodes end
  ~550 so average rollout cost drops once the policy is competent).
- `src/j_metric_preview.py` (data/j_metric_preview.csv): headline scalar
  J = -sum(reward) to t_fire (= turn energy + rho*dt*t, per-agent mean), C2 gate,
  L=250: FC J=228 (success 1.00, deg 19); knn k=8/10/12 J=174/171/160 (success
  0.91/0.91/0.97); disc R125 J=206 (0.88); NN ckpt J=166 (success 16/16, deg 10.0,
  t_med 537); random p0.3 J=1079 (churn cost exposed). Note: NN 16/16 vs knn10
  29/32 is suggestive of adaptive-early-phase value but not significant at these
  n (Fisher p~0.5) -> needs more seeds if claimed.
- REPORT_KO.md: added "확정 기준 C2" section; corrected the earlier "swarm 문헌
  표준" overclaim (no single canonical OP threshold exists; 0.97/0.98 justified
  by common ad-hoc range + repo-internal consistency). PLAN.md Phase 6 added.

## 2026-08-06 (later) — evaluation metrics settled; user corrections

- User correction 1 (accepted): realized degree is NOT a communication proxy —
  observation is global (ego obs contains all agents, comm_range=None), so
  information is already communicated; selection only prunes the ACS control-law
  input. Degree demoted from evaluation axis to descriptive diagnostic (detecting
  "policy == k-NN(k)" mimicry). The value of sparse selection is dynamical and is
  captured directly by J: FC J=228 vs k12 J=160 (pruning far neighbors lowers
  control effort intrinsically). Honest consequence: current NN winner (J=166,
  16/16) sits ON the k-NN frontier (k12 J=160, 31/32), not clearly above it;
  the exploitable margin for learning is adaptivity (init-dependent / per-agent /
  time-varying selection), which becomes the training-design agenda.
- User correction 2 (accepted): compactness is NOT an objective; revisit only if
  near-non-compact flocks actually appear (tripwire in eval reporting, not in
  reward/criterion). R2 frontier-ratio term shelved.
- FINAL evaluation scheme: gate = C2 success rate (Wilson CI); on successes:
  t_conv and J = turn energy + rho*dt*t_conv (rho is the time-vs-energy exchange
  rate; report lambda x0.5/x2 sensitivity); quality margins phi_ss, sigma_v_ss,
  NND_min (collision margin); descriptive diagnostics: sigma_p_ss, degree, churn.
  Paired seeds + Welch, deterministic eval. REPORT_KO RL-implications line
  updated accordingly.

## 2026-08-06 (later 2) — architecture diagnosis for the training-design discussion

- Fact check via code read (models/ppo.py, train_hardtopk.py, beta_dist.py,
  grad_logging_ppo.py, callbacks.py):
  * Winner run hardtopk10_distaux came from train_hardtopk.py: BINARY action,
    hard top-K in the MODEL (ppo.py:306-323, K=top_k=10 from model config),
    dist_aux_coef=1.0 (rank-by-distance target, K=10 HARDCODED at ppo.py:441).
    train.py is a different (continuous) variant: squashed-Gaussian "beta_dist"
    weights, env clips to [0.2,1.0] weight floor and feeds WEIGHTED adjacency to
    ACS (env.py:587-592,681) — never binarized, edges cannot be removed.
  * Saturation diagnosis: binary path logits are +/-(0.1*att +/- 20) =>
    p_select = sigmoid(0.2*att +/- 40) ~ 1/0. PPO ratio ~= 1, entropy ~ 0,
    d(logp)/d(att) ~ 4e-18 -> POLICY GRADIENT THROUGH SELECTION IS ~ZERO.
    Learning was carried by aux losses (dist_aux 1.0 + pair_embedding 0.3),
    which by construction push att to distance rank -> policy == kNN(10) mimic.
    Behavioral confirmation from our rollouts: deg exactly 10.00, churn ~0,
    sigma_p_ss ~ kNN10 equilibrium, J=166 ~ kNN10's 171. "Beat FC +15.6%" ==
    kNN10 beats FC. The architecture structurally cannot exceed the k-NN
    frontier; adaptivity margin requires reopening the gradient path.
  * Other notables: no positional encoding (permutation-equivariant, good);
    critic = separate tower on double-mean-pooled embedding; callbacks'
    flocking_success = (episode.length < max_steps) -> becomes meaningful
    automatically once C2 early termination is wired; GradLoggingPPO = logging
    only; continuous path logp sums over ALL N^2 entries incl. padded/diag and
    env clips the executed action (ratio bias smells) — avoid as-is.

## 2026-08-06 (later 3) — handoff to successor study acs-c2-train

- User decisions on the training redesign: action space = BOTH A (plain
  bernoulli, no distance prior) and B (structured/adaptive; must NOT be rigged
  into kNN mimicry like the old winner); success bonus small (start ~10);
  dist_aux annealing included but careful (1.0->0.2 over 400k steps, never 0);
  repo code MAY now be edited for the successor study (additive + flag-gated);
  resources updated: cuda:1 AND cuda:3 available, CPU <= 64 threads.
- Implementation + training happen in a NEW session. Created
  `studies/acs-c2-train/` with PROBLEM.md (goal/background/success criteria/
  constraints), PLAN.md (Phases 0-5: env C2 termination + c2_shaping reward +
  global-stats obs; two selection heads with saturation-regression guards;
  training scripts train_c2_a/b.py; frontier evaluation + adaptivity
  forensics; pre-registered success = success>=31/32 AND J<=160), RUNLOG.md,
  .gitignore, KICKOFF_PROMPT_KO.md (paste-ready Korean kickoff for the next
  session). B's "learned K" is specced as a per-agent threshold head
  (differentiable generalization of top-K; composite action dist avoided on
  Ray 2.1.0). This study becomes reference material for the successor.
