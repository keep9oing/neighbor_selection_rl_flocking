# PROBLEM.md — acs-robust-r3-stress

Pre-registered 2026-08-08, BEFORE any new data collection. User decision
(2026-08-08, AskUserQuestion): start ONLY candidate #1 from r2 REPORT §7-4
(adversarial baseline stress-test). No training, no reward/env/repo changes,
CPU-only. w_ctrl pilot / pool extension / criteria redesign all NOT approved.

## Goal

The r2 statistics audit reduced the learned method's demonstrated advantage to
an "insurance" structure: slightly worse typical-seed J, large win on k12's
failure/blowup tail (8-9% -> 0.1%), plus N-transfer and low-N dominance. The
strongest remaining threat to these claims is a *better-chosen fixed-k
baseline*, in two forms:

1. A LARGER fixed k (k16/k20) might buy near-policy reliability at N=20 for a
   much smaller typical-J premium than the policies pay. S1 only ever measured
   k12 at n=500 — the reliability of larger k at scale is UNKNOWN.
2. An a-priori ratio rule k = round(0.65N) (band 0.6-0.7N, low-N-corrected)
   already beat C1's typical J at N=40 (k28, audit). Its exact instantiations
   (k7@N10, k13@N20, k26@N40) were never measured.

This study measures both threats at audit-grade statistical resolution and
judges three pre-registered hypotheses. Outcomes directly gate the remaining
r2 §7 candidates (see Decision mapping).

## Hypotheses and operational verdicts (pre-registered)

Failure = offline C2 judge t_fire == -1 within 6000 steps (eval_c2.t_fire_c2,
PHI_GOAL/W_A/W/EPS = 0.98/50/300/0.05 — unchanged from r1/r2).
J = -sum(reward[1:t_fire+1]). Seeds: 500-seed arms use 1000-1499, 32-seed
arms use 1000-1031 (identical to r2; all comparisons seed-paired).

### H-A — insurance uniqueness (N=20, L in {250, 500}, n=500)

Claim under stress: "no fixed k buys policy-grade reliability without a
policy-grade typical-J premium."

H-A is REFUTED for policy P in {R1_i110, C1_i80} at level L iff there exists
an arm A in {k13, k14, k16, k18} with BOTH:
  (i)  reliability: fail_A <= 1.0% at BOTH L values (<= 5/500 each; this
       captures >= ~88% of the k12 -> R1 reliability gap), AND
  (ii) efficiency: P typically worse than A at that L — co-success paired
       median dJ(P - A) > 0 with Wilcoxon signed-rank p < 0.05.
Full refutation = refuted for both policies at both L. Partial outcomes are
reported exactly as measured. Regardless of verdict we publish the frontier
table (fail% + Wilson CI, typical J_med, CVaR10) for
{k12, k13, k14, k16, k18, FC*, R1, C1, A60**} at L250/L500
(* FC at 32 seeds only — known success-safe, J-dominated; ** A60 L250 only).

### H-B — a-priori ratio rule across N (32-seed cells)

Rule R065: k = round(0.65N) -> {7@N10, 13@N20, 26@N40}; band variant R07:
k = round(0.70N) -> {7, 14, 28}.
  (b1) Low-N survival: C1 beats k7@(N10, L177) — co-success paired Wilcoxon
       p < 0.05 with median dJ(k7 - C1) > 0, or exact McNemar p < 0.05 in
       C1's favor on success. (Already true vs k6/k8/FC; k7 closes the gap.)
  (b2) N=40 bracket: k26@(N40, L354) completes {k24, k26, k28}; report C1/R1
       vs k26 paired. No win claim pre-registered (r2 audit already retracted
       N=40 superiority).
  (b3) N=20 instantiation: k13/k14 fail rates at n=500 (from H-A arms) — does
       the rule's own k inherit k12's failure tail (>= ~3%) or not — plus
       their 32-seed J-grid rows at L in {125, 250, 500} (sliced from the
       500-seed data, seeds 1000-1031).

H-B survives iff (b1) holds AND (b3) shows the rule's own k (13 or 14) is
NOT policy-grade reliable (fail > 1% at some in-pool L). Else H-B is refuted
or weakened exactly as measured.

### H-C — the r2 winner's in-pool insurance (C1, n=500)

C1_i80 has NO big-n data today. H-C: C1 carries the insurance property
in-pool — fail <= 1.0% pooled over L in {125, 250, 500} x 500 AND per-L
exact McNemar significantly below k12 (p < 0.05). If C1 fails at 2-4%, the
r2 "2-policy Pareto" framing gets a reliability asterisk on the C1 side —
that is a finding, not a study failure. (L125 leg belongs to tier T4; if T4
is cut for resource reasons, H-C is judged on {250, 500} and the L125 gap is
stated explicitly.)

## Arms and tiers

All new k-NN cells via run_knn_refs3.py (copy of r2 run_knn_refs.py with
STUDY -> r3), policy lanes via eval_c2_r3.py (same treatment). New data under
studies/acs-robust-r3-stress/data/ ONLY; r2/r1/predecessor data read-only.

- T1 knnref n=500 (seeds 1000-1499): k in {13, 14, 16, 18} x L in
  {125, 250, 500} + k12@L125 -> 13 lanes. (k12@L250/L500 n=500 exist in r2.)
- T2 policy n=500: C1_i80 @ L250 and @ L500 (wave 1).
- T3 knnref n=32 (seeds 1000-1031): k in {12, 16, 19=FC} @ L in {375, 750}
  (first-ever k-NN refs at 375/750 — lets the L-axis frontier speak where
  only policy evals existed); k7 @ (N10, L177); k26 @ (N40, L354). 8 cells.
- T4 policy n=500, tier-2 (run unless resources force cuts; cuts logged):
  C1_i80 @ L125 and R1_i110 @ L125 (wave 2). H-C's L125 leg + R1's L125
  reliability.

NOT run (pre-registered scope cuts, stated as limitations): FC at n=500
(success-safe, J-dominated — position on frontier already determined);
A60 @ L500 n=500 (off-scale specialist, not central to H-A/H-C); 500-seed
refs at L375/L750 (extrapolation axis stays 32-seed this round); adaptive/
non-fixed-k heuristics (backlog: related-work sweep).

### Amendment A1 (2026-08-08 23:40, PRE-DATA)

The originally registered arm k20 at N=20 is mathematically impossible
(k <= N-1 = 19; k19 IS the FC baseline) — caught by the runner's own
assert at launch, before any N=20 k>12 lane produced data. Replacement:
H-A/T1 arm k20 -> k18 (near-FC bracket of the reliability-vs-J curve);
T3 extrapolation refs k20 -> k19 (adds the FC comparator at L375/L750).
No other change; no data existed for any affected arm at amendment time.

### Amendment A2 (2026-08-09 ~00:55, after T1 L250/L500 lanes, before any FC n=500 data)

T1 surprise: failure rate is FLAT-TO-WORSE in k across 12-18 (L500: 7-15%;
L250: 5-14%) — no arm approaches the 1% bar, and near-FC k18 fails 9-14%.
Failure-mode inspection (6 seeds, k16@L500 + k18@L250): ALL are
merge/cohesion failures (phi=1.0, spatial band stable, final n_comp_r0=2 —
parallel sub-flocks that never coalesce), not alignment failures. This makes
the pre-registered scope cut "FC not re-run at n=500 (success-safe)" unsafe
to assume: it rested on 32-seed evidence, and k18 vs k19 would be a sharp
one-edge cliff. ADD: FC (k19) @ L in {125, 250, 500} x 500 seeds. Purely
additive arm; H-A verdict rules unchanged (they quantify over k13-k18; FC
enters the frontier table and the honest narrative only). Registered before
any FC n=500 data existed.

### Amendment A3 (2026-08-09 ~07:10, POST-HOC extension — descriptive only; time corrected 2026-08-11 vs mtimes)

Launched in response to a user question ("isn't there a better k for each N
and L?"), AFTER the H-A/H-B/H-C verdicts were computed and reported. Adds
k-NN n=500 arms: k in {8,9,10,11,15,17} x L in {125,250,500} at N=20
(making the N=20 sweep contiguous k=8..19 at n=500), k in {6,7,8} at
(N=10, L=177), k in {24,26,28} at (N=40, L=354).

Status of these arms: DESCRIPTIVE EXTENSION, not part of the pre-registered
judgment. The three verdicts stand as computed on the registered arms. Note
the asymmetry that makes this safe: additional arms can only ever REFUTE
H-A (any arm with fail <= 1% + typical-J win would refute it), never
strengthen it by construction — so reporting them is a strictly harder test
of our own claim, not criteria-fitting. Any arm meeting the H-A refutation
bar must be reported as refuting it despite arriving post-hoc.

## Gates (before any 500-seed lane)

- G1 script-copy fidelity, k-NN side: run_knn_refs3.py k12@L250 seeds
  1000-1007 into r3 data; per-seed npz arrays and judged (t_fire, J) must be
  bit-exact vs r2's k12_L250_N20 npz for the same seeds.
- G2 script-copy fidelity, policy side: eval_c2_r3.py C1@L250 seeds
  1000-1003; (t_fire, J) must match r2's C1_i80_L250_s32 rows (dt=0, dJ=0.0).

## Statistics (pre-registered; audit_stats.py conventions)

- Failure axis: per-arm fail count + Wilson 95% CI; paired exact McNemar vs
  R1 and vs C1 (same L, seed intersection); unpaired Fisher vs k12 as sanity.
- J axis: co-success paired dJ — Wilcoxon signed-rank + exact sign test +
  median + q10/q90; typical-set (reference k-NN arm's J <= its own median) vs
  tail-set (ref J > ref q90) decomposition.
- CVaR10 = mean of the worst decile of success-J within an arm, reported
  ALONGSIDE fail% (no composite penalized metric — metric/criteria redesign
  is r2 §7 candidate 3, user-owned, out of scope here).
- Named claims are judged ONLY by the operational definitions above; all
  other numbers are descriptive.

## Decision mapping (pre-registered)

- H-A fully refuted -> the insurance claim narrows to "no-prior-knowledge +
  low-N"; w_ctrl candidate loses its motivation (cheap reliability exists
  without reward surgery); pool extension loses most value; criteria-redesign
  discussion refocuses on the N-axis. 
- H-A survives -> paper story hardened with the measured frontier; criteria
  redesign (fail-rate/CVaR primary) becomes well-motivated; w_ctrl and pool
  extension remain optional and unblocked.
- H-C fails -> R1 is the only insurance-grade checkpoint; C1's winner status
  carries an asterisk; affects which checkpoint the paper leads with.

## Constraints (inherited, binding)

CPU-only (no GPU use at all); total concurrent pool workers <= 45 (machine
cap 64 threads); no training; no reward/env/repo source changes; new
artifacts under this study dir only; heavy artifacts (npz) NOT committed;
commit only on explicit user approval, no AI attribution; report in Korean,
code/internal docs in English; background/chained commands use absolute
paths; RUNLOG wall-clock claims verified against artifact mtimes.
