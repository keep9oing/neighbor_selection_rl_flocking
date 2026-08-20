# RUNLOG — acs-robust-r3-stress

Append-only, newest at bottom. Wall-clock claims verified against artifact
mtimes at close.

## 2026-08-08 ~23:00-23:38 — grounding, pre-registration, gates

- User decision (AskUserQuestion): start ONLY r2 §7 candidate #1 (baseline
  stress-test). w_ctrl pilot / pool extension / criteria redesign NOT started.
- Inventory (Explore subagent): 500-seed S1 set = seeds 1000-1499 contiguous,
  all arms; 32-seed = 1000-1031. frontier_L.csv lives in acs-c2-train (N=20,
  k {8,10,12,19} x L {125,250,500}). r2 knnref adds L75, N10 {6,8,9}, N40
  {12,16,20,24,28,39}, k12@L250/500 x500. ABSENT everywhere: k13/k14@N20,
  k7@N10, k26@N40, ANY k-NN ref at L375/L750, ANY C1 500-seed data.
  Runtimes (mtime-derived): k-NN 500-ep lane ~4 min @15w; policy 500-ep lane
  ~60-70 min @15w; 32-ep k-NN cell ~0.5 min.
- PROBLEM.md + PLAN.md pre-registered BEFORE data collection: H-A insurance
  uniqueness (refutation = some k in {13,14,16,20} with fail<=1% at both
  L250/L500 AND policy typically worse, Wilcoxon p<.05), H-B ratio rule
  k=round(0.65N) (k7/k13/k26 cells), H-C C1 in-pool insurance (<=1% pooled +
  McNemar < k12 per L). Arms/tiers T1-T4, stats conventions, decision
  mapping — all in PROBLEM.md.
- Scripts: eval_c2.py/run_knn_refs.py copied to src/*_r3.py|*3.py, ONLY
  STUDY constant (+ internal import) changed; new data isolated under this
  study; r2/r1/predecessor read-only.
- Gate G1 (k-NN copy fidelity): k12@L250 seeds 1000-1007 -> first comparison
  FALSELY failed on 6 arrays; root cause = np.array_equal without equal_nan
  on NaN-sentinel entries (1-2 NaN per array, identical positions/patterns
  both sides, real value diffs = 0). Re-judged with equal_nan=True:
  **PASS 8/8 bit-exact** (all arrays + meta).
- Gate G2 (policy copy fidelity): C1_i80@L250 seeds 1000-1003 vs r2
  C1_i80_L250_s32 rows: dt=0, dJ=0.000000 on 4/4 -> **PASS**.

## 2026-08-08 23:38 — T1/T3 + T2 wave-1 LAUNCHED

- Job A (chain, 15w then 10w): k13,14,16,20 @ L500 x500 -> @ L250 x500 ->
  k12,13,14,16,20 @ L125 x500 -> T3 cells (k12,16,20 @ L375, L750 x32;
  k7 @ N10 L177 x32; k26 @ N40 L354 x32). Expected ~70 min.
- Job B: eval_c2_r3 C1_i80 @ L250, seeds 1000-1499, 15w. Expected ~65-70 min.
- Job C: eval_c2_r3 C1_i80 @ L500, seeds 1000-1499, 15w. Expected ~65-70 min.
- Peak workers 45 (cap 45, machine 72 cores, baseline load ~3).
- T4 (C1@L125, R1@L125 x500) launches after B/C complete.

## 2026-08-08 23:40 — Job A aborted at launch; Amendment A1; relaunch

- Job A died instantly on the runner's own guard: assert k=20 >= N=20.
  Registered arm k20@N=20 was impossible (k <= N-1 = 19 = FC) — design
  error caught pre-data. PROBLEM.md Amendment A1: T1 k20 -> k18; T3
  extrapolation refs k20 -> k19 (FC comparator at L375/L750). No data
  existed for any affected arm. Jobs B/C unaffected, still running.
- Job A relaunched with corrected arms (same chain structure).

## 2026-08-09 00:39-00:5x — T2 wave-1 done; T4 launched

- Job B C1_i80@L250 x500: **499/500** (1 fail: seed 1090), succ-J med 158.6.
  Wall ~60 min (23:39 -> 00:39).
- Job C C1_i80@L500 x500: **498/500** (2 fails: 1130, 1333), succ-J med
  205.5 (n=32 grid had said 196.3 — n=500 shifts it up ~9).
- H-C interim (L250+L500): C1 pooled 3/1000 = 0.3% vs k12 85/1000.
- Jobs D/E launched: C1_i80@L125 x500, R1_i110@L125 x500 (15w each; Job A
  still finishing L125 k-NN lanes + T3 cells).
- Job A interim: k13@L125 479/500 (4.2% fail), k14@L125 483/500 (3.4%) —
  the ratio rule's own k carries a NON-trivial failure tail at L125 too.

## 2026-08-09 ~00:45-01:00 — T1 headline: larger k does NOT buy reliability

- Fail rates (n=500 each): L500 k13 7.2% / k14 10.2% / k16 15.0% / k18 9.2%
  (k12 was 9.0%); L250 k13 5.0% / k14 7.8% / k16 10.2% / k18 13.8% (k12
  8.0%); L125 k12 4.0% / k13 4.2% / k14 3.4%. NO fixed-k arm is anywhere
  near the pre-registered 1% bar -> the "bigger k = cheap reliability"
  threat is refuted by measurement; H-A's reliability leg cannot be met by
  any registered arm. succ-J medians: k13 161.6/167.3 (L250/L500) — the
  rule's k13 IS frontier-grade on J, with the same failure class.
- Failure-mode check (6 failed seeds, k16@L500 + k18@L250): all have
  phi=1.0 (perfect alignment), stable s_ent, final n_comp_r0=2 — parallel
  sub-flock lock-in (merge failure), NOT misalignment. Suspected mechanism:
  k-NN's asymmetric farthest-neighbor abandonment (group drops the
  straggler, straggler chases at equal speed forever). Judge artifact ruled
  out (same offline C2 judge, G1 bit-exact).
- Amendment A2: FC(k19) @ 3L x 500 added (32-seed "FC is safe" now needs
  n=500 proof given the k18 cliff would be one edge away). Launches after
  Job A frees workers.

## 2026-08-09 ~01:25 — Job A complete (T1+T3); H-A effectively decided; F launched

- T1 final: k18@L125 451/500 (9.8%). Incremental stress_stats run:
  H-A criterion (i) met by NO arm (min fail = k13@L250 5.0%);
  **H-A SURVIVES at all four policy x L cells**. Sanity section 0
  reproduced r2's k12 460/455. Bonus: at L250 C1 Pareto-dominates
  k13-k18 (paired medians -1.7..-17.7 AND 1 vs 25-69 fails).
- T3 extrapolation refs (NEW — no k-NN refs existed at 375/750):
  k12@L375 26/32 J_med 179.1 vs C1 32/32 178.3 -> the fixed-k frontier has
  an INTERPOLATION reliability hole; C1's "specialist-grade" L375 actually
  beats k12 on both axes. L750: k12 26/32 J 200.1, k16 28/32, FC 32/32
  J 304 — C1's 29/32 there is BETTER than fixed-k's 26-28/32; R1 unique at
  32/32. k7@N10: 30/32 J 186.3 (C1 32/32 J 145 — b1 material).
  k26@N40: 30/32 J 147.9 (bracket k24/k28 consistent).
- Job F (A2) launched: FC k19 @ L500 -> L250 -> L125, 500 seeds, 15w.

## 2026-08-09 ~01:45 — Job F done: FC is truly failure-free; cliff at one edge

- FC(k19) n=500: **500/500 at ALL of L125/L250/L500**; succ-J med 236.7 /
  199.9 / 226.4. The merge-failure tail exists for every k <= 18 (3-15%)
  and vanishes exactly at FC — one-edge cliff k18->k19 confirmed at n=500.
- Frontier reading: the only reliable fixed topology is FC, whose typical-J
  premium over k12 (+40..+90) EXCEEDS the policies' premium; C1 beats FC on
  typical J at L250 (158.6 vs 199.9) and L500 (205.5 vs 226.4) with 1-2
  fails/500. At L500 FC's J_med 226.4 looks better than R1's 236.4.
  [CORRECTED 2026-08-09 ~06:4x: that was an UNPAIRED median comparison —
  the exact error the r2 audit flagged. Paired (n=499): med +14.9,
  Wilcoxon n.s. (p=0.75), sign test marginal (271/499 R1 worse, p=0.06)
  -> no significant typical-J difference at L500, weak trend favoring FC.
  C1 vs FC is a genuine paired win at all three L: med -91.8 / -25.5 /
  -9.8, w_p 2.6e-58 / 3.6e-19 / 3e-05 (sign_p at L500 0.12 — the L500
  edge is tail-driven, typical parity).]
- Awaiting D/E (C1/R1 @ L125 x500, ~55% done) for the final full-table run.

## 2026-08-09 01:35-02:00 — D/E done; FINAL verdicts; STUDY CLOSED

- Job D C1_i80@L125 x500: 480/500 (4.0%), succ-J med 139.3 -> H-C leg
  FAILS (k12-grade rate, McNemar 18:18 p=1; typical-J still -7.7 BETTER,
  p=1.6e-4). Job E R1_i110@L125 x500: **500/500**, J med 185.5.
- Final stress_stats.py --census run: WARN 0, sanity 460/455 reproduced.
  **H-A SURVIVES (4/4), H-B SURVIVES (b1 med -48.4 w_p 8.7e-4 + rule-k
  unreliable), H-C NOT MET (pooled 23/1500 = 1.53%)**. Census: [19+1]
  straggler-abandonment dominates every arm's failures (k18: 164/164);
  FC cliff confirmed (k18 9-14% vs k19 0/1500). Full output ->
  data/stress_report.txt, paired rows -> data/stress_stats.csv.
- REPORT_KO.md finalized (frontier table, mechanism, H-A/B/C verdicts,
  implications for r2 §7 candidates, limitations).
- Wall-clock (artifact mtimes): launches 23:38; T1 first lane 23:45;
  B/C (C1@250/500) 00:39; Job A chain end ~00:55; F (FC x3L) 01:13;
  D 01:35; E 01:37; final analysis 01:37. Total ~2h, ~10,250 episodes,
  CPU-only, peak 45 workers, GPUs untouched.
- All background jobs (A-F) confirmed exited; no monitors were started.
  New artifacts confined to studies/acs-robust-r3-stress/ (data/ heavy,
  uncommitted per constraint).

## 2026-08-09 ~06:40-07:30 — A3 post-hoc extension (user question: "isn't
## there a better k per N and L?")  [times corrected 2026-08-11 vs mtimes;
## originally misdated ~02:05-02:55]

- Jobs G/H/I (35 workers, ~50 min, ~11,900 fresh episodes — 12,000 npz
  incl. 64 cached from T3): N=20 k in
  {8,9,10,11,15,17} x 3L; N=10 k in {6,7,8}; N=40 k in {24,26,28} — all
  n=500. N=20 sweep is now CONTIGUOUS k=8..19 at n=500.
- Result: **every k from 8 to 18 fails 2.8-15.0%** at N=20; per-condition
  minima 2.8% (k11@L125) / 5.0% (k13@L250) / 7.2% (k10,k13@L500). FC cliff
  now verified with no gaps (k18 9.2-13.8% -> k19 0.0%).
- N-universality: N=10 k6/k7/k8 = 4.2/6.8/8.2%; N=40 k24/k26/k28 =
  3.6/5.6/3.6%. The straggler-abandonment tail is not an N=20 artifact.
- Efficiency-optimal k DOES drift with L (k10@L125 142.2, k12@L250 160.3,
  k14/k15@L500 161.3/161.2 — effective tie) — the user's intuition holds on
  the J axis — but the tuning gain is ~3% and buys ZERO reliability;
  J-optimal and fail-minimal k differ at L500 (k14/k15 vs k10/k13).
- H-A is UNCHANGED and strengthened: no arm, registered or post-hoc, comes
  within 2.8x of the 1% bar. Recorded as Amendment A3 (descriptive; the
  pre-registered verdicts were computed and reported before these lanes).
- Also corrected in this pass: the earlier RUNLOG line claiming FC beats R1
  at L500 was an unpaired-median error (paired: n.s., p=0.75).
- Jobs G/H/I confirmed exited. Turn-energy decomposition and paired
  policy-vs-FC stats added to the report (§4.8, §1 table unchanged).

## 2026-08-11 — user-requested VERIFICATION of the post-question edits

- Independently recomputed every number quoted in the A3-era edits from the
  raw summary CSVs. REPRODUCED EXACTLY: all 36 sweep cells (fail%, J_med),
  N-axis (N10 4.2/6.8/8.2, N40 3.6/5.6/3.6), per-condition fail minima,
  turn-energy medians, all paired policy-vs-FC/k12 stats (incl. R1-FC L500
  med +14.9 w_p 0.75). Verdicts H-A/H-B/H-C unaffected.
- ERRORS found and fixed: (1) A3 episode count ~10,500 -> ~11,900 fresh
  (12,000 npz - 64 cached); (2) A3 timestamps were ~4.5 h early
  (02:05-02:55 -> 06:40-07:30 by artifact mtimes; PROBLEM A3 header too);
  (3) L500 J-optimum: k15 161.16 < k14 161.27 (effective tie — was stated
  as k14 alone; table bold moved); (4) REPORT §4.1 "k10-15 flat (160+-7)"
  was false at L500 (k10 176.5, k11 172.4) — replaced with per-L statement;
  also R1 range 0.0-0.2% (was "0.0-0.4%"); (5) §4.8 "oracle 7-36x worse
  than policies" mixed conditions and hid the L125 exception — replaced:
  oracle k11@L125 (14 fails) vs C1 (20) is a statistical tie (McNemar
  p=0.39, J med -6.2 favors C1), R1 beats k11 outright (0 vs 14, p=1.2e-4);
  (6) "R1 and FC STATISTICALLY TIED" wording refined (Wilcoxon n.s.,
  sign test p=0.06 weak FC trend; C1-FC L500 edge is tail-driven,
  sign_p=0.12). New paired cells computed: C1/R1 vs k11@L125.
