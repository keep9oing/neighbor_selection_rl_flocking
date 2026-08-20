# RUNLOG.md — acs-robust-r2

> Append-only log of what was ACTUALLY executed. Newest at the bottom.

## 2026-08-07 — Study created (planning by the round-1 closing session)

- Created at the round-1 (acs-robust-train) close-out after user discussion.
  All round-2 design decisions are user-approved and frozen in PROBLEM.md +
  PLAN.md ("Design decisions already made" table). Round-1 tree is committed
  (code 150a141, study 0425a5e); this study dir + KICKOFF_PROMPT_KO.md are
  intentionally uncommitted (session artifacts, previous-round convention).
- Nothing implemented, nothing trained yet. Execution starts in the NEXT
  session via KICKOFF_PROMPT_KO.md.

## 2026-08-07 ~23:30-00:05 — Phase 0/1 done (gates ALL GREEN); Phase 2+3 LAUNCHED

- Phase 0: cuda:1/3 free (0%), ctx 52K, ckpts verified (A it40/it60, R1 it110).
- train_robust2.py (repo root, NEW file; train_robust.py untouched): weights-only
  init via policy_state.pkl -> set_weights + workers.sync_weights() (+ eval
  workers), fresh optimizer/iteration; manual train loop (tune.run can't take a
  pre-built algo); UnifiedLogger at test_results/<run>/manual/ so result.json /
  params.json / progress.csv appear exactly like tune runs (monitor + eval_c2
  compatible). --lr-flat 1e-4 (lr_schedule None), --iters 80, --gpu {1,3},
  --init-check (build+load+save ckpt0, CPU), --smoke, --resume. Post-set_weights
  assert compares the largest tensor vs the pickle.
- r2 src: eval_c2.py copied from round 1 (STUDY -> r2, NEW --n-agents for the
  N-axis), grid_judge.py copied (STUDY -> r2), monitor_runs2.py (reads
  manual/result.json), run_knn_refs.py (fresh k-NN refs via common.run_episode
  + same offline C2 judge; rows mergeable with frontier_L.csv), s1_analyze.py
  (Fisher + Wilson, S1 verdict).
- Smokes (2-iter CPU, both inits): F1 gnorm 0.74/1.44, C1 gnorm 0.35/1.20,
  entropy ~262 all — NOTE this is NOT a random-init signature: the A-line
  itself trains at entropy ~262 throughout (it1 263.2 -> it60 262.3, bernoulli
  p stays ~0.5; sat_p_dev 0.03 matches it60's 0.031). No crash. Smoke run dirs
  (r2smoke_*) deleted after.
- INIT-FIDELITY GATE (the decisive check): checkpoint_000000 saved via
  --init-check, then offline argmax eval seeds 1000-1003 at L=250:
  * F1 (A it60 init): 4/4, per-seed t_fire/J EXACTLY equal to A_i60_s32
    reference (dt=0, dJ=0.0 on all 4) — bit-exact reproduction through the
    full save->load->eval pipeline.
  * C1 (A it40 init): 4/4, EXACT match vs A_i40_s32 likewise.
- test_baselines.py not required: no env/repo file touched (new files only).
- Phase 2 LAUNCH ~00:00: F1 pid 315848 (cuda:1), C1 pid 315889 (cuda:3);
  lr flat 1e-4, 80 iters, pool {125,250,500}, BASE_SEED 42 (same as R1 —
  init ckpt is the ONLY variable). monitor_runs2.py attached (pids above,
  exits on first death/error signature).
- Phase 3 LAUNCH (CPU, parallel with training; ~30 sweep workers total):
  * S1 lane A: k12 refs L=250 seeds 1032-1499 (15w) -> R1 it110 L=250
    seeds 1000-1499 (32 npz reused from round-1 grid, renamed labels —
    protocol-identical: steps 6000, N=20, same ckpt; verified via npz meta).
  * S1 lane B: k12 refs L=500 -> R1 it110 L=500 -> A60 L=250 (all 500 seeds;
    A60 reuses 32 npz from c2-train A_i60_s32).
  * S2 smoke: R1 it110 2 rollouts at (N=10,L=177) + (N=40,L=354) — labels
    sized for later 32-seed extension (cache).

## 2026-08-08 ~00:05 — training healthy; S2 smoke PASS; k12 big-n refs DONE (big finding)

- Trainings it1 (both 278 s/iter, lr flat 1e-4 confirmed in learner_stats):
  F1 len 639 succ 1.0 J 369; C1 len 620 succ 1.0 J 351. NO early argmax
  degeneracy (train succ 1.0 from it1) — the specialist init already behaves
  under L-mix, unlike scratch R1 (degenerate it10-30). ETA ~06:15-06:45.
- S2 smoke PASS: R1 it110 acts/converges at both probe conditions without
  retraining — (N=10,L=177): 2/2, t_conv 442, J 158.3, deg_ss 7.2;
  (N=40,L=354): 2/2, t_conv 552, J 180.3, deg_ss 37.7. Full 32-seed sweeps
  deferred until S1 lanes free CPU.
- k12 big-n refs COMPLETE (seeds 1032-1499, 468 per L, ~46 s/ep at 30 w):
  * L=250: 429/468 success (39 FAILURES, 8.3%), t_conv med 521, J med 160.4
  * L=500: 423/468 success (45 FAILURES, 9.6%), t_conv med 569, J med 166.3
  * The original 32-seed set (1/32, 0/32 failures) badly underestimated the
    k12 failure tail; J medians match the 32-seed values (160.0/165.6) so
    only the tail was missed, not the center.
  * PROTOCOL VALIDATION before believing it: reran overlap seeds 1000-1007
    at both L through run_knn_refs.py — per-seed t_fire/J EXACTLY equal to
    frontier_L.csv (16/16, dt=0, dJ=0.0). Same pipeline, same judge; the
    fresh failure counts are real and paired-comparable.
- Pooled k12 failures on 1000-1499 so far: L250 40/500, L500 45/500. S1
  verdict now hinges on R1's own big-n failure count (lanes running).

## 2026-08-08 ~00:35 — Phase 3a L=75 probe COMPLETE (compressed regime mapped)

- k-NN frontier at L=75 (32 seeds): k8 28/32-160.1, k10 32/32-168.2,
  k12 30/32-182.9, k19/FC 32/32-248.6. The compressed regime is HARDER for
  fixed-k: k8/k12 drop successes and the frontier J rises (k12 183 vs 160 at
  L=250); best-k shifts to k10.
- R1 it110: 32/32, t_conv 576, J 207.7 (+39.5 vs k10-best, +24.8 vs k12) —
  success-perfect again; with round-1 grid this makes R1 192/192 across
  L in {75,125,250,375,500,750} (10x scale range). rank_dev_ss 0.545 (much
  higher hold-phase non-nearest selection than at large L).
- A60 specialist: 32/32 BUT J 322.9 (t_conv 818) — WORSE THAN FC (248.6).
  The specialist collapses out-of-distribution on the compressed side too,
  mirroring its L=750 explosion (401): L-mix flatness pays on BOTH ends.
- Turn-energy-share analysis (user hypothesis: compressed starts reward
  control frugality) deferred to the report phase — npz series are on disk.

## 2026-08-08 ~01:00 — Phase 3c S2 sweeps COMPLETE (R1 arm + fresh k-NN frontier)

- k choice at new N (study decision, descriptive-only): N=10 -> {6,8,9=FC}
  (k12 impossible, k>=N); N=40 -> {12=same-k, 24=0.6N analog, 39=FC}.
- (N=10, L=177), 32 seeds: k6 31/32-176.7, k8 31/32-159.6, FC(k9) 32/32-231.1;
  R1 it110 32/32, t_conv 516, J 190.0.
- (N=40, L=354): k12 25/32-172.7 (same-k transfer LOSES reliability),
  k24 30/32-151.9, FC(k39) 32/32-241.0; R1 it110 32/32, t_conv 579, J 193.4.
- S2 pre-registered criterion (success >= 30/32 at both, no retraining):
  **MET by R1 it110 (32/32 + 32/32)**. Winner arm to be added post-Phase-4.
- Pattern: fixed-k does not transfer across N (k12: 31/31/32 at N=20 grid ->
  25/32 at N=40); the learned policy stays perfect. R1 cumulative success
  record now 256/256 (192 L-axis + 64 N-axis).

## 2026-08-08 ~01:25 — S1 COMPLETE: pre-registered reliability claim MET (p=4e-25)

- ARTIFACT FIX first: the overlap-validation call had clobbered the two k12
  summary CSVs down to 8 rows (run_knn_refs rebuilt summaries from the
  current call's seeds only). Fixed the script to rebuild from ALL npz in the
  dir; regenerated both summaries at full 500 seeds (npz were never lost;
  +24 fresh seeds 1008-1031 filled). First s1_analyze run on the clobbered
  CSVs (n=32, "NOT MET") is VOID — superseded by the full-n rerun below.
- S1 verdict (paired seeds 1000-1499, offline C2, argmax):
  * L=250: R1 it110 fail 0/500 (Wilson [0, 0.0076]) vs k12 40/500
    ([0.059, 0.107]); Fisher two-sided p = 8.1e-13.
  * L=500: R1 fail 1/500 ([0.0004, 0.0112]) vs k12 45/500; p = 5.0e-13.
  * POOLED: 1/1000 vs 85/1000, p = 3.6e-25 -> **S1 MET** (pre-registered:
    fewer failures, p<0.05 — achieved with ~12 orders to spare).
  * J at n=500 (descriptive): R1 192.9/236.4, paired dJ +26.1/+75.7 —
    consistent with the 32-seed grid; efficiency gap unchanged.
- R1's ONLY failure (seed 1127, L=500): borderline miss — n_comp 1,
  sigma_p_ss 45.7, phi_ss 0.9804 vs the 0.98 windowed-min bar (hovering at
  the alignment threshold; not a scatter/split failure).
- Third arm: A60 specialist AT ITS OWN SCALE fails 11/500 (2.2%) — beats k12
  (p=3.7e-5) but is ~11x R1's failure count. REVISES round-1's "+2 episodes,
  robustness nearly free" reading: at n=500 resolution the L-mix reliability
  gain is real and large vs both the frontier AND the specialist.
- k12 failure anatomy availability: 85 failing npz on disk for the report.

## 2026-08-08 ~04:20 — both trainings COMPLETE (80/80, ~4.6 h, no crash)

- Online eval traces (argmax L=250, 16 eps — NOISY, screening arbiter is
  offline): F1 J 151/171/171/157/169/212/196/186 (it10..80; succ dips 0.81-
  0.94 mid, 1.0 at 60-80). C1 J 160/189/166/196/166/175/156/153 (succ
  0.81-1.0). Both dramatically below R1's online J (~205-247) throughout —
  specialist init preserved efficiency through L-mix fine-tuning (train succ
  1.0 every iter, no degeneracy ever).
- Phase 4 screening LAUNCHED: offline 16-seed (1000-1015) argmax screens at
  L=250 AND L=500 for ALL checkpoints it10-80 of both runs (4 lanes x 10
  workers) — full J(iter) preservation curve, richer than the round-1
  3-ckpt recipe at trivial cost. Final judgment remains ONE ckpt per run.

## 2026-08-08 ~05:00 — 16-seed screens (both L) DONE; grid candidates picked

- F1 screens (succ250/succ500 | J250/J500): it10 14/16|151/237,
  it20 15/16|160/217, it30 15/16|152/264, it40 15/16|158/238,
  it50 16/16|158.8/209.9, it60 16/15|218/225, it70 16/16|167.8/250.2,
  it80 16/16|191.1/246.5.
- C1 screens: it10 16/13|155/246, it20 16/15|168/223, it30 15/16|166/202,
  it40 16/16|170.7/241.0, it50 16/16|172.5/208.9, it60 13/15|169/233,
  it70 16/15|160/233, it80 16/16|160.5/198.4.
- Success-first rule -> perfect-success (16/16 both L) sets:
  F1 {it50, it70, it80} -> **F1 it50** (158.8/209.9, dominant);
  C1 {it40, it50, it80} -> **C1 it80** (160.5/198.4, dominant).
- Context vs round-1 16-seed screens: R1 it110 was 212.5/255.3 — both
  round-2 candidates are ~45-57 J BELOW the round-1 generalist at BOTH L,
  and AT the k12 frontier at L=250 (k12 32-seed J_med: 160.0).
- Grid LAUNCHED: 32 seeds x L={125,250,500} for F1 it50 + C1 it80 (2 lanes).

## 2026-08-08 ~05:40 — GRID FINALS + formal judgment (grid_judge.py, criteria unchanged)

- F1 it50 (32 paired seeds, argmax, offline C2):
  * L=125: 32/32, J_med 225.0 (dJ +76.2, p=5e-7 — WORSE than k12)
  * L=250: 32/32, J_med 161.1 (dJ -3.7 n.s. — AT the frontier)
  * L=500: 31/32, J_med 202.8 (dJ +5.9 n.s.)
  * pooled dJ +26.1 (p=0.217). PRIMARY NOT MET (L125 J, L500 succ 31<32,
    L500 J). The ft-from-CONVERGED-specialist run kept L250 efficiency but
    never became cheap at L125 and dropped one L500 episode.
- C1 it80:
  * L=125: 31/32, J_med 135.9 (dJ -3.8 n.s.; J_med 19 BELOW k12's 155.0)
  * L=250: 32/32, J_med 159.0 (dJ +6.4 n.s.; J_med 1.0 below k12)
  * L=500: 32/32, J_med 196.3 (dJ mean -29.3 n.s. — long k12 tail pulls the
    paired MEAN negative while medians differ +30.7)
  * pooled dJ -9.1 (p=0.657) — FIRST negative pooled dJ in the program.
  * PRIMARY NOT MET on exactly ONE condition: L500 J_med 196.3 > 170.6
    (=k12+5). Success half fully met (31/32/32 >= 31/31/32). STRETCH not met.
- Reading: init strategy CLOSED most of the efficiency gap (round-1 +35..+90
  at every L -> C1: -19/-1/+30.7 by medians). The PLASTIC mid-training init
  (A it40 = curriculum-equivalent) beat the converged init (A it60) across
  the pool — F1 stayed L250-specialized (L125 +76), C1 spread its gains.
- Per PLAN Phase-4 stop rule: both runs miss primary on J -> NO further
  training iterations this round; document honestly. Winner = C1 it80
  (pooled dJ, near-primary). Probes launched: C1 it80 at L={375,750,75} +
  S2 winner arms (N=10,L=177)/(N=40,L=354), 32 seeds each.

## 2026-08-08 ~06:30 — winner probes + S2 winner arms + turn-energy; STUDY CLOSED

- C1 it80 probes (32 seeds): L=375 32/32-178.3 (specialist-grade, A60 181.3;
  R1 220.3); L=750 EXTRAP 29/32-282.3 (3 failures — R1 kept 32/32 there);
  L=75 32/32 but J 315.2, t_conv 962 (R1 207.7/576). The efficiency-
  robustness tradeoff MOVED, not vanished: C1 = frontier-grade inside
  125-500, fragile at both extrapolation ends; R1 = flat everywhere, costly.
- C1 it80 S2 arms: (N=10,L=177) 32/32, J 145.0 — BEATS the whole fixed-k
  frontier (best k8 159.6 at 31/32); (N=40,L=354) 32/32, J 168.1 (k12 172.7
  at 25/32, k24 151.9 at 30/32). S2 criterion met by BOTH R1 and C1.
- Turn-energy decomposition (J_time=0.1*t_fire, J_turn=J-J_time, medians):
  C1 L500 t_conv 574 vs k12 568 (+6 steps) -> remaining +30.7 J_med is
  entirely turn energy (139.7 vs 108.1). Round-1 diagnosis stands, magnitude
  halved (R1's L500 turn was 175.2). L=75: frontier turn cost FLAT (k10
  104.8 ~ L250 k12 108.1) — the compressed regime's real story is the
  reliability trap of sparse fixed-k (k8 28/32, k12 30/32), not a frugality
  premium; R1's turn is also flat (147.7), C1/A60 blow up in both time and
  turn there.
- REPORT_KO.md finalized (round-2 + S1 + S2 + L75 integrated; 2-policy
  Pareto reading: R1 = reliability champion 288/288 cumulative + 1/1000
  big-n; C1 = in-pool frontier-grade, pooled dJ -9.1 first-ever negative).
- Wall-clock summary (artifact mtimes): trainings 23:43->~04:15 (80 iters,
  ~4.6 h each, parallel); screens ~04:30-05:10; grids ~05:10-05:40; probes
  ~05:45-06:25; S1 lanes 23:45->~02:50; L75 00:10-00:35; S2 00:15-01:00.
- All monitors/background tasks confirmed dead at close (training monitor
  exited on process-exit detection ~04:20; no live tasks).

## 2026-08-08 (오전) — user-requested statistics AUDIT; several claims corrected

- User critique: mean-only paired comparisons can hide the shape (policy may
  fix k-NN's worst cases while being slightly worse on typical seeds); also
  asked for ratio-matched k (k~0.6N) at the N-axis, and flagged that a pool
  extension would move the extrapolation test to NEW L values.
- src/audit_stats.py (NEW): per comparison — Wilcoxon signed-rank, exact
  sign test, dJ quantiles, typical-set (ref J <= ref median) vs tail-set
  (ref J > ref q90) decomposition, exact McNemar on paired success. Full
  table -> data/audit_stats.csv. N=40 frontier densified with fresh k-NN
  runs k={16,20,28} x 32 seeds (k28 = best fixed-k there: 30/32, J 142.7).
- CONFIRMED user suspicion (corrections now in REPORT §4.7 + §1/§4.4/§4.6):
  * C1 pooled mean dJ -9.1 is a k12-tail artifact: paired MEDIAN +11.5,
    worse on 54/93 (n.s.); L500 typical-set +72.1, tail-set -567. C1 L500
    Wilcoxon p=0.034 WORSE on typical seeds (med +25.2, 22/32).
  * C1 L125 "-19 better" was an unpaired-median illusion: paired med -2.0.
  * Round-1 "L500 n.s. / pooled p=0.116" was a t-test-killed-by-tails
    artifact: R1 is CONSISTENTLY worse (pooled med +55.3, Wilcoxon
    p=3.8e-07, 69/94; L500 med +72.7 p=0.002). Round-1 report annotated.
  * F1 pooled med +41.0 (p=1e-06) — worse than its mean suggested.
  * S1 J at n=460/454: R1 typical premium +27.2/+68.2 (p=9e-23/5e-52),
    tail return ~-100 -> "insurance policy" structure exact. Success side
    McNemar 1.8e-12/1.3e-12. A60: J-parity with k12 (med -2.2 n.s.),
    fewer failures (11 vs 40, McNemar 5.7e-5).
  * N=40 "beats frontier" RETRACTED: vs k28 C1 is significantly worse on
    typical J (med +23.8, w_p=0.038); success edge 32vs30 n.s. (McNemar
    0.5). Same-k k12 collapse (25/32, McNemar 0.016 vs C1) stands.
  * N=10 superiority SURVIVES strengthened: C1 beats ratio-k6, best-J k8,
    and FC all with p<=0.005 (worse on only 5-7/31-32 pairs).
- Formal pre-registered verdicts UNCHANGED (bars were pre-registered as
  unpaired J_med + pooled mean t): primary still NOT MET, S1/S2 still MET.
  The audit changes the INTERPRETATION layer only (insurance framing).
- REPORT §7 rewritten: next-round candidates honestly downgraded (w_ctrl
  pilot-first; pool extension fixes 75/750 but does NOT prove extrapolation
  — new probes at L=50/~1000 would judge that; optional criteria-redesign
  around failure-rate/CVaR = user decision, new seeds required).
