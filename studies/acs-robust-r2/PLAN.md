# PLAN.md — acs-robust-r2

> Statuses `[ ]`/`[~]`/`[x]`/`[!]`. Log executions in RUNLOG.md. Anchors
> verified 2026-08-07; re-grep before patching.

## Phase 0 — Grounding  [x]
- [ ] Read PROBLEM.md here; round-1 REPORT_KO.md (at least §1/§4.2/§4.4) and
      RUNLOG entries of 2026-08-07; skim train_robust.py.
- [ ] `nvidia-smi` (cuda:1/3 must be free), `uptime`, `cat /tmp/ctx`.
- [ ] Verify checkpoints exist: A it40/it60, R1 it110 (paths in PROBLEM.md).

## Phase 1 — Weights-only init mechanism + gates  [x]
Extend `train_robust.py` (or clone to `train_robust2.py` — keep the old file
working): new args `--init-ckpt <checkpoint_dir>` and `--run-name`, plus
`--lr-flat 1e-4`.
- Mechanism (weights only, NO tune restore — restore would drag in the old
  iteration/timestep counters and lr-schedule position):
  ```python
  algo = GradLoggingPPO(config=config)
  import pickle
  with open(os.path.join(init_ckpt, "policies/default_policy/policy_state.pkl"), "rb") as f:
      state = pickle.load(f)
  algo.get_policy().set_weights(state["weights"])
  algo.workers.sync_weights()          # push to rollout workers
  # then the usual training loop / tune.run? -> tune.run cannot take a
  # pre-built algo; use the manual loop: for i in range(80): algo.train()
  # + algo.save() every 10 iters into /workspace/test_results/<run_name>/
  ```
  NOTE tune.run() constructs its own trainer, so init-from-ckpt uses a MANUAL
  train loop with explicit algo.save() checkpointing (mirror checkpoint_freq
  10 + final). Keep result.json-equivalent metric lines by appending
  `json.dumps(result)` per iter to a file the monitor can read, or simply log
  the metric subset (len/succ/J/eval) to logs/ — monitor_runs.py must be
  pointed at whatever this produces (adapt it; it currently globs
  GradLogging*/result.json).
- Config deltas vs round-1 R1: `lr: 1e-4` FLAT (no schedule) for BOTH runs —
  pre-registered rationale: protect the specialist init from early
  destruction; identical between F1/C1 so init remains the only variable.
  Everything else IDENTICAL to R1 (pool {125,250,500}, cap 2000, entropy
  1e-3 flat, batch 16000, eval every 10 at fixed L=250, 16 eps).
- Iters: 80 per run (~5.5-6 h at ~250-270 s/iter).
- Gates:
  - [ ] Init-fidelity gate: after set_weights, BEFORE training, offline
        4-seed argmax eval at L=250 must reproduce the source policy
        (F1 init ~ A it60 grade: 4/4 success, J ~140-180; C1 init ~ A it40
        grade: 3-4/4, J ~140-160). A random-init policy would give dense
        p~0.5 behavior (J ~350) — unmistakable difference.
  - [ ] 2-iter CPU smoke per variant (--smoke path must still work with
        --init-ckpt): gnorm nonzero, entropy ~262, no crash.
  - [ ] test_baselines.py green if any env/repo file was touched (expected:
        none — this phase should touch only the train script).

## Phase 2 — Launch + monitor  [x]
- [ ] F1: `python /workspace/train_robust2.py --variant legacy --init-ckpt
      <A_it60_dir> --run-name c2F1_ft60_lmix_<date>` on cuda:1.
      C1: same with A it40 -> `c2C1_ft40_lmix_<date>` on cuda:3.
      ABSOLUTE paths, separate single-command launches (cwd trap killed one
      launch in round 1 — see round-1 RUNLOG ~01:25).
- [ ] Adapt + start monitor (exits on first process exit / error signature).
- [ ] While training: check `cat /tmp/ctx` after each notification; keep CPU
      sweeps (Phase 3) running — they are independent of the GPU runs.

## Phase 3 — CPU evidence sweeps (run DURING training)  [x]
All CPU-only; keep total workers <= ~40 while training runs (2 ray drivers
use ~16 CPUs each headroom; sweeps at 10 workers per job, 2-3 jobs at a time).
- [ ] 3a Small-L probe (L=75): k-NN frontier k={8,10,12,19} x 32 seeds via
      common.run_episode (new refs; frontier_L.csv does not cover 75), then
      eval R1 it110 + A it60 (`eval_c2.py --bound 75`). Deliverable: extend
      the regime table; check the user hypothesis (do current policies waste
      control in compressed starts? compare turn-energy share).
- [ ] 3b Big-n reliability (S1): seeds 1000-1499 (500), L={250,500}:
      k12 refs (run_episode) + R1 it110 rollouts; A it60 at L=250 optional
      third arm. ~2000-2500 rollouts total, ~2.5-3.5 h at 30 workers —
      schedule as the long background job EARLY. Deliverable: failure counts
      + two-sided Fisher + Wilson CIs (S1 verdict). NOTE eval_c2.py caches
      per-seed npz — re-runs are cheap; extend --seeds parsing if needed
      (currently "a-b" ranges are fine).
- [ ] 3c N-transfer smoke THEN sweep (S2): first 2 rollouts each at
      (N=10,L=177) and (N=40,L=354) to prove the checkpoint loads and acts
      at different N (transformer is N-agnostic; num_agents_pool=[N] via
      build_config n_agents param; watch d_subobs/global_stats shapes).
      Then 32 seeds x {R1 it110, round-2 winner} + k-NN frontier at those
      (N,L). Deliverable: S2 verdict + descriptive J table.

## Phase 4 — Selection + grid judgment (round 2)  [x]
- [ ] Screening: online eval traces -> offline 16-seed screens (L=250+500)
      on 3-4 late checkpoints per run (round-1 recipe).
- [ ] ONE checkpoint per run -> full grid 32 seeds x L={125,250,500}
      (+375/750 probes for the winner). Judge with grid_judge.py (primary
      unchanged). Key comparisons: F1 vs C1 vs R1 curves; vs specialist
      at 125-375 (did fine-tuning preserve low-L efficiency?).
- [ ] If BOTH runs still miss primary on J: do NOT iterate further this
      round; document the init-strategy answer honestly and stop (the
      remaining lever c2_w_ctrl is a reward change = user decision).

## Phase 5 — Wrap-up  [x]
- [ ] REPORT_KO.md (round-2 + S1/S2/small-L integrated); RUNLOG complete;
      pointer line in round-1 REPORT_KO.
- [ ] Korean report; commit discussion (explicit approval only; kickoff
      files stay uncommitted and get deleted on user instruction, as in
      previous rounds).
- [ ] Handoff hygiene at ANY exit (incl. mid-round at ctx 319K): stop/kill
      monitor + all background tasks FIRST, write KICKOFF_NEXT, then final
      message. No work may continue after the handoff message.

## Design decisions already made (do not relitigate without user input)

| Topic | Decision | Source |
|---|---|---|
| Round-2 runs | F1 = ft from A it60; C1 = ft from A it40 (curriculum-equivalent); one GPU each | user 2026-08-07 |
| Training pool | UNCHANGED {125,250,500} (single-variable vs R1) | user 2026-08-07 |
| Small-L | eval probe L=75 only this round; pool extension = later round | user 2026-08-07 |
| Paper sweeps | big-n reliability (S1) + N-axis (S2) this round; distillation DEFERRED to paper stage | user 2026-08-07 |
| Fine-tune lr | flat 1e-4 both runs (identical; init = only variable) | this plan, pre-registered |
| Reward weights | UNTOUCHED (c2_w_ctrl change = separate user decision) | inherited |
| GPUs | cuda:1 (F1) / cuda:3 (C1) | user |
| Session stop | ctx 319K via `cat /tmp/ctx` -> clean boundary -> kill monitors/tasks -> handoff | user 2026-08-07 |
| Primary bar | unchanged round-1 criteria | inherited pre-registration |
