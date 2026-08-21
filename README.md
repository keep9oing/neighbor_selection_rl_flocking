# neighbor_selection_rl_flocking

RL for **which neighbors to listen to** in a flocking swarm: a PPO policy outputs a
binary adjacency matrix per step (not motion commands); the low-level ACS/Vicsek
controller inside the env turns the selected-neighbor subgraph into velocity updates.

## Current results (2026-08)

The confirmed research line lives in `studies/` — a chronological chain ending at
**`studies/acs-confirm/`** (pre-registered fresh-seed confirmation, 35/37 PASS).
Start with `studies/acs-confirm/REPORT_KO.md` (Korean). Policies of record:

- **π_E** — efficiency policy (`c2C1` fine-tune, it80)
- **π_R** — reliability/"insurance" policy (`c2R1` scratch L-mix, it110)

Checkpoint binaries are **not in git**: see `checkpoints/PROVENANCE.md` for exact
paths, provenance, and reproduction commands (copies live on the lab machine).

## Setup

Pinned stack — do not upgrade any of these without a coordinated bump (details in
`CLAUDE.md`): Python 3.9, `torch==1.12.1+cu113`, `ray==2.1.0` (RLlib),
`gym==0.23.1` (not Gymnasium), `pydantic==1.10.13` (v1 API), `numpy==1.23.4`.

- pip: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113`
- or Docker: see `docker/` (same pins baked in; mounts the repo at `/workspace`)

Many studies/figures scripts hardcode `/workspace/...` paths — keep the repo at
`/workspace` (the docker setup does) or edit their `STUDY`/`REPO` constants.

## Layout

| path | what |
|---|---|
| `envs/env.py` | the entire simulator (`NeighborSelectionFlockingEnv`) |
| `models/` | PPO models — `ppo.py` (ego-centric, current line), `ppo_centralized.py` (legacy variant) |
| `baselines.py` | heuristic baselines + `create_baseline` factory |
| `callbacks.py`, `grad_logging_ppo.py` | shared RLlib callbacks / PPO subclass used by every trainer |
| `train*.py` | current-line trainers (see below) |
| `evaluate_checkpoint.py` | MC eval harness (centralized-variant checkpoints) |
| `test_baselines.py` | repo-wide smoke/regression gate — keep it green |
| `studies/` | research record; per study: `PROBLEM` / `PLAN` / `RUNLOG` / `REPORT_KO` |
| `figures/` | paper-figure pipeline (`figures/README.md`) |
| `checkpoints/` | canonical policy copies (untracked; `PROVENANCE.md`) |
| `docs/` | baseline catalog + guide for adding a heuristic |
| `legacy/` | dormant Jan–May 2026 experiment scripts + frozen era log (`legacy/HANDOFF.md`) |

Root trainers: `train.py` (documented baseline entry point; not the winning recipe),
`train_c2_a.py` / `train_c2_b.py` (C2 arms A/B), `train_robust.py` (→ π_R),
`train_robust2.py` (weights-only fine-tune, → π_E), `train_hardtopk.py`
(Phase-14 ancestor of the C2 line).

## Evaluating / extending

- Criterion-of-record evaluation: `studies/acs-confirm/src/` (`eval_c2_r3.py`,
  `run_knn_refs3.py`, `confirm_judge.py`). Earlier studies carry older copies of
  `eval_c2.py` — **always use the acs-confirm copies**.
- Adding a heuristic baseline: `docs/FOR_HEURISTIC_DEVELOPERS.md`, then run
  `python test_baselines.py`.
