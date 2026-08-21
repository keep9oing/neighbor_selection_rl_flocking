# Canonical checkpoints — provenance

Copies of the three policies of record (2026-08 line), mirrored from `test_results/`
with the same relative layout so eval tooling (which reads `params.json` from the
checkpoint's parent directory) works unchanged. The checkpoint binaries are **not
git-tracked** (only this file is); on a fresh clone this directory is empty — copy
from the machine that has `test_results/`, or reproduce with the commands below.

| policy | path (under `checkpoints/`) | role |
|---|---|---|
| **π_E** | `c2C1_ft40_lmix_260808/manual/checkpoint_000080` | Headline efficiency policy (C1): weights-only fine-tune from A it40 over L-mix. Closes the J gap (−19/−1/+31 vs specialist). Confirmed in acs-confirm; sole registered miss: N10 failure 2.2% (n.s.), N-claim demoted per pre-registered mapping. |
| **π_R** | `c2R1_lmix_legacy_260807/GradLoggingPPO_…_cc4e6_…/checkpoint_000110` | Reliability/insurance policy (R1): scratch L-mix training, legacy obs. Failures 1/1000 vs k12 85/1000 (S1); insurance confirmed on every axis in acs-confirm (L 2/1500, N 1/1000). |
| **A it40** | `c2A_bernoulli_260806/GradLoggingPPO_…_d5510_…/checkpoint_000040` | L=250 specialist (variant A, bernoulli head) at iter 40 — the fine-tune init that produced π_E. Kept for lineage/reproduction. |

## Reproduction (repo root, pinned stack, GPU)

- A / it40: `python train_c2_a.py` — NOTE: its `RUN_NAME` constant currently points at
  the ablation run (`c2A2_…`); set it back to `c2A_bernoulli_260806`-style to
  reproduce the canonical run. Study: `studies/acs-c2-train/`.
- π_R: `python train_robust.py --variant legacy` (120 it; ckpt of record = it110).
  Study: `studies/acs-robust-train/`.
- π_E: `python train_robust2.py --run-name <name> --init-ckpt <A it40 path>`
  (weights-only init + init-fidelity gate, flat lr 1e-4, 80 it, L pool {125,250,500}).
  Study: `studies/acs-robust-r2/`.

Original artifacts remain in `test_results/<run>/…` (full checkpoint series, tune
logs, `result.json`). Training was seeded but full bit-reproducibility across
hardware is not guaranteed — for exact numbers use these checkpoint copies.

## Evaluation

Criterion-of-record offline eval: `studies/acs-confirm/src/` (`eval_c2_r3.py` +
`run_knn_refs3.py` + `confirm_judge.py`); results: `studies/acs-confirm/REPORT_KO.md`.
Earlier studies' `eval_c2.py` copies are outdated (acs-c2-train's lacks the
`obs_position_scale` handling) — always use the acs-confirm copies.
`evaluate_checkpoint.py` is for the centralized-obs model variant (see CLAUDE.md)
and is not the judge for these ego-centric policies.
