# legacy/ — dormant experiment scripts (Jan–May 2026)

Historical one-off and superseded scripts from the pre-`studies/` era (Phases 1–14),
moved out of the repo root on 2026-08-21. Nothing in the current line references them;
they are kept for provenance, not for use.

- `HANDOFF.md` — the frozen Phase 1–14 research log (see its banner). Start there to
  understand what each script family was for. `REPORT_hardtopk_distaux_KO.md` is the
  Phase-14 writeup (untracked by choice).
- `train_*.py`, `eval_*.py`, `check_*.py`, `diagnostic_*.py`, `run_eval.sh` — one-off
  trainers/monitors from the May sweep era. Their runs live under `test_results/`.
- `verify_*.py` — per-baseline sanity checks, functionally subsumed by root
  `test_baselines.py` (the maintained regression gate).

**These scripts do not run from this directory as-is**: they import root modules
(`from callbacks import …`, `from envs.env import …`) relative to the repo root.
If you ever need to run one, execute it from the repo root, e.g.
`python legacy/train_topk.py` will still fail on the bare `callbacks` import —
copy it back to the root instead. Some also hardcode dead run names/paths.
