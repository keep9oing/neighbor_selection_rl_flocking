# RUNLOG.md — acs-c2-train

> **Purpose of this file**: Append-only log of what was ACTUALLY executed in this
> study (commands, configs, numbers, deviations from PLAN). Newest entries at the
> bottom. Keep entries dense and citable — a fresh session reconstructs state
> from here.

## 2026-08-06 — Study created (planning session, no implementation yet)

- Created by the criterion/metric-design session (predecessor:
  `studies/acs-conv-knn/`, see its RUNLOG 2026-08-06 entries for: C2 window
  sweep, evaluation-metric decisions, architecture saturation diagnosis).
- User decisions recorded in PROBLEM.md/PLAN.md: variants A+B, B un-rigged;
  repo edits allowed (additive/flag-gated); success bonus small (10);
  dist_aux annealed carefully (1.0->0.2/400k, never 0); GPUs cuda:1 (A) and
  cuda:3 (B); implementation + training happen in the NEXT session using
  KICKOFF_PROMPT_KO.md.
- Nothing implemented, nothing trained, nothing committed yet.
