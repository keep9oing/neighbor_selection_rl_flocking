# PROBLEM.md

> **Purpose of this file**: Immutable problem statement for this study. Defines WHAT we are investigating and WHY, the constraints, and the success criteria. A fresh Claude session should read this first to understand the goal without any prior conversation context. This file should NOT change as work progresses (only clarified); progress lives in PLAN.md / RUNLOG.md.

## Background

The RL project in /workspace trains a neighbor-selection policy on top of an ACS
(Augmented Cucker-Smale: CS heading alignment + inter-particle bonding/repulsion
forces, cf. Park et al.; treat the code in `envs/env.py` as ground truth) flocking
controller. An episode is currently judged "converged" when BOTH:

1. **Heading alignment** — velocity entropy (equivalent to order parameter) below a
   threshold, and
2. **Spatial distribution** — xy standard deviation ("spatial entropy") below a
   threshold,

each sustained for a fixed window.

## The problem

The two components of ACS have fundamentally different convergence properties w.r.t.
the interaction topology:

- **Heading consensus** is achievable on sparse (non-fully-connected) topologies;
  Vicsek-type results show fast alignment unless noise is near the phase transition.
- **Spatial equilibrium** depends strongly on the topology: with only a subset of
  neighbors in the bonding/repulsion sums, the equilibrium spatial distribution
  (flock spread) at steady state differs from the fully-connected case — plausibly
  monotonically in neighbor count.

This breaks fairness of the current absolute-threshold convergence criterion across
neighbor-selection strategies:

- **NN (RL policy) selection**: neighbor counts change actively and stochastically →
  equilibrium spread is ill-defined / fluctuating; absolute spatial threshold may be
  unreachable or trivially reachable depending on learned degree.
- **Disc (metric radius) model**: neighbor count varies over time with local density.
- **Fixed k-NN**: relatively stable, but the equilibrium spread still depends
  strongly on k, so a single absolute threshold favors some k over others.

## Goal

Redesign the ACS convergence criterion so that it is **fair across neighbor-selection
strategies** (NN policy, disc, k-NN, FC) and **stabilizes RL training**, while still
demanding the original intent of ACS: heading alignment at equilibrium **plus** some
form of spatial flocking/cohesion (not merely alignment).

## Requested deliverables

1. **k-NN convergence-trend study**: empirically map how flock convergence
   (equilibrium reach, equilibrium spread level, time scales) depends on k, initial
   density, and other necessary variables, using N=20 agents by default (other N
   allowed if variable-controlled). Archive experiment data; report to the user only
   the digestible essentials (user-facing report in Korean).
2. **Criterion discussion**: grounded in the ACS equations, discuss what convergence
   criterion should be used so it works for disc model, NN policy, k-NN, etc.

## Constraints

- All artifacts live under `/workspace/studies/acs-conv-knn/`.
- Heavy files (logs, data, figures) must NOT be committed (see .gitignore here).
- Never `git push` (even if asked — standing user rule; commits only with explicit
  user approval).
- May USE the existing RL env (`envs/env.py` etc.) but MUST NOT modify any existing
  repo code.
- Default N = 20 agents; changes allowed for the study if variables stay controlled.
- Compute: CPU up to 64 threads. GPU cuda:1 only if needed (this study is
  heuristic-rollout-based → CPU only, no GPU use expected).

## Language convention

- User-facing text (chat, final report) → Korean.
- Everything Claude/code reads (these files, code, comments, data schemas) → English.
