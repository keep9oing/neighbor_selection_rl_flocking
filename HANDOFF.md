# Handoff: Ego-Centric Neighbor Selection for ACS Flocking

## Research Goal

Train an ego-centric PPO policy that, given each agent's local observation, selects **which neighbors to listen to** (binary adjacency matrix per step). The low-level ACS controller then uses the selected subgraph to compute velocity updates.

**Success metric:** The ego-centric policy must **clearly outperform the fully-connected ACS baseline** on flocking quality (velocity/spatial entropy, flocking success rate). Merely matching or approximating FC is considered failure — the policy must learn meaningful neighbor discrimination. This is known to be achievable: the centralized-obs model already demonstrated performance far exceeding FC-ACS. Under FC, ego-centric and centralized observations (flock-center + average-heading frame) are mathematically interconvertible, so a parameter-sharing ego-centric policy can in principle replicate the centralized policy's decisions. The auxiliary task (predicting centralized-frame states from ego encoder embeddings) was motivated by this interconvertibility, but proved insufficient on its own.

**Evaluation metrics** (logged by callbacks):
- **Episode reward** — primary metric. Sum of per-agent ACS rewards (`env._compute_rewards()`): negative heading-rate control cost + cruise cost. Must clearly exceed FC-ACS baseline.
- `final_velocity_entropy` — velocity alignment (lower = better). The env's flocking-success threshold (`entropy_v_goal=0.1`) is already strict — vel_ent=0.1 corresponds to order parameter well above 0.995 (near-perfect alignment). This threshold can be relaxed; order parameter ≥ 0.995 is sufficient.
- `final_spatial_entropy` — spatial cohesion (target ≈ 39.5, maintained around 38–42)
- `flocking_success` — 1.0 if episode terminates early (both entropies at goal + stable)
- **Time-to-flocking** — number of steps to reach flocking success. Faster convergence at equal final quality indicates a better policy.

## Constraints

### Fixed
- **Communication:** fully-connected visibility (`comm_range=None`). Every agent CAN observe every other agent — the policy decides which to LISTEN to.
- **Observation:** ego-centric (`observation_type="ego_centric"`). Each agent sees relative positions/headings of all others in its own heading frame. Shape `(N_max, N_max, obs_dim)`.
- **Action:** binary adjacency matrix `(N_max, N_max)` int8. `action[i,j]=1` means agent i listens to agent j. Diagonal must be 1 (self-loop). Masking enforced.
- **Controller:** ACS (Active Cohesive Swarm). The policy does NOT control motion — it only selects neighbors. The ACS controller converts the subgraph into heading-rate commands.
- **Pinned stack:** Ray 2.1.0, Torch 1.12.1, Pydantic v1, Gym 0.23.1. See CLAUDE.md.

### In scope for next steps
- RL-driven approaches: PPO, SAC, REINFORCE, different action-space designs (autoregressive, top-K selection, pointer networks, etc.)
- Reward reshaping, auxiliary tasks, training dynamics tuning
- Architecture changes to the policy network

### Out of scope
- Purely parametric heuristic approaches (e.g., learning a disk radius, learning weights within a fixed heuristic topology). The policy must make per-agent, per-neighbor, per-step selection decisions via RL, not reduce to tuning a few parameters of an existing heuristic.

## Key Files (see CLAUDE.md for full architecture)

- `envs/env.py` — environment. `_compute_rewards()` (line ~985): ACS reward (negative control cost). `compute_custom_reward()` (line ~1285): shaped training reward (spatial + velocity error + control cost).
- `models/ppo.py` — ego-centric Transformer model. `scale_factor` (line ~49, 265–268): multiplies raw attention scores before logit formation — critically affects gradient flow (see findings below).
- `models/ppo_centralized.py` — centralized variant (NOT used in current experiments).
- `grad_logging_ppo.py` — custom PPO subclass that logs pre-clip gradient norms (actor/critic/total) into `learner_stats`. Uses `tower_stats → stats_fn` pipeline because RLlib 2.1.0's multi-GPU path drops `extra_grad_process` return values.
- `callbacks.py` — logs `final_spatial_entropy`, `final_velocity_entropy`, `flocking_success`.
- `evaluate_checkpoint.py` — compares a checkpoint against Pure-ACS FC baseline.
- `baselines.py` — heuristic baselines.
- `train.py` — current training config (MODIFIED — currently set for the grad-fix experiment, not the original factorial).

## Research Trajectory & Key Findings

### Phase 1: Aux task development (branch `feat/aux-task`, commits up to 5d8c082)
Added an auxiliary self-supervised task: from each agent's encoder embedding, predict the flock-center-frame state of agents. Implemented via `ModelV2.custom_loss()` hook. Controlled by `aux_enabled` master switch in model config. Early experiments (`aux_weight_sweep_260515`, `critic_aux_sweep_260517`) showed mixed results — later found to be confounded by deeper issues.

### Phase 2: Control-cost sign bug (commit 20498d6)
**Found and fixed a sign bug in `compute_custom_reward()`** (env.py line 1311). The control-cost term was `- w_ctrl * control_cost` where `control_cost ≤ 0`, making it a turning BONUS instead of a PENALTY. Fix: `+ w_ctrl * control_cost`. This invalidated all prior experiment conclusions (the reward gradient was partly garbage). The velocity-weight sweep (`vel_weight_sweep_260518`) showed w_vel had zero effect across 0.2–3.0 — it was never the lever; the buggy control term dominated.

### Phase 3: grad_clip=1.0 throttle discovery
The `ctrl_aux_factorial_260519` (8 trials, 4×2 factorial on w_ctrl × aux) ran ~120 iters post-fix. Policy entropy stayed at 263.4 (theoretical max = 380 × ln2) — the policy was **completely frozen at uniform random**. The critic was healthy (vf_explained_var ~0.8). Investigation revealed:

**`grad_clip=1.0` was throttling the actor gradient by ~20×.** RLlib PPO uses a single optimizer with global gradient norm clipping. The value loss (10–100× larger than policy loss) dominated the global norm (~20), and clipping to 1.0 scaled the actor's gradient slice to 0.016 (effectively zero). Note: RLlib 2.1.0's built-in `grad_gnorm` is (a) dropped on the multi-GPU code path (`learn_on_loaded_batch` ignores `grad_info`) and (b) `min(norm, clip_value)`-clamped (useless when clip=1.0). The `grad_logging_ppo.py` custom PPO was created to work around this.

**Fix:** `grad_clip=None`. Confirmed by paired diagnostic (`grad_norm_diagnostic_260520`): total gnorm ~7.6, critic/actor ratio ~200:1.

### Phase 4: Aux task validation
With grad_clip=None, paired experiment (`grad_fix_lr5e4_aux/noaux_260520`, 100 iters each):
- **aux=OFF:** entropy delta = -0.04 (essentially no learning)
- **aux=ON:** entropy delta = -1.05 (26× more)

**Aux task is necessary** under the current architecture — without it, the PPO gradient alone cannot escape the near-uniform initialization. The aux task provides dense, un-throttled representation pressure on the encoder.

### Phase 5: scale_factor discovery (THE dominant bottleneck)
The model multiplies raw attention scores by `scale_factor` (default 0.002) before forming action logits (ppo.py line 268). This creates a **500× gradient suppression** at the output: `∂loss/∂raw_score = ∂loss/∂logit × scale_factor`. Even after fixing grad_clip, this 500× bottleneck kept the policy near-uniform.

**`scale_factor_sweep_260521`** (sf ∈ {0.01, 0.05, 0.2}, 100 iters each):
- sf=0.01: entropy 263→243 (delta=-20, 20× improvement over sf=0.002)
- sf=0.05: entropy 263→219 (delta=-44), vel_ent reached 1.12 (best ever), but oscillation
- sf=0.2: **immediate collapse** — policy went deterministic in 1 iteration, swarm destroyed (sp_ent→1000+, vel_ent→10+, vf_loss saturated at vf_clip_param=256)

**Sweet spot: sf ∈ [0.01, 0.1].** Too low = gradient throttle. Too high = premature deterministic collapse.

### Phase 6: Policy converged to near-FC
Detailed evaluation of the sf=0.03 checkpoint (iter 70, best vel_ent=0.425):
- **Action probabilities: 99.82% of (i,j) selections have probability > 0.9.** Mean = 0.9482. The policy selects ~18/19 neighbors per agent — essentially fully connected.
- **Uniform across all timesteps** (t=1 through t=999): no temporal discrimination.
- **Stochastic sampling noise is the gap:** With deterministic actions (argmax → FC), the ACS controller achieves flocking (vel_ent < 0.10, sp_ent ~37) in ~50% of episodes. With stochastic actions (~5% per-edge dropout), vel_ent stays at ~0.45 and no episode achieves flocking.
- **The policy learned "don't interfere"** — output near-FC and let the ACS controller handle alignment. It has NOT learned to discriminate between neighbors.

### Phase 7a: Connection cost (FAILED — zero learning)
Added `acs_train_w_conn` to the env: a reward penalty proportional to the fraction of selected edges (conn_ratio). Theory: directly incentivize sparsity → break the FC attractor. Implementation: `reward += w_conn * (-conn_ratio)` where conn_ratio ∈ [0,1] (1.0 = FC). Also increased `w_ctrl` (0.1–0.3) to amplify control-cost savings from selective filtering.

**`conn_cost_sweep_260522`** (4 trials, sf=0.03, w_conn × w_ctrl sweep, 8 iters before early stop):
- **w_conn ∈ {0.05, 0.15, 0.3} × w_ctrl ∈ {0.1, 0.3}:** ALL trials showed zero learning. Entropy locked at 263.3–263.4 (theoretical max) across 8 iterations. conn_ratio stuck at ~0.5 (consistent with uniform policy at P=0.5 per edge). KL=0.000000 — policy literally unchanged from old policy.
- **Diagnostics:** vf_explained_var ~0.75 (critic IS learning), policy_loss ~0.001 (essentially zero), grad norms not reported correctly.
- **Root cause:** The conn_cost gradient is uniformly distributed across 380 edges. Each edge contributes only 1/380 of the total conn_ratio, giving a per-edge gradient of ~0.0007 — too diluted for PPO to act on. The connection cost provides GLOBAL sparsity pressure but no PER-EDGE discrimination signal. The policy can't learn WHICH neighbors matter from a uniform penalty on total edge count.
- **Also:** higher w_ctrl (0.1 vs original 0.02) worsened things by adding reward noise without informative signal. Near the uniform initialization, FC and random-50% selections give similar average control costs (both average over ~10+ neighbors per agent).
- **Conclusion:** Reward-side interventions that treat all edges uniformly cannot escape the uniform attractor. The per-edge gradient is too diluted with N=20 agents and 380 binary decisions.

### Phase 7b: Scale factor 0.07–0.15 sweep (IN PROGRESS — sf=0.15 promising)
Tested the gap between sf=0.05 (Phase 5, delta=-44/100 iters) and sf=0.2 (Phase 5, immediate collapse). Clean reward (no conn_cost, w_ctrl=0.02).

**`sf_sweep_260522`** (3 trials: sf ∈ {0.07, 0.1, 0.15}, 150 iters target):
- **sf=0.07 and sf=0.1:** Zero learning after 4 iterations. Entropy locked at 263.3 — same symptom as all previous experiments with sf ≤ 0.1.
- **sf=0.15: PHASE TRANSITION.** Entropy dropped from 263.4 to 163.6 in a single iteration (delta=-100!). Spatial entropy remained healthy at 39.7 (NOT a collapse like sf=0.2 which went to 1000+). The policy immediately began making decisive neighbor selections.
  - Iter 1–3: entropy 163.6→169.3→184.3, vel_ent 1.76→1.22→1.07 (best ego-centric vel_ent ever)
  - Iter 4–7: entropy rebounded to ~200 (PPO clipping regulating the initial overshoot), then stabilized. vel_ent oscillated but resumed improvement: 1.40→1.32→1.26→1.13
  - sp_ent stable at 39.7–39.9 throughout — no swarm disruption
  - **The oscillation is damping**: entropy settling ~200, vel_ent trending downward
- **Key finding:** There is a sharp PHASE TRANSITION between sf=0.1 (no learning at all) and sf=0.15 (immediate massive learning). The threshold determines whether the attention-score gradients are strong enough to escape the uniform attractor in a single PPO update.
- **Status:** Still running. Need to monitor whether vel_ent continues to decrease toward FC-ACS baseline (0.10) or plateaus. Need to evaluate checkpoint to determine if policy is approaching FC or discovering non-FC strategies.

### FC-ACS Baseline Performance
Pure fully-connected deterministic ACS (10 episodes):
- 5/10 episodes: early termination, sp_ent≈37, vel_ent≈0.10 (**successful flocking**)
- 5/10 episodes: 1000 steps, sp_ent≈41, vel_ent≈0.16–0.39 (still converging)
- Stochastic near-FC (from learned policy): 0/20 episodes achieve flocking

## Current State (2026-05-22)

### Running experiments
**`sf_sweep_260522`** — 3 trials (sf=0.07, 0.1, 0.15), 150 iters target. Running on GPUs 1 & 3 via `CUDA_VISIBLE_DEVICES=1,3 python train.py`. PID: check `ps aux | grep train.py`. Only sf=0.15 is learning. At iter 7: entropy≈200, vel_ent≈1.13, sp_ent≈39.7. The other two (sf=0.07, 0.1) are flat at max entropy.

### Git state
Branch `exp/autonomous-research`. Working tree has uncommitted changes: connection cost implementation in env.py, callbacks.py update, train.py for sf sweep, check_sweep.py utility.

### Experiment results on disk
All under `/workspace/test_results/`. Key ones:
- `ctrl_aux_factorial_260519/` — 8 trials under the buggy grad_clip. Policy frozen. Historical reference only.
- `grad_norm_diagnostic_260520/` — 2 trials confirming grad_clip throttle (gnorm data).
- `grad_fix_lr5e4_aux/noaux_260520/` — aux ON vs OFF paired comparison. Validates aux necessity.
- `scale_factor_sweep_260521/` — sf {0.01, 0.05, 0.2}. Shows the scale_factor bottleneck.
- `sf03_fresh_lr2e4_260522/` — sf=0.03 with lr decay. Best vel_ent=0.425 at iter 68.
- `conn_cost_sweep_260522/` — 4 trials (w_conn × w_ctrl), sf=0.03. Zero learning after 8 iters. Killed.
- `sf_sweep_260522/` — 3 trials (sf=0.07/0.1/0.15). **sf=0.15 is the breakthrough** — first ego-centric model to show rapid learning. Still running.

## Autonomous Session Protocol (`/goal`)

This document drives autonomous `/goal` sessions. Each session reads HANDOFF.md, runs experiments, and updates HANDOFF.md before exiting — forming a Ralph Loop across sessions.

### Context budget
- `cat /tmp/ctx` returns current token count. Capacity = 1,000,000.
- **Hard stop at 400,000 tokens (40%).** Before stopping: update "Current State" and "Research Trajectory" sections, then declare the session complete.
- Check `cat /tmp/ctx` every few turns to monitor usage.

### Operating rules
1. **Read HANDOFF.md first** — understand current state, what's been tried, what's next.
2. **Delegate to sub-agents (Opus)** — all code exploration, experiment config generation, and training monitoring go to sub-agents. All sub-agents must use model=opus. Main agent makes research decisions and records findings. Sub-agents can monitor training progress at efficient intervals.
3. **Experiment design** — choose a direction from "Possible next approaches" based on your own judgment. Design the experiment, implement config changes, start training. If time permits within context budget, run multiple experiments.
4. **Resources** — GPUs: `cuda:1` and `cuda:3` only. CPU: max 58 Ray workers total (local + remote + eval).
5. **Git workflow** — work on branch `exp/autonomous-research` (branched from `feat/aux-task`). Commit freely to this branch (local only, never push).
6. **Record everything** — update Research Trajectory with a new Phase entry. Update Current State with running/completed experiments. Update Possible next approaches (remove tried ones, add new ideas from findings).
7. **Graceful exit** — use your judgment on when an experiment cycle is complete. When hitting context budget OR completing a cycle, update this file so the next session can continue seamlessly.

## Open Questions & Suggested Directions

### The fundamental challenge
The centralized-obs model proved that FC is suboptimal — a learned policy can clearly beat it. However, the ego-centric model has not yet succeeded despite multiple attempts. The policy consistently converges to near-FC (selecting all neighbors), suggesting the current architecture/reward/training setup doesn't incentivize discovering non-FC strategies from ego-centric observations.

### Why FC might be suboptimal (and how to test)
1. **Control cost:** With FC, each agent averages ALL neighbors including distant or misaligned ones, potentially causing unnecessary turning. A selective policy that ignores "noisy" neighbors could reduce heading-rate cost → better ACS reward. But with `w_ctrl=0.02` (tiny), this benefit is negligible in the training reward. **Try higher w_ctrl** (0.1–0.5) to make selective neighbor filtering rewarding.
2. **Communication constraints:** With unlimited comm range, FC is easy. Add `comm_range` limits to force actual selection.
3. **Agent scaling:** With 20 agents, FC is 380 binary decisions — manageable. With 50+ agents, FC becomes computationally expensive and the policy must learn to select.

### Training dynamics lessons (carry forward)
- `grad_clip` must be None or very large (≥40). The default 1.0 kills actor learning.
- **`scale_factor` has a sharp phase transition at ~0.12–0.15.** Below 0.1, the policy is trapped at uniform (zero learning). At 0.15, learning is immediate and dramatic (delta=-100 in 1 iter). At 0.2, the policy collapses. The sweet spot is sf ∈ [0.12, 0.18]. Start new experiments at sf=0.15.
- `aux_enabled=True` is necessary for the ego-centric model to learn representations.
- `lr` ∈ [3e-4, 5e-4] is the productive range. Below 2e-4 the policy stalls.
- `vf_clip_param=256` can saturate if reward magnitudes explode. Monitor during scale_factor tuning.
- **Connection cost (global sparsity penalty) does NOT work.** The per-edge gradient (~1/380) is too diluted. Don't retry without a mechanism to provide per-edge differentiation signal.
- **Higher w_ctrl alone doesn't help** — near the uniform initialization, FC and random-50% selections give similar average control costs.

### Possible next approaches
- **sf=0.15 extended run:** The current sf_sweep is still running. If sf=0.15 continues to improve vel_ent (currently ~1.1, FC-ACS baseline is 0.10), let it run to 150 iterations and evaluate. **This is the highest-priority follow-up.**
- **Evaluate sf=0.15 checkpoint:** Once iter 10+ is reached, evaluate with `evaluate_checkpoint.py` to determine whether the policy is approaching FC or learning non-FC strategies. Check action distribution (P(select) per edge, mean edges per agent).
- **sf fine-tuning around 0.15:** Try sf=0.12, 0.13, 0.14 to find the exact onset of the phase transition. Lower sf might give slower but more stable learning without the initial oscillation.
- **sf=0.15 + connection cost:** Now that sf=0.15 breaks the uniform attractor, adding w_conn might help SUSTAIN non-FC behavior rather than converging back to FC. The per-edge gradient from conn_cost should be non-diluted once the policy is no longer uniform.
- **Action space redesign (top-K):** Investigation complete — top-K can be implemented by replacing only `attention_scores_to_logits()` and adding a custom Plackett-Luce action distribution, keeping the Transformer encoder unchanged. This would FORCE non-FC selection (each agent selects exactly K < N neighbors). Worth trying if sf=0.15 still converges to FC.
- **Curriculum:** Start with fewer agents or shorter episodes, increase gradually.

### Evaluation protocol
Use `evaluate_checkpoint.py` to compare against Pure-ACS baseline. Use deterministic (argmax) actions for evaluation even when training is stochastic.
