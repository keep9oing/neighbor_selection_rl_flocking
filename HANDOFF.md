# Handoff: Ego-Centric Neighbor Selection for ACS Flocking

## Research Goal

Train an ego-centric PPO policy that, given each agent's local observation, selects **which neighbors to listen to** (binary adjacency matrix per step). The low-level ACS controller then uses the selected subgraph to compute velocity updates.

**Success metric:** The ego-centric policy must **clearly outperform the fully-connected ACS baseline** — faster convergence to flocking and/or better eval reward. The centralized-obs model already achieved this under a **relaxed convergence condition (vel_ent < 1.0 instead of the default 0.1)**: it converges faster and achieves better eval reward than FC-ACS. The ego-centric model should target the same. Under FC, ego-centric and centralized observations are mathematically interconvertible, so a parameter-sharing ego-centric policy can in principle replicate the centralized policy's decisions.

**Current status: FC-ACS has NOT been beaten.** No ego-centric checkpoint has been formally evaluated as outperforming FC-ACS. All trained policies so far either (a) converged to near-FC, or (b) became unstable before being evaluated.

**Evaluation metrics** (logged by callbacks):
- **Episode reward** — primary metric. Sum of per-agent ACS rewards (`env._compute_rewards()`): negative heading-rate control cost + cruise cost. Must clearly exceed FC-ACS baseline.
- `final_velocity_entropy` — velocity alignment (lower = better). The env's flocking-success threshold (`entropy_v_goal=0.1`) is already strict — vel_ent=0.1 corresponds to order parameter well above 0.995. This threshold can be relaxed; order parameter ≥ 0.995 is sufficient.
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

- `envs/env.py` — environment. `_compute_rewards()` (line ~985): ACS reward (negative control cost). `compute_custom_reward()` (line ~1285): shaped training reward (spatial + velocity error + control cost + optional connection cost).
- `models/ppo.py` — ego-centric Transformer model. `scale_factor` (line ~49, 265–268): multiplies raw attention scores before logit formation — critically affects gradient flow.
- `models/ppo_centralized.py` — centralized variant (NOT used in current experiments).
- `grad_logging_ppo.py` — custom PPO subclass that logs pre-clip gradient norms into `learner_stats`.
- `callbacks.py` — logs `final_spatial_entropy`, `final_velocity_entropy`, `flocking_success`, `final_conn_ratio`.
- `evaluate_checkpoint.py` — compares a checkpoint against Pure-ACS FC baseline. **NOTE: loads `NeighborSelectionPPORLlibCentralized` — cannot evaluate ego-centric checkpoints without modification.**
- `baselines.py` — heuristic baselines.
- `train.py` — current training config. Set for sf sweep (sf grid_search over {0.07, 0.1, 0.15}).
- `check_sweep.py` — utility to monitor sweep progress.

## FC-ACS Baseline Performance
Pure fully-connected deterministic ACS (10 episodes):
- 5/10 episodes: early termination, sp_ent≈37, vel_ent≈0.10 (**successful flocking**)
- 5/10 episodes: 1000 steps, sp_ent≈41, vel_ent≈0.16–0.39 (still converging)
- Stochastic near-FC (from learned policy, ~5% per-edge dropout): 0/20 episodes achieve flocking

## Research Trajectory & Key Findings

### Phase 1: Aux task development (branch `feat/aux-task`, commits up to 5d8c082)
Added an auxiliary self-supervised task: from each agent's encoder embedding, predict the flock-center-frame state of agents. Implemented via `ModelV2.custom_loss()` hook. Controlled by `aux_enabled` master switch in model config. Early experiments (`aux_weight_sweep_260515`, `critic_aux_sweep_260517`) showed mixed results — later found to be confounded by deeper issues.

### Phase 2: Control-cost sign bug (commit 20498d6)
**Found and fixed a sign bug in `compute_custom_reward()`** (env.py line 1311). The control-cost term was `- w_ctrl * control_cost` where `control_cost ≤ 0`, making it a turning BONUS instead of a PENALTY. Fix: `+ w_ctrl * control_cost`. This invalidated all prior experiment conclusions. The velocity-weight sweep (`vel_weight_sweep_260518`) showed w_vel had zero effect across 0.2–3.0 — it was never the lever; the buggy control term dominated.

### Phase 3: grad_clip=1.0 throttle discovery
The `ctrl_aux_factorial_260519` (8 trials, 4×2 factorial on w_ctrl × aux) ran ~120 iters post-fix. Policy entropy stayed at 263.4 (theoretical max = 380 × ln2) — completely frozen at uniform random. Investigation revealed:

**`grad_clip=1.0` was throttling the actor gradient by ~20×.** RLlib PPO uses a single optimizer with global gradient norm clipping. The value loss dominated the global norm (~20), and clipping to 1.0 scaled the actor's gradient to ~0.016 (effectively zero).

**Fix:** `grad_clip=None`. Confirmed by paired diagnostic (`grad_norm_diagnostic_260520`): total gnorm ~7.6, critic/actor ratio ~200:1.

### Phase 4: Aux task validation
With grad_clip=None, paired experiment (`grad_fix_lr5e4_aux/noaux_260520`, 100 iters each):
- **aux=OFF:** entropy delta = -0.04 (essentially no learning)
- **aux=ON:** entropy delta = -1.05 (26× more)

**Aux task is necessary** under the current architecture — without it, the PPO gradient alone cannot escape the near-uniform initialization.

### Phase 5: scale_factor sweep (sf=0.01, 0.05, 0.2)
The model multiplies raw attention scores by `scale_factor` (default 0.002) before forming action logits. This creates gradient suppression proportional to 1/scale_factor.

**`scale_factor_sweep_260521`** (sf ∈ {0.01, 0.05, 0.2}, 100 iters each):
- sf=0.01: entropy 263→243 (delta=-20)
- sf=0.05: entropy 263→219 (delta=-44), vel_ent reached 1.12
- sf=0.2: **immediate collapse** — policy went deterministic in 1 iteration, sp_ent→1000+

### Phase 6: Policy converged to near-FC
Detailed evaluation of the sf=0.03 checkpoint (iter 70, best training vel_ent=0.425):
- **Action probabilities: 99.82% of (i,j) selections have probability > 0.9.** The policy selects ~18/19 neighbors per agent — essentially FC.
- **With deterministic actions (argmax → FC):** vel_ent < 0.10, sp_ent ~37 in ~50% of episodes. This matches pure FC-ACS behavior.
- **The policy learned "don't interfere"** — output near-FC and let the ACS controller handle alignment. It did NOT learn to discriminate between neighbors.

**This is the core problem:** under the current reward, FC appears to be a local optimum. The policy converges to FC regardless of how well it learns (sf=0.03, 0.05).

### Phase 7a: Connection cost (FAILED — zero learning)
Added `acs_train_w_conn` to the env: a reward penalty proportional to the fraction of selected edges. Theory: directly incentivize sparsity → break the FC attractor.

**`conn_cost_sweep_260522`** (4 trials, sf=0.03, w_conn ∈ {0.05, 0.15, 0.3} × w_ctrl ∈ {0.1, 0.3}, 8 iters before early stop):
- ALL trials showed zero learning. Entropy locked at 263.3–263.4. KL=0.000000.
- **Root cause:** The conn_cost gradient is uniformly distributed across 380 edges (~0.0007 per edge) — too diluted for PPO. Global sparsity pressure provides no per-edge discrimination signal.

### Phase 7b: Scale factor 0.07–0.15 sweep
Tested the gap between sf=0.05 (Phase 5) and sf=0.2 (Phase 5, collapse).

**`sf_sweep_260522`** (3 trials: sf ∈ {0.07, 0.1, 0.15}, stopped at iter 78 for sf=0.15):
- **sf=0.07 and sf=0.1:** Zero learning after 12 iterations. Entropy stuck at 263.3.
- **sf=0.15:** Entropy dropped from 263→164 in 1 iteration. Showed genuine policy learning for ~31 iterations (vel_ent improved from 1.76 to 0.99, entropy stabilized ~192, sp_ent stable ~39.7).
  - **Then collapsed at iter 32:** entropy crashed to 0 (fully deterministic), sp_ent spiked to 190. Same failure mode as sf=0.2, but delayed.
  - **Recovered at iter 40–49:** sp_ent returned to ~40, training vel_ent reached 0.06. But entropy=0 means the policy was fully deterministic — likely rediscovering FC (Phase 6 showed deterministic argmax policies behave as FC). **This is a training metric, not formal evaluation. No checkpoint at iter 49.**
  - **Second collapse at iter 63, partial recovery, then instability through iter 78.**
- **Available checkpoints:** iter 50, 60, 70 (keep_checkpoints_num=3). Checkpoint 70 has training vel_ent=0.57, sp_ent=39.9, entropy=15.2. **None have been formally evaluated.**
- **Key finding:** Sharp phase transition between sf=0.1 (no learning) and sf=0.15 (immediate learning + eventual collapse). sf=0.15 is ON the collapse boundary — same instability as sf=0.2, just delayed.

## Current State (2026-05-23)

### Running experiments
None. All experiments stopped.

### Git state
Branch `exp/autonomous-research`. Clean working tree.

### Experiment results on disk
All under `/workspace/test_results/`. Key ones:
- `scale_factor_sweep_260521/` — sf {0.01, 0.05, 0.2}. Phase 5 data.
- `sf03_fresh_lr2e4_260522/` — sf=0.03. Converged to near-FC (Phase 6).
- `conn_cost_sweep_260522/` — Connection cost sweep. Zero learning (Phase 7a).
- `sf_sweep_260522/` — sf {0.07, 0.1, 0.15}. sf=0.15 learned then collapsed (Phase 7b). **Checkpoints 50/60/70 available, NOT evaluated.**

### What has NOT been done
- **No ego-centric checkpoint has been formally evaluated against FC-ACS.** `evaluate_checkpoint.py` only works with the centralized model. Need modification or a new script.
- **No verification of whether any learned policy is genuinely selective (non-FC).** Phase 6 showed the only previously evaluated policy was near-FC.
- **The sf=0.15 checkpoints (50/60/70) have not been evaluated at all.**

## Autonomous Session Protocol (`/goal`)

This document drives autonomous `/goal` sessions. Each session reads HANDOFF.md, runs experiments, and updates HANDOFF.md before exiting.

### Context budget
- `cat /tmp/ctx` returns current token count. Capacity = 1,000,000.
- **Hard stop at 400,000 tokens (40%).** Before stopping: update this document, then declare the session complete.
- Check `cat /tmp/ctx` periodically. Do not waste context on repetitive polling — use background tasks and monitors efficiently.

### Operating rules
1. **Read HANDOFF.md first** — understand current state, what's been tried, what's next.
2. **Delegate to sub-agents (Opus)** — all code exploration, experiment config generation, evaluation runs, and training monitoring go to sub-agents. All sub-agents must use model=opus. Main agent makes research decisions and records findings only.
3. **Training monitoring** — do NOT poll iteration progress from the main agent. Instead, launch a sub-agent with `run_in_background=true` to monitor and report back when a target iteration is reached. Estimate iteration time from prior data (sf=0.15 with 3 trials: ~6 min/iter) and check at reasonable intervals (e.g., if 30 iters will take ~3 hours, check at 2.5h and 3h — not every 30 seconds).
4. **Evaluate before claiming results.** Training metrics (episode_reward_mean, custom_metrics) are noisy indicators. Only formal multi-episode evaluation with deterministic actions counts as evidence. Never write "proves" or "CAN beat" in HANDOFF without eval data.
5. **Resources** — GPUs: `cuda:1` and `cuda:3` only. CPU: max 58 Ray workers total.
6. **Git workflow** — work on branch `exp/autonomous-research`. Commit freely (local only, never push).
6. **Record accurately** — update Research Trajectory with factual findings. Do not overstate results. Distinguish training metrics from formal evaluation.
7. **Graceful exit** — update this file so the next session can continue seamlessly.

## Open Questions & Directions

### The fundamental problem
Every trained ego-centric policy has either:
1. **Converged to near-FC** (selecting all neighbors) — Phase 6 confirmed this for sf=0.03
2. **Become unstable before evaluation** — sf=0.15 collapsed at iter 32

The question is whether FC is the true optimum under the current reward, or whether the training dynamics prevent discovering better strategies.

### Why FC might be suboptimal
The centralized model proved FC is suboptimal (verified with vel_ent < 1.0 convergence criterion — faster convergence and better eval reward than FC-ACS). Possible reasons:
1. **Control cost:** With FC, each agent averages ALL neighbors including distant/misaligned ones → unnecessary turning. Selective filtering reduces this. But with w_ctrl=0.02, this benefit is negligible in training reward.
2. **Convergence speed:** A selective policy that ignores noisy/distant neighbors may converge to alignment faster than FC, which dilutes signal with noise from all 19 neighbors.

### Training dynamics lessons
- `grad_clip` must be None. Default 1.0 kills actor learning.
- `scale_factor`: sf ≤ 0.1 → no learning; sf=0.15 → learning but collapses by iter 32; sf=0.2 → immediate collapse. The narrow window (0.1–0.2) between "no learning" and "collapse" is the core training difficulty.
- `aux_enabled=True` is necessary for learning.
- `lr` ∈ [3e-4, 5e-4] is the productive range.
- Connection cost (global sparsity penalty) does NOT work — per-edge gradient too diluted at 1/380.

### Possible next approaches (prioritized)
1. **Evaluate sf=0.15 checkpoint_000070** — Modify `evaluate_checkpoint.py` to load `NeighborSelectionPPORLlib` (ego-centric). Run deterministic eval, check action distribution. This determines whether the learned policy is FC or genuinely selective.
2. **Stabilize sf=0.15 training** — Options:
   - lr schedule: start at 5e-4, decay to 1e-4 after iter 20 (collapse happened at iter 32 with constant lr)
   - entropy bonus (entropy_coeff=0.001) to prevent entropy from crashing to 0
   - clip_param=0.1 (tighter PPO clipping to prevent overshoot)
   - sf=0.12 or 0.13 (below collapse threshold but above no-learning threshold)
3. **Top-K action space** — Replace independent binary per-edge with "select exactly K neighbors." Implementation: replace `attention_scores_to_logits()` + add Plackett-Luce custom action distribution. Forces non-FC by construction.
4. **Higher w_ctrl (0.5–1.0)** — Make control cost savings from selective filtering visible in reward. Only useful if there IS a benefit to selective filtering (which eval of checkpoint 70 would reveal).
