# Handoff: Ego-Centric Neighbor Selection for ACS Flocking

## Research Goal

Train an ego-centric RL policy that, given each agent's local observation, selects **which neighbors to listen to** (binary adjacency matrix per step). The low-level ACS controller then uses the selected subgraph to compute velocity updates.

**Success metric:** The ego-centric policy must **clearly outperform the fully-connected ACS baseline** — faster convergence to flocking and/or better eval reward. The centralized-obs model already achieved this under a **relaxed convergence condition (vel_ent < 1.0 instead of the default 0.1)**: it converges faster and achieves better eval reward than FC-ACS. The ego-centric model should target the same. Under FC, ego-centric and centralized observations are mathematically interconvertible, so a parameter-sharing ego-centric policy can in principle replicate the centralized policy's decisions.

**Current status: FC-ACS has NOT been beaten by RL.** K=10 nearest heuristic beats FC by 17% (t=4.94), confirming selective IS better. But no RL-trained ego-centric policy has matched KNN performance. The binary action space + scalar reward architecture prevents RL from learning per-edge credit assignment.

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
- **Available checkpoints:** iter 50, 60, 70 (keep_checkpoints_num=3). Checkpoint 70 has training vel_ent=0.57, sp_ent=39.9, entropy=15.2.
- **Checkpoint 70 formally evaluated (50 episodes, deterministic):**
  - RL: reward=-258.4±89.7, vel_ent=0.15±0.71, sp_ent=39.6±1.4, flocking_success=0/50, **mean_edges/agent=19.0±0.0 (exactly FC)**
  - FC-ACS: reward=-274.4±106.9, vel_ent=0.23±1.11, sp_ent=39.8±1.6, flocking_success=0/50, edges=19.0
  - **Conclusion: policy converged to FC.** No selective neighbor selection. Marginal reward difference (+5.8%) is within noise.
- **Key finding:** Sharp phase transition between sf=0.1 (no learning) and sf=0.15 (immediate learning + eventual collapse). sf=0.15 is ON the collapse boundary — same instability as sf=0.2, just delayed. All learned policies converge to FC regardless of training dynamics.

## Current State (2026-05-27)

### Running experiments
- `continuous_sf05_floor02_260526` (GPU 1): sf=0.05, continuous action, weight floor 0.2. ~20 iterations done.
- `continuous_sf15_floor02_260527` (GPU 3): sf=0.15, continuous action, weight floor 0.2. ~15 iterations done.
Both running to 100 iterations.

### FC-ACS has NOT been beaten by RL
**K=10 nearest heuristic beats FC** (paired t=4.94, p<0.001, reward -225 vs -272, 17% better). No RL-trained policy has matched or exceeded KNN performance. Continuous action space (Phase 11) eliminates binary bistability and enables genuine per-edge differentiation, but has not yet beaten FC in formal evaluation. V3 (sf=0.15, floor 0.2) is closest at +0.7% reward (within noise), training continuing.

### Git state
Branch `exp/autonomous-research`. Modified: `evaluate_checkpoint.py`, `train.py`, `envs/env.py`.

### Experiment results on disk
All under `/workspace/test_results/`. Key ones:
- `scale_factor_sweep_260521/` — sf {0.01, 0.05, 0.2}. Phase 5 data.
- `sf03_fresh_lr2e4_260522/` — sf=0.03. Converged to near-FC (Phase 6).
- `conn_cost_sweep_260522/` — Connection cost sweep. Zero learning (Phase 7a).
- `sf_sweep_260522/` — sf {0.07, 0.1, 0.15}. sf=0.15 learned then collapsed (Phase 7b). **Checkpoint 70 evaluated: exactly FC (19.0 edges/agent).**
- `wctrl_sweep_260523/` — w_ctrl {0.3, 0.5, 1.0}. Entropy stuck at 263 (entropy_coeff too strong).
- `topk10_260523/` — top-K=10 v1 (collapsed) and v2 (bias=+2.0, checkpoints at 10/20/30).

### What has been done this session
- **Modified `evaluate_checkpoint.py`** to support ego-centric model (`NeighborSelectionPPORLlib`). Auto-detects observation type from checkpoint params. Added mean_edges_per_agent and flocking_success metrics.
- **Modified `envs/env.py`** to always compute `_conn_ratio` for logging, even when `w_conn=0`.
- **Formally evaluated sf_sweep checkpoint 70** (50 episodes): exactly FC, 19.0 edges/agent, no selective behavior.
- **Ran K-nearest diagnostic**: K=10 gets 23% less control cost than FC — confirmed selectivity IS beneficial.
- **Ran 9 training experiments** testing w_ctrl sweep, entropy_coeff, sf values, batch sizes, lr schedules, fine-tuning from FC checkpoint, and curriculum approaches. All failed due to the bistability of the binary action space.
- **Diagnosed the fundamental problem**: the independent binary action space with sf-scaled logits creates two strong attractors (FC and empty) with no stable intermediate. This is an architecture problem, not a reward problem.
- **Implemented top-K action space** in `models/ppo.py`: `top_k` config parameter masks `attention_scores_to_logits()` to allow only K highest-scoring neighbors. +2.0 bias prevents empty-attractor collapse.
- **Tested top-K=10**: FC attractor eliminated, conn stable at ~0.50 for 40 iters. But vel_ent worse than random top-K selection — credit assignment is the remaining bottleneck. Collapsed after ~48 iters when model overcame the +2.0 bias.
- **Identified next step**: Initialize top-K from K-nearest heuristic (warm start from vel_ent=0.23) rather than learning from scratch.

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
- `scale_factor`: sf ≤ 0.12 → no learning (gradient too weak); sf=0.15 → learning but bistable; sf=0.2 → immediate collapse. The narrow window (0.12–0.2) is NOT tunable — it's a symptom of the binary action space architecture.
- `aux_enabled=True` is necessary for learning.
- `lr` ∈ [3e-4, 5e-4] is the productive range.
- Connection cost (global sparsity penalty) does NOT work — per-edge gradient too diluted at 1/380.
- `entropy_coeff` ≥ 0.005 kills learning for 380-edge binary action space (bonus 1.3/sample overwhelms policy gradient). entropy_coeff=0.001 is too weak to break FC. entropy_coeff=0.01 breaks FC but causes value function mismatch when later removed.
- **w_ctrl > 0.02 with sf=0.15 DOES incentivize selectivity** — the policy immediately moves away from FC — but it overshoots to near-empty (conn<0.1) because the binary action space has no stable intermediate state.
- **Higher w_ctrl with sf=0.05: alignment and control cost gradients partially cancel → zero learning.** The opposing reward components make the net per-edge gradient negligible at low sf.

### Phase 8: Formal eval of checkpoint 70 + w_ctrl sweep
**Formal eval of sf_sweep checkpoint 70** (50 episodes, deterministic, `evaluate_checkpoint.py` modified to support ego-centric model):
- RL: reward=-258.4±89.7, vel_ent=0.15±0.71, sp_ent=39.6±1.4, flocking_success=0/50, **edges/agent=19.0 (exactly FC)**
- FC-ACS: reward=-274.4±106.9, vel_ent=0.23±1.11, sp_ent=39.8±1.6, flocking_success=0/50, edges=19.0
- **Policy is FC.** No selective neighbor selection learned.

**Diagnostic: FC vs sparse control cost** (10 episodes each, K-nearest heuristic):
- FC (19 edges): return=-300.7±96, vel_ent=0.05, sp_ent=39.8
- K=10: return=-231.7±56, vel_ent=0.23, sp_ent=61.3 — **23% less control cost than FC**
- K=5: return=-248.2±32, vel_ent=3.44, sp_ent=297.7 — too sparse, swarm fragments
- **Confirms selectivity IS beneficial** — the optimal edge count is well below FC.

**`wctrl_sweep_260523`** (3 trials: w_ctrl ∈ {0.3, 0.5, 1.0}, sf=0.15, entropy_coeff=0.005, clip=0.15):
- **All stuck at near-random (entropy≈263) after 4-8 iterations.** entropy_coeff=0.005 is too strong for 380 binary edges — the entropy bonus (0.005 × 263 = 1.3/sample) overwhelms the policy gradient.
- **Killed after iter 8.** entropy_coeff must be ≤0.0001 for this action space, or 0.

**`wctrl05_noent_260523`** (1 trial: w_ctrl=0.5, sf=0.15, entropy_coeff=0, clip=0.15):
- Iter 1: entropy=211.6, conn=0.518, vel_ent=2.3 — **learning and selective!**
- Iter 2-4: entropy collapsed to ~53, conn dropped to 0.20 (~4 edges/agent), vel_ent=10.0, reward=-1100.
- **Same collapse as Phase 7b but in reverse direction** — instead of converging to FC, over-selected to near-empty. sf=0.15 gradient too aggressive.
- **Killed after iter 4.**

**`stable_wctrl_260523`** (2 trials: sf ∈ {0.10, 0.12}, w_ctrl=0.3, clip=0.1, lr=3e-4):
- Both stuck at entropy=263 after 2 iterations. **Confirms: sf ≤ 0.12 = zero learning regardless of w_ctrl.** Killed.

**`sf15_stable_260523`** (2 trials: w_ctrl ∈ {0.1, 0.2}, sf=0.15, clip=0.1, sgd_iter=5, lr=3e-4):
- w_ctrl=0.1: entropy=263 at iter 2 — too little selectivity pressure. Dead.
- w_ctrl=0.2: **Showed learning** (entropy oscillated 201→212→192→232 over 4 iters, conn oscillated 0.50→0.64→0.65→0.57). The stabilization prevented collapse, but gradient variance caused persistent oscillation. Killed at iter 4.

**`sf15_bigbatch_260523`** (1 trial: sf=0.15, w_ctrl=0.2, batch=64k):
- Entropy barely moved (263→262 over 4 iters). Conservative settings + large batch made learning too slow. Killed.

**Fine-tune from FC checkpoint** (3 attempts with entropy_coeff ∈ {0, 0.01, 0.001}):
- entropy_coeff=0: entropy stayed at 13.4 (vanishing gradient at FC — per-edge gradient ∝ p(1-p) ≈ 0.001). No learning.
- entropy_coeff=0.01: entropy jumped 13→216 in 1 iter (broke out of FC!), but subsequent iter with entropy_coeff=0 immediately reverted to random (263) due to value function mismatch. The value function was calibrated to entropy-bonus rewards.
- entropy_coeff=0.001: entropy barely moved (13→14). Push too weak.
- **Conclusion: Two-phase fine-tune fails because the value function doesn't transfer across entropy_coeff changes.** Would need to re-train the value function gradually.

**`sf05_wctrl02_260523`** (1 trial: sf=0.05, w_ctrl=0.2, Phase 5's settings):
- Zero learning after 15 iterations (entropy=263.4, flat). With w_ctrl=0.2, the alignment and control cost gradients partially cancel → net gradient too weak for sf=0.05. Killed.

**`sf15_longrun_260523`** (1 trial: sf=0.15, w_ctrl=0.2, clip=0.1, sgd_iter=5, lr_schedule 3e-4→5e-5, 20 iters):
- Collapsed to over-sparse by iter 8: conn=0.073 (1.4 edges/agent), vel_ent=14 (terrible), reward=-782. Stabilized at conn≈0.065 for iters 9-20 but could not recover. **Same bistability as wctrl05_noent.**

**`curriculum_step1_260523`** (fine-tune from checkpoint 70 with w_ctrl=0.05, entropy_coeff=0):
- Stuck at FC (conn=0.994, entropy=13-14) for 5 iterations. **Vanishing gradient at FC attractor**: per-edge gradient ∝ p(1-p) ≈ 0.001 at p=0.999. Even increasing w_ctrl from 0.02→0.05 produces negligible per-edge signal.

## Fundamental Diagnosis (Phase 8 conclusion)

### The binary action space is bistable
The independent binary per-edge action space with sf-scaled logits creates **two strong attractors**:
1. **FC attractor** (all edges selected, entropy≈13): vanishing gradient because p≈0.999 → p(1-p)≈0.001
2. **Empty attractor** (no edges selected, entropy≈90): vanishing gradient because p≈0.001 → p(1-p)≈0.001

Any intermediate state is **unstable** — sf=0.15's gradient pushes each edge toward the nearest extreme. This creates the observed dynamics:
- From random (entropy≈263): w_ctrl=0 → converges to FC; w_ctrl>0 → collapses to empty
- From FC: gradient too small to drop any edge; entropy_coeff breaks out but jumps to random
- Oscillation with conservative settings, but no convergence to a stable intermediate

### Evidence this is an architecture problem, not a reward problem
1. K=10 heuristic achieves 23% better return than FC — selectivity IS genuinely optimal
2. w_ctrl=0.2 correctly incentivizes selectivity (verified: all w_ctrl>0 experiments moved away from FC)
3. But the policy can only reach FC or empty, never the optimal intermediate (~10 edges/agent)

### Phase 9: Top-K action space (implemented and tested)
**Implementation** (models/ppo.py): Added `top_k` parameter to `custom_model_config`. In `attention_scores_to_logits()`, masks out non-top-K attention scores per row, adds +2.0 bias to top-K positions. Non-top-K get -1e9 (blocked). Minimal change — no env/RLlib modifications needed.

**`topk10_260523` v1** (top-K=10, sf=0.15, w_ctrl=0.02, no bias on top-K):
- Top-K as CANDIDATES (model can choose to not select any). Collapsed to conn=0.027 (~0.5 edges/agent) by iter 30. Same empty attractor — model learned to reject all candidates. Killed.

**`topk10_260523` v2** (top-K=10, sf=0.15, w_ctrl=0.02, +2.0 bias):
- **Conn stable at ~0.50 for 40 iterations** — the FC attractor is eliminated, and bias prevents empty attractor.
- vel_ent trajectory: 3.8 (iter 1, random top-K) → 10.5 (iter 5, worse) → 5.0 (iter 20, improving) → 8.4 (iter 40, worsening) → collapse at iter 48 (model overcame +2.0 bias).
- **Formal eval of checkpoint 30**: vel_ent=10.5±4.7, reward=-684±39, edges=10.0. Much worse than FC (vel_ent=0.13, reward=-277).
- **Critical finding: random top-K selection (iter 1, vel_ent=3.8) outperforms the trained policy (vel_ent=5-8).** The PPO gradient doesn't provide useful per-edge credit assignment — the model learns to confidently select BAD neighbors.

### Diagnosis: credit assignment is the core remaining problem
The top-K constraint successfully eliminates the FC/empty bistability, but the model can't learn WHICH K neighbors are best because:
1. **Per-edge reward signal is too dilute**: each edge contributes ~1/10 of the alignment change. PPO's advantage estimates can't reliably attribute reward to individual edge decisions.
2. **Random selection within top-K is a strong baseline**: by chance, random top-K includes enough useful neighbors for reasonable alignment (vel_ent=3.8). The trained policy's confident-but-wrong selections are worse.
3. **Entropy collapse to 0 prevents recovery**: once the policy becomes deterministic about a bad selection, it can't explore better options.

### Phase 10: Distance bias + BC+RL (2026-05-26)

**Rank-biased model (disguised heuristic — user correctly identified):**
- rank bias scale=5: entropy=0, pure KNN, RL contributes nothing
- rank bias scale=0.5: entropy=1.9, near-pure KNN, RL degrades by 2.2 points vs KNN
- KNN heuristic (t=4.94) > RL checkpoint (t=4.54) > FC — RL made KNN WORSE

**Distance-based bias (absolute, threshold=1.4):**
- scale=1.0, sf=0.05: stable training, vel_ent improving, but policy converged toward FC (conn 0.90→0.94). Eval at iter 80: FC still wins (paired t-test).
- scale=3.0, sf=0.05: conn=0.999 (FC) because agents cluster mid-episode → all distances shrink below threshold.
- scale=10, sf=0.15: same clustering issue, plus instability.

**Rank-based bias (scale-invariant — the fix for clustering):**
- Maintains K=10 nearest regardless of spatial clustering. Verified: 10 edges/agent at both episode start and after 200 steps.
- But: scale≥2 → entropy=0 (pure KNN, no RL learning). Scale=0.5 → RL degrades KNN.

**BC + RL fine-tuning:**
- Phase 1 (BC): 99.4% accuracy, 96% action match with KNN. Model genuinely learns distance-based selection from weights.
- Phase 2 (RL, sf=0.15): BC initialization destroyed by iter 2 (conn 0.531→0.411→collapse to 0.123 by iter 10). Same bistability.

**Core conclusion:** The per-edge credit assignment problem is structural, not fixable by initialization, biases, or hyperparameter tuning. 380 independent binary decisions with one scalar reward → PPO's gradient is noise that degrades any good policy.

### Recommended next approaches (prioritized)

**The blocker is credit assignment, not architecture.** All attempts with independent binary edges + scalar reward failed because PPO can't attribute the episode reward to individual edge decisions. Solutions must address this directly.

1. **Per-agent reward decomposition** — compute per-agent rewards in `compute_custom_reward()` (each agent's control cost + alignment contribution separately). Sum for the env scalar, but use per-agent values in a **custom PPO loss** (similar to the existing aux task hook). Each agent's 19 edge decisions share only that agent's advantage — 19× better credit assignment than sharing across 380 edges.
2. **Continuous attention weights** — replace binary 0/1 with continuous [0,1] weights. The ACS controller uses `weight[i,j] * neighbor_j_heading` instead of `action[i,j] * neighbor_j_heading`. Smooth gradients everywhere; no bistability. Requires `env_transition()` modification (~20 lines).
3. **Autoregressive neighbor selection** — select neighbors one at a time per agent, conditioning on previous selections. Use a Plackett-Luce distribution over the pointer-net scores. Each selection gets its own reward attribution. Requires custom action distribution + env wrapper.
4. **Multi-agent RL** — treat each agent as a separate agent in RLlib's multi-agent mode. Per-agent obs, per-agent reward, per-agent policy. The action per agent is a 19-dim binary vector — much smaller than 380. The env already supports `multi_env` mode.

**What's been ruled out:**
- Architectural biases (distance, rank, top-K): all reduce to hardcoded heuristics. RL either contributes nothing or degrades.
- BC + RL fine-tuning: sf=0.15 destroys any initialization in 1-2 iters. Lower sf → no learning.
- Reward weight tuning (w_ctrl, w_vel, w_pos): doesn't address the credit assignment gap.
- Hyperparameter tuning (lr, clip, batch size, entropy_coeff): exhaustively explored. No combination stabilizes the binary action space.

### Phase 11: Continuous attention weights (2026-05-27, in progress)

**Implementation** (envs/env.py, models/ppo.py, models/beta_dist.py):
- Env: `continuous_action=True` in EnvConfig. Action space changes from `Box(int8)` to `Box(float32, [0,1])`. `env_transition()` uses `neighbor_masks * action` (float multiply) instead of `np.logical_and`. Weight floor at 0.2 prevents spatial fragmentation.
- Model: `attention_scores_to_logits()` outputs `[mean_logits, log_std_logits]` instead of `[-score, +score]` pairs. Mean logits = sf-scaled attention scores + 10.0 diagonal boost. Log_std = learnable global parameter (init -1.0, giving std≈0.37).
- Distribution: `TorchContinuousWeightDist` (squashed Gaussian) — sigmoid(Normal(mean, std)). Avoids torch.distributions.Beta which triggers CUDA JIT failures on newer GPUs with CUDA 11.3. Actions reshaped to (N, N) to match action space.
- Training: `normalize_actions=False` to prevent RLlib's default unsquashing.

**V1: sf=0.15, no floor, log_std=0** (continuous_sf15_260526, 8 iters before kill):
- Iter 1-3: Promising — vel_ent improved 0.68→0.49, conn stable 0.49→0.67. No binary bistability.
- Iter 5-8: Weights drifted toward 0 (conn: 0.67→0.52→0.44→0.36→0.34). Spatial fragmentation at iter 7 (sp_ent=87, reward=-4480). Killed.
- **Root cause:** High std (exp(0)=1.0) in pre-sigmoid space creates excessive weight variance. Combined with sf=0.15, the mean logits drift negative under noisy gradients.

**V2: sf=0.05, floor=0.2, log_std=-1** (continuous_sf05_floor02_260526, running):
- Very stable: vel_ent oscillates 0.29-0.35, conn stable at 0.50-0.58, sp_ent stable at 38-40.
- But learning is slow — vel_ent plateaued after iter 2. sf=0.05 may be too conservative for even continuous actions.
- Running to 100 iters to see if slow learning eventually differentiates neighbor weights.

**V3: sf=0.15, floor=0.2, log_std=-1** (continuous_sf15_floor02_260527, running):
- Hypothesis: the 0.2 weight floor prevents V1's fragmentation failure. sf=0.15's faster gradient should enable actual per-edge differentiation.
- Running on GPU 3 in parallel with V2.

**V2 checkpoint 10 formal eval** (100 episodes, deterministic):
- RL: reward=-292.0±122.0, vel_ent=0.626±2.200, sp_ent=39.8, **edges/agent=8.17** (43% of FC)
- FC: reward=-282.3±106.6, vel_ent=0.122±0.394, sp_ent=39.7, edges=19.0
- RL -3.4% reward (WORSE), vel_ent worse. **50-episode eval showed +7.6% but was sample noise — 100-episode eval corrects to -3.4%.**
- RL learned genuine selectivity (8.17 effective edges vs FC's 19.0) — first RL policy with real per-edge differentiation. But the selectivity is suboptimal: wrong neighbors downweighted.

**Key finding:** Continuous weights eliminate binary bistability and enable per-edge differentiation. But credit assignment remains: 400 continuous decisions with scalar reward → the policy differentiates weights but can't attribute which edges help. The structural selectivity (8.17 vs 19.0 edges) is a genuine RL contribution, but it hurts rather than helps at checkpoint 10.

**V3 checkpoint 10 formal eval** (100 episodes, deterministic):
- RL: reward=-274.5±94.6, vel_ent=0.195±1.103, sp_ent=39.6, **edges/agent=15.85** (83% of FC)
- FC: reward=-276.3±100.9, vel_ent=0.065±0.199, sp_ent=39.7, edges=19.0
- RL +0.7% reward (essentially equal). V3 is less selective than V2 (15.85 vs 8.17 edges) but closer to FC performance.

**Summary of continuous action space experiments (as of iter ~20):**
- Continuous weights successfully eliminate binary bistability. Policy learns stably for 20+ iterations without collapsing to FC or empty.
- Both V2 (sf=0.05) and V3 (sf=0.15) learn genuine per-edge differentiation (conn < 1.0).
- Neither has beaten FC-ACS in formal evaluation. V2 is 3.4% worse; V3 is 0.7% better (within noise).
- The per-edge credit assignment problem persists: 400 continuous decisions with one scalar reward → noisy gradient doesn't reliably identify which neighbors to downweight.
- Training continues on both GPUs. Later checkpoints (50-100 iters) may show improvement as the policy gradually refines its neighbor weighting.

**Remaining challenge:** The credit assignment problem limits per-edge learning regardless of action space type (binary or continuous). Continuous weights provide smoother gradients and eliminate bistability, but can't overcome the fundamental issue of attributing a scalar reward to 400 individual edge decisions. Per-agent reward decomposition remains the most promising next step.

Training continues on V2 (GPU 1) and V3 (GPU 3) toward 100 iterations.
