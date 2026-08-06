# NOTES_env.md — condensed env/baseline reference for this study

> Distilled from two code-exploration passes over `/workspace/envs/env.py` (1396
> lines; contains env AND pydantic configs) and `/workspace/baselines.py` /
> `evaluate_checkpoint.py`. Line numbers = current working tree. This is the
> ground-truth reference for the harness in `src/` and for the criterion
> discussion. Do not re-derive; verify against these line numbers if code changes.

## ACS dynamics (task_type='acs', env.py:666-913)

Constant-speed unicycle, heading-rate control only:

- State per agent: `[x, y, vx, vy, theta]`, `agent_states` (N_max,5); theta UNWRAPPED.
- Applied network = `neighbor_masks AND action` (env.py:683); with `comm_range=None`
  (default) neighbor_masks is all-ones, so applied network = action.
- Neighbor count N_i = 1 + (#selected neighbors) — self-loop forced to 1
  (env.py:595-597). Both control terms are normalized by N_i → **mean over selected
  neighbors, not sum** (env.py:820-821).
- Alignment (Cucker-Smale, env.py:833-840):
  `u_cs[i] = (lam/N_i) * sum_j a_ij * (1+r_ij^2)^(-beta) * sin(theta_j - theta_i)`
  with lam=5, beta=1/3. NOTE: r in meters, so psi=(1+r^2)^(-1/3) ≈ 0.046 at r=100
  → long-range alignment coupling is weak.
- Cohesion/separation (bonding, env.py:842-863):
  `u_coh[i] = sig/(N_i*speed) * sum_j a_ij * [ k1/(2 r_ij^2) * <v_j-v_i, x_j-x_i>
              + k2/(2 r_ij) * (r_ij - r0) ] * <nhat_i, x_j-x_i>`
  where nhat_i = [-sin th_i, cos th_i] (left body normal) → steering command.
  sig=1, k1=1 (damping), k2=3 (spring), r0=60 m. For r >> r0 the spring part grows
  ~ k2*r/2 * sin(bearing) → distant attraction SATURATES the turn-rate clip.
- `u = clip(u_cs + u_coh, ±max_turn_rate=8/15 rad/s)` (env.py:866).
- Integration (env.py:890-913): `x += v_old*dt; theta += u*dt; v = speed*[cos,sin]`,
  dt=0.1, speed=15 m/s (1.5 m/step).

## Metrics (env.py:1283-1293)

- `spatial_entropy  sigma_p = sqrt(Var(x)+Var(y))` over active agents [m].
- `velocity_entropy sigma_v = sqrt(Var(vx)+Var(vy))` [m/s], max ≈ speed = 15.
- Relation to order parameter phi = |mean(v_i/|v_i|)|: with constant speed,
  `sigma_v = speed * sqrt(1 - phi^2)` → sigma_v<0.1 ⟺ phi>0.99998 (VERY tight).
- Useful identity: mean squared pairwise distance E[r_ij^2] = 2*sigma_p^2
  (population var, so if all pairs sat at r0, sigma_p = r0/sqrt(2) ≈ 42.4).
- Computed every step in `check_episode_termination` (even when fixed-length),
  stored in `env.spatial_entropy_hist/velocity_entropy_hist[t]` and echoed in
  `info['spatial_entropy'/'velocity_entropy']` (post-step values; t=0 initial value
  never reported by env — harness computes it itself).

## Current convergence criterion (env.py:1236-1259)

Episode ends ("converged") iff ALL of, at current step t:
1. sigma_p < entropy_p_goal (default None → **0.7*r0 = 42.0**, set at env.py:309)
2. sigma_v < entropy_v_goal (default **0.1**)
3. t >= 49 and peak-to-peak of last 50 sigma_p samples < entropy_p_rate_goal (0.1)
4. peak-to-peak of last 50 sigma_v samples < entropy_v_rate_goal (0.2)
Plus hard stop at max_time_steps. `use_fixed_episode_length=True` disables 1-4
entirely (gate at env.py:1244) → our harness uses this for fixed-horizon runs.

## Key empirical anchors (calibration batch, see RUNLOG)

- FC (k=19, N=20) equilibrium: sigma_p ≈ 38.7-40.9, i.e. ~2-8% BELOW the 42
  threshold → the current absolute threshold is effectively "the FC equilibrium
  plus small margin".
- Single-cluster k=5 equilibrium: sigma_p ≈ 90 stable; phi>0.99 reached in ~400
  steps but sigma_v stays ≈ 0.2 (>0.1 goal) indefinitely — station-keeping heading
  jitter keeps sigma_v above goal on sparse topologies.
- k=1-2: flock fragments into ~pairs/triplets (n_comp_r0 10-17); fragmented runs
  have sigma_p growing linearly forever (clusters of size > k are self-contained
  under k-NN → no inter-cluster attraction → never re-merge).

## Baselines used (baselines.py)

- k-NN: `create_baseline('nearest', k=...)` (class FixedNearestNeighborSelection,
  baselines.py:249-332). **k EXCLUDES self**; row sum = k+1; k capped at n-1;
  stateless; REQUIRES observation_type='ego_centric'; distances from
  `obs['local_agent_infos'][:,:,:2]` (norm-invariant to ego rotation).
- Disc: `create_baseline('distance', distance_threshold=...)` — threshold in units
  of initial_position_bound/2 (=125 m by default → r0=60 m ⟺ 0.48).
- Registry: random / distance / nearest / farthest / mti / voronoi / highest_degree
  / mfnn / visual_attention / active_search / gazing_preference / motion_salience
  (baselines.py:2075-2088). Factory key is 'nearest', NOT 'knn'.

## Standalone rollout contract (gym 0.23.1)

- Build: `cfg = load_config('/workspace/envs/default_env_config.yaml')`; set
  `cfg.env.*` / `cfg.control.*` attributes; env =
  `NeighborSelectionFlockingEnv(config_to_env_input(cfg, seed_id))`.
- `env.seed(s)` BEFORE `reset()`; `reset()` returns obs only; `step` returns
  `(obs, reward, done, info)`; exactly max_time_steps steps (stepping past done →
  IndexError). Action must be np int dtype, shape (N_max,N_max); diagonal
  auto-fixed to 1 (in-place mutation + printed warning).
- Mask violations are NOT caught by validate_action (bitwise-~ bug makes the
  assert vacuous, env.py:598-599) — they are silently ANDed away at env.py:683.
- `is_training=False` → reward = -(dt*speed*|u| + rho*dt) averaged over agents
  (per-agent turn cost; rho=1).
- Ground truth after step: `env.state['agent_states']` (+ padding_mask); applied
  adjacency = `env.current_action` (== action AND masks when comm_range=None).
- Init: positions uniform in [-L/2, L/2]^2 with L=initial_position_bound (250
  default), headings uniform(-pi,pi), speed always 15. `num_agents_pool=[20]` →
  N_max=20, no padding rows.
- Import only `envs.env` + `baselines` (NOT evaluate_checkpoint — pulls torch/ray).

## Default parameter card (yaml + EnvConfig/ControlConfig defaults)

speed=15, max_turn_rate=8/15, initial_position_bound=250, beta=1/3, lam=5, sig=1,
k1=1, k2=3, r0=60, rho=1.0 | dt=0.1, max_time_steps=1000 (harness overrides),
comm_range=None, periodic_boundary=False, obs_dim=4, task_type='acs',
observation_type='ego_centric', entropy_v_goal=0.1, entropy_p_goal→42.0,
entropy_p_rate_goal=0.1, entropy_v_rate_goal=0.2, entropy_rate_window_length=50.
