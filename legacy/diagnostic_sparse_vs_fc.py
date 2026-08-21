"""
Diagnostic: does sparse neighbor selection reduce control cost vs fully-connected (FC)?

Runs 10 episodes each of FC, K=5, and K=10 in the ACS env with 20 agents, 1000 steps.
Reports mean/std of total_episode_return, final_vel_ent, final_sp_ent, mean_edges_per_agent.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from envs.env import Config, ControlConfig, EnvConfig, config_to_env_input, NeighborSelectionFlockingEnv
from baselines import create_baseline

NUM_AGENTS = 20
MAX_STEPS = 1000
NUM_EPISODES = 10
SEED_BASE = 12345


def make_env(seed_id=0):
    config = Config(
        control=ControlConfig(),
        env=EnvConfig(
            num_agents_pool=[NUM_AGENTS],
            max_time_steps=MAX_STEPS,
            is_training=False,
            observation_type="ego_centric",
            expose_aux_target=True,
            task_type="acs",
            use_fixed_episode_length=True,
        ),
    )
    env_input = config_to_env_input(config, seed_id=seed_id)
    return NeighborSelectionFlockingEnv(env_input)


def fc_policy(obs):
    """Fully connected: select all valid neighbors."""
    pm = obs["padding_mask"]
    nm = obs["neighbor_masks"]
    N = pm.shape[0]
    action = np.eye(N, dtype=np.int8)
    pm2d = pm[:, None] & pm[None, :]
    action[pm2d & nm] = 1
    return action


def run_episode(env, policy):
    obs = env.reset()
    total_reward = 0.0
    edge_counts = []
    last_vel_ent = None
    last_sp_ent = None

    for t in range(MAX_STEPS):
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Count edges per real agent (excluding self-loops)
        pm = obs["padding_mask"]
        N_real = int(pm.sum())
        if N_real > 0:
            real_block = action[pm][:, pm]
            edges = int(real_block.sum()) - N_real  # subtract self-loops
            edge_counts.append(edges / N_real)

        last_vel_ent = info.get("velocity_entropy")
        last_sp_ent = info.get("spatial_entropy")

        if done:
            break

    mean_edges = np.mean(edge_counts) if edge_counts else 0.0
    return total_reward, last_vel_ent, last_sp_ent, mean_edges


def run_condition(name, policy_fn):
    returns, vel_ents, sp_ents, edges_list = [], [], [], []
    for ep in range(NUM_EPISODES):
        env = make_env(seed_id=SEED_BASE + ep)
        ret, ve, se, me = run_episode(env, policy_fn)
        returns.append(ret)
        vel_ents.append(ve)
        sp_ents.append(se)
        edges_list.append(me)
        print(f"  [{name}] ep {ep}: return={ret:.2f}, vel_ent={ve:.4f}, sp_ent={se:.4f}, edges/agent={me:.2f}")

    print(f"\n  [{name}] SUMMARY ({NUM_EPISODES} episodes):")
    for label, vals in [("total_return", returns), ("final_vel_ent", vel_ents),
                        ("final_sp_ent", sp_ents), ("mean_edges/agent", edges_list)]:
        arr = np.array(vals)
        print(f"    {label:20s}: mean={arr.mean():.4f}  std={arr.std():.4f}")
    print()
    return returns, vel_ents, sp_ents, edges_list


if __name__ == "__main__":
    print("=" * 70)
    print("Sparse vs FC control cost diagnostic")
    print(f"Agents={NUM_AGENTS}, Steps={MAX_STEPS}, Episodes={NUM_EPISODES}")
    print("=" * 70)

    # 1. Fully connected
    print("\n--- Fully Connected (FC) ---")
    fc_results = run_condition("FC", fc_policy)

    # 2. K=5
    print("\n--- K-Nearest (K=5) ---")
    knn5 = create_baseline("nearest", k=5)
    knn5_results = run_condition("K=5", knn5)

    # 3. K=10
    print("\n--- K-Nearest (K=10) ---")
    knn10 = create_baseline("nearest", k=10)
    knn10_results = run_condition("K=10", knn10)

    # Final comparison table
    print("=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    header = f"{'Condition':>12s} | {'return':>12s} | {'vel_ent':>12s} | {'sp_ent':>12s} | {'edges/agent':>12s}"
    print(header)
    print("-" * len(header))
    for name, (rets, ves, ses, eds) in [("FC", fc_results), ("K=5", knn5_results), ("K=10", knn10_results)]:
        r, v, s, e = np.mean(rets), np.mean(ves), np.mean(ses), np.mean(eds)
        rs, vs, ss, es = np.std(rets), np.std(ves), np.std(ses), np.std(eds)
        print(f"{name:>12s} | {r:>5.2f}+{rs:>4.2f} | {v:>5.4f}+{vs:.4f} | {s:>5.2f}+{ss:>4.2f} | {e:>5.2f}+{es:>4.2f}")
    print("=" * 70)
