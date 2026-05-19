"""
Evaluate an always-select-all-neighbors policy on multiple random environments.

This script:
1. Rolls out the environment for multiple episodes with different seeds.
2. Uses a policy that selects every valid neighbor allowed by the environment.
3. Saves per-step rewards and per-episode summaries.
4. Plots reward curves across random environments.

Example:
    python plot_all_neighbors_rewards.py --num_episodes 30 --base_seed 42
"""

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from envs.env import NeighborSelectionFlockingEnv, load_config


class AllNeighborPolicy:
    """Select all valid neighbors, including self-loops for active agents only."""

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        neighbor_masks = obs["neighbor_masks"]
        padding_mask = obs["padding_mask"]

        padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]
        action = (neighbor_masks & padding_mask_2d).astype(np.int8)
        np.fill_diagonal(action, padding_mask.astype(np.int8))
        return action


def create_env(
    config_path: str,
    seed: int,
    num_agents: int = None,
    max_steps: int = None,
    use_fixed_episode_length: bool = True,
    use_training_reward: bool = True,
) -> NeighborSelectionFlockingEnv:
    config = load_config(config_path)
    config.env.is_training = use_training_reward
    config.env.get_state_hist = False
    config.env.get_action_hist = False

    if num_agents is not None:
        config.env.num_agents_pool = [num_agents]
    if max_steps is not None:
        config.env.max_time_steps = max_steps
    config.env.use_fixed_episode_length = use_fixed_episode_length

    env_context = {"seed_id": seed, "config": config.dict()}
    return NeighborSelectionFlockingEnv(env_context)


def run_episode(
    env: NeighborSelectionFlockingEnv,
    policy: AllNeighborPolicy,
    collect_trajectory: bool = True,
) -> Dict[str, object]:
    obs = env.reset()
    done = False

    rewards: List[float] = []
    original_rewards: List[float] = []
    spatial_entropies: List[float] = []
    velocity_entropies: List[float] = []
    trajectory = [] if collect_trajectory else None

    if collect_trajectory:
        positions = env.state["agent_states"][:, :2].copy()
        padding_mask = env.state["padding_mask"].copy()
        trajectory.append((positions, padding_mask))

    while not done:
        action = policy(obs)
        obs, reward, done, info = env.step(action)

        rewards.append(float(reward))
        original_rewards.append(float(info.get("original_reward", reward)))

        if collect_trajectory:
            positions = env.state["agent_states"][:, :2].copy()
            padding_mask = env.state["padding_mask"].copy()
            trajectory.append((positions, padding_mask))

        if info.get("spatial_entropy") is not None:
            spatial_entropies.append(float(info["spatial_entropy"]))
        if info.get("velocity_entropy") is not None:
            velocity_entropies.append(float(info["velocity_entropy"]))

    reward_array = np.asarray(rewards, dtype=np.float64)
    original_reward_array = np.asarray(original_rewards, dtype=np.float64)

    result = {
        "rewards": reward_array,
        "original_rewards": original_reward_array,
        "episode_return": float(reward_array.sum()),
        "tensorboard_episode_reward_mean_value": float(reward_array.sum()),
        "original_episode_return": float(original_reward_array.sum()),
        "episode_length": int(len(reward_array)),
        "mean_step_reward": float(reward_array.mean()) if len(reward_array) > 0 else np.nan,
        "mean_original_step_reward": float(original_reward_array.mean()) if len(original_reward_array) > 0 else np.nan,
        "num_agents": int(env.num_agents),
        "spatial_entropy_final": float(spatial_entropies[-1]) if spatial_entropies else np.nan,
        "velocity_entropy_final": float(velocity_entropies[-1]) if velocity_entropies else np.nan,
    }

    if collect_trajectory:
        result["trajectory"] = trajectory

    return result


def plot_single_trajectory(trajectory, episode_idx: int, save_dir: str, title: str, episode_return: float) -> str:
    fig, ax = plt.subplots(figsize=(8, 7))

    num_agents = int(np.sum(trajectory[0][1]))
    colors = plt.cm.tab20(np.linspace(0, 1, max(num_agents, 1)))
    padding_mask = trajectory[0][1]
    active_indices = np.where(padding_mask)[0]

    for color_index, agent_idx in enumerate(active_indices):
        agent_positions = np.array([time_step[0][agent_idx] for time_step in trajectory])
        ax.plot(agent_positions[:, 0], agent_positions[:, 1],
                color=colors[color_index % len(colors)], alpha=0.7, linewidth=0.9)
        ax.scatter(agent_positions[0, 0], agent_positions[0, 1],
                   color=colors[color_index % len(colors)], marker='o', s=40,
                   edgecolors='black', linewidths=0.4)
        ax.scatter(agent_positions[-1, 0], agent_positions[-1, 1],
                   color=colors[color_index % len(colors)], marker='^', s=65,
                   edgecolors='black', linewidths=0.4)

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(f"{title}\nEpisode {episode_idx + 1}, training return={episode_return:.2f}")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    file_name = f"trajectory_{title.lower().replace(' ', '_')}_episode_{episode_idx + 1:03d}.png"
    file_path = os.path.join(save_dir, file_name)
    fig.savefig(file_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return file_path


def stack_with_nan_padding(series_list: List[np.ndarray]) -> np.ndarray:
    max_len = max(len(series) for series in series_list)
    stacked = np.full((len(series_list), max_len), np.nan, dtype=np.float64)
    for index, series in enumerate(series_list):
        stacked[index, :len(series)] = series
    return stacked


def save_step_rewards_csv(output_dir: str, episode_records: List[Dict[str, object]]) -> str:
    file_path = os.path.join(output_dir, "episode_step_rewards.csv")
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "episode",
            "seed",
            "step",
            "training_reward",
            "original_reward",
            "cumulative_training_reward",
            "cumulative_original_reward",
            "num_agents",
        ])

        for episode_index, record in enumerate(episode_records):
            cumulative_reward = np.cumsum(record["rewards"])
            cumulative_original_reward = np.cumsum(record["original_rewards"])
            for step_index, reward in enumerate(record["rewards"]):
                writer.writerow([
                    episode_index,
                    record["seed"],
                    step_index,
                    float(reward),
                    float(record["original_rewards"][step_index]),
                    float(cumulative_reward[step_index]),
                    float(cumulative_original_reward[step_index]),
                    record["num_agents"],
                ])
    return file_path


def save_episode_summary_csv(output_dir: str, episode_records: List[Dict[str, object]]) -> str:
    file_path = os.path.join(output_dir, "episode_summary.csv")
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "episode",
            "seed",
            "num_agents",
            "episode_length",
            "episode_return",
            "tensorboard_episode_reward_mean_value",
            "mean_step_reward",
            "original_episode_return",
            "mean_original_step_reward",
            "spatial_entropy_final",
            "velocity_entropy_final",
        ])

        for episode_index, record in enumerate(episode_records):
            writer.writerow([
                episode_index,
                record["seed"],
                record["num_agents"],
                record["episode_length"],
                record["episode_return"],
                record["tensorboard_episode_reward_mean_value"],
                record["mean_step_reward"],
                record["original_episode_return"],
                record["mean_original_step_reward"],
                record["spatial_entropy_final"],
                record["velocity_entropy_final"],
            ])
    return file_path


def save_curve_summary_csv(output_dir: str, episode_records: List[Dict[str, object]]) -> str:
    reward_matrix = stack_with_nan_padding([record["rewards"] for record in episode_records])
    cumulative_matrix = stack_with_nan_padding([
        np.cumsum(record["rewards"]) for record in episode_records
    ])

    file_path = os.path.join(output_dir, "reward_curve_summary.csv")
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "step",
            "training_reward_mean",
            "training_reward_std",
            "training_reward_min",
            "training_reward_max",
            "cumulative_return_mean",
            "cumulative_return_std",
            "original_reward_mean",
            "original_reward_std",
            "num_active_episodes",
        ])

        original_reward_matrix = stack_with_nan_padding([record["original_rewards"] for record in episode_records])

        for step_index in range(reward_matrix.shape[1]):
            reward_step = reward_matrix[:, step_index]
            cumulative_step = cumulative_matrix[:, step_index]
            original_reward_step = original_reward_matrix[:, step_index]
            active_mask = ~np.isnan(reward_step)

            writer.writerow([
                step_index,
                float(np.nanmean(reward_step)),
                float(np.nanstd(reward_step)),
                float(np.nanmin(reward_step)),
                float(np.nanmax(reward_step)),
                float(np.nanmean(cumulative_step)),
                float(np.nanstd(cumulative_step)),
                float(np.nanmean(original_reward_step)),
                float(np.nanstd(original_reward_step)),
                int(active_mask.sum()),
            ])
    return file_path


def save_metadata(output_dir: str, args: argparse.Namespace, episode_records: List[Dict[str, object]]) -> str:
    episode_returns = [record["episode_return"] for record in episode_records]
    original_episode_returns = [record["original_episode_return"] for record in episode_records]
    episode_lengths = [record["episode_length"] for record in episode_records]

    file_path = os.path.join(output_dir, "metadata.json")
    with open(file_path, "w") as jsonfile:
        json.dump(
            {
                "created_at": datetime.now().isoformat(),
                "arguments": vars(args),
                "summary": {
                    "num_episodes": len(episode_records),
                    "tensorboard_episode_reward_mean": float(np.mean(episode_returns)),
                    "tensorboard_episode_reward_std": float(np.std(episode_returns)),
                    "episode_return_mean": float(np.mean(episode_returns)),
                    "episode_return_std": float(np.std(episode_returns)),
                    "original_episode_return_mean": float(np.mean(original_episode_returns)),
                    "original_episode_return_std": float(np.std(original_episode_returns)),
                    "episode_length_mean": float(np.mean(episode_lengths)),
                    "episode_length_std": float(np.std(episode_lengths)),
                },
            },
            jsonfile,
            indent=2,
        )
    return file_path


def plot_reward_curves(output_dir: str, episode_records: List[Dict[str, object]]) -> List[str]:
    reward_matrix = stack_with_nan_padding([record["rewards"] for record in episode_records])
    cumulative_matrix = stack_with_nan_padding([
        np.cumsum(record["rewards"]) for record in episode_records
    ])
    original_reward_matrix = stack_with_nan_padding([record["original_rewards"] for record in episode_records])
    original_cumulative_matrix = stack_with_nan_padding([
        np.cumsum(record["original_rewards"]) for record in episode_records
    ])

    steps = np.arange(reward_matrix.shape[1])
    reward_mean = np.nanmean(reward_matrix, axis=0)
    reward_std = np.nanstd(reward_matrix, axis=0)
    cumulative_mean = np.nanmean(cumulative_matrix, axis=0)
    cumulative_std = np.nanstd(cumulative_matrix, axis=0)
    original_reward_mean = np.nanmean(original_reward_matrix, axis=0)
    original_reward_std = np.nanstd(original_reward_matrix, axis=0)
    original_cumulative_mean = np.nanmean(original_cumulative_matrix, axis=0)
    original_cumulative_std = np.nanstd(original_cumulative_matrix, axis=0)

    saved_plots = []

    fig, ax = plt.subplots(figsize=(10, 6))
    for record in episode_records:
        ax.plot(record["rewards"], alpha=0.25, linewidth=1)
    ax.plot(steps, reward_mean, color="black", linewidth=2, label="mean training reward")
    ax.fill_between(steps, reward_mean - reward_std, reward_mean + reward_std,
                    color="gray", alpha=0.25, label="±1 std")
    ax.set_title("All-neighbor policy: per-step training reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Training reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    reward_plot_path = os.path.join(output_dir, "training_reward_per_step.png")
    fig.savefig(reward_plot_path, dpi=200)
    plt.close(fig)
    saved_plots.append(reward_plot_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    for record in episode_records:
        ax.plot(np.cumsum(record["rewards"]), alpha=0.25, linewidth=1)
    ax.plot(steps, cumulative_mean, color="tab:blue", linewidth=2,
            label="mean cumulative training return")
    ax.fill_between(steps, cumulative_mean - cumulative_std, cumulative_mean + cumulative_std,
                    color="tab:blue", alpha=0.2, label="±1 std")
    ax.set_title("All-neighbor policy: cumulative training return")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative training return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    cumulative_plot_path = os.path.join(output_dir, "cumulative_training_return.png")
    fig.savefig(cumulative_plot_path, dpi=200)
    plt.close(fig)
    saved_plots.append(cumulative_plot_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    for record in episode_records:
        ax.plot(record["original_rewards"], alpha=0.2, linewidth=1)
    ax.plot(steps, original_reward_mean, color="tab:orange", linewidth=2, label="mean original reward")
    ax.fill_between(steps, original_reward_mean - original_reward_std, original_reward_mean + original_reward_std,
                    color="tab:orange", alpha=0.2, label="±1 std")
    ax.set_title("All-neighbor policy: per-step original reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Original reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    original_reward_plot_path = os.path.join(output_dir, "original_reward_per_step.png")
    fig.savefig(original_reward_plot_path, dpi=200)
    plt.close(fig)
    saved_plots.append(original_reward_plot_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    for record in episode_records:
        ax.plot(np.cumsum(record["original_rewards"]), alpha=0.2, linewidth=1)
    ax.plot(steps, original_cumulative_mean, color="tab:red", linewidth=2,
            label="mean cumulative original return")
    ax.fill_between(steps, original_cumulative_mean - original_cumulative_std,
                    original_cumulative_mean + original_cumulative_std,
                    color="tab:red", alpha=0.2, label="±1 std")
    ax.set_title("All-neighbor policy: cumulative original return")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative original return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    original_cumulative_plot_path = os.path.join(output_dir, "cumulative_original_return.png")
    fig.savefig(original_cumulative_plot_path, dpi=200)
    plt.close(fig)
    saved_plots.append(original_cumulative_plot_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    episode_returns = [record["episode_return"] for record in episode_records]
    ax.hist(episode_returns, bins=min(20, max(5, len(episode_returns))), color="tab:green", alpha=0.8)
    ax.set_title("TensorBoard-style episode reward distribution")
    ax.set_xlabel("Episode reward (training return)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    hist_plot_path = os.path.join(output_dir, "tensorboard_episode_reward_histogram.png")
    fig.savefig(hist_plot_path, dpi=200)
    plt.close(fig)
    saved_plots.append(hist_plot_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    original_episode_returns = [record["original_episode_return"] for record in episode_records]
    ax.hist(original_episode_returns, bins=min(20, max(5, len(original_episode_returns))),
            color="tab:purple", alpha=0.8)
    ax.set_title("Original episode return distribution")
    ax.set_xlabel("Original episode return")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    original_hist_plot_path = os.path.join(output_dir, "original_episode_return_histogram.png")
    fig.savefig(original_hist_plot_path, dpi=200)
    plt.close(fig)
    saved_plots.append(original_hist_plot_path)

    return saved_plots


def build_output_dir(base_dir: str) -> str:
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"all_neighbors_reward_{timestamp}")
    os.makedirs(output_dir, exist_ok=False)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all-neighbor baseline on multiple random environments and plot rewards."
    )
    parser.add_argument("--config", type=str, default="envs/default_env_config.yaml",
                        help="Path to environment config yaml")
    parser.add_argument("--num_episodes", type=int, default=30,
                        help="Number of random environments / episodes to evaluate")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base seed. Episode i uses base_seed + i")
    parser.add_argument("--num_agents", type=int, default=None,
                        help="Override num_agents_pool with a fixed agent count")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max_time_steps")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Base directory where results folder will be created")
    parser.add_argument("--use_original_reward_only", action="store_true",
                        help="Disable training reward and analyze only original_reward")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = build_output_dir(args.output_dir)

    policy = AllNeighborPolicy()
    episode_records: List[Dict[str, object]] = []

    print("=" * 72)
    print("All-neighbor policy evaluation")
    print("=" * 72)
    print(f"Config: {args.config}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Base seed: {args.base_seed}")
    print(f"Training reward mode: {not args.use_original_reward_only}")
    print("Fixed episode length: True")
    print(f"Output directory: {output_dir}")

    for episode_index in range(args.num_episodes):
        seed = args.base_seed + episode_index
        env = create_env(
            config_path=args.config,
            seed=seed,
            num_agents=args.num_agents,
            max_steps=args.max_steps,
            use_fixed_episode_length=True,
            use_training_reward=not args.use_original_reward_only,
        )
        result = run_episode(env, policy, collect_trajectory=True)
        result["seed"] = seed
        episode_records.append(result)

        print(
            f"[Episode {episode_index + 1:03d}/{args.num_episodes:03d}] "
            f"seed={seed}, agents={result['num_agents']}, "
            f"length={result['episode_length']}, "
            f"tb_episode_reward={result['tensorboard_episode_reward_mean_value']:.4f}, "
            f"tb_mean_step_reward={result['mean_step_reward']:.6f}, "
            f"original_episode_return={result['original_episode_return']:.4f}"
        )

    step_csv_path = save_step_rewards_csv(output_dir, episode_records)
    episode_csv_path = save_episode_summary_csv(output_dir, episode_records)
    curve_csv_path = save_curve_summary_csv(output_dir, episode_records)
    metadata_path = save_metadata(output_dir, args, episode_records)
    plot_paths = plot_reward_curves(output_dir, episode_records)

    best_episode_index = int(np.argmax([record["episode_return"] for record in episode_records]))
    worst_episode_index = int(np.argmin([record["episode_return"] for record in episode_records]))
    best_trajectory_path = plot_single_trajectory(
        episode_records[best_episode_index]["trajectory"],
        best_episode_index,
        output_dir,
        "Best Training Return",
        episode_records[best_episode_index]["episode_return"],
    )
    worst_trajectory_path = plot_single_trajectory(
        episode_records[worst_episode_index]["trajectory"],
        worst_episode_index,
        output_dir,
        "Worst Training Return",
        episode_records[worst_episode_index]["episode_return"],
    )
    plot_paths.extend([best_trajectory_path, worst_trajectory_path])

    episode_returns = np.array([record["episode_return"] for record in episode_records], dtype=np.float64)
    original_episode_returns = np.array([record["original_episode_return"] for record in episode_records], dtype=np.float64)
    print("\nFinished.")
    print(f"TensorBoard-style episode_reward_mean: {episode_returns.mean():.4f} ± {episode_returns.std():.4f}")
    print(f"Mean training return per episode: {episode_returns.mean():.4f}")
    print(f"Mean training reward per step: {np.mean([record['mean_step_reward'] for record in episode_records]):.6f}")
    print(f"Original episode return mean ± std: {original_episode_returns.mean():.4f} ± {original_episode_returns.std():.4f}")
    print(f"Best training-return episode: #{best_episode_index + 1} ({episode_records[best_episode_index]['episode_return']:.4f})")
    print(f"Worst training-return episode: #{worst_episode_index + 1} ({episode_records[worst_episode_index]['episode_return']:.4f})")
    print(f"Saved step rewards: {step_csv_path}")
    print(f"Saved episode summary: {episode_csv_path}")
    print(f"Saved curve summary: {curve_csv_path}")
    print(f"Saved metadata: {metadata_path}")
    print("Saved plots:")
    for plot_path in plot_paths:
        print(f"  - {plot_path}")


if __name__ == "__main__":
    main()