import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from make_radius_vs_acs_media import (
    LearnedRadiusPolicy,
    collect_rollout,
    find_trial_dir,
    load_trial_config,
    make_env,
    select_best_checkpoint,
)


def json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def save_radius_csv(radius_by_step: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + [f"agent_{idx:02d}" for idx in range(radius_by_step.shape[1])])
        for step, row in enumerate(radius_by_step):
            writer.writerow([step] + ["" if np.isnan(value) else f"{value:.8f}" for value in row])


def plot_episode_radius_trend(
    rollout: Dict[str, Any],
    output_path: Path,
    radius_min: float,
    radius_max: float,
    color_vmin: float,
    color_vmax: float,
    dpi: int,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    radii = np.asarray(rollout["radii"], dtype=np.float64)
    active_masks = np.asarray(rollout["masks"][:-1], dtype=bool)
    radius_by_step = radii.copy()
    radius_by_step[~active_masks] = np.nan

    num_steps, num_agents = radius_by_step.shape
    agent_means = np.nanmean(radius_by_step, axis=0)
    agent_stds = np.nanstd(radius_by_step, axis=0)
    global_mean = float(np.nanmean(radius_by_step))
    global_std = float(np.nanstd(radius_by_step))

    fig = plt.figure(figsize=(13, 8.5), dpi=dpi, constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.45], hspace=0.32)

    ax_heatmap = fig.add_subplot(gs[0])
    image = ax_heatmap.imshow(
        radius_by_step.T,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="viridis",
        vmin=color_vmin,
        vmax=color_vmax,
        extent=[0, num_steps - 1, -0.5, num_agents - 0.5],
    )
    ax_heatmap.set_title(
        f"learned_radius_best | seed={rollout['seed']} | "
        f"train={rollout['train_return']:.2f} | original={rollout['original_return']:.2f} | "
        f"mean radius={global_mean:.2f} +/- {global_std:.2f}",
        fontsize=10,
    )
    ax_heatmap.set_xlabel("step")
    ax_heatmap.set_ylabel("agent id")
    ax_heatmap.set_yticks(np.arange(num_agents))
    ax_heatmap.grid(False)
    colorbar = fig.colorbar(image, ax=ax_heatmap, pad=0.01)
    colorbar.set_label(f"selected disk radius ({color_vmin:.1f}-{color_vmax:.1f})")

    ax_bar = fig.add_subplot(gs[1])
    agent_ids = np.arange(num_agents)
    colors = plt.get_cmap("tab20")(agent_ids % 20)
    ax_bar.bar(agent_ids, agent_means, yerr=agent_stds, color=colors, alpha=0.85, capsize=2.5)
    ax_bar.axhline(global_mean, color="#111827", linewidth=1.0, linestyle="--", label="episode mean")
    ax_bar.set_xlim(-0.7, num_agents - 0.3)
    ax_bar.set_ylim(max(0.0, radius_min - 3.0), radius_max + 3.0)
    ax_bar.set_xlabel("agent id")
    ax_bar.set_ylabel("mean radius")
    ax_bar.set_xticks(agent_ids)
    ax_bar.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax_bar.legend(loc="upper right", frameon=False)

    fig.savefig(output_path)
    plt.close(fig)
    return radius_by_step, agent_means, agent_stds


def main():
    parser = argparse.ArgumentParser(description="Plot learned-radius per-agent radius trends.")
    parser.add_argument("--experiment-dir", default="ray_results/radius_action_20agents_2gpu_aggressive_5m")
    parser.add_argument("--metric", default="episode_reward_mean")
    parser.add_argument("--mode", choices=("max", "min"), default="max")
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=20260520)
    parser.add_argument("--output-dir", default="results/radius_vs_acs_best_checkpoint/media/radius_trends")
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--color-vmin", type=float, default=None)
    parser.add_argument("--color-vmax", type=float, default=None)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    trial_dir = find_trial_dir(experiment_dir)
    best = select_best_checkpoint(trial_dir, args.metric, args.mode)
    checkpoint_dir = Path(best["checkpoint"])
    config = load_trial_config(trial_dir)

    probe_env = make_env(config, "radius", args.base_seed, is_training=True)
    learned_policy = LearnedRadiusPolicy(checkpoint_dir, config, probe_env)
    radius_min = float(probe_env.config.env.radius_min)
    radius_max = float(probe_env.config.env.radius_max)
    color_vmin = radius_min if args.color_vmin is None else float(args.color_vmin)
    color_vmax = radius_max if args.color_vmax is None else float(args.color_vmax)
    if color_vmax <= color_vmin:
        raise ValueError("--color-vmax must be greater than --color-vmin")

    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "plots"
    csv_dir = output_dir / "csv"
    records = []

    for episode_idx in range(args.num_episodes):
        seed = args.base_seed + episode_idx
        rollout = collect_rollout(config, "learned_radius_best", learned_policy, "radius", seed)
        stem = f"episode_{episode_idx + 1:03d}_seed_{seed}_learned_radius_best_radius_trend"
        plot_path = plot_dir / f"{stem}.png"
        csv_path = csv_dir / f"{stem}.csv"

        radius_by_step, agent_means, agent_stds = plot_episode_radius_trend(
            rollout=rollout,
            output_path=plot_path,
            radius_min=radius_min,
            radius_max=radius_max,
            color_vmin=color_vmin,
            color_vmax=color_vmax,
            dpi=args.dpi,
        )
        save_radius_csv(radius_by_step, csv_path)

        record = {
            "episode": episode_idx,
            "seed": seed,
            "plot_path": str(plot_path),
            "csv_path": str(csv_path),
            "train_return": rollout["train_return"],
            "original_return": rollout["original_return"],
            "episode_length": rollout["episode_length"],
            "mean_radius": float(np.nanmean(radius_by_step)),
            "std_radius": float(np.nanstd(radius_by_step)),
            "min_radius": float(np.nanmin(radius_by_step)),
            "max_radius": float(np.nanmax(radius_by_step)),
            "agent_mean_radius": agent_means,
            "agent_std_radius": agent_stds,
        }
        records.append(record)
        print(f"wrote {plot_path} and {csv_path}", flush=True)

    summary = {
        "best_checkpoint": best,
        "num_episodes": args.num_episodes,
        "base_seed": args.base_seed,
        "configured_radius_min": radius_min,
        "configured_radius_max": radius_max,
        "color_vmin": color_vmin,
        "color_vmax": color_vmax,
        "records": records,
    }
    summary_path = output_dir / "radius_trend_summary.json"
    with summary_path.open("w") as f:
        json.dump(json_safe(summary), f, indent=2)
    print(f"Wrote radius trend summary to {summary_path}")


if __name__ == "__main__":
    main()
