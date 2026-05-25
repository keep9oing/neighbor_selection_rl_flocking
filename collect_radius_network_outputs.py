import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from make_radius_vs_acs_media import (
    LearnedRadiusPolicy,
    find_trial_dir,
    load_trial_config,
    make_env,
    radius_selection,
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


class LearnedRadiusPolicyWithRawOutput(LearnedRadiusPolicy):
    def forward_raw(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            input_dict = {
                "obs": {
                    "local_agent_infos": torch.from_numpy(obs["local_agent_infos"][None]).float().to(self.device),
                    "neighbor_masks": torch.from_numpy(obs["neighbor_masks"][None]).float().to(self.device),
                    "padding_mask": torch.from_numpy(obs["padding_mask"][None]).float().to(self.device),
                    "is_from_my_env": torch.ones((1,), dtype=torch.bool, device=self.device),
                }
            }
            output, _ = self.model.forward(input_dict, state=[], seq_lens=None)
            action_dim = obs["padding_mask"].shape[0]
            mean = output[:, :action_dim].cpu().numpy()[0].astype(np.float64)
            log_std = output[:, action_dim:].cpu().numpy()[0].astype(np.float64)
            clipped_action = np.clip(mean, 0.0, 1.0)
            return {
                "raw_mean": mean,
                "log_std": log_std,
                "std": np.exp(log_std),
                "clipped_action": clipped_action,
            }


def write_step_csv(path: Path, values: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + [f"agent_{idx:02d}" for idx in range(values.shape[1])])
        for step, row in enumerate(values):
            writer.writerow([step] + ["" if np.isnan(value) else f"{value:.8f}" for value in row])


def collect_episode_outputs(config: Dict[str, Any], policy: LearnedRadiusPolicyWithRawOutput, seed: int):
    env = make_env(config, "radius", seed, is_training=True)
    obs = env.reset()
    done = False

    raw_means = []
    clipped_actions = []
    log_stds = []
    stds = []
    physical_radii = []
    masks = []
    train_rewards = []
    original_rewards = []

    while not done:
        state_before = {
            "agent_states": np.asarray(env.state["agent_states"]).copy(),
            "neighbor_masks": np.asarray(env.state["neighbor_masks"]).copy(),
            "padding_mask": np.asarray(env.state["padding_mask"]).copy(),
        }
        padding_mask = state_before["padding_mask"].astype(bool)
        output = policy.forward_raw(obs)
        selection = radius_selection(env, state_before, output["clipped_action"])

        raw_mean = output["raw_mean"].copy()
        clipped_action = output["clipped_action"].copy()
        log_std = output["log_std"].copy()
        std = output["std"].copy()
        physical_radius = selection["radii"].copy()

        raw_mean[~padding_mask] = np.nan
        clipped_action[~padding_mask] = np.nan
        log_std[~padding_mask] = np.nan
        std[~padding_mask] = np.nan
        physical_radius[~padding_mask] = np.nan

        raw_means.append(raw_mean)
        clipped_actions.append(clipped_action)
        log_stds.append(log_std)
        stds.append(std)
        physical_radii.append(physical_radius)
        masks.append(padding_mask)

        obs, reward, done, info = env.step(output["clipped_action"].astype(np.float32))
        train_rewards.append(float(reward))
        original_rewards.append(float(info.get("original_reward", reward)))

    return {
        "raw_mean": np.asarray(raw_means),
        "clipped_action": np.asarray(clipped_actions),
        "log_std": np.asarray(log_stds),
        "std": np.asarray(stds),
        "physical_radius": np.asarray(physical_radii),
        "mask": np.asarray(masks),
        "train_return": float(np.sum(train_rewards)),
        "original_return": float(np.sum(original_rewards)),
        "episode_length": int(len(train_rewards)),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect raw learned-radius network outputs per timestep.")
    parser.add_argument("--experiment-dir", default="ray_results/radius_action_20agents_2gpu_aggressive_5m")
    parser.add_argument("--metric", default="episode_reward_mean")
    parser.add_argument("--mode", choices=("max", "min"), default="max")
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=20260520)
    parser.add_argument("--output-dir", default="results/radius_vs_acs_best_checkpoint/media/network_outputs")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    trial_dir = find_trial_dir(experiment_dir)
    best = select_best_checkpoint(trial_dir, args.metric, args.mode)
    checkpoint_dir = Path(best["checkpoint"])
    config = load_trial_config(trial_dir)

    probe_env = make_env(config, "radius", args.base_seed, is_training=True)
    policy = LearnedRadiusPolicyWithRawOutput(checkpoint_dir, config, probe_env)
    radius_min = float(probe_env.config.env.radius_min)
    radius_max = float(probe_env.config.env.radius_max)

    output_dir = Path(args.output_dir)
    csv_dirs = {
        "raw_mean": output_dir / "csv" / "raw_mean",
        "clipped_action": output_dir / "csv" / "clipped_action",
        "log_std": output_dir / "csv" / "log_std",
        "std": output_dir / "csv" / "std",
        "physical_radius": output_dir / "csv" / "physical_radius",
    }
    npz_dir = output_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for episode_idx in range(args.num_episodes):
        seed = args.base_seed + episode_idx
        episode = collect_episode_outputs(config, policy, seed)
        stem = f"episode_{episode_idx + 1:03d}_seed_{seed}_learned_radius_best"

        paths = {}
        for key, values in episode.items():
            if key in csv_dirs:
                path = csv_dirs[key] / f"{stem}_{key}.csv"
                write_step_csv(path, values)
                paths[f"{key}_csv_path"] = str(path)

        npz_path = npz_dir / f"{stem}_network_outputs.npz"
        np.savez_compressed(
            npz_path,
            raw_mean=episode["raw_mean"],
            clipped_action=episode["clipped_action"],
            log_std=episode["log_std"],
            std=episode["std"],
            physical_radius=episode["physical_radius"],
            mask=episode["mask"],
        )

        record = {
            "episode": episode_idx,
            "seed": seed,
            "npz_path": str(npz_path),
            "train_return": episode["train_return"],
            "original_return": episode["original_return"],
            "episode_length": episode["episode_length"],
            "raw_mean_min": float(np.nanmin(episode["raw_mean"])),
            "raw_mean_mean": float(np.nanmean(episode["raw_mean"])),
            "raw_mean_max": float(np.nanmax(episode["raw_mean"])),
            "clipped_action_min": float(np.nanmin(episode["clipped_action"])),
            "clipped_action_mean": float(np.nanmean(episode["clipped_action"])),
            "clipped_action_max": float(np.nanmax(episode["clipped_action"])),
            "log_std_min": float(np.nanmin(episode["log_std"])),
            "log_std_mean": float(np.nanmean(episode["log_std"])),
            "log_std_max": float(np.nanmax(episode["log_std"])),
            "std_min": float(np.nanmin(episode["std"])),
            "std_mean": float(np.nanmean(episode["std"])),
            "std_max": float(np.nanmax(episode["std"])),
            **paths,
        }
        records.append(record)
        print(
            f"wrote {stem}: raw_mean {record['raw_mean_min']:.4f}-"
            f"{record['raw_mean_max']:.4f}, log_std {record['log_std_min']:.4f}-"
            f"{record['log_std_max']:.4f}",
            flush=True,
        )

    summary = {
        "best_checkpoint": best,
        "num_episodes": args.num_episodes,
        "base_seed": args.base_seed,
        "radius_min": radius_min,
        "radius_max": radius_max,
        "note": "raw_mean is the model Gaussian mean before action clipping; clipped_action is clip(raw_mean, 0, 1).",
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "network_outputs_summary.json"
    with summary_path.open("w") as f:
        json.dump(json_safe(summary), f, indent=2)
    print(f"Wrote network output summary to {summary_path}")


if __name__ == "__main__":
    main()
