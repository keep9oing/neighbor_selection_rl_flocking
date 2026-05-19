import argparse
import csv
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from envs.env import NeighborSelectionFlockingEnv, load_config
from models.ppo_radius import NeighborSelectionRadiusPPORLlib
from radius_checkpoint_config import load_config_from_checkpoint


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    config.pop("smoke", None)
    return config


def make_env(config: Dict[str, Any], action_type: str, seed_id: Optional[int], overrides: Optional[Dict[str, Any]] = None):
    default_config = load_config("./envs/default_env_config.yaml")
    for key, value in config.get("env", {}).items():
        setattr(default_config.env, key, value)
    for key, value in config.get("control", {}).items():
        setattr(default_config.control, key, value)
    default_config.env.action_type = action_type
    default_config.env.is_training = False
    default_config.env.get_action_hist = False
    default_config.env.get_state_hist = False
    if overrides:
        for key, value in overrides.items():
            if hasattr(default_config.env, key):
                setattr(default_config.env, key, value)
            elif hasattr(default_config.control, key):
                setattr(default_config.control, key, value)
            else:
                raise ValueError(f"Unknown config override: {key}")
    return NeighborSelectionFlockingEnv({"seed_id": seed_id, "config": default_config.dict()})


class FixedNormalizedRadiusPolicy:
    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return np.full(obs["padding_mask"].shape, self.value, dtype=np.float32)


class FixedPhysicalRadiusPolicy:
    def __init__(self, radius: float, radius_min: float, radius_max: float):
        self.action_value = 0.0 if radius_max <= radius_min else (float(radius) - radius_min) / (radius_max - radius_min)

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return np.full(obs["padding_mask"].shape, self.action_value, dtype=np.float32)


class AllNeighborPolicy:
    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        padding_mask = obs["padding_mask"]
        padding_mask_2d = padding_mask[:, None] & padding_mask[None, :]
        return (obs["neighbor_masks"] & padding_mask_2d).astype(np.int8)


class RadiusRLPolicy:
    def __init__(self, checkpoint_path: str, config: Dict[str, Any], env):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_outputs = int(np.prod(env.action_space.shape) * 2)
        self.model = NeighborSelectionRadiusPPORLlib(
            obs_space=env.observation_space,
            action_space=env.action_space,
            num_outputs=num_outputs,
            model_config={"custom_model_config": config.get("model", {})},
            name="radius_policy",
        ).to(self.device)
        self._load_weights(checkpoint_path)
        self.model.eval()

    def _load_weights(self, checkpoint_path: str):
        policy_state_path = os.path.join(checkpoint_path, "policies", "default_policy", "policy_state.pkl")
        with open(policy_state_path, "rb") as f:
            state = pickle.load(f)
        weights = state["weights"]
        torch_state = {
            key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for key, value in weights.items()
        }
        self.model.load_state_dict(torch_state, strict=False)

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
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
            mean = output[:, :action_dim]
            return torch.clamp(mean, 0.0, 1.0).cpu().numpy()[0].astype(np.float32)


def has_converged(info: Dict[str, Any], env) -> bool:
    if env.config.env.task_type != "acs":
        return False
    spatial_entropy = info.get("spatial_entropy")
    velocity_entropy = info.get("velocity_entropy")
    if spatial_entropy is None or velocity_entropy is None:
        return False
    return spatial_entropy < env.config.env.entropy_p_goal and velocity_entropy < env.config.env.entropy_v_goal


def run_episode(env, policy) -> Dict[str, Any]:
    obs = env.reset()
    done = False
    episode_return = 0.0
    original_episode_return = 0.0
    first_converged_step = None
    radius_means: List[float] = []
    selected_counts: List[float] = []
    steps = 0
    last_info: Dict[str, Any] = {}

    while not done:
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        steps += 1
        episode_return += float(reward)
        original_episode_return += float(info.get("original_reward", 0.0))
        if first_converged_step is None and has_converged(info, env):
            first_converged_step = steps
        if info.get("radius_mean") is not None:
            radius_means.append(float(info["radius_mean"]))
        if info.get("selected_neighbor_count_mean") is not None:
            selected_counts.append(float(info["selected_neighbor_count_mean"]))
        last_info = info

    max_steps = env.config.env.max_time_steps
    return {
        "first_converged_step": first_converged_step if first_converged_step is not None else max_steps,
        "success": first_converged_step is not None,
        "episode_length": steps,
        "episode_return": episode_return,
        "original_episode_return": original_episode_return,
        "final_spatial_entropy": last_info.get("spatial_entropy"),
        "final_velocity_entropy": last_info.get("velocity_entropy"),
        "mean_radius": float(np.mean(radius_means)) if radius_means else np.nan,
        "std_radius": float(np.std(radius_means)) if radius_means else np.nan,
        "mean_selected_neighbor_count": float(np.mean(selected_counts)) if selected_counts else np.nan,
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {}
    keys = rows[0].keys()
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        if np.isnan(values).all():
            summary[f"{key}_mean"] = np.nan
            summary[f"{key}_std"] = np.nan
        else:
            summary[f"{key}_mean"] = float(np.nanmean(values))
            summary[f"{key}_std"] = float(np.nanstd(values))
    return summary


def evaluate_policy(name: str, env, policy, num_episodes: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    for episode_idx in range(num_episodes):
        result = run_episode(env, policy)
        result["policy"] = name
        result["episode"] = episode_idx
        rows.append(result)
    return rows, summarize([{k: v for k, v in row.items() if k not in ("policy", "episode")} for row in rows])


def write_outputs(output_dir: str, episode_rows: List[Dict[str, Any]], summaries: Dict[str, Dict[str, Any]]):
    os.makedirs(output_dir, exist_ok=True)
    episode_path = os.path.join(output_dir, "radius_evaluation_episodes.csv")
    with open(episode_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

    summary_path = os.path.join(output_dir, "radius_evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"Wrote episode metrics to {episode_path}")
    print(f"Wrote summary metrics to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate radius-action policies and baselines.")
    parser.add_argument("--config", default="configs/radius_action_train.yaml")
    parser.add_argument("--checkpoint", default=None, help="RLlib checkpoint directory for learned radius policy.")
    parser.add_argument(
        "--use-checkpoint-config",
        action="store_true",
        help="Load env/control/model settings from the checkpoint trial's params.json.",
    )
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./results/radius_evaluation")
    parser.add_argument("--baselines-only", action="store_true")
    args = parser.parse_args()

    if args.use_checkpoint_config:
        if args.checkpoint is None:
            raise ValueError("--use-checkpoint-config requires --checkpoint")
        config = load_config_from_checkpoint(args.checkpoint)
    else:
        config = load_yaml_config(args.config)
    env_for_radius = make_env(config, "radius", args.seed)
    radius_min = env_for_radius.config.env.radius_min
    radius_max = env_for_radius.config.env.radius_max
    r0 = env_for_radius.config.control.r0

    policies = []
    for value in (0.25, 0.5, 0.75, 1.0):
        policies.append((f"fixed_normalized_{value:g}", "radius", FixedNormalizedRadiusPolicy(value)))
    for radius in (r0, 2 * r0, 4 * r0):
        policies.append((f"fixed_physical_{radius:g}", "radius", FixedPhysicalRadiusPolicy(radius, radius_min, radius_max)))
    policies.append(("all_neighbor", "binary_vector", AllNeighborPolicy()))

    if not args.baselines_only:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required unless --baselines-only is set")
        policies.insert(0, ("learned_radius", "radius", RadiusRLPolicy(args.checkpoint, config, env_for_radius)))

    all_rows = []
    summaries = {}
    for name, action_type, policy in policies:
        env = make_env(config, action_type, args.seed)
        rows, summary = evaluate_policy(name, env, policy, args.num_episodes)
        all_rows.extend(rows)
        summaries[name] = summary
        print(f"{name}: success_rate={summary['success_mean']:.3f}, return={summary['episode_return_mean']:.3f}")

    write_outputs(args.output_dir, all_rows, summaries)


if __name__ == "__main__":
    main()
