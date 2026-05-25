import argparse
import csv
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from envs.env import NeighborSelectionFlockingEnv, load_config
from models.ppo_radius import NeighborSelectionRadiusPPORLlib


def checkpoint_iteration(path: Path) -> Optional[int]:
    try:
        return int(path.name.split("_", 1)[1])
    except Exception:
        return None


def find_trial_dir(experiment_dir: Path) -> Path:
    trial_dirs = [
        p
        for p in experiment_dir.iterdir()
        if p.is_dir() and (p / "progress.csv").exists() and (p / "params.json").exists()
    ]
    if not trial_dirs:
        raise FileNotFoundError(f"No Ray trial dirs found under {experiment_dir}")
    if len(trial_dirs) > 1:
        trial_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return trial_dirs[0]


def select_best_checkpoint(trial_dir: Path, metric: str, mode: str) -> Dict[str, Any]:
    checkpoints = {
        checkpoint_iteration(p): p
        for p in trial_dir.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint_") and checkpoint_iteration(p) is not None
    }
    best = None
    with (trial_dir / "progress.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iteration = int(float(row["training_iteration"]))
            if iteration not in checkpoints:
                continue
            try:
                value = float(row[metric])
            except Exception:
                continue
            candidate = {
                "checkpoint": str(checkpoints[iteration]),
                "iteration": iteration,
                "metric": metric,
                "metric_value": value,
            }
            if best is None:
                best = candidate
            elif mode == "max" and value > best["metric_value"]:
                best = candidate
            elif mode == "min" and value < best["metric_value"]:
                best = candidate
    if best is None:
        raise FileNotFoundError(f"No checkpoint matched metric {metric} under {trial_dir}")
    return best


def load_trial_config(trial_dir: Path) -> Dict[str, Any]:
    with (trial_dir / "params.json").open() as f:
        params = json.load(f)
    env_config = params["env_config"]["config"]
    return {
        "env": env_config["env"],
        "control": env_config["control"],
        "model": params["model"]["custom_model_config"],
        "params": params,
    }


def make_env(config: Dict[str, Any], action_type: str, seed: int, is_training: bool = True):
    default_config = load_config("./envs/default_env_config.yaml")
    for key, value in config["env"].items():
        setattr(default_config.env, key, value)
    for key, value in config["control"].items():
        setattr(default_config.control, key, value)
    default_config.env.action_type = action_type
    default_config.env.is_training = is_training
    default_config.env.get_action_hist = False
    default_config.env.get_state_hist = False
    return NeighborSelectionFlockingEnv({"seed_id": seed, "config": default_config.dict()})


def load_checkpoint_weights(checkpoint_dir: Path):
    policy_state = checkpoint_dir / "policies" / "default_policy" / "policy_state.pkl"
    if policy_state.exists():
        with policy_state.open("rb") as f:
            state = pickle.load(f)
        return state["weights"]

    payloads = [
        p for p in checkpoint_dir.iterdir()
        if p.name.startswith("checkpoint-") and not p.name.endswith(".tune_metadata")
    ]
    if not payloads:
        raise FileNotFoundError(f"No checkpoint payload found in {checkpoint_dir}")
    with payloads[0].open("rb") as f:
        payload = pickle.load(f)
    worker_state = pickle.loads(payload["worker"])
    return worker_state["state"]["default_policy"]["weights"]


class LearnedRadiusPolicy:
    def __init__(self, checkpoint_dir: Path, config: Dict[str, Any], env):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_outputs = int(np.prod(env.action_space.shape) * 2)
        self.model = NeighborSelectionRadiusPPORLlib(
            obs_space=env.observation_space,
            action_space=env.action_space,
            num_outputs=num_outputs,
            model_config={"custom_model_config": config["model"]},
            name="radius_policy_media",
        ).to(self.device)
        weights = load_checkpoint_weights(checkpoint_dir)
        state = {
            key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for key, value in weights.items()
        }
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

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


class AllNeighborPolicy:
    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        padding_mask = obs["padding_mask"]
        active_pair_mask = padding_mask[:, None] & padding_mask[None, :]
        return (obs["neighbor_masks"] & active_pair_mask).astype(np.int8)


def radius_selection(env, state: Dict[str, np.ndarray], normalized_action: np.ndarray) -> Dict[str, np.ndarray]:
    padding_mask = state["padding_mask"].astype(bool)
    radius_min = float(env.config.env.radius_min)
    radius_max = float(env.config.env.radius_max)
    action = np.clip(np.asarray(normalized_action, dtype=np.float32), 0.0, 1.0)
    physical_radii = radius_min + action * (radius_max - radius_min)
    rel_state = env.get_relative_state(state=state)
    distances = np.asarray(rel_state["rel_agent_dists"])
    active_pair_mask = padding_mask[:, None] & padding_mask[None, :]
    selected = state["neighbor_masks"].astype(bool) & (distances <= physical_radii[:, None]) & active_pair_mask
    active = np.nonzero(padding_mask)[0]
    selected[active, active] = True
    return {"radii": physical_radii, "selected": selected}


def collect_rollout(config: Dict[str, Any], policy_name: str, policy, action_type: str, seed: int):
    env = make_env(config, action_type, seed, is_training=True)
    obs = env.reset()
    done = False

    positions = []
    headings = []
    masks = []
    radii = []
    selected_masks = []
    train_rewards = []
    original_rewards = []
    spatial_entropy = []
    velocity_entropy = []

    while not done:
        state_before = {
            "agent_states": np.asarray(env.state["agent_states"]).copy(),
            "neighbor_masks": np.asarray(env.state["neighbor_masks"]).copy(),
            "padding_mask": np.asarray(env.state["padding_mask"]).copy(),
        }
        action = policy(obs)
        positions.append(state_before["agent_states"][:, :2])
        headings.append(state_before["agent_states"][:, 4])
        masks.append(state_before["padding_mask"].astype(bool))

        if action_type == "radius":
            selection = radius_selection(env, state_before, action)
            radii.append(selection["radii"])
            selected_masks.append(selection["selected"])
        else:
            radii.append(np.full(env.num_agents_max, np.nan, dtype=np.float64))
            selected_masks.append(np.asarray(action, dtype=bool))

        obs, reward, done, info = env.step(action)
        train_rewards.append(float(reward))
        original_rewards.append(float(info.get("original_reward", reward)))
        spatial_entropy.append(float(info.get("spatial_entropy", np.nan)))
        velocity_entropy.append(float(info.get("velocity_entropy", np.nan)))

    positions.append(np.asarray(env.state["agent_states"])[:, :2].copy())
    headings.append(np.asarray(env.state["agent_states"])[:, 4].copy())
    masks.append(np.asarray(env.state["padding_mask"]).astype(bool).copy())

    return {
        "policy": policy_name,
        "seed": seed,
        "action_type": action_type,
        "positions": np.asarray(positions),
        "headings": np.asarray(headings),
        "masks": np.asarray(masks),
        "radii": np.asarray(radii),
        "selected_masks": np.asarray(selected_masks),
        "train_rewards": np.asarray(train_rewards),
        "original_rewards": np.asarray(original_rewards),
        "spatial_entropy": np.asarray(spatial_entropy),
        "velocity_entropy": np.asarray(velocity_entropy),
        "train_return": float(np.sum(train_rewards)),
        "original_return": float(np.sum(original_rewards)),
        "episode_length": int(len(train_rewards)),
    }


def axis_limits(positions: np.ndarray, radii: np.ndarray, masks: np.ndarray, show_radii: bool):
    active_positions = positions[masks]
    if active_positions.size == 0:
        return (-1, 1), (-1, 1)
    xmin, ymin = active_positions.min(axis=0)
    xmax, ymax = active_positions.max(axis=0)
    pad = 20.0
    if show_radii and np.isfinite(radii).any():
        pad += float(np.nanpercentile(radii, 95)) * 0.25
    width = max(xmax - xmin, ymax - ymin, 1.0)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * width + pad
    return (cx - half, cx + half), (cy - half, cy + half)


def agent_colors(num_agents: int):
    cmap = plt.get_cmap("tab20")
    return np.asarray([cmap(i % 20) for i in range(num_agents)])


def selected_segments(positions: np.ndarray, selected: np.ndarray, active: np.ndarray):
    segments = []
    for i in np.nonzero(active)[0]:
        for j in np.nonzero(active & selected[i])[0]:
            if i < j:
                segments.append([positions[i], positions[j]])
    return segments


def render_animation(rollout: Dict[str, Any], output_path: Path, frame_stride: int, fps: int, dpi: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positions = rollout["positions"][:-1]
    headings = rollout["headings"][:-1]
    masks = rollout["masks"][:-1]
    radii = rollout["radii"]
    selected = rollout["selected_masks"]
    show_radii = rollout["action_type"] == "radius"
    colors = agent_colors(positions.shape[1])
    xlim, ylim = axis_limits(rollout["positions"], radii, rollout["masks"], show_radii)
    frame_indices = list(range(0, len(positions), frame_stride))
    if frame_indices[-1] != len(positions) - 1:
        frame_indices.append(len(positions) - 1)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    edge_collection = LineCollection([], colors=(0.2, 0.2, 0.2, 0.16), linewidths=0.7, zorder=1)
    ax.add_collection(edge_collection)
    scatter = ax.scatter([], [], s=34, c=[], edgecolors="white", linewidths=0.7, zorder=4)
    circles = []
    for color in colors:
        circles.append(Circle((0, 0), 0, facecolor=(*color[:3], 0.045), edgecolor=(*color[:3], 0.40), linewidth=0.8, visible=False, zorder=2))
        ax.add_patch(circles[-1])
    initial_active = masks[frame_indices[0]] & np.isfinite(positions[frame_indices[0]]).all(axis=1)
    initial_q_pos = positions[frame_indices[0]][initial_active]
    quiver_count = len(initial_q_pos)
    quiver = ax.quiver(
        initial_q_pos[:, 0] if quiver_count else [],
        initial_q_pos[:, 1] if quiver_count else [],
        np.zeros(quiver_count),
        np.zeros(quiver_count),
        color=colors[initial_active] if quiver_count else [],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0035,
        zorder=5,
    )
    arrow_length = 0.035 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=7,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
        macro_block_size=16,
    )
    try:
        for idx in frame_indices:
            ax.set_title(
                f"{rollout['policy']} | seed={rollout['seed']} | step={idx}/{len(positions)-1} | "
                f"train={np.sum(rollout['train_rewards'][:idx+1]):.2f} | original={np.sum(rollout['original_rewards'][:idx+1]):.2f}",
                fontsize=9,
            )
            active = masks[idx] & np.isfinite(positions[idx]).all(axis=1)
            edge_collection.set_segments(selected_segments(positions[idx], selected[idx], active))
            visible_positions = positions[idx][active]
            scatter.set_offsets(visible_positions if len(visible_positions) else np.empty((0, 2)))
            scatter.set_color(colors[active] if active.any() else [])

            for agent_idx, circle in enumerate(circles):
                visible = bool(show_radii and active[agent_idx] and np.isfinite(radii[idx, agent_idx]))
                circle.set_visible(visible)
                if visible:
                    circle.center = tuple(positions[idx, agent_idx])
                    circle.radius = max(float(radii[idx, agent_idx]), 0.0)

            if active.any():
                q_pos = positions[idx][active]
                q_head = headings[idx][active]
                q_u = np.cos(q_head) * arrow_length
                q_v = np.sin(q_head) * arrow_length
                if len(q_pos) != quiver_count:
                    quiver.remove()
                    quiver_count = len(q_pos)
                    quiver = ax.quiver(
                        q_pos[:, 0],
                        q_pos[:, 1],
                        q_u,
                        q_v,
                        color=colors[active],
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                        width=0.0035,
                        zorder=5,
                    )
                else:
                    quiver.set_offsets(q_pos)
                    quiver.set_UVC(q_u, q_v)
                    quiver.set_color(colors[active])
            else:
                if quiver_count:
                    quiver.remove()
                    quiver_count = 0
                    quiver = ax.quiver([], [], [], [], color=[], angles="xy", scale_units="xy", scale=1.0, width=0.0035, zorder=5)

            canvas.draw()
            width, height = canvas.get_width_height()
            frame = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape((height, width, 3))
            writer.append_data(frame)
    finally:
        writer.close()
        plt.close(fig)


def plot_trajectory(rollout: Dict[str, Any], output_path: Path, dpi: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positions = rollout["positions"]
    masks = rollout["masks"]
    colors = agent_colors(positions.shape[1])
    xlim, ylim = axis_limits(positions, rollout["radii"], masks, False)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    active = masks[0]
    for agent_idx in np.nonzero(active)[0]:
        traj = positions[:, agent_idx, :]
        ax.plot(traj[:, 0], traj[:, 1], color=colors[agent_idx], linewidth=1.1, alpha=0.85)
        ax.scatter(traj[0, 0], traj[0, 1], marker="o", s=26, color=colors[agent_idx], edgecolors="white", linewidths=0.5)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="x", s=28, color=colors[agent_idx], linewidths=1.0)
    ax.set_title(
        f"{rollout['policy']} | seed={rollout['seed']} | train={rollout['train_return']:.2f} | "
        f"original={rollout['original_return']:.2f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


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


def main():
    parser = argparse.ArgumentParser(description="Render per-episode learned-radius vs pure-ACS media.")
    parser.add_argument("--experiment-dir", default="ray_results/radius_action_20agents_2gpu_aggressive_5m")
    parser.add_argument("--metric", default="episode_reward_mean")
    parser.add_argument("--mode", choices=("max", "min"), default="max")
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=20260520)
    parser.add_argument("--output-dir", default="results/radius_vs_acs_best_checkpoint/media")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=112)
    args = parser.parse_args()

    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"Using ffmpeg: {ffmpeg_exe}")

    experiment_dir = Path(args.experiment_dir)
    trial_dir = find_trial_dir(experiment_dir)
    best = select_best_checkpoint(trial_dir, args.metric, args.mode)
    checkpoint_dir = Path(best["checkpoint"])
    config = load_trial_config(trial_dir)

    output_dir = Path(args.output_dir)
    animation_dir = output_dir / "animations"
    trajectory_dir = output_dir / "trajectories"
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    probe_env = make_env(config, "radius", args.base_seed, is_training=True)
    learned_policy = LearnedRadiusPolicy(checkpoint_dir, config, probe_env)
    all_policy = AllNeighborPolicy()

    records = []
    for episode_idx in range(args.num_episodes):
        seed = args.base_seed + episode_idx
        jobs = [
            ("learned_radius_best", learned_policy, "radius"),
            ("pure_acs_all_neighbor", all_policy, "binary_vector"),
        ]
        for policy_name, policy, action_type in jobs:
            rollout = collect_rollout(config, policy_name, policy, action_type, seed)
            stem = f"episode_{episode_idx + 1:03d}_seed_{seed}_{policy_name}"
            animation_path = animation_dir / f"{stem}.mp4"
            trajectory_path = trajectory_dir / f"{stem}_trajectory.png"
            metadata_path = metadata_dir / f"{stem}.json"

            render_animation(rollout, animation_path, args.frame_stride, args.fps, args.dpi)
            plot_trajectory(rollout, trajectory_path, args.dpi)

            metadata = {
                "policy": policy_name,
                "episode": episode_idx,
                "seed": seed,
                "animation_path": str(animation_path),
                "trajectory_path": str(trajectory_path),
                "train_return": rollout["train_return"],
                "original_return": rollout["original_return"],
                "episode_length": rollout["episode_length"],
                "best_checkpoint": best,
                "frame_stride": args.frame_stride,
                "fps": args.fps,
                "dpi": args.dpi,
            }
            if action_type == "radius":
                metadata["mean_radius"] = float(np.nanmean(rollout["radii"]))
                metadata["mean_selected_neighbor_count"] = float(np.nanmean(rollout["selected_masks"].sum(axis=2)[rollout["masks"][:-1]]))
            with metadata_path.open("w") as f:
                json.dump(json_safe(metadata), f, indent=2)
            records.append(metadata)
            print(f"wrote {animation_path} and {trajectory_path}", flush=True)

    summary = {
        "best_checkpoint": best,
        "num_episodes": args.num_episodes,
        "base_seed": args.base_seed,
        "frame_stride": args.frame_stride,
        "fps": args.fps,
        "dpi": args.dpi,
        "records": records,
    }
    with (output_dir / "media_summary.json").open("w") as f:
        json.dump(json_safe(summary), f, indent=2)
    print(f"Wrote media summary to {output_dir / 'media_summary.json'}")


if __name__ == "__main__":
    main()
