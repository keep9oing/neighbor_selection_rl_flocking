"""Trace collection for radius-policy flocking animations.

This module intentionally has no rendering or CLI side effects.  It provides a
small callable API that builds the radius environment, runs one episode or a
bounded prefix of one episode, and returns JSON-friendly trace data.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from evaluate_radius_policy import (
    FixedNormalizedRadiusPolicy,
    RadiusRLPolicy,
    has_converged,
    load_yaml_config,
    make_env,
)


def _as_jsonable(value: Any) -> Any:
    """Convert numpy-heavy values into JSON-friendly Python containers."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _as_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(val) for val in value]
    return value


def _state_snapshot(state: Dict[str, np.ndarray]) -> Dict[str, Any]:
    agent_states = np.asarray(state["agent_states"])
    padding_mask = np.asarray(state["padding_mask"], dtype=bool)
    return {
        "agent_states": _as_jsonable(agent_states),
        "positions": _as_jsonable(agent_states[:, :2]),
        "velocities": _as_jsonable(agent_states[:, 2:4]),
        "headings": _as_jsonable(agent_states[:, 4]),
        "padding_mask": _as_jsonable(padding_mask),
        "neighbor_masks": _as_jsonable(np.asarray(state["neighbor_masks"], dtype=bool)),
    }


def _normalize_radius_action(action: Any, env) -> np.ndarray:
    action_array = np.asarray(action, dtype=np.float32)
    expected_shape = (env.num_agents_max,)
    if action_array.shape != expected_shape:
        raise ValueError(
            f"radius policy returned action with shape {action_array.shape}; "
            f"expected {expected_shape}"
        )

    if env.config.env.radius_clip_actions:
        action_array = np.clip(action_array, 0.0, 1.0)
    else:
        padding_mask = np.asarray(env.state["padding_mask"], dtype=bool)
        active_actions = action_array[padding_mask]
        if np.any(active_actions < 0.0) or np.any(active_actions > 1.0):
            raise ValueError("radius action must be in [0, 1] for all active agents")

    return action_array.astype(np.float32, copy=False)


def _radius_selection_from_state(
    env,
    state: Dict[str, np.ndarray],
    physical_radii: np.ndarray,
) -> Dict[str, Any]:
    rel_state = env.get_relative_state(state=state)
    rel_agent_dists = np.asarray(rel_state["rel_agent_dists"])
    padding_mask = np.asarray(state["padding_mask"], dtype=bool)
    padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]

    selected = (
        np.asarray(state["neighbor_masks"], dtype=bool)
        & (rel_agent_dists <= physical_radii[:, np.newaxis])
        & padding_mask_2d
    )
    active_indices = np.nonzero(padding_mask)[0]
    selected[active_indices, active_indices] = True

    selected_counts = selected.sum(axis=1).astype(np.int64)
    selected_counts[~padding_mask] = 0

    if padding_mask.any():
        active_radii = physical_radii[padding_mask]
        active_counts = selected_counts[padding_mask]
        radius_mean = float(active_radii.mean())
        radius_min = float(active_radii.min())
        radius_max = float(active_radii.max())
        selected_count_mean = float(active_counts.mean())
    else:
        radius_mean = radius_min = radius_max = selected_count_mean = None

    return {
        "selected_neighbor_masks": _as_jsonable(selected),
        "selected_neighbor_counts": _as_jsonable(selected_counts),
        "radius_mean": radius_mean,
        "radius_min": radius_min,
        "radius_max": radius_max,
        "selected_neighbor_count_mean": selected_count_mean,
    }


def collect_radius_trace(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    fixed_normalized_radius: Optional[float] = None,
    seed: int = 42,
    max_steps: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect one radius-policy rollout trace for animation.

    Exactly one policy mode is used:
    - learned mode: pass ``checkpoint_path`` and leave ``fixed_normalized_radius`` unset
    - fixed mode: pass ``fixed_normalized_radius`` for smoke tests

    The returned dict is JSON-friendly.  Per-step entries store the state before
    the radius action, then reward/entropy/done values observed after applying it.
    """
    if fixed_normalized_radius is not None and checkpoint_path is not None:
        raise ValueError("Provide either checkpoint_path or fixed_normalized_radius, not both")
    if fixed_normalized_radius is None and checkpoint_path is None:
        raise ValueError("checkpoint_path is required unless fixed_normalized_radius is provided")
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative or None")

    if config is None:
        if config_path is None:
            raise ValueError("config_path is required when config is not provided")
        config = load_yaml_config(config_path)
    env = make_env(config, "radius", seed)

    radius_min = float(env.config.env.radius_min)
    radius_max = float(env.config.env.radius_max)
    if radius_max < radius_min:
        raise ValueError(f"Invalid radius bounds: min={radius_min}, max={radius_max}")

    if fixed_normalized_radius is not None:
        fixed_value = float(fixed_normalized_radius)
        if fixed_value < 0.0 or fixed_value > 1.0:
            raise ValueError("fixed_normalized_radius must be in [0, 1]")
        policy = FixedNormalizedRadiusPolicy(fixed_value)
        policy_mode = "fixed_normalized_radius"
    else:
        policy = RadiusRLPolicy(str(checkpoint_path), config, env)
        policy_mode = "learned_radius"

    obs = env.reset()
    done = False
    steps = []
    episode_return = 0.0
    original_episode_return = 0.0
    first_converged_step = None
    step_idx = 0
    step_limit = env.config.env.max_time_steps if max_steps is None else int(max_steps)

    while not done and step_idx < step_limit:
        state_before = {
            "agent_states": np.asarray(env.state["agent_states"]).copy(),
            "neighbor_masks": np.asarray(env.state["neighbor_masks"]).copy(),
            "padding_mask": np.asarray(env.state["padding_mask"]).copy(),
        }
        action = _normalize_radius_action(policy(obs), env)
        physical_radii = radius_min + action * (radius_max - radius_min)
        selection = _radius_selection_from_state(env, state_before, physical_radii)

        obs, reward, done, info = env.step(action)
        step_idx += 1

        reward_value = float(reward)
        original_reward_value = float(info.get("original_reward", 0.0))
        episode_return += reward_value
        original_episode_return += original_reward_value

        if first_converged_step is None and has_converged(info, env):
            first_converged_step = step_idx

        step_record = {
            "step": step_idx - 1,
            "state": _state_snapshot(state_before),
            "positions": _as_jsonable(state_before["agent_states"][:, :2]),
            "headings": _as_jsonable(state_before["agent_states"][:, 4]),
            "padding_mask": _as_jsonable(state_before["padding_mask"].astype(bool)),
            "normalized_radius_action": _as_jsonable(action),
            "physical_radii": _as_jsonable(physical_radii.astype(np.float64)),
            "selected_neighbor_masks": selection["selected_neighbor_masks"],
            "selected_neighbor_counts": selection["selected_neighbor_counts"],
            "reward": reward_value,
            "original_reward": original_reward_value,
            "spatial_entropy": _as_jsonable(info.get("spatial_entropy")),
            "velocity_entropy": _as_jsonable(info.get("velocity_entropy")),
            "done": bool(done),
            "radius_mean": selection["radius_mean"],
            "radius_min": selection["radius_min"],
            "radius_max": selection["radius_max"],
            "selected_neighbor_count_mean": selection["selected_neighbor_count_mean"],
        }
        steps.append(step_record)

    terminated = bool(done)
    trace = {
        "metadata": {
            "policy_mode": policy_mode,
            "episode_return": float(episode_return),
            "original_episode_return": float(original_episode_return),
            "success": first_converged_step is not None,
            "first_converged_step": first_converged_step,
            "episode_length": step_idx,
            "radius_min": radius_min,
            "radius_max": radius_max,
            "checkpoint": checkpoint_path,
            "fixed_normalized_radius": fixed_normalized_radius,
            "seed": int(seed),
            "config_path": config_path,
            "config_source": config.get("checkpoint_config_source", config_path),
            "num_agents": int(env.num_agents),
            "num_agents_max": int(env.num_agents_max),
            "terminated": terminated,
            "truncated": not terminated and step_idx >= step_limit,
            "max_steps": None if max_steps is None else int(max_steps),
            "env_max_time_steps": int(env.config.env.max_time_steps),
        },
        "steps": steps,
    }
    return trace


__all__ = ["collect_radius_trace"]
