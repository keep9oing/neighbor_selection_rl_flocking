"""
Smoke test for the radius-action RLlib model.

This script verifies the model-level radius action contract:
    action_space: Box(0, 1, shape=(N,), dtype=float32)
    model output: (batch, 2 * N), ordered as Gaussian mean then log_std
"""

import sys
import traceback

import numpy as np
import torch
from gym.spaces import Box

from envs.env import NeighborSelectionFlockingEnv, config_to_env_input, load_config
from models.ppo_radius import NeighborSelectionRadiusPPORLlib


def build_radius_env():
    config = load_config("./envs/default_env_config.yaml")
    config.env.action_type = "radius"
    config.env.env_mode = "single_env"
    config.env.observation_type = "ego_centric"
    config.env.num_agents_pool = [3, 5]
    config.env.obs_dim = 4
    config.env.comm_range = None
    config.env.max_time_steps = 50
    config.env.use_fixed_episode_length = True

    return NeighborSelectionFlockingEnv(config_to_env_input(config, seed_id=42))


def make_padded_obs(env):
    num_active = 3
    positions = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
        ],
        dtype=np.float64,
    )
    headings = np.zeros((num_active, 1), dtype=np.float64)
    velocities = env.config.control.speed * np.concatenate(
        [np.cos(headings), np.sin(headings)],
        axis=1,
    )
    return env.custom_reset(
        positions,
        velocities,
        headings,
        num_agents_max=env.num_agents_max,
        comm_range=env.config.env.comm_range,
    )


def batch_obs(obs):
    return {
        key: torch.as_tensor(value).unsqueeze(0)
        for key, value in obs.items()
        if key in ("local_agent_infos", "neighbor_masks", "padding_mask", "is_from_my_env")
    }


def main():
    print("=" * 60)
    print("Testing radius-action PPO model")
    print("=" * 60)

    try:
        env = build_radius_env()
    except Exception as exc:
        print("Failed to construct radius env.")
        print("This verifier requires env support for action_type='radius'.")
        print(f"{type(exc).__name__}: {exc}")
        return 1

    obs = make_padded_obs(env)
    num_agents_max = int(env.num_agents_max)
    assert not obs["padding_mask"].all(), "Verifier expected at least one padded agent."

    action_space = env.action_space
    if action_space.shape != (num_agents_max,):
        print(
            "Radius env action_space has the wrong shape: "
            f"expected ({num_agents_max},), got {action_space.shape}."
        )
        return 1
    if not isinstance(action_space, Box):
        print(f"Radius env action_space must be gym.spaces.Box, got {type(action_space).__name__}.")
        return 1

    model_config = {
        "custom_model_config": {
            "d_embed_input": 64,
            "d_ff": 128,
            "d_subobs": obs["local_agent_infos"].shape[-1],
            "dr_rate": 0.0,
            "initial_log_std": -0.5,
            "n_layers_encoder": 1,
            "norm_eps": 1e-5,
            "num_heads": 4,
            "share_layers": False,
        }
    }

    model = NeighborSelectionRadiusPPORLlib(
        obs_space=env.observation_space,
        action_space=action_space,
        num_outputs=2 * num_agents_max,
        model_config=model_config,
        name="radius_model_verify",
    )

    input_dict = {"obs": batch_obs(obs)}
    with torch.no_grad():
        outputs, state = model(input_dict, [], None)
        values = model.value_function()

    expected_output_shape = (1, 2 * num_agents_max)
    expected_value_shape = (1,)

    assert state == [], f"Unexpected model state: {state}"
    assert tuple(outputs.shape) == expected_output_shape, (
        f"Wrong output shape: expected {expected_output_shape}, got {tuple(outputs.shape)}"
    )
    assert tuple(values.shape) == expected_value_shape, (
        f"Wrong value shape: expected {expected_value_shape}, got {tuple(values.shape)}"
    )
    assert torch.isfinite(outputs).all(), "Model outputs contain non-finite values."
    assert torch.isfinite(values).all(), "Value function contains non-finite values."

    means = outputs[:, :num_agents_max]
    log_stds = outputs[:, num_agents_max:]
    assert means.shape == log_stds.shape == (1, num_agents_max)

    print("Radius env and model forward pass succeeded.")
    print(f"num_agents_max: {num_agents_max}")
    print(f"model output shape: {tuple(outputs.shape)}")
    print(f"value shape: {tuple(values.shape)}")
    print("All finite checks passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Verification failed with {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
