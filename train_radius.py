import argparse
import copy
from typing import Any, Dict

import ray
import yaml
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from envs.env import NeighborSelectionFlockingEnv, load_config
from models.ppo_radius import NeighborSelectionRadiusPPORLlib


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml_config(path: str, smoke: bool = False) -> Dict[str, Any]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config = copy.deepcopy(config)
    if smoke and config.get("smoke"):
        smoke_config = config.pop("smoke")
        for section in ("run", "env", "control", "model", "ppo"):
            if section in smoke_config:
                deep_update(config.setdefault(section, {}), smoke_config[section])
        config["stop"] = smoke_config.get("stop", {"training_iteration": 1})
    else:
        config.pop("smoke", None)
        config.setdefault("stop", None)
    return config


def build_env_config(config: Dict[str, Any]) -> Dict[str, Any]:
    default_config = load_config("./envs/default_env_config.yaml")

    for key, value in config.get("env", {}).items():
        setattr(default_config.env, key, value)
    for key, value in config.get("control", {}).items():
        setattr(default_config.control, key, value)

    return {
        "seed_id": config.get("run", {}).get("seed_id", 42),
        "config": default_config.dict(),
    }


def build_ppo_config(config: Dict[str, Any], env_name: str, model_name: str, env_config: Dict[str, Any]) -> Dict[str, Any]:
    ppo_config = copy.deepcopy(config.get("ppo", {}))
    ppo_config.update(
        {
            "env": env_name,
            "env_config": env_config,
            "model": {
                "custom_model": model_name,
                "custom_model_config": copy.deepcopy(config.get("model", {})),
            },
        }
    )
    if config.get("run", {}).get("num_gpus") is not None:
        ppo_config["num_gpus"] = config["run"]["num_gpus"]
    return ppo_config


def checkpoint_score_attr(run_config: Dict[str, Any]):
    attr = run_config.get("checkpoint_score_attr")
    if attr is None:
        return None

    score_order = run_config.get("checkpoint_score_order", "max")
    if score_order not in ("max", "min"):
        raise ValueError("run.checkpoint_score_order must be either 'max' or 'min'")
    if score_order == "min" and not attr.startswith("min-"):
        return f"min-{attr}"
    if score_order == "max" and attr.startswith("min-"):
        return attr[len("min-"):]
    return attr


def main():
    parser = argparse.ArgumentParser(description="Train a radius-action neighbor-selection PPO policy.")
    parser.add_argument("--config", default="configs/radius_action_train.yaml", help="Radius training YAML config.")
    parser.add_argument("--smoke", action="store_true", help="Run one small PPO iteration for sanity checking.")
    parser.add_argument("--local-mode", action="store_true", help="Start Ray in local_mode for debugging.")
    args = parser.parse_args()

    config = load_yaml_config(args.config, smoke=args.smoke)
    run_config = config.get("run", {})

    if not ray.is_initialized():
        ray_init_kwargs = {"local_mode": args.local_mode}
        if run_config.get("num_cpus") is not None:
            ray_init_kwargs["num_cpus"] = run_config["num_cpus"]
        ray.init(**ray_init_kwargs)

    env_name = run_config.get("env_name", "neighbor_selection_flocking_radius_env")
    model_name = run_config.get("model_name", "radius_neighbor_selector_rl")
    env_config = build_env_config(config)

    register_env(env_name, lambda cfg: NeighborSelectionFlockingEnv(cfg))
    ModelCatalog.register_custom_model(model_name, NeighborSelectionRadiusPPORLlib)

    score_attr = checkpoint_score_attr(run_config)
    tune_kwargs = {}
    if score_attr is not None:
        tune_kwargs["checkpoint_score_attr"] = score_attr

    tune.run(
        "PPO",
        name=run_config.get("name", "radius_action_training"),
        local_dir=run_config.get("local_dir", "./ray_results"),
        checkpoint_freq=run_config.get("checkpoint_freq", 8),
        keep_checkpoints_num=run_config.get("keep_checkpoints_num", 8),
        checkpoint_at_end=run_config.get("checkpoint_at_end", True),
        stop=config.get("stop"),
        config=build_ppo_config(config, env_name, model_name, env_config),
        **tune_kwargs,
    )


if __name__ == "__main__":
    main()
