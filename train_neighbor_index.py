"""
PPO trainer for the `neighbor_index` action type with contextual-bandit wrapper.

Config-driven: loads a YAML file (default `configs/neighbor_index_bandit_train.yaml`) and
applies its env/control/model/ppo sections, mirroring the layout used by the radius branch.

Usage:
  python train_neighbor_index.py
  python train_neighbor_index.py --config configs/neighbor_index_bandit_train.yaml
  python train_neighbor_index.py --smoke               # short run from the YAML's `smoke` block
  python train_neighbor_index.py --local-mode          # ray.init(local_mode=True) for debugging
  python train_neighbor_index.py --restore <path>      # resume policy weights from a checkpoint
"""

import argparse
import copy
from typing import Any, Dict

import ray
import yaml
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from envs.env import load_config
from envs.bandit_env import NeighborIndexBanditEnv
from models.ppo_neighbor_index import NeighborIndexPPORLlib


DEFAULT_CONFIG_PATH = "configs/neighbor_index_bandit_train.yaml"


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
        smoke_overrides = config.pop("smoke")
        for section in ("run", "env", "control", "model", "ppo"):
            if section in smoke_overrides:
                deep_update(config.setdefault(section, {}), smoke_overrides[section])
        if "stop" in smoke_overrides:
            config["stop"] = smoke_overrides["stop"]
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
    run_cfg = config.get("run", {})
    return {
        "seed_id": run_cfg.get("seed_id", 42),
        "inner_gamma": run_cfg.get("inner_gamma", 1.0),
        "config": default_config.dict(),
    }


def build_ppo_config(config: Dict[str, Any], env_name: str, model_name: str,
                     env_config: Dict[str, Any]) -> Dict[str, Any]:
    ppo_config = copy.deepcopy(config.get("ppo", {}))
    ppo_config.update({
        "env": env_name,
        "env_config": env_config,
        "model": {
            "custom_model": model_name,
            "custom_model_config": copy.deepcopy(config.get("model", {})),
        },
    })
    run_cfg = config.get("run", {})
    if run_cfg.get("num_gpus") is not None:
        ppo_config["num_gpus"] = run_cfg["num_gpus"]
    return ppo_config


def checkpoint_score_attr(run_cfg: Dict[str, Any]):
    attr = run_cfg.get("checkpoint_score_attr")
    if attr is None:
        return None
    order = run_cfg.get("checkpoint_score_order", "max")
    if order not in ("max", "min"):
        raise ValueError("run.checkpoint_score_order must be either 'max' or 'min'")
    if order == "min" and not attr.startswith("min-"):
        return f"min-{attr}"
    if order == "max" and attr.startswith("min-"):
        return attr[len("min-"):]
    return attr


def main():
    parser = argparse.ArgumentParser(description="Train neighbor_index bandit-style PPO.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument("--smoke", action="store_true", help="Apply the `smoke` override block for a tiny run.")
    parser.add_argument("--local-mode", action="store_true", help="ray.init(local_mode=True) for debugging.")
    parser.add_argument("--restore", default=None, help="Restore trainer state from a checkpoint path.")
    args = parser.parse_args()

    config = load_yaml_config(args.config, smoke=args.smoke)
    run_cfg = config.get("run", {})

    if not ray.is_initialized():
        ray_init_kwargs = {"local_mode": args.local_mode}
        if run_cfg.get("num_cpus") is not None:
            ray_init_kwargs["num_cpus"] = run_cfg["num_cpus"]
        ray.init(**ray_init_kwargs)

    env_name = run_cfg.get("env_name", "neighbor_index_bandit_env")
    model_name = run_cfg.get("model_name", "neighbor_index_rl")

    env_config = build_env_config(config)
    register_env(env_name, lambda cfg: NeighborIndexBanditEnv(cfg))
    ModelCatalog.register_custom_model(model_name, NeighborIndexPPORLlib)

    tune_kwargs = {}
    score_attr = checkpoint_score_attr(run_cfg)
    if score_attr is not None:
        tune_kwargs["checkpoint_score_attr"] = score_attr
    if args.restore:
        tune_kwargs["restore"] = args.restore

    tune.run(
        "PPO",
        name=run_cfg.get("name", "neighbor_index_bandit"),
        local_dir=run_cfg.get("local_dir", "./ray_results"),
        checkpoint_freq=run_cfg.get("checkpoint_freq", 8),
        keep_checkpoints_num=run_cfg.get("keep_checkpoints_num", 8),
        checkpoint_at_end=run_cfg.get("checkpoint_at_end", True),
        stop=config.get("stop"),
        config=build_ppo_config(config, env_name, model_name, env_config),
        **tune_kwargs,
    )


if __name__ == "__main__":
    main()
