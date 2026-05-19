import json
import os
from pathlib import Path
from typing import Any, Dict


PPO_CONFIG_KEYS = (
    "framework",
    "num_workers",
    "num_envs_per_worker",
    "rollout_fragment_length",
    "train_batch_size",
    "sgd_minibatch_size",
    "num_sgd_iter",
    "lr",
    "lr_schedule",
    "vf_loss_coeff",
    "use_critic",
    "use_gae",
    "gamma",
    "lambda",
    "kl_coeff",
    "clip_param",
    "vf_clip_param",
    "grad_clip",
    "kl_target",
)


def checkpoint_trial_dir(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    if path.name.startswith("checkpoint_"):
        return path.parent
    if path.parent.name.startswith("checkpoint_"):
        return path.parent.parent
    raise ValueError(f"Could not infer trial directory from checkpoint path: {checkpoint_path}")


def checkpoint_params_path(checkpoint_path: str) -> Path:
    params_path = checkpoint_trial_dir(checkpoint_path) / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Could not find Ray Tune params.json next to checkpoint: {params_path}")
    return params_path


def load_config_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Reconstruct this repo's radius YAML config shape from an RLlib trial params.json."""
    params_path = checkpoint_params_path(checkpoint_path)
    with params_path.open() as f:
        params = json.load(f)

    env_config = params.get("env_config", {}).get("config")
    if not isinstance(env_config, dict) or "env" not in env_config or "control" not in env_config:
        raise ValueError(f"{params_path} does not contain env_config.config.env/control")

    model_config = params.get("model", {})
    custom_model_config = model_config.get("custom_model_config", {})

    ppo_config = {key: params[key] for key in PPO_CONFIG_KEYS if key in params}
    run_config = {
        "name": checkpoint_trial_dir(checkpoint_path).parent.name,
        "env_name": params.get("env"),
        "model_name": model_config.get("custom_model"),
        "seed_id": params.get("env_config", {}).get("seed_id", 42),
        "num_gpus": params.get("num_gpus"),
    }
    if "num_workers" in params:
        run_config["num_cpus"] = int(params["num_workers"]) + 1

    return {
        "run": run_config,
        "env": env_config["env"],
        "control": env_config["control"],
        "model": custom_model_config,
        "ppo": ppo_config,
        "checkpoint_config_source": str(params_path),
    }


def checkpoint_num_agents_pool(checkpoint_path: str):
    config = load_config_from_checkpoint(checkpoint_path)
    return config.get("env", {}).get("num_agents_pool")
