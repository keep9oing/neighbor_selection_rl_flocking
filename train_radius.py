import argparse
import copy
import csv
import json
import pickle
import shutil
from pathlib import Path
from typing import Optional
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


def checkpoint_iteration(path: Path) -> Optional[int]:
    try:
        return int(path.name.split("_", 1)[1])
    except Exception:
        return None


def experiment_dir(run_config: Dict[str, Any]) -> Path:
    return Path(run_config.get("local_dir", "./ray_results")) / run_config.get("name", "radius_action_training")


def find_latest_trial_dir(exp_dir: Path) -> Path:
    trial_dirs = [
        path for path in exp_dir.iterdir()
        if path.is_dir() and (path / "progress.csv").exists()
    ]
    if not trial_dirs:
        raise FileNotFoundError(f"No Ray trial directory with progress.csv found under {exp_dir}")
    return max(trial_dirs, key=lambda path: path.stat().st_mtime)


def find_latest_checkpoint(exp_dir: Path) -> Path:
    candidates = []
    for checkpoint_dir in exp_dir.glob("*/checkpoint_*"):
        if not checkpoint_dir.is_dir():
            continue
        iteration = checkpoint_iteration(checkpoint_dir)
        if iteration is not None:
            candidates.append((iteration, checkpoint_dir.stat().st_mtime, checkpoint_dir))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_* directories found under {exp_dir}")
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def latest_progress_iteration(exp_dir: Path) -> Optional[int]:
    trial_dir = find_latest_trial_dir(exp_dir)
    latest = None
    with (trial_dir / "progress.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                latest = int(float(row["training_iteration"]))
            except Exception:
                continue
    return latest


def set_training_iteration_stop(config: Dict[str, Any], target_iteration: int) -> None:
    if target_iteration <= 0:
        raise ValueError("target training_iteration must be positive")
    stop_config = copy.deepcopy(config.get("stop") or {})
    stop_config["training_iteration"] = target_iteration
    config["stop"] = stop_config


def checkpoint_payload_file(checkpoint_dir: Path) -> Optional[Path]:
    payloads = [
        path for path in checkpoint_dir.iterdir()
        if path.name.startswith("checkpoint-") and not path.name.endswith(".tune_metadata")
    ]
    if not payloads:
        return None
    return payloads[0]


def remove_none_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: remove_none_values(child)
            for key, child in value.items()
            if child is not None
        }
    if isinstance(value, list):
        return [remove_none_values(child) for child in value]
    if isinstance(value, tuple):
        return tuple(remove_none_values(child) for child in value)
    return value


def sanitize_checkpoint_optimizer_state(checkpoint_dir: Path) -> bool:
    payload_path = checkpoint_payload_file(checkpoint_dir)
    if payload_path is None:
        return False

    with payload_path.open("rb") as f:
        payload = pickle.load(f)
    if "worker" not in payload:
        return False

    worker = pickle.loads(payload["worker"])
    changed = False
    for policy_state in worker.get("state", {}).values():
        optimizer_variables = policy_state.get("_optimizer_variables")
        if optimizer_variables:
            cleaned = remove_none_values(optimizer_variables)
            if cleaned != optimizer_variables:
                policy_state["_optimizer_variables"] = cleaned
                changed = True

    if not changed:
        return False

    backup_path = payload_path.with_suffix(payload_path.suffix + ".before_none_sanitize")
    if not backup_path.exists():
        shutil.copy2(payload_path, backup_path)

    payload["worker"] = pickle.dumps(worker)
    with payload_path.open("wb") as f:
        pickle.dump(payload, f)
    return True


def sanitize_experiment_checkpoints(exp_dir: Path) -> int:
    count = 0
    for checkpoint_dir in exp_dir.glob("*/checkpoint_*"):
        if checkpoint_dir.is_dir() and sanitize_checkpoint_optimizer_state(checkpoint_dir):
            count += 1
    return count


def patch_latest_experiment_state_stop(exp_dir: Path, target_iteration: int) -> Optional[Path]:
    state_files = sorted(exp_dir.glob("experiment_state-*.json"), key=lambda path: path.stat().st_mtime)
    if not state_files:
        return None

    state_path = state_files[-1]
    with state_path.open() as f:
        state = json.load(f)

    changed = False
    patched_checkpoints = []
    for trial_json in state.get("checkpoints", []):
        trial_state = json.loads(trial_json)
        stopping_criterion = trial_state.setdefault("stopping_criterion", {})
        if stopping_criterion.get("training_iteration") != target_iteration:
            stopping_criterion["training_iteration"] = target_iteration
            changed = True
        patched_checkpoints.append(json.dumps(trial_state))

    if not changed:
        return state_path

    backup_path = state_path.with_suffix(state_path.suffix + ".before_stop_patch")
    if not backup_path.exists():
        shutil.copy2(state_path, backup_path)

    state["checkpoints"] = patched_checkpoints
    with state_path.open("w") as f:
        json.dump(state, f)
    return state_path


def main():
    parser = argparse.ArgumentParser(description="Train a radius-action neighbor-selection PPO policy.")
    parser.add_argument("--config", default="configs/radius_action_train.yaml", help="Radius training YAML config.")
    parser.add_argument("--smoke", action="store_true", help="Run one small PPO iteration for sanity checking.")
    parser.add_argument("--local-mode", action="store_true", help="Start Ray in local_mode for debugging.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="LOCAL",
        default=False,
        help="Resume a Ray Tune experiment. Use without a value for LOCAL, or pass AUTO/LOCAL/REMOTE/PROMPT/ERRORED_ONLY.",
    )
    parser.add_argument(
        "--restore",
        default=None,
        help="Restore trainer weights/state from a checkpoint path. Use 'latest' for the latest checkpoint in this run.name.",
    )
    parser.add_argument(
        "--stop-iters",
        type=int,
        default=None,
        help="Override stop.training_iteration with an absolute target iteration.",
    )
    parser.add_argument(
        "--additional-iters",
        type=int,
        default=None,
        help="Train this many more iterations from the latest progress row, or from --restore checkpoint iteration.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config, smoke=args.smoke)
    run_config = config.get("run", {})
    exp_dir = experiment_dir(run_config)
    target_iteration = None

    restore_path = args.restore
    if restore_path == "latest":
        restore_path = str(find_latest_checkpoint(exp_dir))

    if args.stop_iters is not None and args.additional_iters is not None:
        raise ValueError("Use only one of --stop-iters or --additional-iters")
    if args.stop_iters is not None:
        set_training_iteration_stop(config, args.stop_iters)
        target_iteration = args.stop_iters
    elif args.additional_iters is not None:
        if args.additional_iters <= 0:
            raise ValueError("--additional-iters must be positive")
        if restore_path:
            base_iteration = checkpoint_iteration(Path(restore_path))
            if base_iteration is None:
                raise ValueError(f"Cannot infer iteration from restore checkpoint path: {restore_path}")
        else:
            base_iteration = latest_progress_iteration(exp_dir)
            if base_iteration is None:
                raise ValueError(f"Cannot infer latest progress iteration under {exp_dir}")
        target_iteration = base_iteration + args.additional_iters
        set_training_iteration_stop(config, target_iteration)

    if restore_path or args.resume:
        sanitized_count = sanitize_experiment_checkpoints(exp_dir)
        if sanitized_count:
            print(f"Sanitized optimizer None fields in {sanitized_count} checkpoint(s) under {exp_dir}.")
    if args.resume and target_iteration is not None:
        patched_state = patch_latest_experiment_state_stop(exp_dir, target_iteration)
        if patched_state is not None:
            print(f"Patched resumed experiment stop.training_iteration={target_iteration} in {patched_state}.")

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
    if restore_path:
        tune_kwargs["restore"] = restore_path
    if args.resume:
        tune_kwargs["resume"] = args.resume

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
