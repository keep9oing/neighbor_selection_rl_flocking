import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Optional


def checkpoint_iteration(path: str) -> Optional[int]:
    name = os.path.basename(path.rstrip(os.sep))
    if not name.startswith("checkpoint_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def find_trial_dirs(experiment_dir: str) -> List[str]:
    if not os.path.isdir(experiment_dir):
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    trial_dirs = []
    for name in sorted(os.listdir(experiment_dir)):
        path = os.path.join(experiment_dir, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "progress.csv")):
            trial_dirs.append(path)
    if not trial_dirs:
        raise FileNotFoundError(f"No trial directories with progress.csv found in {experiment_dir}")
    return trial_dirs


def load_metric_by_iteration(progress_path: str, metric: str) -> Dict[int, float]:
    metric_by_iteration: Dict[int, float] = {}
    with open(progress_path, newline="") as f:
        reader = csv.DictReader(f)
        if "training_iteration" not in (reader.fieldnames or []):
            raise ValueError(f"{progress_path} does not contain training_iteration")
        if metric not in (reader.fieldnames or []):
            raise ValueError(f"{progress_path} does not contain metric '{metric}'")
        for row in reader:
            try:
                iteration = int(float(row["training_iteration"]))
                value = float(row[metric])
            except (TypeError, ValueError):
                continue
            metric_by_iteration[iteration] = value
    return metric_by_iteration


def iter_checkpoint_candidates(trial_dirs: Iterable[str], metric: str):
    for trial_dir in trial_dirs:
        metric_by_iteration = load_metric_by_iteration(os.path.join(trial_dir, "progress.csv"), metric)
        for name in sorted(os.listdir(trial_dir)):
            checkpoint_dir = os.path.join(trial_dir, name)
            if not os.path.isdir(checkpoint_dir):
                continue
            iteration = checkpoint_iteration(checkpoint_dir)
            if iteration is None:
                continue
            policy_state = os.path.join(checkpoint_dir, "policies", "default_policy", "policy_state.pkl")
            if not os.path.exists(policy_state):
                continue
            if iteration not in metric_by_iteration:
                continue
            yield {
                "checkpoint": checkpoint_dir,
                "trial_dir": trial_dir,
                "iteration": iteration,
                "metric": metric,
                "metric_value": metric_by_iteration[iteration],
            }


def select_best_checkpoint(experiment_dir: str, metric: str = "episode_reward_mean", mode: str = "max"):
    if mode not in ("max", "min"):
        raise ValueError("mode must be either 'max' or 'min'")
    candidates = list(iter_checkpoint_candidates(find_trial_dirs(experiment_dir), metric))
    if not candidates:
        raise FileNotFoundError(
            f"No usable checkpoints with metric '{metric}' found under {experiment_dir}"
        )
    key = lambda candidate: candidate["metric_value"]
    return max(candidates, key=key) if mode == "max" else min(candidates, key=key)


def main():
    parser = argparse.ArgumentParser(description="Select the best radius-policy checkpoint from Ray Tune results.")
    parser.add_argument(
        "--experiment-dir",
        default="ray_results/radius_action_sqrt2_bound_100iter",
        help="Ray Tune experiment directory containing one or more trial dirs.",
    )
    parser.add_argument("--metric", default="episode_reward_mean")
    parser.add_argument("--mode", choices=("max", "min"), default="max")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    best = select_best_checkpoint(args.experiment_dir, metric=args.metric, mode=args.mode)
    if args.json:
        print(json.dumps(best, indent=2))
    else:
        print(best["checkpoint"])
        print(f"metric={best['metric']} value={best['metric_value']} iteration={best['iteration']}")


if __name__ == "__main__":
    main()
