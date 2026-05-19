import argparse
import json
import os
from datetime import datetime
from typing import Any

import numpy as np

from radius_checkpoint_config import load_config_from_checkpoint
from visualization.radius_animation import render_radius_animation
from visualization.radius_trace import collect_radius_trace


def json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def build_output_paths(output_dir: str, policy_name: str, seed: int):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{policy_name}_seed_{seed}_{timestamp}"
    return (
        os.path.join(output_dir, f"{stem}.mp4"),
        os.path.join(output_dir, f"{stem}_metadata.json"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Animate an ACS rollout with per-agent learned interaction radii."
    )
    parser.add_argument("--config", default="configs/radius_action_train.yaml")
    parser.add_argument("--checkpoint", default=None, help="RLlib checkpoint directory for learned radius policy.")
    parser.add_argument(
        "--use-checkpoint-config",
        action="store_true",
        help="Load env/control/model settings from the checkpoint trial's params.json.",
    )
    parser.add_argument(
        "--fixed-normalized-radius",
        type=float,
        default=None,
        help="Use a constant normalized radius in [0, 1] for renderer smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--output-dir", default="results/radius_animation")
    args = parser.parse_args()

    using_checkpoint = args.checkpoint is not None
    using_fixed = args.fixed_normalized_radius is not None
    if using_checkpoint == using_fixed:
        raise ValueError("Provide exactly one of --checkpoint or --fixed-normalized-radius.")
    if using_fixed and not (0.0 <= args.fixed_normalized_radius <= 1.0):
        raise ValueError("--fixed-normalized-radius must be in [0, 1].")
    if args.use_checkpoint_config and not using_checkpoint:
        raise ValueError("--use-checkpoint-config requires --checkpoint.")

    policy_name = "learned_radius" if using_checkpoint else f"fixed_radius_{args.fixed_normalized_radius:g}"
    output_path, metadata_path = build_output_paths(args.output_dir, policy_name, args.seed)
    config = load_config_from_checkpoint(args.checkpoint) if args.use_checkpoint_config else None
    config_path = None if args.use_checkpoint_config else args.config

    trace = collect_radius_trace(
        config_path=config_path,
        config=config,
        checkpoint_path=args.checkpoint,
        fixed_normalized_radius=args.fixed_normalized_radius,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    render_radius_animation(trace, output_path=output_path, fps=args.fps, dpi=args.dpi)

    metadata = dict(trace.get("metadata", {}))
    metadata.update(
        {
            "animation_path": output_path,
            "metadata_path": metadata_path,
            "config_path": config_path,
            "config_source": trace.get("metadata", {}).get("config_source", config_path),
            "checkpoint": args.checkpoint,
            "fixed_normalized_radius": args.fixed_normalized_radius,
            "seed": args.seed,
            "fps": args.fps,
            "dpi": args.dpi,
        }
    )
    with open(metadata_path, "w") as f:
        json.dump(json_safe(metadata), f, indent=2)

    print(f"Wrote radius animation to {output_path}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
