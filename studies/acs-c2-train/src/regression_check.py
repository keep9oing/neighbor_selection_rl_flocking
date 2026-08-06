"""Byte-level regression check for env defaults (study acs-c2-train, Phase 0/1).

Runs deterministic episodes with the 'nearest' (k-NN) baseline under LEGACY
default flags and records md5 hashes of every obs key, reward, done per step.

Usage:
  python regression_check.py --out ref_pre.json          # capture reference
  python regression_check.py --compare ref_pre.json      # compare after edits

Covers both reward paths (is_training True/False) and expose_aux_target on/off.
Any behavioral drift of the default env shows up as a hash mismatch.
"""
import argparse
import hashlib
import json
import sys

import numpy as np

REPO = "/workspace"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from envs.env import NeighborSelectionFlockingEnv, load_config, config_to_env_input  # noqa: E402
from baselines import create_baseline  # noqa: E402

DEFAULT_YAML = "/workspace/envs/default_env_config.yaml"


def h(arr):
    a = np.ascontiguousarray(arr)
    return hashlib.md5(a.tobytes()).hexdigest()[:16]


def run_case(seed, n_steps, is_training, expose_aux, use_fixed):
    cfg = load_config(DEFAULT_YAML)
    cfg.env.task_type = "acs"
    cfg.env.observation_type = "ego_centric"
    cfg.env.num_agents_pool = [20]
    cfg.env.max_time_steps = 1000
    cfg.env.use_fixed_episode_length = use_fixed
    cfg.env.is_training = is_training
    cfg.env.expose_aux_target = expose_aux
    cfg.env.continuous_action = False

    env = NeighborSelectionFlockingEnv(config_to_env_input(cfg, seed_id=seed))
    env.seed(seed)
    obs = env.reset()
    policy = create_baseline("nearest", k=10)

    rec = []
    rec.append({k: h(v) for k, v in sorted(obs.items())})
    done = False
    t = 0
    while not done and t < n_steps:
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        t += 1
        entry = {k: h(v) for k, v in sorted(obs.items())}
        entry["reward"] = f"{float(reward):.12e}"
        entry["done"] = bool(done)
        entry["sp"] = f"{float(info['spatial_entropy']):.12e}"
        entry["vp"] = f"{float(info['velocity_entropy']):.12e}"
        rec.append(entry)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()

    cases = {
        "train_aux": run_case(seed=1000, n_steps=120, is_training=True, expose_aux=True, use_fixed=True),
        "eval_plain": run_case(seed=1001, n_steps=120, is_training=False, expose_aux=False, use_fixed=True),
        "eval_earlyterm": run_case(seed=1002, n_steps=400, is_training=False, expose_aux=False, use_fixed=False),
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(cases, f, indent=0)
        print(f"reference written: {args.out}")
    if args.compare:
        with open(args.compare) as f:
            ref = json.load(f)
        n_bad = 0
        for name, rec in cases.items():
            r = ref[name]
            if len(r) != len(rec):
                print(f"[FAIL] {name}: length {len(rec)} != ref {len(r)}")
                n_bad += 1
                continue
            for t, (a, b) in enumerate(zip(rec, r)):
                if a != b:
                    print(f"[FAIL] {name} step {t}: {a} != {b}")
                    n_bad += 1
                    break
            else:
                print(f"[OK] {name}: {len(rec)} steps identical")
        if n_bad:
            sys.exit(1)
        print("REGRESSION CHECK PASSED")


if __name__ == "__main__":
    main()
