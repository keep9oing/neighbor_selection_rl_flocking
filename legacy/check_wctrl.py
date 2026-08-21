"""Quick check of w_ctrl sweep training progress."""
import os
import json
import pandas as pd

base = "/workspace/test_results/wctrl_sweep_260523"
trials = sorted([d for d in os.listdir(base) if d.startswith("GradLoggingPPO_") and "91e0d" in d])

for trial_dir in trials:
    path = os.path.join(base, trial_dir, "progress.csv")
    wctrl = trial_dir.split("acs_train_w_ctrl=")[1].split("_")[0]
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"w_ctrl={wctrl}: no data yet")
        continue
    df = pd.read_csv(path)
    if len(df) == 0:
        print(f"w_ctrl={wctrl}: no iterations yet")
        continue
    last = df.iloc[-1]
    cols = {
        "iter": "training_iteration",
        "reward": "episode_reward_mean",
        "entropy": "info/learner/default_policy/learner_stats/entropy",
        "vel_ent": "custom_metrics/final_velocity_entropy_mean",
        "sp_ent": "custom_metrics/final_spatial_entropy_mean",
        "conn_ratio": "custom_metrics/final_conn_ratio_mean",
    }
    vals = {}
    for label, col in cols.items():
        if col in df.columns:
            vals[label] = last[col]
        else:
            vals[label] = "N/A"
    print(f"w_ctrl={wctrl}: iter={vals['iter']}, reward={vals.get('reward','?'):.2f}, "
          f"entropy={vals.get('entropy','?'):.1f}, vel_ent={vals.get('vel_ent','?'):.3f}, "
          f"sp_ent={vals.get('sp_ent','?'):.1f}, conn_ratio={vals.get('conn_ratio','?')}")
