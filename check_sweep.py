"""Quick progress check for the latest sweep experiment."""
import json, glob, os, sys

# Auto-detect the latest sweep directory
base_candidates = [
    "/workspace/test_results/sf_sweep_260522",
    "/workspace/test_results/conn_cost_sweep_260522",
]
base = None
for b in base_candidates:
    if os.path.exists(b):
        base = b
        break

if base is None:
    print("No sweep directory found")
    sys.exit(1)

trials = sorted(glob.glob(os.path.join(base, "GradLogging*")))

for t in trials:
    result_file = os.path.join(t, "result.json")
    params_file = os.path.join(t, "params.json")

    label = "?"
    if os.path.exists(params_file):
        with open(params_file) as f:
            params = json.load(f)
        mc = params.get("model", {}).get("custom_model_config", {})
        sf = mc.get("scale_factor", "?")
        ec = params.get("env_config", {}).get("config", {}).get("env", {})
        w_conn = ec.get("acs_train_w_conn", 0)
        w_ctrl = ec.get("acs_train_w_ctrl", 0.02)
        label = f"sf={sf}"
        if w_conn > 0:
            label += f" w_conn={w_conn}"
        if w_ctrl != 0.02:
            label += f" w_ctrl={w_ctrl}"

    results = []
    if os.path.exists(result_file):
        with open(result_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if not results:
        print(f"{label}: no results yet")
        continue

    r = results[-1]
    it = r.get("training_iteration", 0)
    ep_rew = r.get("episode_reward_mean", 0)
    ls = r.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {})
    entropy = ls.get("entropy", 0)
    cm = r.get("custom_metrics", {})
    vel = cm.get("final_velocity_entropy_mean", -1)
    sp = cm.get("final_spatial_entropy_mean", -1)
    conn = cm.get("final_conn_ratio_mean", -1)
    flock = cm.get("flocking_success_mean", -1)

    conn_str = f" | conn={conn:.3f}" if conn >= 0 else ""
    print(f"{label:20s} | iter={it:3d} | rew={ep_rew:8.1f} | entropy={entropy:6.1f} | "
          f"vel_ent={vel:5.2f} | sp_ent={sp:5.1f}{conn_str} | flock={flock:.2f}")

    if len(results) > 5:
        best_vel = min(r2.get("custom_metrics", {}).get("final_velocity_entropy_mean", 999) for r2 in results)
        best_iter = next(r2.get("training_iteration", 0) for r2 in results
                        if r2.get("custom_metrics", {}).get("final_velocity_entropy_mean", 999) == best_vel)
        ent_first = results[0].get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {}).get("entropy", 0)
        ent_last = entropy
        print(f"  best vel_ent={best_vel:.3f}@iter{best_iter}, entropy_delta={ent_last-ent_first:.1f}")
