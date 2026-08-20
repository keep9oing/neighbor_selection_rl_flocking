"""Worst-case topology-churn stress test for criterion C1.

The 'random' baseline re-samples the neighbor set i.i.d. EVERY step (churn ~ max
possible — harsher than any learned policy's switching). If sigma_p is stationary
within C1's 2%/500-step window under this, per-step stochastic selection per se
does not break the criterion.

Runs selection_probability p in {0.15, 0.3, 0.6} x 16 seeds (N=20, L=250, 6000
steps) -> data/stress_random/, then prints C1/C0 stats and churn levels.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import multiprocessing as mp  # noqa: E402
import sys                    # noqa: E402

STUDY = "/workspace/studies/acs-conv-knn"
sys.path.insert(0, os.path.join(STUDY, "src"))

from common import build_config, rollout, save_run, create_baseline  # noqa: E402


def _one(task):
    p, rep = task
    seed = 1000 + rep
    out = os.path.join(STUDY, "data", "stress_random", f"rnd{int(p*100):03d}_L250_s{seed}.npz")
    if os.path.exists(out):
        return out
    cfg = build_config(n_agents=20, max_steps=6000, initial_position_bound=250.0)
    pol = create_baseline("random", selection_probability=p, seed=seed)
    rec, snaps, ts, meta = rollout(pol, cfg, seed, pos_stride=10,
                                   extra_meta=dict(policy="random", sel_p=p))
    save_run(out, rec, snaps, ts, meta)
    return out


def main():
    tasks = [(p, rep) for p in (0.15, 0.3, 0.6) for rep in range(16)]
    with mp.Pool(48) as pool:
        for _ in pool.imap_unordered(_one, tasks):
            pass
    print("stress batch done:", len(tasks), "runs")


if __name__ == "__main__":
    main()
