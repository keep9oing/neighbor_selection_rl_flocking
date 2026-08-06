"""Parallel sweep runner for the ACS k-NN convergence study.

Usage examples (run from anywhere; paths are absolute):
  python run_sweep.py --batch smoke --ks 5 --bounds 250 --reps 1 --steps 200 --workers 1
  python run_sweep.py --batch main --ks 1,2,3,4,5,6,7,8,10,12,15,19 \
      --bounds 125,250,500 --reps 20 --steps 3000 --workers 56

Disc-model batch (threshold in units of initial_position_bound/2):
  python run_sweep.py --batch disc --thresholds 0.2,0.35,0.48,0.7,1.0 \
      --bounds 250 --reps 20 --steps 3000 --workers 40

Output: one npz per run under data/<batch>/, progress in logs/<batch>.log.
Seeds are paired across k (and thresholds) within (bound, rep): seed = 1000 + rep,
so identical initial conditions are used for every k at a given (bound, rep).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse            # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys                 # noqa: E402
import time                # noqa: E402
import traceback           # noqa: E402

STUDY = "/workspace/studies/acs-conv-knn"
sys.path.insert(0, os.path.join(STUDY, "src"))

from common import run_episode, save_run  # noqa: E402


def _one(task):
    import numpy as np  # noqa: F401  (ensure per-worker import after env vars)
    kind, val, bound, rep, steps, n_agents, batch, stride = task
    seed = 1000 + rep
    if kind == "k":
        tag = f"k{val:02d}_L{int(bound)}_s{seed}"
        kw = dict(k=val)
    else:
        tag = f"th{val:.2f}_L{int(bound)}_s{seed}"
        kw = dict(distance_threshold=val)
    out = os.path.join(STUDY, "data", batch, tag + ".npz")
    if os.path.exists(out):
        return tag, "skip (exists)"
    try:
        t0 = time.time()
        rec, snaps, ts, meta = run_episode(
            n_agents=n_agents, max_steps=steps, initial_position_bound=bound,
            seed=seed, pos_stride=stride, **kw)
        save_run(out, rec, snaps, ts, meta)
        last = slice(max(0, meta["steps_done"] - 200), meta["steps_done"] + 1)
        import numpy as np
        msg = (f"ok {time.time()-t0:5.1f}s  s_end={np.nanmean(rec['s_ent'][last]):7.2f}"
               f" v_end={np.nanmean(rec['v_ent'][last]):6.3f}"
               f" phi_end={np.nanmean(rec['phi'][last]):5.3f}"
               f" comp_r0={rec['n_comp_r0'][meta['steps_done']]:2.0f}"
               f" dbg_diff={meta['env_metric_max_diff_spatial']:.2e}")
        return tag, msg
    except Exception:
        return tag, "FAIL\n" + traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--ks", default="", help="comma list of k values (nearest baseline)")
    ap.add_argument("--thresholds", default="", help="comma list (distance baseline)")
    ap.add_argument("--bounds", default="250", help="comma list of initial_position_bound")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-agents", type=int, default=20)
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x]
    ths = [float(x) for x in args.thresholds.split(",") if x]
    bounds = [float(x) for x in args.bounds.split(",") if x]
    tasks = []
    for b in bounds:
        for rep in range(args.reps):
            for k in ks:
                tasks.append(("k", k, b, rep, args.steps, args.n_agents,
                              args.batch, args.stride))
            for th in ths:
                tasks.append(("th", th, b, rep, args.steps, args.n_agents,
                              args.batch, args.stride))

    log_path = os.path.join(STUDY, "logs", f"{args.batch}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0 = time.time()
    with open(log_path, "a", buffering=1) as lf:
        lf.write(f"=== batch {args.batch}: {len(tasks)} tasks, "
                 f"{args.workers} workers ===\n")
        with mp.Pool(args.workers) as pool:
            for i, (tag, msg) in enumerate(pool.imap_unordered(_one, tasks, chunksize=1)):
                lf.write(f"[{i+1:4d}/{len(tasks)}] {tag}: {msg}\n")
        lf.write(f"=== done in {time.time()-t0:.0f}s ===\n")
    print(f"batch {args.batch} finished: {len(tasks)} tasks in {time.time()-t0:.0f}s; "
          f"log: {log_path}")


if __name__ == "__main__":
    main()
