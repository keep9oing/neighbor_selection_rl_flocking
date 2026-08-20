"""Phase 1 validation gates (study acs-c2-train):

Gate 2 — online/offline C2 parity: run k-NN(10) episodes with the env's
  termination_mode="c2" and independently re-judge the logged series with the
  offline (pandas rolling) reference semantics. Env t_conv must equal offline
  t_fire exactly; failures (cap-terminated) must also not fire offline.

Gate 3 — c2_shaping reward audit: with is_training=True and
  reward_mode="c2_shaping", reconstruct the reward from info stats + control
  inputs and compare to the env-returned reward each step; print per-term
  magnitudes for a transient and a settled snapshot.

Usage: python c2_crosscheck.py [--seeds 1000-1007] [--max-steps 1500]
"""
import argparse
import sys

import numpy as np
import pandas as pd

REPO = "/workspace"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from envs.env import NeighborSelectionFlockingEnv, load_config, config_to_env_input  # noqa: E402
from baselines import create_baseline  # noqa: E402

DEFAULT_YAML = "/workspace/envs/default_env_config.yaml"
PHI_GOAL, W_A, W, EPS = 0.98, 50, 300, 0.05


def build_cfg(max_steps, is_training, reward_mode, expose_gs=False):
    cfg = load_config(DEFAULT_YAML)
    cfg.env.task_type = "acs"
    cfg.env.observation_type = "ego_centric"
    cfg.env.num_agents_pool = [20]
    cfg.env.max_time_steps = int(max_steps)
    cfg.env.use_fixed_episode_length = False
    cfg.env.termination_mode = "c2"
    cfg.env.reward_mode = reward_mode
    cfg.env.is_training = is_training
    cfg.env.continuous_action = False
    cfg.env.expose_global_stats = expose_gs
    return cfg


def swarm_series_entry(env):
    st = env.state["agent_states"]
    pm = env.state["padding_mask"]
    pos = st[pm, 0:2].astype(np.float64)
    vel = st[pm, 2:4].astype(np.float64)
    unit = vel / np.maximum(np.linalg.norm(vel, axis=1), 1e-12)[:, None]
    phi = float(np.linalg.norm(unit.mean(axis=0)))
    s = float(np.sqrt(pos.var(axis=0).sum()))
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    # offline judge component count (bool-semiring closure, same as common.py)
    adj = d < 60.0
    n = adj.shape[0]
    reach = (adj | adj.T | np.eye(n, dtype=bool)).astype(np.float32)
    for _ in range(max(1, int(np.ceil(np.log2(max(n, 2)))))):
        reach = ((reach @ reach) > 0).astype(np.float32)
    n_comp = int(np.unique(reach > 0, axis=0).shape[0])
    return phi, s, n_comp


def offline_t_fire(phi, s, comp):
    pphi, ps, pcomp = pd.Series(phi), pd.Series(s), pd.Series(comp)
    align = (pphi.rolling(W_A).min() > PHI_GOAL).values
    coh = (pcomp.rolling(W).max() == 1).values
    band = ((ps.rolling(W).max() - ps.rolling(W).min()) / ps.rolling(W).mean()).values
    with np.errstate(invalid="ignore"):
        ok = align & coh & (band < EPS)
    hit = np.flatnonzero(ok)
    return int(hit[0]) if hit.size else -1


def gate2(seeds, max_steps):
    print(f"=== Gate 2: online/offline C2 parity (k-NN(10), seeds {seeds}) ===")
    n_bad = 0
    for seed in seeds:
        cfg = build_cfg(max_steps, is_training=False, reward_mode="legacy")
        env = NeighborSelectionFlockingEnv(config_to_env_input(cfg, seed_id=seed))
        env.seed(seed)
        obs = env.reset()
        policy = create_baseline("nearest", k=10)
        phis, ss, comps = [], [], []
        p0, s0, c0 = swarm_series_entry(env)
        phis.append(p0); ss.append(s0); comps.append(c0)
        done, t, info = False, 0, {}
        while not done and t < max_steps:
            obs, r, done, info = env.step(policy(obs))
            t += 1
            p, s, c = swarm_series_entry(env)
            phis.append(p); ss.append(s); comps.append(c)
        t_off = offline_t_fire(np.array(phis), np.array(ss), np.array(comps))
        c2s = info.get("c2_success", None)
        t_env = info.get("t_conv", -1)
        if c2s:
            ok = (t_env == t_off == t)
            print(f"seed {seed}: SUCCESS env t_conv={t_env}, offline t_fire={t_off}, steps={t} -> {'OK' if ok else 'MISMATCH'}")
        else:
            ok = (t_off == -1) and (t == max_steps)
            print(f"seed {seed}: FAILURE ran to {t} (cap {max_steps}), offline t_fire={t_off} -> {'OK' if ok else 'MISMATCH'}")
        n_bad += 0 if ok else 1
    return n_bad


def gate3(seed, max_steps):
    print(f"\n=== Gate 3: c2_shaping reward audit (seed {seed}) ===")
    cfg = build_cfg(max_steps, is_training=True, reward_mode="c2_shaping", expose_gs=True)
    env = NeighborSelectionFlockingEnv(config_to_env_input(cfg, seed_id=seed))
    env.seed(seed)
    obs = env.reset()
    policy = create_baseline("nearest", k=10)
    dt, speed, rho = cfg.env.dt, cfg.control.speed, cfg.control.rho
    n_bad, done, t = 0, False, 0
    printed = set()
    while not done and t < max_steps:
        obs, r, done, info = env.step(policy(obs))
        t += 1
        # reconstruct: per-agent raw reward mean + rho*dt = -dt*speed*mean|u|
        par = info["per_agent_rewards"]
        pm = env.state["padding_mask"]
        ctrl = par[pm].mean() + rho * dt
        pos_term = -4.0 * (1.0 - info["c2_f_largest"]) ** 2
        vel_term = -0.2 * max(0.98 - info["c2_phi"], 0.0) ** 2
        bonus = 10.0 if (done and info.get("c2_success")) else 0.0
        recon = pos_term + vel_term + 0.1 * ctrl + bonus
        # float32 op-order differences between env internals and this recon
        # are ~1e-8; anything beyond 2e-6 is a real formula drift.
        if abs(recon - r) > 2e-6:
            n_bad += 1
            if n_bad < 4:
                print(f"  step {t}: recon {recon:.6f} != env {r:.6f}")
        tag = "early" if t == 5 else ("mid" if t == 300 else ("fire" if done else None))
        if tag and tag not in printed:
            printed.add(tag)
            print(f"  [{tag} t={t}] pos={pos_term:+.4f} vel={vel_term:+.4f} "
                  f"ctrl={0.1*ctrl:+.4f} bonus={bonus:+.1f} total={r:+.4f} "
                  f"(phi={info['c2_phi']:.3f}, f_largest={info['c2_f_largest']:.2f}, "
                  f"gs={np.array2string(obs['global_stats'], precision=3)})")
    print(f"  episode end: t={t}, c2_success={info.get('c2_success')}, t_conv={info.get('t_conv')}, "
          f"reward mismatches={n_bad}")
    return n_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1000-1007")
    ap.add_argument("--max-steps", type=int, default=1500)
    args = ap.parse_args()
    a, b = args.seeds.split("-")
    seeds = list(range(int(a), int(b) + 1))
    bad = gate2(seeds, args.max_steps)
    bad += gate3(seeds[0], args.max_steps)
    if bad:
        print(f"\nGATES FAILED ({bad} problems)")
        sys.exit(1)
    print("\nALL PHASE-1 GATES PASSED")


if __name__ == "__main__":
    main()
