"""Phase 1 validation gates (study acs-robust-train).

Gate A — initial_position_bound_pool: per-episode L sampling is seeded-
  deterministic, covers the pool, bounds the initial positions, and the
  legacy obs normalization follows the EPISODE's L (not the static config).
Gate B — obs_position_scale="r0_log": exact transform math on the ego obs
  (direction preserved, magnitude log1p(r/r0), diag zero, finite, in range),
  on the centralized aux target (norm mapping), and on global_stats sigma_p.
Gate C — C2 online/offline parity at L=500 (k-NN(12)), plus action/trajectory
  equivalence of the k-NN baseline under r0_log obs (log1p is monotone ->
  identical neighbor ordering -> identical trajectory and t_conv).

Usage: python scale_unit_check.py   (CPU, ~3 min)
"""
import sys

import numpy as np
import pandas as pd

REPO = "/workspace"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from envs.env import NeighborSelectionFlockingEnv, load_config, config_to_env_input  # noqa: E402
from baselines import create_baseline  # noqa: E402

DEFAULT_YAML = "/workspace/envs/default_env_config.yaml"
R0 = 60.0
PHI_GOAL, W_A, W, EPS = 0.98, 50, 300, 0.05
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def build_cfg(pool=None, scale="legacy", bound=250.0, rotated=True, max_steps=1500,
              aux=False, gs=False):
    cfg = load_config(DEFAULT_YAML)
    cfg.env.task_type = "acs"
    cfg.env.observation_type = "ego_centric"
    cfg.env.num_agents_pool = [20]
    cfg.env.max_time_steps = int(max_steps)
    cfg.env.use_fixed_episode_length = False
    cfg.env.termination_mode = "c2"
    cfg.env.reward_mode = "legacy"
    cfg.env.is_training = False
    cfg.env.continuous_action = False
    cfg.env.use_rotated_ego_obs = rotated
    cfg.env.expose_aux_target = aux
    cfg.env.expose_global_stats = gs
    cfg.env.initial_position_bound_pool = pool
    cfg.env.obs_position_scale = scale
    cfg.control.initial_position_bound = float(bound)
    return cfg


def fresh_env(cfg, seed):
    env = NeighborSelectionFlockingEnv(config_to_env_input(cfg, seed_id=seed))
    env.seed(seed)
    return env


def active_pos(env):
    st, pm = env.state["agent_states"], env.state["padding_mask"]
    return st[pm, 0:2].astype(np.float64)


# ------------------------------------------------------------------ Gate A
def gate_a():
    print("=== Gate A: initial_position_bound_pool ===")
    pool = [125.0, 250.0, 500.0]
    env = fresh_env(build_cfg(pool=pool, rotated=False), seed=7)
    seen, ok_bound = [], True
    for _ in range(30):
        env.reset()
        L = env.episode_position_bound
        seen.append(L)
        p = active_pos(env)
        ok_bound &= bool(np.max(np.abs(p)) <= L / 2 + 1e-9)
    check("pool coverage over 30 resets", set(seen) == set(pool), f"seen {sorted(set(seen))}")
    check("positions bounded by episode L/2", ok_bound)

    env2 = fresh_env(build_cfg(pool=pool, rotated=False), seed=7)
    seq2 = []
    for _ in range(10):
        env2.reset()
        seq2.append((env2.episode_position_bound, active_pos(env2)[0].copy()))
    env3 = fresh_env(build_cfg(pool=pool, rotated=False), seed=7)
    same = True
    for k in range(10):
        env3.reset()
        same &= env3.episode_position_bound == seq2[k][0]
        same &= bool(np.allclose(active_pos(env3)[0], seq2[k][1]))
    check("seed -> identical (L, init) sequence", same)

    # legacy obs normalization follows the EPISODE L (world-frame obs)
    obs = env.reset()
    L = env.episode_position_bound
    p = active_pos(env)
    d01 = p[1] - p[0]
    o01 = obs["local_agent_infos"][0, 1, :2]
    conv = None
    if np.allclose(o01, d01 / (L / 2), atol=1e-12):
        conv = "d = p_j - p_i"
    elif np.allclose(o01, -d01 / (L / 2), atol=1e-12):
        conv = "d = p_i - p_j"
    check("legacy obs == rel/(episode L/2)", conv is not None, f"convention: {conv}, L={L}")
    return 1 if conv == "d = p_i - p_j" else 0  # sign for Gate B reuse


# ------------------------------------------------------------------ Gate B
def gate_b(sign_flip):
    print("=== Gate B: obs_position_scale='r0_log' ===")
    env = fresh_env(build_cfg(pool=[500.0], scale="r0_log", rotated=False,
                              aux=True, gs=True), seed=11)
    obs = env.reset()
    p = active_pos(env)
    n = p.shape[0]
    X = obs["local_agent_infos"][:n, :n, :2]
    rel = p[None, :, :] - p[:, None, :]           # d = p_j - p_i
    if sign_flip:
        rel = -rel
    r = np.linalg.norm(rel, axis=2, keepdims=True)
    expected = rel * (np.log1p(r / R0) / np.maximum(r, 1e-12))
    check("ego obs == unit(d)*log1p(|d|/r0)", bool(np.allclose(X, expected, atol=1e-9)),
          f"max dev {np.max(np.abs(X - expected)):.2e}")
    check("diagonal zero", bool(np.allclose(np.diagonal(X, axis1=0, axis2=1), 0.0)))
    check("finite + range at L=500", bool(np.all(np.isfinite(X)) and np.max(np.abs(X)) < 3.0),
          f"max |obs| {np.max(np.abs(X)):.3f}")
    cosang = np.sum(X * rel, axis=2) / np.maximum(
        np.linalg.norm(X, axis=2) * np.linalg.norm(rel, axis=2), 1e-12)
    offdiag = ~np.eye(n, dtype=bool)
    check("direction preserved", bool(np.all(cosang[offdiag] > 1 - 1e-9)))

    aux = obs["global_agent_infos"][:n, :2]
    center = p.mean(axis=0)
    mag = np.linalg.norm(p - center, axis=1)
    check("aux target |xy| == log1p(|p-center|/r0)",
          bool(np.allclose(np.linalg.norm(aux, axis=1), np.log1p(mag / R0), atol=1e-9)))

    sigma_p = float(np.sqrt(p.var(axis=0).sum()))
    check("global_stats sigma term == log1p(sigma_p/r0)",
          bool(abs(obs["global_stats"][3] - np.log1p(sigma_p / R0)) < 1e-9))

    # legacy + pool: aux target and global_stats use the EPISODE L
    env2 = fresh_env(build_cfg(pool=[125.0], scale="legacy", rotated=False,
                               aux=True, gs=True), seed=11)
    obs2 = env2.reset()
    p2 = active_pos(env2)
    c2 = p2.mean(axis=0)
    mag2 = np.linalg.norm(p2 - c2, axis=1)
    aux2 = obs2["global_agent_infos"][:p2.shape[0], :2]
    check("legacy aux |xy| == |p-center|/(episode L/2)",
          bool(np.allclose(np.linalg.norm(aux2, axis=1), mag2 / (125.0 / 2), atol=1e-9)))
    sp2 = float(np.sqrt(p2.var(axis=0).sum()))
    check("legacy global_stats sigma == sigma_p/(episode L/2)",
          bool(abs(obs2["global_stats"][3] - sp2 / 62.5) < 1e-9))


# ------------------------------------------------------------------ Gate C
def swarm_entry(env):
    p = active_pos(env)
    st, pm = env.state["agent_states"], env.state["padding_mask"]
    vel = st[pm, 2:4].astype(np.float64)
    unit = vel / np.maximum(np.linalg.norm(vel, axis=1), 1e-12)[:, None]
    phi = float(np.linalg.norm(unit.mean(axis=0)))
    s = float(np.sqrt(p.var(axis=0).sum()))
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    adj = d < R0
    nn = adj.shape[0]
    reach = (adj | adj.T | np.eye(nn, dtype=bool)).astype(np.float32)
    for _ in range(max(1, int(np.ceil(np.log2(max(nn, 2)))))):
        reach = ((reach @ reach) > 0).astype(np.float32)
    return phi, s, int(np.unique(reach > 0, axis=0).shape[0])


def offline_t_fire(phi, s, comp):
    pphi, ps, pcomp = pd.Series(phi), pd.Series(s), pd.Series(comp)
    align = (pphi.rolling(W_A).min() > PHI_GOAL).values
    coh = (pcomp.rolling(W).max() == 1).values
    band = ((ps.rolling(W).max() - ps.rolling(W).min()) / ps.rolling(W).mean()).values
    with np.errstate(invalid="ignore"):
        ok = align & coh & (band < EPS)
    hit = np.flatnonzero(ok)
    return int(hit[0]) if hit.size else -1


def run_knn_episode(scale, seed, max_steps=3000):
    cfg = build_cfg(scale=scale, bound=500.0, max_steps=max_steps)
    env = fresh_env(cfg, seed)
    obs = env.reset()
    policy = create_baseline("nearest", k=12)
    phis, ss, comps = [], [], []
    e = swarm_entry(env)
    phis.append(e[0]); ss.append(e[1]); comps.append(e[2])
    done, t, info = False, 0, {}
    while not done and t < max_steps:
        obs, r, done, info = env.step(policy(obs))
        t += 1
        e = swarm_entry(env)
        phis.append(e[0]); ss.append(e[1]); comps.append(e[2])
    return t, info, offline_t_fire(np.array(phis), np.array(ss), np.array(comps))


def gate_c():
    print("=== Gate C: C2 parity at L=500 (k-NN(12)) + r0_log ordering ===")
    legacy_tconv = {}
    for seed in (1000, 1001, 1002, 1003):
        t, info, t_off = run_knn_episode("legacy", seed)
        succ = bool(info.get("c2_success"))
        t_env = info.get("t_conv", -1)
        ok = (succ and t_env == t_off == t) or (not succ and t_off == -1 and t == 3000)
        legacy_tconv[seed] = t_env if succ else -1
        check(f"seed {seed} parity", ok,
              f"{'SUCCESS' if succ else 'FAILURE'} env={t_env} offline={t_off} steps={t}")
    t, info, t_off = run_knn_episode("r0_log", 1000)
    same = bool(info.get("c2_success")) and info.get("t_conv") == legacy_tconv[1000]
    check("r0_log preserves k-NN trajectory (seed 1000)", same,
          f"t_conv {info.get('t_conv')} vs legacy {legacy_tconv[1000]}")


def main():
    sign = gate_a()
    gate_b(sign)
    gate_c()
    if FAILS:
        print(f"\nGATES FAILED: {FAILS}")
        sys.exit(1)
    print("\nALL SCALE GATES PASSED")


if __name__ == "__main__":
    main()
