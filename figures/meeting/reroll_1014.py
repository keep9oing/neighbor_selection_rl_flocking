"""Re-roll seed 1014 @ L250 N20 for C1 (policy) and k12 (baseline), capturing
full per-step actions to t<=2000. Saves positions, actions, and per-step
non-nearest-edge shares (global + edges from the main flock toward the agent
k12 abandons). Verifies reproduction against the archived npz snapshots."""
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, "/workspace/studies/acs-robust-r3-stress/src")
sys.path.insert(0, "/workspace/studies/acs-conv-knn/src")
sys.path.insert(0, "/workspace")

from common import build_config  # noqa: E402
from eval_c2_r3 import C2Policy  # noqa: E402
from envs.env import NeighborSelectionFlockingEnv, config_to_env_input  # noqa: E402
from baselines import create_baseline  # noqa: E402

SEED, TMAX = 1014, 2000
CKPT = "/workspace/test_results/c2C1_ft40_lmix_260808/manual/checkpoint_000080"
OUT = "/workspace/figures/meeting/reroll_1014.npz"


def run(policy_kind):
    cfg = build_config(n_agents=20, max_steps=6000, initial_position_bound=250)
    if policy_kind == "C1":
        cfg.env.expose_aux_target = True
        cfg.env.expose_global_stats = True
        with open(os.path.join(os.path.dirname(CKPT), "params.json")) as f:
            _p = json.load(f)
        cfg.env.obs_position_scale = (_p.get("env_config", {}).get("config", {})
                                      .get("env", {}).get("obs_position_scale", "legacy"))
    env = NeighborSelectionFlockingEnv(config_to_env_input(cfg, seed_id=SEED))
    env.seed(SEED)
    obs = env.reset()
    policy = (C2Policy(CKPT, env) if policy_kind == "C1"
              else create_baseline("nearest", k=12))
    pos = [env.state["agent_states"][:, 0:2].copy()]
    acts = []
    for t in range(TMAX):
        a = policy(obs)
        obs, r, done, info = env.step(a)
        acts.append(a.astype(np.int8))
        pos.append(env.state["agent_states"][:, 0:2].copy())
    return np.stack(pos), np.stack(acts)


pos_c1, act_c1 = run("C1")
pos_k12, act_k12 = run("k12")

# verify reproduction vs archived rollouts (pos_snaps stride 10)
zc = np.load("/workspace/studies/acs-robust-r3-stress/data/eval/C1_i80_L250_s500/"
             "C1_i80_L250_s500_s1014.npz", allow_pickle=True)
zk = np.load("/workspace/studies/acs-robust-r2/data/knnref/k12_L250_N20/"
             "k12_L250_N20_s1014.npz", allow_pickle=True)
for tag, pos, z in (("C1", pos_c1, zc), ("k12", pos_k12, zk)):
    ts = z["snap_ts"]
    keep = ts <= TMAX
    diff = np.max(np.abs(z["pos_snaps"][keep] - pos[ts[keep]]))
    print(f"{tag} reproduction max|dpos| vs archive: {diff:.2e}")

# straggler = singleton component of k12 final archived frame
pf = zk["pos_snaps"][-1]
d = np.linalg.norm(pf[:, None] - pf[None, :], axis=-1)
adj = d < 60.0
lab = -np.ones(len(pf), int)
c = 0
for i in range(len(pf)):
    if lab[i] >= 0:
        continue
    st = [i]
    lab[i] = c
    while st:
        u = st.pop()
        for v in np.where(adj[u])[0]:
            if lab[v] < 0:
                lab[v] = c
                st.append(v)
    c += 1
sizes = np.bincount(lab)
strag = int(np.where(lab == np.argmin(sizes))[0][0]) if sizes.min() == 1 else -1
print("k12 final split:", sorted(sizes.tolist(), reverse=True), "straggler idx:", strag)


def nonnearest_share(pos, acts):
    """Per-step: share of selected off-diag edges outside each agent's
    nearest-deg_i set; plus count of flock->straggler in-edges that are
    non-nearest picks."""
    T = len(acts)
    share = np.zeros(T)
    strag_in = np.zeros(T, int)
    strag_in_nonnear = np.zeros(T, int)
    for t in range(T):
        p = pos[t]
        dd = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
        np.fill_diagonal(dd, np.inf)
        order = np.argsort(dd, axis=1)
        a = acts[t].astype(bool).copy()
        np.fill_diagonal(a, False)
        tot = out = 0
        for i in range(len(p)):
            deg = int(a[i].sum())
            if deg == 0:
                continue
            nearest = set(order[i, :deg].tolist())
            sel = np.where(a[i])[0]
            o = sum(1 for j in sel if j not in nearest)
            tot += deg
            out += o
            if strag >= 0 and i != strag and a[i, strag]:
                strag_in[t] += 1
                if strag not in nearest:
                    strag_in_nonnear[t] += 1
        share[t] = out / max(tot, 1)
    return share, strag_in, strag_in_nonnear

sh_c1, si_c1, sin_c1 = nonnearest_share(pos_c1, act_c1)
sh_k12, si_k12, _ = nonnearest_share(pos_k12, act_k12)
print("C1 non-nearest share: t<100 %.3f | 100-400 %.3f | 1000-2000 %.3f"
      % (sh_c1[:100].mean(), sh_c1[100:400].mean(), sh_c1[1000:].mean()))
print("straggler in-edges (C1): t<100 %.1f | 100-400 %.1f | 1000+ %.1f ; nonnear %.1f/%.1f/%.1f"
      % (si_c1[:100].mean(), si_c1[100:400].mean(), si_c1[1000:].mean(),
         sin_c1[:100].mean(), sin_c1[100:400].mean(), sin_c1[1000:].mean()))
print("straggler in-edges (k12): t<100 %.1f | 100-400 %.1f | 1000+ %.1f"
      % (si_k12[:100].mean(), si_k12[100:400].mean(), si_k12[1000:].mean()))

np.savez_compressed(OUT, pos_c1=pos_c1, act_c1=act_c1, pos_k12=pos_k12,
                    act_k12=act_k12, share_c1=sh_c1, share_k12=sh_k12,
                    strag=strag, strag_in_c1=si_c1, strag_in_nonnear_c1=sin_c1,
                    strag_in_k12=si_k12)
print("saved", OUT)
