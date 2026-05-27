"""
Sanity checks for the `neighbor_index` action_type (k-NN semantics, persistent k per agent).

Semantics under test:
- Step 0: action is an anchor index j_i per agent. The env converts to k_i = rank of j_i in i's
  current sorted distances to non-self active agents (k=0 if j_i == i).
- k is frozen for the rest of the episode; every step the env recomputes top-k_i nearest non-self
  active agents under the CURRENT positions.
- Different agents can hold different k. Pure k-NN: ranking ignores neighbor_masks (but excludes padding).
- Self-loop is always forced True on the diagonal for active agents.
"""

import numpy as np

from envs.env import NeighborSelectionFlockingEnv, load_config
from envs.bandit_env import NeighborIndexBanditEnv


def make_env(num_agents=5, num_agents_max=None, seed=0):
    if num_agents_max is None:
        num_agents_max = num_agents
    cfg = load_config("./envs/default_env_config.yaml")
    cfg.env.action_type = "neighbor_index"
    cfg.env.num_agents_pool = [num_agents_max]
    cfg.env.max_time_steps = 50
    cfg.env.use_fixed_episode_length = True
    cfg.env.comm_range = None  # full connectivity
    env_context = {"seed_id": seed, "config": cfg.dict()}
    env = NeighborSelectionFlockingEnv(env_context)
    return env


def test_step_returns_expected_info_keys():
    env = make_env(num_agents=5, seed=1)
    env.reset()
    n = env.num_agents_max
    # All agents pick agent 0 as anchor.
    action = np.zeros(n, dtype=np.int64)
    _, _, _, info = env.step(action)
    diag = env._last_neighbor_index_diagnostics
    assert diag is not None, "diagnostics should be populated after first step"
    assert diag["selected_neighbor_count_mean"] >= 1.0, "self-loop alone is 1; mean must be >= 1"
    for key in ("k_mean", "k_min", "k_max", "selected_neighbor_count_mean", "isolated_ratio"):
        assert key in info, f"info must include {key!r} for neighbor_index action"
    print("[OK] test_step_returns_expected_info_keys")


def test_k_freezes_after_first_step():
    env = make_env(num_agents=5, seed=2)
    env.reset()
    n = env.num_agents_max

    # Step 0: anchor = self for all -> each k_i should be 0 (no non-self with dist <= 0).
    a0 = np.arange(n, dtype=np.int64)
    env.step(a0)
    assert env._cached_k_per_agent is not None, "k vector must be cached after step 0"
    cached_k = env._cached_k_per_agent.copy()
    assert np.array_equal(cached_k, np.zeros(n, dtype=np.int64)), \
        f"self-anchor everywhere should yield k=0 for all; got {cached_k}"
    # Anchor we stored on step 0 must be the step-0 action.
    assert np.array_equal(env._last_anchor_idx, a0)

    # Subsequent steps with different anchors must not change cached k.
    a_other = np.zeros(n, dtype=np.int64)
    for _ in range(3):
        env.step(a_other)
    assert np.array_equal(env._cached_k_per_agent, cached_k), \
        "cached k must persist unchanged after subsequent steps"
    print("[OK] test_k_freezes_after_first_step")


def test_self_anchor_gives_k_zero():
    env = make_env(num_agents=4, seed=3)
    env.reset()
    n = env.num_agents_max
    # All agents pick themselves as anchor -> k_i = 0 -> only self-loop True per row.
    action = np.arange(n, dtype=np.int64)
    env.step(action)
    pad = env.state["padding_mask"]
    assert np.array_equal(env._cached_k_per_agent[pad], np.zeros(pad.sum(), dtype=np.int64))
    # Rebuild mask from cached k and inspect (the env applied it identically internally).
    mask = env.to_binary_action(env._cached_k_per_agent)
    for i in np.nonzero(pad)[0]:
        row = mask[i]
        active_true = np.nonzero(row)[0]
        assert len(active_true) == 1 and active_true[0] == i, \
            f"agent {i} should have only self-loop True; got {active_true}"
    # isolated_ratio in the diagnostics must reflect all-zero k.
    assert env._last_neighbor_index_diagnostics["isolated_ratio"] == 1.0
    print("[OK] test_self_anchor_gives_k_zero")


def test_k_matches_rank_at_step_zero():
    """Hand-craft positions and verify k_i equals rank-of-anchor (with <= tie semantics)."""
    env = make_env(num_agents=4, seed=4)
    env.reset()
    n = env.num_agents_max
    # 4 agents on a line at x = 0, 1, 2, 3 (y=0).
    p = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]], dtype=np.float64)
    v = np.zeros((4, 2), dtype=np.float64)
    th = np.zeros(4, dtype=np.float64)
    agent_states = np.concatenate([p, v, th[:, np.newaxis]], axis=1)
    env.state["agent_states"][:n] = agent_states
    env.rel_state = env.get_relative_state(env.state)

    # Agent 0 picks anchor = 2 (which is 2 units away, the 2nd closest non-self).
    # Agent 1 picks anchor = 3 (which is 2 units away from agent 1; agents 0 and 2 are 1 unit away tied).
    # Agent 2 picks anchor = 0 (which is 2 units away from agent 2; agents 1 and 3 are 1 unit away tied).
    # Agent 3 picks itself -> k=0.
    action = np.array([2, 3, 0, 3], dtype=np.int64)
    env.step(action)
    k = env._cached_k_per_agent
    # Agent 0: dists {1:1, 2:2, 3:3}; anchor=2 -> dist=2 -> {agents with dist <= 2} = {1, 2} -> k=2.
    assert k[0] == 2, f"expected k[0]=2, got {k[0]}"
    # Agent 1: dists {0:1, 2:1, 3:2}; anchor=3 -> dist=2 -> all three (0,2,3) have dist<=2 -> k=3.
    assert k[1] == 3, f"expected k[1]=3, got {k[1]}"
    # Agent 2: dists {0:2, 1:1, 3:1}; anchor=0 -> dist=2 -> all three (0,1,3) have dist<=2 -> k=3.
    assert k[2] == 3, f"expected k[2]=3, got {k[2]}"
    # Agent 3: self-anchor -> k=0.
    assert k[3] == 0, f"expected k[3]=0, got {k[3]}"
    print("[OK] test_k_matches_rank_at_step_zero")


def test_tie_distance_at_anchor_counted_in_k():
    """Square layout: tie-distance peer of the anchor must be counted in k (<= semantics)."""
    env = make_env(num_agents=4, seed=5)
    env.reset()
    n = env.num_agents_max
    # 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1). From 0: dist(1)=dist(2)=1, dist(3)=sqrt(2).
    p = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=np.float64)
    v = np.zeros((4, 2), dtype=np.float64)
    th = np.zeros(4, dtype=np.float64)
    agent_states = np.concatenate([p, v, th[:, np.newaxis]], axis=1)
    env.state["agent_states"][:n] = agent_states
    env.rel_state = env.get_relative_state(env.state)

    # Agent 0 picks anchor = 1 (dist 1.0); agent 2 also at dist 1.0 -> both counted -> k[0]=2.
    action = np.array([1, 0, 0, 0], dtype=np.int64)
    env.step(action)
    assert env._cached_k_per_agent[0] == 2, \
        f"tie-distance peer must count in k; got k[0]={env._cached_k_per_agent[0]}"
    # The selected row 0 should include self + agents 1 and 2 (the two closest non-self), exclude 3.
    mask = env.to_binary_action(env._cached_k_per_agent)
    row0 = mask[0]
    assert row0[0] == 1, "self-loop must be True"
    assert row0[1] == 1 and row0[2] == 1, "both tie-distance neighbors (1 and 2) must be selected"
    assert row0[3] == 0, "farther agent (3) must be excluded"
    print("[OK] test_tie_distance_at_anchor_counted_in_k")


def test_knn_selection_tracks_moving_neighbors():
    """After k is frozen at step 0, selection still recomputes against current distances every step."""
    env = make_env(num_agents=5, seed=6)
    env.reset()
    n = env.num_agents_max
    # Step 0 with arbitrary anchors; we only care that some agents end up with k > 0.
    # Pick anchor = (i+1) % n for everyone -> a cyclic non-self choice; k_i will be a positive rank.
    action = np.array([(i + 1) % n for i in range(n)], dtype=np.int64)
    env.step(action)
    k_per_agent = env._cached_k_per_agent.copy()

    # Step a few times; every step, the selected neighbors of i must be i's current top-k_i closest non-self.
    for _ in range(5):
        env.step(np.zeros(n, dtype=np.int64))  # ignored
        mask = env.to_binary_action(env._cached_k_per_agent)
        rel_dists = env.get_relative_state(env.state)["rel_agent_dists"]
        pad = env.state["padding_mask"]
        for i in np.nonzero(pad)[0]:
            ki = k_per_agent[i]
            # Compute expected top-k_i closest non-self active agents from rel_dists.
            valid = pad.copy()
            valid[i] = False
            d = np.where(valid, rel_dists[i], np.inf)
            expected_neighbors = set(np.argsort(d)[:ki].tolist()) if ki > 0 else set()
            row_neighbors = set(np.nonzero(mask[i])[0].tolist()) - {i}  # drop self-loop
            assert row_neighbors == expected_neighbors, \
                f"agent {i}: expected top-{ki} neighbors {expected_neighbors}; got {row_neighbors}"
    print("[OK] test_knn_selection_tracks_moving_neighbors")


def test_step_rejects_padding_anchor():
    # Build env with num_agents_pool that has padding: pool=[3,5] -> num_agents_max=5, pick 3 active.
    cfg = load_config("./envs/default_env_config.yaml")
    cfg.env.action_type = "neighbor_index"
    cfg.env.num_agents_pool = [3, 5]
    cfg.env.max_time_steps = 50
    cfg.env.use_fixed_episode_length = True
    cfg.env.comm_range = None
    env = NeighborSelectionFlockingEnv({"seed_id": 5, "config": cfg.dict()})
    # Reset until we draw 3 active agents.
    for _ in range(10):
        env.reset()
        if env.num_agents == 3:
            break
    assert env.num_agents == 3, "expected to land on num_agents=3 within 10 reset tries"
    n = env.num_agents_max  # 5
    pad = env.state["padding_mask"]
    padded_idx = int(np.where(~pad)[0][0])  # first padded agent index

    # Build invalid action: agent 0 points at a padded agent. step() must raise on step 0.
    action = np.zeros(n, dtype=np.int64)
    action[0] = padded_idx
    try:
        env.step(action)
    except AssertionError:
        print("[OK] test_step_rejects_padding_anchor")
        return
    raise AssertionError("step() should have rejected anchor pointing to padding agent")


def make_bandit_env(num_agents=5, seed=0, max_time_steps=20):
    cfg = load_config("./envs/default_env_config.yaml")
    cfg.env.action_type = "neighbor_index"
    cfg.env.num_agents_pool = [num_agents]
    cfg.env.max_time_steps = max_time_steps
    cfg.env.use_fixed_episode_length = True
    cfg.env.comm_range = None
    return NeighborIndexBanditEnv({"seed_id": seed, "config": cfg.dict(), "inner_gamma": 1.0})


def test_bandit_one_outer_step_runs_full_inner_episode():
    env = make_bandit_env(num_agents=5, seed=10, max_time_steps=25)
    env.reset()
    n = env.action_space.nvec.shape[0]
    action = np.zeros(n, dtype=np.int64)  # all anchor = agent 0
    _, reward, done, info = env.step(action)
    assert done is True, "outer step must always terminate the bandit episode"
    assert info["episode_length_inner"] == 25, \
        f"inner should run full max_time_steps=25; got {info['episode_length_inner']}"
    assert isinstance(reward, float), "outer reward must be a python float"
    print("[OK] test_bandit_one_outer_step_runs_full_inner_episode")


def test_bandit_action_only_set_once_per_outer_step():
    env = make_bandit_env(num_agents=4, seed=11, max_time_steps=10)
    env.reset()
    n = env.action_space.nvec.shape[0]
    # Anchor = self for all => k=0 for all => every agent isolated => isolated_ratio == 1.0.
    action_self = np.arange(n, dtype=np.int64)
    _, _, _, info_self = env.step(action_self)
    assert info_self["isolated_ratio"] == 1.0, \
        "all agents picked self; isolated_ratio must be 1.0"

    # New outer episode: different action -> not every agent isolated.
    env.reset()
    action_zero = np.zeros(n, dtype=np.int64)
    _, _, _, info_zero = env.step(action_zero)
    assert info_zero["isolated_ratio"] < 1.0, \
        "cached k should NOT leak across outer episodes (reset must clear it)"
    print("[OK] test_bandit_action_only_set_once_per_outer_step")


def test_bandit_undiscounted_return_matches_inner_sum():
    """Outer reward should equal the sum of inner per-step rewards when inner_gamma=1.0."""
    env = make_bandit_env(num_agents=4, seed=12, max_time_steps=15)
    env.reset()
    n = env.action_space.nvec.shape[0]
    action = np.zeros(n, dtype=np.int64)
    _, outer_reward, _, info = env.step(action)
    assert abs(outer_reward - info["episode_return_inner"]) < 1e-9
    print("[OK] test_bandit_undiscounted_return_matches_inner_sum")


if __name__ == "__main__":
    test_step_returns_expected_info_keys()
    test_k_freezes_after_first_step()
    test_self_anchor_gives_k_zero()
    test_k_matches_rank_at_step_zero()
    test_tie_distance_at_anchor_counted_in_k()
    test_knn_selection_tracks_moving_neighbors()
    test_step_rejects_padding_anchor()
    test_bandit_one_outer_step_runs_full_inner_episode()
    test_bandit_action_only_set_once_per_outer_step()
    test_bandit_undiscounted_return_matches_inner_sum()
    print("\nAll neighbor_index sanity checks passed.")
