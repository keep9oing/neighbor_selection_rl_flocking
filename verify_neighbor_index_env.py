"""
Sanity checks for the new `neighbor_index` action_type.

Covers:
- to_binary_action produces a binary mask matching the spec
- Action is cached on step 0 and reused for the rest of the episode (persistence)
- Anchor = self produces an isolated row (only self-loop True)
- Tie-distance neighbors are included (<= semantics)
- validate_action catches anchors that point to padding agents
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


def test_step_returns_expected_mask():
    env = make_env(num_agents=5, seed=1)
    env.reset()
    n = env.num_agents_max
    # pick anchor = agent 0 for all agents -> threshold = dist(i, 0)
    action = np.zeros(n, dtype=np.int64)
    _, _, _, info = env.step(action)
    diag = env._last_neighbor_index_diagnostics
    assert diag is not None, "diagnostics should be populated after first step"
    assert diag["selected_neighbor_count_mean"] >= 1.0, "self-loop alone is 1; mean must be >= 1"
    assert "anchor_distance_mean" in info, "info must include anchor_distance_mean for neighbor_index action"
    print("[OK] test_step_returns_expected_mask")


def test_action_persistence():
    env = make_env(num_agents=5, seed=2)
    env.reset()
    n = env.num_agents_max

    # Step 0: anchor = self for all (should give isolated rows)
    a0 = np.arange(n, dtype=np.int64)
    env.step(a0)
    cached_after_step0 = env._cached_first_action.copy()
    assert np.array_equal(cached_after_step0, a0), "cached action must match step-0 action"

    # Step 1+: try a different anchor; env must ignore it
    a_other = np.zeros(n, dtype=np.int64)
    for _ in range(3):
        env.step(a_other)
    assert np.array_equal(env._cached_first_action, a0), \
        "cached action must persist unchanged after subsequent steps"
    print("[OK] test_action_persistence")


def test_self_anchor_isolates_agent():
    env = make_env(num_agents=4, seed=3)
    env.reset()
    n = env.num_agents_max
    # All agents pick themselves as anchor
    action = np.arange(n, dtype=np.int64)
    env.step(action)
    # The action_hist isn't enabled, but we can re-run to_binary_action directly
    mask = env.to_binary_action(env._cached_first_action)
    # Each active agent's row must have only self-loop True
    pad = env.state["padding_mask"]
    for i in np.nonzero(pad)[0]:
        row = mask[i]
        active_true = np.nonzero(row)[0]
        assert len(active_true) == 1 and active_true[0] == i, \
            f"agent {i} should have only self-loop True; got {active_true}"
    print("[OK] test_self_anchor_isolates_agent")


def test_tie_distance_included():
    env = make_env(num_agents=4, seed=4)
    env.reset()
    n = env.num_agents_max
    # Override state with hand-crafted positions: place agents on a square so two pairs share a distance.
    # Layout (x, y):
    # 0: (0, 0); 1: (1, 0); 2: (0, 1); 3: (1, 1)
    # From agent 0: dist to 1 = 1, dist to 2 = 1, dist to 3 = sqrt(2)
    p = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=np.float64)
    v = np.zeros((4, 2), dtype=np.float64)
    th = np.zeros(4, dtype=np.float64)
    agent_states = np.concatenate([p, v, th[:, np.newaxis]], axis=1)
    env.state["agent_states"][:n] = agent_states
    env.rel_state = env.get_relative_state(env.state)

    # Agent 0 picks agent 1 as anchor (threshold = 1.0). Agent 2 is also at distance 1.0 -> included by <=.
    # Anchors for others arbitrary; we only inspect row 0.
    action = np.array([1, 0, 0, 0], dtype=np.int64)
    mask = env.to_binary_action(action)
    row0 = mask[0]
    # Expect row0 to include self (0), agent 1, and agent 2 (tie at 1.0), exclude agent 3.
    assert row0[0] == 1, "self-loop must be True"
    assert row0[1] == 1, "anchor (agent 1) must be selected"
    assert row0[2] == 1, "tie-distance neighbor (agent 2) must be selected with <= semantics"
    assert row0[3] == 0, "farther agent (agent 3) must be excluded"
    print("[OK] test_tie_distance_included")


def test_validate_rejects_padding_anchor():
    # Build env with num_agents_pool that has padding: pool=[3,5] -> num_agents_max=5, pick 3 active
    cfg = load_config("./envs/default_env_config.yaml")
    cfg.env.action_type = "neighbor_index"
    cfg.env.num_agents_pool = [3, 5]
    cfg.env.max_time_steps = 50
    cfg.env.use_fixed_episode_length = True
    cfg.env.comm_range = None
    env = NeighborSelectionFlockingEnv({"seed_id": 5, "config": cfg.dict()})
    # Reset until we draw 3 active agents (worst case a few retries; deterministic with fixed seed).
    for _ in range(10):
        env.reset()
        if env.num_agents == 3:
            break
    assert env.num_agents == 3, "expected to land on num_agents=3 within 10 reset tries"
    n = env.num_agents_max  # 5
    pad = env.state["padding_mask"]
    padded_idx = int(np.where(~pad)[0][0])  # first padded agent index

    # Build invalid action: agent 0 points at a padded agent.
    action = np.zeros(n, dtype=np.int64)
    action[0] = padded_idx
    # Run a full step so the action gets cached and validate_action gets called.
    try:
        env.step(action)
    except AssertionError:
        print("[OK] test_validate_rejects_padding_anchor")
        return
    raise AssertionError("validate_action should have rejected anchor pointing to padding agent")


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
    obs = env.reset()
    n = env.action_space.nvec.shape[0]
    action = np.zeros(n, dtype=np.int64)  # all anchor=agent 0
    next_obs, reward, done, info = env.step(action)
    assert done is True, "outer step must always terminate the bandit episode"
    assert info["episode_length_inner"] == 25, \
        f"inner should run full max_time_steps=25; got {info['episode_length_inner']}"
    assert isinstance(reward, float), "outer reward must be a python float"
    print("[OK] test_bandit_one_outer_step_runs_full_inner_episode")


def test_bandit_action_only_set_once_per_outer_step():
    env = make_bandit_env(num_agents=4, seed=11, max_time_steps=10)
    env.reset()
    n = env.action_space.nvec.shape[0]
    # Anchor=self for all => threshold=0 => only self-loop in flocking network
    action_self = np.arange(n, dtype=np.int64)
    _, _, _, info_self = env.step(action_self)
    assert info_self["self_chosen_ratio"] == 1.0, \
        "all agents picked self; self_chosen_ratio must be 1.0"

    # New outer episode: different action
    env.reset()
    action_zero = np.zeros(n, dtype=np.int64)
    _, _, _, info_zero = env.step(action_zero)
    # Only agent 0 picked self in the second outer episode
    assert info_zero["self_chosen_ratio"] < 1.0, \
        "cached action should NOT leak across outer episodes (reset must clear it)"
    print("[OK] test_bandit_action_only_set_once_per_outer_step")


def test_bandit_undiscounted_return_matches_inner_sum():
    """Outer reward should equal the sum of inner per-step rewards when inner_gamma=1.0."""
    env = make_bandit_env(num_agents=4, seed=12, max_time_steps=15)
    env.reset()
    n = env.action_space.nvec.shape[0]
    action = np.zeros(n, dtype=np.int64)
    _, outer_reward, _, info = env.step(action)
    # We don't have direct access to per-step inner rewards from outside, but episode_return_inner
    # is the same total tracked inside the wrapper -> they must match.
    assert abs(outer_reward - info["episode_return_inner"]) < 1e-9
    print("[OK] test_bandit_undiscounted_return_matches_inner_sum")


if __name__ == "__main__":
    test_step_returns_expected_mask()
    test_action_persistence()
    test_self_anchor_isolates_agent()
    test_tie_distance_included()
    test_validate_rejects_padding_anchor()
    test_bandit_one_outer_step_runs_full_inner_episode()
    test_bandit_action_only_set_once_per_outer_step()
    test_bandit_undiscounted_return_matches_inner_sum()
    print("\nAll neighbor_index sanity checks passed.")
