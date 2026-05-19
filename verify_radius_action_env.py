import numpy as np
from gym.spaces import Box

from envs.env import Config, NeighborSelectionFlockingEnv, config_to_env_input, load_dict


def make_env(radius_max=None, comm_range=None, radius_clip_actions=True, seed=123):
    config_dict = load_dict("envs/default_env_config.yaml")
    config_dict["env"].update({
        "num_agents_pool": [3, 4],
        "action_type": "radius",
        "comm_range": comm_range,
        "max_time_steps": 5,
        "use_fixed_episode_length": True,
        "radius_min": 0.0,
        "radius_max": radius_max,
        "radius_clip_actions": radius_clip_actions,
    })
    config = Config(**config_dict)
    return NeighborSelectionFlockingEnv(config_to_env_input(config, seed_id=seed))


def reset_three_active_agents(env, comm_range):
    positions = np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [20.0, 0.0],
    ], dtype=np.float64)
    velocities = np.array([
        [15.0, 0.0],
        [15.0, 0.0],
        [15.0, 0.0],
    ], dtype=np.float64)
    headings = np.zeros((3, 1), dtype=np.float64)
    env.custom_reset(
        p_=positions,
        v_=velocities,
        th_=headings,
        num_agents_max=4,
        comm_range=comm_range,
    )


def assert_active_self_loops_and_no_padding(mask):
    assert np.all(np.diag(mask)[:3]), "active self-loops must be selected"
    assert not mask[3, :].any(), "padding row must not select anything"
    assert not mask[:, 3].any(), "padding column must not be selected"


def test_action_space_and_default_radius_max():
    env = make_env(radius_max=None)

    assert isinstance(env.action_space, Box), "radius action space must be a Box"
    assert env.action_space.shape == (4,), f"unexpected action shape: {env.action_space.shape}"
    assert env.action_space.dtype == np.float32, f"unexpected action dtype: {env.action_space.dtype}"
    assert env.config.env.radius_max == env.config.control.initial_position_bound, \
        "radius_max=None must resolve to initial_position_bound"


def test_comm_range_none_selects_from_full_active_flock():
    env = make_env(radius_max=30.0, comm_range=None)
    reset_three_active_agents(env, comm_range=None)

    selected = env.to_binary_action(np.ones(4, dtype=np.float32))
    expected = np.array([
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 0],
    ], dtype=bool)

    assert np.array_equal(selected, expected), f"unexpected comm_range=None mask:\n{selected}"
    assert_active_self_loops_and_no_padding(selected)


def test_clipping_self_loops_padding_and_info():
    env = make_env(radius_max=10.0, comm_range=None, radius_clip_actions=True)
    reset_three_active_agents(env, comm_range=None)

    raw_action = np.array([0.5, 1.2, -0.5, 0.9], dtype=np.float32)
    selected = env.to_binary_action(raw_action)
    expected = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
    ], dtype=bool)

    assert np.array_equal(selected, expected), f"unexpected clipped radius mask:\n{selected}"
    assert_active_self_loops_and_no_padding(selected)

    _, _, _, info = env.step(raw_action)
    expected_keys = {
        "radius_mean",
        "radius_min",
        "radius_max",
        "radius_action_mean",
        "selected_neighbor_count_mean",
    }
    missing_keys = expected_keys - set(info)
    assert not missing_keys, f"missing radius info keys: {missing_keys}"
    assert np.isclose(info["radius_action_mean"], 0.5), info
    assert np.isclose(info["radius_mean"], 5.0), info
    assert np.isclose(info["radius_min"], 0.0), info
    assert np.isclose(info["radius_max"], 10.0), info
    assert np.isclose(info["selected_neighbor_count_mean"], 5.0 / 3.0), info


def test_finite_comm_range_limits_radius_selection():
    env = make_env(radius_max=30.0, comm_range=8.0)
    reset_three_active_agents(env, comm_range=8.0)

    selected = env.to_binary_action(np.ones(4, dtype=np.float32))
    expected = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
    ], dtype=bool)

    assert np.array_equal(selected, expected), f"finite comm_range was not enforced:\n{selected}"
    assert_active_self_loops_and_no_padding(selected)


def main():
    tests = [
        test_action_space_and_default_radius_max,
        test_comm_range_none_selects_from_full_active_flock,
        test_clipping_self_loops_padding_and_info,
        test_finite_comm_range_limits_radius_selection,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("All radius-action environment checks passed.")


if __name__ == "__main__":
    main()
