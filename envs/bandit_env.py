"""
Contextual-bandit wrapper around NeighborSelectionFlockingEnv for the `neighbor_index`
action type.

One outer step = one full inner episode. Inside outer.step():
  1. The given action is fed to the inner env, which caches it on the first inner step.
  2. The inner env is rolled out to completion (the cached action is reused for every
     subsequent inner step, as designed by the persistence logic in env.py).
  3. The cumulative inner return is reported as the outer reward.
  4. done=True is returned, so PPO sees a single (s_0, a, R) transition per episode.

This removes the wasted policy-gradient signal at inner steps >= 1, where the policy's
emitted action has no causal effect on rewards. The PPO advantage becomes r - V(s_0)
and the gradient at the only outer step carries the full attribution for the decision.
"""

import gym
import numpy as np

from envs.env import NeighborSelectionFlockingEnv


class NeighborIndexBanditEnv(gym.Env):
    def __init__(self, env_context):
        super().__init__()
        self.inner = NeighborSelectionFlockingEnv(env_context)
        assert self.inner.config.env.action_type == "neighbor_index", \
            "NeighborIndexBanditEnv requires inner action_type='neighbor_index'"
        self.observation_space = self.inner.observation_space
        self.action_space = self.inner.action_space
        # Discount factor used to combine inner per-step rewards into the outer return.
        # 1.0 = undiscounted sum, matching the natural bandit framing.
        self.inner_gamma = float(env_context.get("inner_gamma", 1.0))

    def seed(self, seed=None):
        return self.inner.seed(seed)

    def reset(self):
        return self.inner.reset()

    def step(self, action):
        total_return = 0.0
        gamma_t = 1.0
        last_obs = None
        last_info = {}
        inner_steps_taken = 0
        max_steps = self.inner.config.env.max_time_steps

        for _ in range(max_steps):
            obs, reward, done, info = self.inner.step(action)
            total_return += gamma_t * float(reward)
            gamma_t *= self.inner_gamma
            last_obs = obs
            last_info = info
            inner_steps_taken += 1
            if done:
                break

        last_info = dict(last_info) if last_info else {}
        last_info["episode_length_inner"] = inner_steps_taken
        last_info["episode_return_inner"] = total_return
        return last_obs, total_return, True, last_info
