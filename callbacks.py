import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch


class FlockingCallbacks(DefaultCallbacks):

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        info = episode.last_info_for()
        if info is None:
            return

        sp = info.get("spatial_entropy")
        vl = info.get("velocity_entropy")
        if sp is not None:
            episode.custom_metrics["final_spatial_entropy"] = float(sp)
        if vl is not None:
            episode.custom_metrics["final_velocity_entropy"] = float(vl)

        cr = info.get("conn_ratio")
        if cr is not None:
            episode.custom_metrics["final_conn_ratio"] = float(cr)

        max_steps = 1000
        try:
            env = base_env.get_sub_environments()[0]
            max_steps = env.config.env.max_time_steps
        except Exception:
            pass
        episode.custom_metrics["flocking_success"] = float(episode.length < max_steps)

    def on_postprocess_trajectory(self, *, worker, episode, agent_id,
                                  policy_id, policies, postprocessed_batch,
                                  original_batches, **kwargs):
        infos = postprocessed_batch[SampleBatch.INFOS]
        first_par = infos[0].get("per_agent_rewards") if len(infos) > 0 else None
        if first_par is not None:
            N = len(first_par)
            per_agent_rewards = np.array(
                [info.get("per_agent_rewards", np.zeros(N, dtype=np.float32))
                 for info in infos],
                dtype=np.float32,
            )
            postprocessed_batch["per_agent_rewards"] = per_agent_rewards
