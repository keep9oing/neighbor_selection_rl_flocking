from ray.rllib.algorithms.callbacks import DefaultCallbacks


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
