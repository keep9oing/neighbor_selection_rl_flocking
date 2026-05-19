import copy

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from envs.env import NeighborSelectionFlockingEnv, Config, load_config
from models.ppo import NeighborSelectionPPORLlib
from callbacks import FlockingCallbacks

if __name__ == "__main__":

    enable_debugging = False
    # enable_debugging = True

    if enable_debugging:
        ray.init(local_mode=True)

    # Set up environment configuration
    default_config_path = "./envs/default_env_config.yaml"
    my_config = load_config(default_config_path)

    # environment configs
    my_config.env.acs_train_w_ctrl = 0.02
    my_config.env.acs_train_w_pos  = 1.0
    my_config.env.acs_train_w_vel  = 0.2  # overridden by grid_search below
    my_config.env.action_type = "binary_vector"
    my_config.env.agent_name_prefix = "agent_"
    my_config.env.alignment_goal = 0.97
    my_config.env.alignment_rate_goal = 0.03
    my_config.env.alignment_window_length = 32
    my_config.env.comm_range = None
    my_config.env.dt = 0.1
    my_config.env.entropy_p_goal = None
    my_config.env.entropy_p_rate_goal = 0.1
    my_config.env.entropy_rate_window_length = 50
    my_config.env.entropy_v_goal = 0.1
    my_config.env.entropy_v_rate_goal = 0.2
    my_config.env.env_mode = "single_env"
    my_config.env.get_action_hist = False
    my_config.env.get_state_hist = False
    my_config.env.ignore_comm_lost_agents = False
    my_config.env.is_training = True
    my_config.env.max_time_steps = 1000
    my_config.env.num_agents_pool = [20]
    my_config.env.obs_dim = 4
    my_config.env.periodic_boundary = False
    my_config.env.seed = None
    my_config.env.task_type = "acs"
    my_config.env.use_fixed_episode_length = True
    my_config.env.expose_aux_target = True

    # control config:
    my_config.control.beta = 1/3
    my_config.control.initial_position_bound = 250.0
    my_config.control.k1 = 1.0
    my_config.control.k2 = 3.0
    my_config.control.lam = 5.0
    my_config.control.max_turn_rate = 8/15
    my_config.control.r0 = 60.0
    my_config.control.rho = 1.0
    my_config.control.sig = 1.0
    my_config.control.speed = 15.0

    # Build env config dict; inject grid_search on velocity reward weight.
    # Tune resolves grid_search before passing to each trial, so both training
    # and eval (via RLlib's recursive merge) inherit the same resolved value.
    env_config_dict = my_config.dict()
    env_config_dict["env"]["acs_train_w_vel"] = tune.grid_search(
        [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
    )

    env_config = {
        "seed_id": 42,
        "config": env_config_dict,
    }
    env_name = "neighbor_selection_flocking_env"
    register_env(env_name, lambda cfg: NeighborSelectionFlockingEnv(cfg))

    # model config (all fixed — no grid_search here)
    custom_model_config = {
        "d_embed_context": 128,
        "d_embed_input": 128,
        "d_ff": 256,
        "d_ff_decoder": 256,
        "d_model": 128,
        "d_model_decoder": 128,
        "d_subobs": 4,
        "dr_rate": 0,
        "is_bias": False,
        "n_layers_decoder": 1,
        "n_layers_encoder": 3,
        "norm_eps": 1e-05,
        "num_heads": 4,
        "scale_factor": 0.002,
        "share_layers": False,
        "use_FNN_in_decoder": True,
        "use_residual_in_decoder": True,
        # Auxiliary task (fixed from prior sweep best)
        "aux_enabled": True,
        "aux_type": "pair_embedding",
        "aux_loss_coef": 0.3,
        "aux_target_dim": 4,
        "aux_loss_coef_critic": 0.05,
    }

    # register model
    model_name = "neighbor_selector_rl"
    ModelCatalog.register_custom_model(model_name, NeighborSelectionPPORLlib)

    # -------------------------------------------------------------------------
    # Velocity reward weight sweep.
    #
    # Prior findings:
    #   - Spatial cohesion achieved (entropy 40 vs goal 42) but velocity
    #     alignment stuck at ~1.2 (goal 0.1, 12× above threshold).
    #   - Current acs_train_w_vel=0.2 is 5× weaker than w_pos=1.0.
    #   - Best PPO settings from prior sweep: lr=1e-4, grad_clip=1.0.
    #   - Critic aux at 0.3 hurt; trying 0.05 (low enough to avoid
    #     competing with vf_loss but nonzero for representation benefit).
    #
    # This sweep tests 8 w_vel values from 0.2 (current) to 3.0 (3× spatial).
    # -------------------------------------------------------------------------

    tune.run(
        "PPO",
        name="vel_weight_sweep_260518",
        local_dir="/workspace/test_results",
        checkpoint_freq=10,
        keep_checkpoints_num=8,
        checkpoint_at_end=True,
        checkpoint_score_attr="evaluation/custom_metrics/flocking_success_mean",
        config={
            "env": env_name,
            "env_config": env_config,
            "framework": "torch",
            "callbacks": FlockingCallbacks,
            "model": {
                "custom_model": model_name,
                "custom_model_config": custom_model_config,
            },
            # --- Resources (per trial) ---
            "num_gpus": 0.25,
            "num_workers": 4,
            "num_envs_per_worker": 4,
            # --- Rollout / batch ---
            "rollout_fragment_length": 4000,
            "train_batch_size": 16000,
            "sgd_minibatch_size": 256,
            "num_sgd_iter": 10,
            # --- Learning rate (fixed from best prior trial) ---
            "lr": 1e-4,
            "lr_schedule": None,
            # --- PPO ---
            "vf_loss_coeff": 0.5,
            "use_critic": True,
            "use_gae": True,
            "gamma": 0.99,
            "lambda": 0.95,
            "kl_coeff": 0,
            "clip_param": 0.2,
            "vf_clip_param": 256,
            "grad_clip": 1.0,
            "kl_target": 0.01,
            "entropy_coeff": 0,
            # --- Evaluation ---
            # RLlib deep-merges evaluation_config with the base config,
            # so the swept acs_train_w_vel is inherited by eval workers.
            "evaluation_interval": 10,
            "evaluation_duration": 100,
            "evaluation_duration_unit": "episodes",
            "evaluation_num_workers": 3,
            "evaluation_config": {
                "env_config": {
                    "seed_id": 0,
                    "config": {
                        "env": {
                            "use_fixed_episode_length": False,
                        }
                    }
                },
            },
        },
    )
