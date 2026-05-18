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
    my_config.env.acs_train_w_vel  = 0.2
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

    # register environment
    env_config = {
        "seed_id": 42,
        "config": my_config.dict(),
    }
    env_name = "neighbor_selection_flocking_env"
    register_env(env_name, lambda cfg: NeighborSelectionFlockingEnv(cfg))

    # eval env config: same except non-fixed episode length (early termination on flocking success)
    eval_my_config = copy.deepcopy(my_config)
    eval_my_config.env.use_fixed_episode_length = False
    eval_env_config = {
        "seed_id": 0,
        "config": eval_my_config.dict(),
    }

    # model config
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
        # Auxiliary task
        "aux_enabled": True,
        "aux_type": "pair_embedding",
        "aux_loss_coef": 0.3,
        "aux_target_dim": 4,
        # Critic aux: sweep coef (0 = off, >0 = on with separate critic aux branch)
        "aux_loss_coef_critic": tune.grid_search([0, 0.3]),
    }

    # register model
    model_name = "neighbor_selector_rl"
    ModelCatalog.register_custom_model(model_name, NeighborSelectionPPORLlib)

    # -------------------------------------------------------------------------
    # PPO hyperparameter choices (with reasoning):
    #
    #   num_sgd_iter: 32 → 10
    #       Fewer SGD passes per batch to reduce overfitting on each rollout and
    #       speed up wall-clock time per training iteration.
    #
    #   rollout_fragment_length: 1024 → 4000
    #       RLlib counts total transitions per worker (across all vectorized envs).
    #       4000 / 4 envs = 1000 steps per env = exactly 1 full episode per env per fragment.
    #
    #   num_workers=4, num_envs_per_worker=4 → 16 env instances
    #       Same effective parallelism as before (16 envs). Fewer workers × more
    #       envs per worker = fewer inter-process comms, slightly more efficient.
    #
    #   train_batch_size: 16384 → 16000
    #       = 4 workers × 4000 fragment_length. Exactly one round of rollouts
    #       fills one train batch — no waiting for partial rounds.
    #
    #   lr: 2e-5 → sweep [5e-5, 1e-4]
    #       num_sgd_iter dropped 3.2×; higher lr compensates for fewer SGD steps.
    #       Both values tested: 5e-5 (conservative) and 1e-4 (aggressive).
    #
    #   grad_clip: sweep [0.5, 1.0]
    #       Prior analysis showed critic gradient dominates the shared clip budget,
    #       compressing actor gradient. grad_clip=1.0 gives the actor more room.
    #
    #   sgd_minibatch_size: 256 (unchanged)
    #       With train_batch_size=16000 and num_sgd_iter=10:
    #       SGD steps per iter = 16000/256 × 10 = 625 (down from prior 2048).
    #
    #   entropy_coeff: 0 (unchanged, low priority per user)
    # -------------------------------------------------------------------------

    tune.run(
        "PPO",
        name="critic_aux_sweep_260517",
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
            # --- Learning rate (swept) ---
            "lr": tune.grid_search([5e-5, 1e-4]),
            "lr_schedule": None,  # use fixed lr (swept); no schedule
            # --- PPO ---
            "vf_loss_coeff": 0.5,
            "use_critic": True,
            "use_gae": True,
            "gamma": 0.99,
            "lambda": 0.95,
            "kl_coeff": 0,
            "clip_param": 0.2,
            "vf_clip_param": 256,
            "grad_clip": tune.grid_search([0.5, 1.0]),
            "kl_target": 0.01,
            "entropy_coeff": 0,
            # --- Evaluation ---
            "evaluation_interval": 10,
            "evaluation_duration": 100,
            "evaluation_duration_unit": "episodes",
            "evaluation_num_workers": 3,
            "evaluation_config": {
                "env_config": eval_env_config,
            },
        },
    )
