"""Two-phase training: behavioral cloning from KNN, then RL fine-tuning.

Phase 1 (BC): Train the model to imitate K-nearest selection via supervised learning.
After BC, the model's learned weights encode distance-based selection without any
hardcoded bias. The encoder learns distance representations from the obs.

Phase 2 (RL): Fine-tune with PPO. The model starts from KNN-equivalent behavior
and can potentially improve by using heading information (cos/sin Δθ) that KNN ignores.
"""
import os, json, pickle, copy
import numpy as np
import torch
import torch.nn as nn
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from envs.env import NeighborSelectionFlockingEnv, Config, load_config
from models.ppo import NeighborSelectionPPORLlib
from baselines import create_baseline
from callbacks import FlockingCallbacks
from grad_logging_ppo import GradLoggingPPO


def collect_bc_data(env, knn_baseline, num_episodes=200):
    """Collect (obs, action) pairs from KNN baseline."""
    obs_list, action_list = [], []
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = knn_baseline(obs)
            obs_tensors = {k: torch.from_numpy(v).float() for k, v in obs.items()}
            obs_list.append(obs_tensors)
            action_list.append(torch.from_numpy(action).long())
            obs, _, done, _ = env.step(action)
    return obs_list, action_list


def train_bc(model, obs_list, action_list, device, num_epochs=50, lr=1e-3, batch_size=32):
    """Train model to imitate KNN actions via cross-entropy."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    N = len(obs_list)
    num_agents_max = action_list[0].shape[0]

    for epoch in range(num_epochs):
        indices = np.random.permutation(N)
        total_loss, total_acc, n_batches = 0.0, 0.0, 0
        for start in range(0, N, batch_size):
            batch_idx = indices[start:start+batch_size]
            B = len(batch_idx)

            batch_obs = {}
            for k in obs_list[0].keys():
                batch_obs[k] = torch.stack([obs_list[i][k] for i in batch_idx]).to(device)
            batch_actions = torch.stack([action_list[i] for i in batch_idx]).to(device)

            input_dict = {"obs": batch_obs}
            logits, _ = model.forward(input_dict, state=[], seq_lens=None)
            logits = logits.reshape(B, num_agents_max, num_agents_max, 2)

            targets = batch_actions.reshape(B, num_agents_max, num_agents_max)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, 2), targets.reshape(-1), reduction='mean'
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1)
            acc = (preds == targets).float().mean().item()
            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"  BC epoch {epoch+1}: loss={total_loss/n_batches:.4f}, acc={total_acc/n_batches:.4f}")

    model.eval()
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_config_path = "./envs/default_env_config.yaml"
    my_config = load_config(default_config_path)
    my_config.env.acs_train_w_ctrl = 0.02
    my_config.env.acs_train_w_pos = 1.0
    my_config.env.acs_train_w_vel = 0.2
    my_config.env.acs_train_w_conn = 0.0
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

    env_config_dict = my_config.dict()
    env = NeighborSelectionFlockingEnv({"seed_id": 42, "config": env_config_dict})

    custom_model_config = {
        "d_embed_context": 128, "d_embed_input": 128, "d_ff": 256, "d_ff_decoder": 256,
        "d_model": 128, "d_model_decoder": 128, "d_subobs": 4, "dr_rate": 0, "is_bias": False,
        "n_layers_decoder": 1, "n_layers_encoder": 3, "norm_eps": 1e-05, "num_heads": 4,
        "scale_factor": 0.15, "share_layers": False,
        "use_FNN_in_decoder": True, "use_residual_in_decoder": True,
        "aux_enabled": True, "aux_type": "pair_embedding",
        "aux_loss_coef": 0.3, "aux_target_dim": 4, "aux_loss_coef_critic": 0.05,
        "top_k": None, "distance_bias_scale": 0.0,
    }

    num_outputs = int(np.sum(env.action_space.nvec)) if hasattr(env.action_space, 'nvec') else 800
    model = NeighborSelectionPPORLlib(
        obs_space=env.observation_space, action_space=env.action_space,
        num_outputs=num_outputs,
        model_config={"custom_model_config": custom_model_config},
        name="bc_model",
    ).to(device)

    # Phase 1: Behavioral Cloning
    print("=" * 60)
    print("PHASE 1: Behavioral Cloning from K=10 Nearest")
    print("=" * 60)
    knn = create_baseline('nearest', k=10)
    print("Collecting BC data (100 episodes)...")
    obs_list, action_list = collect_bc_data(env, knn, num_episodes=100)
    print(f"Collected {len(obs_list)} timestep samples")
    print("Training BC...")
    model = train_bc(model, obs_list, action_list, device, num_epochs=50, lr=1e-3, batch_size=64)

    # Verify BC quality
    print("\nVerifying BC model...")
    obs = env.reset()
    with torch.no_grad():
        input_dict = {"obs": {k: torch.from_numpy(v[np.newaxis]).float().to(device) for k, v in obs.items()}}
        logits, _ = model.forward(input_dict, state=[], seq_lens=None)
    logits_np = logits.cpu().numpy()[0].reshape(20, 20, 2)
    bc_action = np.argmax(logits_np, axis=-1).astype(np.int8)
    knn_action = knn(obs)
    match = (bc_action == knn_action).mean()
    edges = bc_action[obs['padding_mask']][:, obs['padding_mask']].sum(axis=1) - 1
    print(f"  BC vs KNN action match: {match:.4f}")
    print(f"  BC edges/agent: {edges.mean():.1f}")

    # Save BC weights
    bc_weights_path = "/workspace/test_results/bc_knn_weights.pt"
    torch.save(model.state_dict(), bc_weights_path)
    print(f"  Saved BC weights to {bc_weights_path}")

    # Phase 2: RL Fine-tuning
    print("\n" + "=" * 60)
    print("PHASE 2: RL Fine-tuning from BC checkpoint")
    print("=" * 60)

    env_name = "neighbor_selection_flocking_env"
    model_name = "neighbor_selector_rl"
    register_env(env_name, lambda cfg: NeighborSelectionFlockingEnv(cfg))
    ModelCatalog.register_custom_model(model_name, NeighborSelectionPPORLlib)

    # Create a temporary RLlib checkpoint from BC weights
    bc_ckpt_dir = "/workspace/test_results/bc_checkpoint"
    os.makedirs(os.path.join(bc_ckpt_dir, "policies/default_policy"), exist_ok=True)
    # Save as RLlib policy state
    numpy_weights = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    policy_state = {"weights": numpy_weights}
    with open(os.path.join(bc_ckpt_dir, "policies/default_policy/policy_state.pkl"), "wb") as f:
        pickle.dump(policy_state, f)
    # Save params.json
    params = {
        "model": {"custom_model": model_name, "custom_model_config": custom_model_config},
        "env_config": {"seed_id": 42, "config": env_config_dict},
    }
    with open(os.path.join(bc_ckpt_dir, "params.json"), "w") as f:
        json.dump(params, f)

    tune.run(
        GradLoggingPPO,
        name="bc_rl_260526",
        local_dir="/workspace/test_results",
        checkpoint_freq=5,
        keep_checkpoints_num=5,
        checkpoint_at_end=True,
        stop={"training_iteration": 50},
        config={
            "env": env_name,
            "env_config": {"seed_id": 42, "config": env_config_dict},
            "framework": "torch",
            "callbacks": FlockingCallbacks,
            "model": {
                "custom_model": model_name,
                "custom_model_config": custom_model_config,
            },
            "num_gpus": 0.5,
            "num_workers": 4,
            "num_envs_per_worker": 4,
            "rollout_fragment_length": 1000,
            "train_batch_size": 16000,
            "sgd_minibatch_size": 256,
            "num_sgd_iter": 10,
            "lr": 5e-4,
            "lr_schedule": None,
            "vf_loss_coeff": 0.5,
            "use_critic": True,
            "use_gae": True,
            "gamma": 0.99,
            "lambda": 0.95,
            "kl_coeff": 0,
            "clip_param": 0.2,
            "vf_clip_param": 256,
            "grad_clip": None,
            "kl_target": 0.01,
            "entropy_coeff": 0,
            "evaluation_interval": None,
        },
    )
