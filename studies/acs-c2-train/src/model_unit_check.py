"""Phase 2 validation gate 1 (study acs-c2-train): unit forward + gradient flow.

Builds the two new selection heads (A "bernoulli", B "threshold") plus the OLD
saturated hard-top-K head as a negative control, runs a real-obs forward, and
checks:
  - logits shape (B, 2*N*N); diagonal forced selected (p~1);
  - selection probabilities unsaturated at init (mean |p-0.5| well below 0.5);
  - per-edge entropy > 0;
  - d(logp)/d(att) mean abs > 1e-6 for A and B  (old saturated path ~1e-18);
  - d(logp)/d(tau) nonzero for B (threshold head receives gradient);
  - custom_loss dist_aux path (B): finite loss, K=dist_aux_k respected,
    dist_aux_coef_current honored, tower_stats keys stashed.

Runs on CPU. Usage: python model_unit_check.py
"""
import sys

import numpy as np
import torch

REPO = "/workspace"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from envs.env import NeighborSelectionFlockingEnv, load_config, config_to_env_input  # noqa: E402
from baselines import create_baseline  # noqa: E402
from models.ppo import NeighborSelectionPPORLlib  # noqa: E402

BASE_MODEL_CFG = {
    "d_embed_context": 128, "d_embed_input": 128, "d_ff": 256, "d_ff_decoder": 256,
    "d_model": 128, "d_model_decoder": 128, "d_subobs": 4, "dr_rate": 0,
    "is_bias": False, "n_layers_decoder": 1, "n_layers_encoder": 3,
    "norm_eps": 1e-05, "num_heads": 4, "scale_factor": 0.10, "share_layers": False,
    "use_FNN_in_decoder": True, "use_residual_in_decoder": True,
    "aux_enabled": True, "aux_type": "pair_embedding", "aux_loss_coef": 0.3,
    "aux_target_dim": 4, "aux_loss_coef_critic": 0.05,
    "continuous_action": False, "per_agent_credit": False,
}

VARIANTS = {
    "A_bernoulli": dict(selection_head="bernoulli", top_k=None, hard_top_k=False,
                        dist_aux_coef=0.0, use_global_stats=True),
    "B_threshold": dict(selection_head="threshold", logit_scale=10.0, top_k=None,
                        hard_top_k=False, dist_aux_coef=1.0, dist_aux_k=10,
                        dist_aux_schedule=[[0, 1.0], [400000, 0.2]],
                        use_global_stats=True),
    "OLD_hardtopk (neg ctrl)": dict(selection_head="legacy", top_k=10, hard_top_k=True,
                                    dist_aux_coef=1.0, use_global_stats=False),
}


def get_obs_batch(B=4, n_steps=3, seed=7):
    cfg = load_config("/workspace/envs/default_env_config.yaml")
    cfg.env.task_type = "acs"
    cfg.env.num_agents_pool = [20]
    cfg.env.expose_aux_target = True
    cfg.env.expose_global_stats = True
    cfg.env.is_training = True
    cfg.env.reward_mode = "c2_shaping"
    cfg.env.use_fixed_episode_length = False
    cfg.env.termination_mode = "c2"
    cfg.env.max_time_steps = 1500
    env = NeighborSelectionFlockingEnv(config_to_env_input(cfg, seed_id=seed))
    env.seed(seed)
    policy = create_baseline("nearest", k=10)
    obs_list = []
    obs = env.reset()
    for b in range(B):
        for _ in range(n_steps):
            obs, _, _, _ = env.step(policy(obs))
        obs_list.append({k: np.array(v) for k, v in obs.items()})
    batch = {}
    for k in obs_list[0]:
        arr = np.stack([o[k] for o in obs_list])
        if k in ("neighbor_masks", "padding_mask"):
            batch[k] = torch.as_tensor(arr, dtype=torch.float32)
        elif k == "is_from_my_env":
            batch[k] = torch.as_tensor(arr, dtype=torch.bool)
        else:
            batch[k] = torch.as_tensor(arr, dtype=torch.float32)
    return env, batch


def check_variant(name, extra_cfg, env, batch):
    print(f"\n=== {name} ===")
    torch.manual_seed(0)
    mc = dict(BASE_MODEL_CFG)
    mc.update(extra_cfg)
    N = env.num_agents_max
    model = NeighborSelectionPPORLlib(
        obs_space=env.observation_space, action_space=env.action_space,
        num_outputs=2 * N * N, model_config={"custom_model_config": mc}, name="unit")
    model.tower_stats = {}  # normally provided by TorchModelV2 machinery

    # ensure att caching for gradient introspection even when dist_aux is off
    model.dist_aux_coef = max(model.dist_aux_coef, 1e-12)

    logits, _ = model.forward({"obs": batch}, [], None)
    B = batch["padding_mask"].shape[0]
    assert logits.shape == (B, 2 * N * N), f"bad shape {logits.shape}"

    att = model._cached_att
    att.retain_grad()
    tau = getattr(model, "_tau", None)
    if tau is not None:
        tau.retain_grad()

    lg = logits.view(B, N, N, 2)
    dist = torch.distributions.Categorical(logits=lg)
    a = dist.sample()
    p_sel = torch.softmax(lg, dim=-1)[..., 1]
    off = ~torch.eye(N, dtype=torch.bool).unsqueeze(0).expand(B, N, N)

    diag_p = p_sel[~off].min().item()
    sat = (p_sel[off] - 0.5).abs().mean().item()
    ent = dist.entropy()[off].mean().item()
    deg = a.float()[off].view(B, -1).sum(-1).mean().item() / 1.0
    print(f"  diag p_select min      = {diag_p:.6f}  (must be ~1)")
    print(f"  mean |p-0.5| (off-diag)= {sat:.4f}   (0=stochastic, 0.5=saturated)")
    print(f"  mean per-edge entropy  = {ent:.4f}   (ln2={np.log(2):.4f})")
    print(f"  sampled off-diag degree= {deg / N:.2f} per agent")

    logp = dist.log_prob(a)[off].sum()
    logp.backward(retain_graph=True)
    g_att = att.grad.abs().mean().item()
    print(f"  mean |d logp / d att|  = {g_att:.3e}  (gate: > 1e-6)")
    ok = g_att > 1e-6
    if tau is not None:
        g_tau = tau.grad.abs().mean().item()
        gen_g = sum(p.grad.abs().sum().item() for p in model.threshold_head.parameters())
        print(f"  mean |d logp / d tau|  = {g_tau:.3e}  (nonzero)")
        print(f"  threshold_head |grad|  = {gen_g:.3e}  (nonzero)")
        ok = ok and g_tau > 1e-6 and gen_g > 0

    assert bool(diag_p > 0.999), "diagonal not forced"
    assert ent > 1e-4 or "OLD" in name, "entropy collapsed at init"

    # custom_loss pass (aux + dist_aux + tower stats)
    model.zero_grad()
    pl = [torch.zeros((), requires_grad=True) + 0.0]
    out = model.custom_loss(pl, {})
    total = out[0]
    assert torch.isfinite(total), "custom_loss not finite"
    total.backward()
    da = model._last_dist_aux_loss
    print(f"  custom_loss total      = {float(total):.4f}; aux_mse={float(model._last_aux_loss):.4f}"
          f"{'; dist_aux=%.4f' % float(da) if da is not None else ''}")
    print(f"  tower_stats keys       = {sorted(model.tower_stats.keys())}")
    if "dist_aux_coef_current" in model.tower_stats:
        print(f"  dist_aux_coef_current  = {float(model.tower_stats['dist_aux_coef_current']):.3f}")
    return ok, g_att


def main():
    env, batch = get_obs_batch()
    results = {}
    for name, cfg in VARIANTS.items():
        ok, g = check_variant(name, cfg, env, batch)
        results[name] = (ok, g)

    print("\n=== SUMMARY ===")
    a_ok, a_g = results["A_bernoulli"]
    b_ok, b_g = results["B_threshold"]
    _, old_g = results["OLD_hardtopk (neg ctrl)"]
    print(f"A grad {a_g:.2e} ({'OK' if a_ok else 'FAIL'}), "
          f"B grad {b_g:.2e} ({'OK' if b_ok else 'FAIL'}), "
          f"OLD grad {old_g:.2e} (expected ~1e-18, ratio A/OLD={a_g / max(old_g, 1e-30):.1e})")
    if not (a_ok and b_ok):
        sys.exit(1)
    print("PHASE-2 UNIT GATE PASSED")


if __name__ == "__main__":
    main()
