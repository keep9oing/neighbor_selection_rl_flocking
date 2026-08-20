"""Custom PPO that logs pre-clip actor/critic gradient norms into learner_stats.

RLlib 2.1.0's multi-GPU code path (learn_on_loaded_batch) drops extra_grad_process's
return dict. tower_stats -> stats_fn is the only route to learner_stats on that path.
"""
import torch
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import apply_grad_clipping
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2


def _grad_norm(*modules):
    """L2 norm of .grad across all params of the given modules (pre-clip)."""
    sq = 0.0
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum().item())
    return torch.tensor(sq ** 0.5)


class GradLoggingPPOTorchPolicy(PPOTorchPolicy):

    @override(TorchPolicyV2)
    def extra_grad_process(self, local_optimizer, loss):
        m = self.model
        m.tower_stats["gnorm_total_preclip"] = _grad_norm(m)
        m.tower_stats["gnorm_actor_preclip"] = _grad_norm(m.actor)
        m.tower_stats["gnorm_critic_preclip"] = _grad_norm(
            getattr(m, "critic", None), m.value_branch
        )
        return apply_grad_clipping(self, local_optimizer, loss)

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch):
        stats = super().stats_fn(train_batch)
        for k in ("gnorm_total_preclip", "gnorm_actor_preclip", "gnorm_critic_preclip",
                  # study acs-c2-train: aux losses + saturation monitor stashed
                  # by NeighborSelectionPPORLlib.custom_loss into tower_stats
                  "aux_mse", "dist_aux", "dist_aux_coef_current", "sat_p_dev"):
            try:
                stats[k] = float(torch.mean(torch.stack(self.get_tower_stats(k))))
            except AssertionError:
                pass
        return stats


class GradLoggingPPO(PPO):
    @override(PPO)
    def get_default_policy_class(self, config):
        if config["framework"] == "torch":
            return GradLoggingPPOTorchPolicy
        return super().get_default_policy_class(config)
