"""RLlib PPO model for pointer-parameterized Dynamic-k NN selection."""

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from ray.rllib.utils.typing import TensorType

from models.ppo import NeighborSelectionPPORLlib


class DynamicKNNPPORLlib(NeighborSelectionPPORLlib):
    """Shared Transformer/pointer policy with one categorical choice per ego.

    The inherited actor applies the same encoder, decoder, and
    :class:`RawAttentionScoreGenerator` to every ego. Unlike the binary policy,
    its raw pointer scores are returned directly as ``N`` categorical logits
    for each of ``N`` egos. The RLlib action distribution is consequently a
    ``MultiDiscrete([N] * N)`` distribution rather than ``N*N`` binary choices.
    """

    respect_padding_mask = True

    @staticmethod
    def required_num_outputs(action_size: int) -> int:
        return action_size ** 2

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        if not hasattr(action_space, "nvec"):
            raise TypeError("DynamicKNNPPORLlib requires a MultiDiscrete action space")

        nvec = np.asarray(action_space.nvec)
        action_size = action_space.shape[0]
        if nvec.shape != (action_size,) or not np.all(nvec == action_size):
            raise ValueError("Dynamic-k NN action space must be MultiDiscrete([N] * N)")

        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name,
            **kwargs,
        )

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        obs_dict = input_dict["obs"]
        pointer_scores, active_ego_context = self.actor(obs_dict)
        logits = self.attention_scores_to_logits(
            pointer_scores, obs_dict["padding_mask"]
        )

        # NeighborSelectorTorch zeros padded ego contexts before taking this
        # mean, so the critic representation contains active egos only.
        if self.share_layers:
            self.values = active_ego_context.squeeze(1)
        else:
            self.values = self.critic(obs_dict)[1].squeeze(1)

        return logits, state

    def attention_scores_to_logits(
        self,
        attention_scores: TensorType,
        padding_mask: TensorType,
    ) -> TensorType:
        """Return flattened categorical pointer logits with padding masked."""
        batch_size, num_egos, num_candidates = attention_scores.shape
        if num_egos != num_candidates:
            raise ValueError("Dynamic-k NN action requires one candidate per ego index")

        active_agents = padding_mask.bool()
        # RawAttentionScoreGenerator already produces scaled dot-product
        # attention scores, so no binary-policy score/negation conversion is
        # applied here.
        logits = attention_scores

        # Every active agent is a candidate regardless of its distance. Only
        # padding columns are excluded from each ego's categorical distribution.
        candidate_mask = active_agents.unsqueeze(1).expand(-1, num_egos, -1)
        logits = logits.masked_fill(~candidate_mask, -1e9)

        # Padded ego action dimensions are ignored by the environment, but each
        # categorical distribution still needs a valid support point. Point it
        # deterministically at the first active agent so ignored dimensions add
        # neither random actions nor spurious entropy to PPO's joint log-prob.
        padded_egos = ~active_agents
        first_active = active_agents.to(torch.int64).argmax(dim=1)
        fallback = F.one_hot(first_active, num_classes=num_candidates).bool().unsqueeze(1)
        fallback = fallback & padded_egos.unsqueeze(-1)
        logits = logits.masked_fill(padded_egos.unsqueeze(-1), -1e9)
        logits = torch.where(fallback, torch.zeros_like(logits), logits)

        return logits.reshape(batch_size, num_egos * num_candidates)
