from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import TensorType


class NeighborSelectionRadiusPPORLlib(TorchModelV2, nn.Module):
    """
    Ego-centric PPO model for radius-action neighbor selection.

    The policy consumes the existing ego-centric observation:
        local_agent_infos: (batch, N, N, obs_dim)
        neighbor_masks:    (batch, N, N)
        padding_mask:      (batch, N)

    It emits RLlib diagonal-Gaussian parameters for a Box action of shape (N,):
        output[:, :N]  = per-agent radius means
        output[:, N:]  = per-agent log standard deviations
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        cfg = (model_config or {}).get("custom_model_config", {})

        if not isinstance(action_space, Box) or len(action_space.shape) != 1:
            raise ValueError(
                "NeighborSelectionRadiusPPORLlib requires a one-dimensional Box action space "
                f"with shape (num_agents_max,), got {action_space}."
            )

        self.num_agents_max = int(action_space.shape[0])
        if num_outputs != 2 * self.num_agents_max:
            raise ValueError(
                "Radius PPO model must output Gaussian mean and log_std for each agent: "
                f"expected {2 * self.num_agents_max}, got {num_outputs}."
            )

        d_subobs = int(cfg["d_subobs"]) if "d_subobs" in cfg else self._infer_obs_dim(obs_space)
        d_embed = int(cfg.get("d_embed_input", 128))
        n_layers_encoder = int(cfg.get("n_layers_encoder", 3))
        n_heads = int(cfg.get("num_heads", 4))
        d_ff = int(cfg.get("d_ff", 256))
        dr_rate = float(cfg.get("dr_rate", 0.0))
        norm_eps = float(cfg.get("norm_eps", 1e-5))

        self.share_layers = bool(cfg.get("share_layers", False))
        self.trainable_log_std = bool(cfg.get("trainable_log_std", cfg.get("free_log_std", True)))
        self.log_std_min = float(cfg.get("log_std_min", -5.0))
        self.log_std_max = float(cfg.get("log_std_max", 2.0))
        self.padding_action_mean = float(cfg.get("padding_action_mean", 0.0))
        self.mean_activation = cfg.get("mean_activation", "identity")
        self.initial_mean = float(cfg.get("initial_mean", 0.5))

        init_log_std = cfg.get("log_std", cfg.get("initial_log_std", cfg.get("init_log_std", -0.5)))
        init_log_std_tensor = self._make_agent_vector(init_log_std, self.num_agents_max, "initial_log_std")

        self.action_low = self._action_bound_tensor(action_space.low, default=0.0)
        self.action_high = self._action_bound_tensor(action_space.high, default=1.0)

        self.token_embedding = nn.Linear(d_subobs, d_embed)
        self.encoder = self._build_encoder(d_embed, n_heads, d_ff, dr_rate, norm_eps, n_layers_encoder)

        self.mean_head = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, 1),
        )
        self._init_mean_head()

        if self.trainable_log_std:
            self.log_std = nn.Parameter(init_log_std_tensor)
        else:
            self.register_buffer("log_std", init_log_std_tensor)

        if not self.share_layers:
            self.critic_token_embedding = nn.Linear(d_subobs, d_embed)
            self.critic_encoder = self._build_encoder(d_embed, n_heads, d_ff, dr_rate, norm_eps, n_layers_encoder)

        self.value_branch = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, 1),
        )

        self._context_for_value: Optional[TensorType] = None

    @staticmethod
    def _build_encoder(d_embed, n_heads, d_ff, dr_rate, norm_eps, n_layers_encoder):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embed,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dr_rate,
            layer_norm_eps=norm_eps,
            norm_first=True,
            batch_first=True,
        )
        return nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers_encoder,
            enable_nested_tensor=False,
        )

    @staticmethod
    def _infer_obs_dim(obs_space):
        if hasattr(obs_space, "original_space"):
            obs_space = obs_space.original_space
        try:
            return int(obs_space["local_agent_infos"].shape[-1])
        except Exception:
            return int(obs_space.spaces["local_agent_infos"].shape[-1])

    @staticmethod
    def _make_agent_vector(value, num_agents_max, name):
        if np.isscalar(value):
            return torch.full((num_agents_max,), float(value), dtype=torch.float32)

        values = torch.as_tensor(value, dtype=torch.float32)
        if values.numel() == 1:
            return values.reshape(1).expand(num_agents_max).clone()
        if values.shape != (num_agents_max,):
            raise ValueError(f"{name} must be scalar or shape ({num_agents_max},), got {tuple(values.shape)}.")
        return values.clone()

    def _init_mean_head(self):
        final_layer = self.mean_head[-1]
        nn.init.zeros_(final_layer.weight)
        if self.mean_activation == "sigmoid":
            nn.init.zeros_(final_layer.bias)
        else:
            nn.init.constant_(final_layer.bias, self.initial_mean)

    @staticmethod
    def _action_bound_tensor(bound, default):
        bound_tensor = torch.as_tensor(bound, dtype=torch.float32)
        finite = torch.isfinite(bound_tensor)
        default_tensor = torch.full_like(bound_tensor, float(default))
        return torch.where(finite, bound_tensor, default_tensor)

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        obs_dict = input_dict["obs"]

        local_agent_infos = obs_dict["local_agent_infos"].float()
        neighbor_masks = obs_dict["neighbor_masks"].bool()
        padding_mask = obs_dict["padding_mask"].bool()

        if local_agent_infos.dim() != 4:
            raise ValueError(
                "local_agent_infos must have shape (batch, N, N, obs_dim), "
                f"got {tuple(local_agent_infos.shape)}."
            )

        batch_size, num_agents_max, neighbor_count, _ = local_agent_infos.shape
        if num_agents_max != self.num_agents_max or neighbor_count != self.num_agents_max:
            raise ValueError(
                "Ego-centric radius model expects local_agent_infos shape (batch, N, N, obs_dim) "
                f"matching action space N={self.num_agents_max}, got {tuple(local_agent_infos.shape)}."
            )

        agent_contexts = self._encode_agent_contexts(
            local_agent_infos,
            neighbor_masks,
            padding_mask,
            self.token_embedding,
            self.encoder,
        )

        mean_raw = self.mean_head(agent_contexts).squeeze(-1)
        means = self._transform_means(mean_raw)
        means = means.masked_fill(~padding_mask, self.padding_action_mean)

        log_stds = self.log_std.to(device=means.device, dtype=means.dtype)
        log_stds = torch.clamp(log_stds, min=self.log_std_min, max=self.log_std_max)
        log_stds = log_stds.unsqueeze(0).expand(batch_size, -1)

        if self.share_layers:
            self._context_for_value = self._masked_pool_agents(agent_contexts, padding_mask)
        else:
            critic_contexts = self._encode_agent_contexts(
                local_agent_infos,
                neighbor_masks,
                padding_mask,
                self.critic_token_embedding,
                self.critic_encoder,
            )
            self._context_for_value = self._masked_pool_agents(critic_contexts, padding_mask)

        outputs = torch.cat((means, log_stds), dim=-1)
        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)

        return outputs, state

    def _encode_agent_contexts(
        self,
        local_agent_infos: TensorType,
        neighbor_masks: TensorType,
        padding_mask: TensorType,
        embedding: nn.Module,
        encoder: nn.Module,
    ) -> TensorType:
        batch_size, num_agents_max, _, obs_dim = local_agent_infos.shape

        focal_valid = padding_mask.unsqueeze(-1)
        neighbor_valid = padding_mask.unsqueeze(1)
        token_mask = neighbor_masks & focal_valid & neighbor_valid

        flat_tokens = local_agent_infos.reshape(batch_size * num_agents_max, num_agents_max, obs_dim)
        flat_token_mask = token_mask.reshape(batch_size * num_agents_max, num_agents_max)

        flat_tokens = torch.where(flat_token_mask.unsqueeze(-1), flat_tokens, torch.zeros_like(flat_tokens))
        token_embeddings = embedding(flat_tokens)

        safe_token_mask = flat_token_mask.clone()
        empty_rows = ~safe_token_mask.any(dim=1)
        if empty_rows.any():
            safe_token_mask[empty_rows, 0] = True

        encoded = encoder(token_embeddings, src_key_padding_mask=~safe_token_mask)

        pool_mask = flat_token_mask.unsqueeze(-1).to(dtype=encoded.dtype)
        pooled = (encoded * pool_mask).sum(dim=1)
        pooled = pooled / pool_mask.sum(dim=1).clamp(min=1.0)

        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        return pooled.reshape(batch_size, num_agents_max, -1)

    def _transform_means(self, mean_raw: TensorType) -> TensorType:
        if self.mean_activation in (None, "identity", "linear"):
            return mean_raw
        if self.mean_activation != "sigmoid":
            raise ValueError(f"Unsupported mean_activation '{self.mean_activation}'.")

        action_low = self.action_low.to(device=mean_raw.device, dtype=mean_raw.dtype).unsqueeze(0)
        action_high = self.action_high.to(device=mean_raw.device, dtype=mean_raw.dtype).unsqueeze(0)
        return action_low + torch.sigmoid(mean_raw) * (action_high - action_low)

    @staticmethod
    def _masked_pool_agents(agent_contexts: TensorType, padding_mask: TensorType) -> TensorType:
        mask = padding_mask.unsqueeze(-1).to(dtype=agent_contexts.dtype)
        pooled = (agent_contexts * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp(min=1.0)
        return torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)

    def value_function(self) -> TensorType:
        assert self._context_for_value is not None, "Must call forward() before value_function()."
        return self.value_branch(self._context_for_value).squeeze(-1)


RadiusActionPPORLlib = NeighborSelectionRadiusPPORLlib
