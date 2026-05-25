"""
PPO model for the `neighbor_index` action type.

The action is a per-agent integer index in [0, num_agents_max - 1]: each agent picks ONE
reference neighbor (self allowed). The env then uses only those neighbors whose Euclidean
distance is <= dist(i, anchor_i) for flocking control. The action is set at episode start
and frozen for the rest of the episode (handled inside the env).

Architecture: reuses the encoder/decoder/pointer-attention stack from `models/ppo.py`. The
existing `NeighborSelectorTorch` already produces `(B, N, N)` raw attention scores via
`RawAttentionScoreGenerator` - these are exactly the per-agent pointer logits we want.
This module differs from `NeighborSelectionPPORLlib` only in how those scores are mapped
to the action distribution: flatten directly to `(B, N*N)` MultiCategorical logits instead
of producing 2-class binary logits.

The pointer mechanism makes the head independent of any fixed N-way output layer; variable
num_agents_max values use the same weights since each (query_i, key_j) score is computed
by the same linear projections in the generator.
"""

import copy

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import TensorType
from typing import List, Dict

import torch
import torch.nn as nn

from models.modules.token_embedding import LinearEmbedding
from models.modules.multi_head_attention_layer import MultiHeadAttentionLayer
from models.modules.position_wise_feed_forward_layer import PositionWiseFeedForwardLayer
from models.modules.encoder_block import EncoderBlock
from models.modules.decoder_block import CustomDecoderBlock as DecoderBlock
from models.modules.encoder import Encoder
from models.modules.decoder import Decoder, DecoderPlaceholder
from models.modules.pointer_net import RawAttentionScoreGenerator, RawAttentionScoreGeneratorPlaceholder
from models.ppo import NeighborSelectorTorch


class NeighborIndexPPORLlib(TorchModelV2, nn.Module):
    """
    PPO model whose action distribution is MultiDiscrete([N] * N): per-agent pick of one
    neighbor index (self allowed). Logits come from the pointer-attention scores produced
    by NeighborSelectorTorch.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        if model_config is None:
            raise ValueError("model_config must be specified")
        cfg = model_config["custom_model_config"]
        share_layers = cfg.get("share_layers", True)
        d_subobs = cfg["d_subobs"]
        d_embed_input = cfg.get("d_embed_input", 128)
        d_embed_context = cfg.get("d_embed_context", 128)
        d_model = cfg.get("d_model", 128)
        d_model_decoder = cfg.get("d_model_decoder", 128)
        n_layers_encoder = cfg.get("n_layers_encoder", 3)
        n_layers_decoder = cfg.get("n_layers_decoder", 1)
        h = cfg.get("num_heads", 8)
        d_ff = cfg.get("d_ff", 512)
        d_ff_decoder = cfg.get("d_ff_decoder", 512)
        dr_rate = cfg.get("dr_rate", 0)
        norm_eps = cfg.get("norm_eps", 1e-5)
        is_bias = cfg.get("is_bias", True)
        use_residual_in_decoder = cfg.get("use_residual_in_decoder", True)
        use_FNN_in_decoder = cfg.get("use_FNN_in_decoder", True)
        self.scale_factor = cfg.get("scale_factor", 1.0)

        # Encoder
        input_embed = LinearEmbedding(d_env=d_subobs, d_embed=d_embed_input)
        mha_encoder = MultiHeadAttentionLayer(
            d_model=d_model, h=h,
            q_fc=nn.Linear(d_embed_input, d_model, is_bias),
            kv_fc=nn.Linear(d_embed_input, d_model, is_bias),
            out_fc=nn.Linear(d_model, d_embed_input, is_bias),
            dr_rate=dr_rate,
        )
        position_ff_encoder = PositionWiseFeedForwardLayer(
            fc1=nn.Linear(d_embed_input, d_ff),
            fc2=nn.Linear(d_ff, d_embed_input),
            dr_rate=dr_rate,
        )
        norm_encoder = nn.LayerNorm(d_embed_input, eps=norm_eps)

        # Decoder
        mha_decoder = MultiHeadAttentionLayer(
            d_model=d_model_decoder, h=h,
            q_fc=nn.Linear(d_embed_context, d_model_decoder, is_bias),
            kv_fc=nn.Linear(d_embed_input, d_model_decoder, is_bias),
            out_fc=nn.Linear(d_model_decoder, d_embed_context, is_bias),
            dr_rate=dr_rate,
        )
        position_ff_decoder = PositionWiseFeedForwardLayer(
            fc1=nn.Linear(d_embed_context, d_ff_decoder),
            fc2=nn.Linear(d_ff_decoder, d_embed_context),
            dr_rate=dr_rate,
        ) if use_FNN_in_decoder else None
        norm_decoder = nn.LayerNorm(d_embed_context, eps=norm_eps)

        encoder_block = EncoderBlock(
            self_attention=copy.deepcopy(mha_encoder),
            position_ff=copy.deepcopy(position_ff_encoder),
            norm=copy.deepcopy(norm_encoder),
            dr_rate=dr_rate,
        )
        decoder_block = DecoderBlock(
            self_attention=None,
            cross_attention=copy.deepcopy(mha_decoder),
            position_ff=position_ff_decoder,
            norm=copy.deepcopy(norm_decoder),
            dr_rate=dr_rate,
            efficient=not use_residual_in_decoder,
        )
        encoder = Encoder(
            encoder_block=encoder_block,
            n_layer=n_layers_encoder,
            norm=copy.deepcopy(norm_encoder),
        )
        decoder = Decoder(
            decoder_block=decoder_block,
            n_layer=n_layers_decoder,
            norm=copy.deepcopy(norm_decoder),
        )
        generator = RawAttentionScoreGenerator(
            d_model=d_model_decoder,
            q_fc=nn.Linear(d_embed_context, d_model_decoder, is_bias),
            k_fc=nn.Linear(d_embed_input, d_model_decoder, is_bias),
            dr_rate=dr_rate,
        )

        # Action space: MultiDiscrete([N, N, ..., N]). nvec sum == N * N.
        action_size = int(action_space.nvec[0])  # N
        assert (action_space.nvec == action_size).all(), \
            "MultiDiscrete nvec must be uniform (= num_agents_max)"
        assert num_outputs == action_size * action_size, \
            f"num_outputs != N*N; num_outputs={num_outputs}, N={action_size}"
        self.num_agents_max = action_size

        # Policy network
        self.actor = NeighborSelectorTorch(
            src_embed=input_embed,
            encoder=encoder,
            decoder=decoder,
            generator=generator,
            d_embed_context=d_embed_context,
        )

        # Value network
        self.values = None
        self.share_layers = share_layers
        if not self.share_layers:
            self.critic = NeighborSelectorTorch(
                src_embed=copy.deepcopy(input_embed),
                encoder=copy.deepcopy(encoder),
                decoder=DecoderPlaceholder(),
                generator=RawAttentionScoreGeneratorPlaceholder(),
                d_embed_context=d_embed_context,
            )

        self.value_branch = nn.Sequential(
            nn.Linear(in_features=d_embed_context, out_features=d_embed_context),
            nn.ReLU(),
            nn.Linear(in_features=d_embed_context, out_features=1),
        )

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ):
        obs_dict = input_dict["obs"]

        # att: (B, N, N) raw pointer-attention scores from the existing NeighborSelectorTorch.
        # h_c_N: (B, 1, d_embed_context)
        att, h_c_N = self.actor(obs_dict)

        logits = self.attention_scores_to_logits(att, obs_dict)

        if self.share_layers:
            self.values = h_c_N.squeeze(1)
        else:
            self.values = self.critic(obs_dict)[1].squeeze(1)

        return logits, state

    def attention_scores_to_logits(self, attention_scores: TensorType, obs_dict) -> TensorType:
        """
        Map per-agent pointer scores (B, N, N) to MultiCategorical logits (B, N*N).

        - Scale by self.scale_factor (matches binary_vector head convention).
        - Mask padded candidate columns to -inf so the policy never picks padding agents.
        - Self is intentionally NOT masked (anchor=self is a valid action meaning "fly solo").
        - Padded agent rows already come out as -1e9 from NeighborSelectorTorch.local_forward
          (uninitialized for padded local_padding_flags); softmax over a uniform -1e9 row is
          uniform, but the env discards those rows anyway via padding_mask.
        """
        batch_size, num_agents_max, _ = attention_scores.shape

        attention_scores = attention_scores * self.scale_factor

        # Mask padded candidate columns (j is padded) so they cannot be sampled by any agent.
        padding_mask = obs_dict["padding_mask"].bool()  # (B, N)
        # Broadcast over query (i) dim: result shape (B, 1, N).
        cand_mask = padding_mask.unsqueeze(1)
        attention_scores = attention_scores.masked_fill(~cand_mask, -1e9)

        logits = attention_scores.reshape(batch_size, num_agents_max * num_agents_max)
        return logits

    def value_function(self) -> TensorType:
        value = self.value_branch(self.values).squeeze(-1)
        return value
