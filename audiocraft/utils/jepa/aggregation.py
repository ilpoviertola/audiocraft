# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
import einops
from timm.layers import trunc_normal_

from .pos_embs import (
    get_1d_sincos_pos_embed,
)
from .masks import apply_masks


class FrameAggregation(nn.Module):
    """
    Process each frame independently and concatenate all tokens
    """

    def __init__(
        self, model, max_frames=10000, use_pos_embed=False, attend_across_segments=False
    ):
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads
        self.attend_across_segments = attend_across_segments
        # 1D-temporal pos-embedding
        self.pos_embed = None
        if use_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, max_frames, embed_dim), requires_grad=False
            )
            sincos = get_1d_sincos_pos_embed(embed_dim, max_frames)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):

        # TODO: implement attend_across_segments=False
        # num_clips = len(x)
        num_views_per_clip = len(x[0])

        # Concatenate views along batch dimension
        x = [torch.cat(xi, dim=0) for xi in x]
        # Concatenate clips along temporal dimension
        x = torch.cat(x, dim=2)
        B, C, T, H, W = x.size()

        # Put each frame along the batch dimension
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        outputs = self.model(x)
        _, N, D = outputs.size()
        outputs = outputs.reshape(B, T, N, D).flatten(1, 2)

        # Separate views into list
        B = B // num_views_per_clip
        all_outputs = []
        for i in range(num_views_per_clip):
            o = outputs[i * B : (i + 1) * B]
            # Compute positional embedding
            if (self.pos_embed is not None) and (clip_indices is not None):
                pos_embed = self.pos_embed.repeat(B, 1, 1)  # [B, F, D]
                pos_embed = apply_masks(
                    pos_embed, clip_indices, concat=False
                )  # list(Tensor([B, T, D]))
                pos_embed = torch.cat(
                    pos_embed, dim=1
                )  # concatenate along temporal dimension
                pos_embed = pos_embed.unsqueeze(2).repeat(
                    1, 1, N, 1
                )  # [B, T*num_clips, N, D]
                pos_embed = pos_embed.flatten(1, 2)
                o += pos_embed
            all_outputs += [o]

        return all_outputs


class ClipAggregation(nn.Module):
    """
    Process each clip independently and concatenate all tokens
    """

    def __init__(
        self,
        model,
        tubelet_size=2,
        max_frames=10000,
        use_pos_embed=False,
        attend_across_segments=False,
        use_spatial_aggregation=False,
        spatial_agg_type="attention",
    ):
        super().__init__()
        self.model = model
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads
        self.attend_across_segments = attend_across_segments
        # 1D-temporal pos-embedding
        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames  # // tubelet_size
            self.pos_embed = nn.Parameter(
                torch.zeros(1, max_T, embed_dim), requires_grad=False
            )
            sincos = get_1d_sincos_pos_embed(embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

        self.use_spatial_aggregation = use_spatial_aggregation
        if self.use_spatial_aggregation:
            if spatial_agg_type == "attention":
                self.spatial_aggregation = SpatialTransformerEncoderLayer(
                    d_model=self.embed_dim,
                    nhead=self.num_heads,
                    activation=nn.GELU(),
                    batch_first=True,
                    dim_feedforward=model.mlp_ratio * self.embed_dim,
                    dropout=model.drop_rate,
                    layer_norm_eps=1e-6,
                    norm_first=True,
                )
            elif spatial_agg_type == "avgpool":
                self.spatial_aggregation = AveragePooling(
                    avg_pattern="bs d t h w -> bs d t",
                    then_permute_pattern="bs D t -> bs t D",
                    reduce_fn="mean",
                )
            elif spatial_agg_type == "maxpool":
                self.spatial_aggregation = AveragePooling(
                    avg_pattern="bs d t h w -> bs d t",
                    then_permute_pattern="bs D t -> bs t D",
                    reduce_fn="max",
                )
            else:
                raise ValueError(
                    f"spatial_agg_type={spatial_agg_type} not supported. "
                    "Supported types are: 'attention', 'avgpool', 'maxpool'"
                )

    def forward(self, x, clip_indices=None):

        num_clips = len(x)  # aka sequences
        num_views_per_clip = len(x[0])
        assert num_views_per_clip == 1, "1 view per clip supported for now."
        B, C, T, H, W = x[0][0].size()

        # Concatenate all spatial and temporal views along batch dimension
        x = [torch.cat(xi, dim=0) for xi in x]
        x = torch.cat(x, dim=0)  # (B*num_clips*num_views_per_clip, C, T, H, W)
        outputs = self.model(x)
        _, N, D = outputs.size()  # (B*num_clips*num_views_per_clip, N(t*h*w), D)

        if self.use_spatial_aggregation:
            # all_outputs is list of clips
            # create a tensor of it
            all_outputs = self.restore_spatio_temp_dims(
                outputs, (num_clips, num_views_per_clip, B, C, T, H, W)
            )  # (B*S, D, t, h, w) <- (B*S, t*h*w, D)
            agg_outputs = self.spatial_aggregation(all_outputs)

            if (self.pos_embed is not None) and (clip_indices is not None):
                clip_indices = torch.tensor(clip_indices, device=outputs.device)
                clip_indices = [c[:, :: self.tubelet_size] for c in clip_indices]
                pos_embed = self.pos_embed.repeat(B, 1, 1)  # [B, F, D]
                pos_embed = apply_masks(
                    pos_embed, clip_indices, concat=False
                )  # list(Tensor([B, T, D]))
                pos_embed = torch.cat(
                    pos_embed, dim=1
                )  # concatenate along temporal dimension
                pos_embed = pos_embed.view(
                    B * num_views_per_clip, num_clips, T // self.tubelet_size, D
                ).transpose(0, 1)
                agg_outputs += pos_embed.reshape(
                    B * num_clips * num_views_per_clip, -1, D
                )

            return agg_outputs.view(
                num_clips * num_views_per_clip * B, *agg_outputs.shape[1:]
            )

        return outputs

    def restore_spatio_temp_dims(
        self, feats: torch.Tensor, orig_shape: tuple
    ) -> torch.Tensor:
        """
        feats are of shape (B*S, T, D) where T = 1 + (224 // 16) * (224 // 16) * 8
        Our goal is to make them of shape (B*S, t, h, w, D) where h, w are the spatial dimensions.
        From `self.patch_embed_3d`, it follows that we could reshape feats with:
            `feats.transpose(1, 2).view(B*S, D, t, h, w)`
        """
        num_clips, num_views_per_clip, B, C, T, H, W = orig_shape
        _, N, _ = feats.size()
        D = self.embed_dim

        # num patches in each dimension
        t = T // self.tubelet_size
        h = int(sqrt(N // t))
        w = h

        feats = feats.permute(0, 2, 1)  # (B*S, D, T)
        feats = feats.view(
            B * num_clips * num_views_per_clip, D, t, h, w
        )  # (B*num_clips*num_views_per_clip, D, t, h, w)

        return feats


class BaseEncoderLayer(nn.TransformerEncoderLayer):
    """
    This is a wrapper around nn.TransformerEncoderLayer that adds a CLS token
    to the sequence and outputs the CLS token's representation.
    This base class parents both SpatialEncoderLayer and TemporalEncoderLayer for the RGB stream
    and the FrequencyEncoderLayer and TemporalEncoderLayer for the audio stream stream.
    We also, optionally, add a positional embedding to the input sequence which
    allows to reuse it for global aggregation (of segments) for both streams.
    """

    def __init__(
        self,
        add_pos_emb: bool = False,
        pos_emb_drop: Optional[float] = None,
        pos_max_len: Optional[int] = None,
        *args_transformer_enc,
        **kwargs_transformer_enc,
    ):
        super().__init__(*args_transformer_enc, **kwargs_transformer_enc)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.self_attn.embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

        # add positional embedding
        self.add_pos_emb = add_pos_emb
        if add_pos_emb:
            assert pos_max_len is not None, "pos_max_len must be provided"
            assert pos_emb_drop is not None, "pos_emb_drop must be provided"
            self.pos_max_len = 1 + pos_max_len  # +1 (for CLS)
            self.pos_emb = nn.Parameter(
                torch.zeros(1, self.pos_max_len, self.self_attn.embed_dim)
            )
            self.pos_drop = nn.Dropout(pos_emb_drop)
            trunc_normal_(self.pos_emb, std=0.02)

        self.apply(self._init_weights)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """x is of shape (B, N, D); if provided x_mask is of shape (B, N)"""
        batch_dim = x.shape[0]

        # add CLS token
        cls_tokens = self.cls_token.expand(
            batch_dim, -1, -1
        )  # expanding to match batch dimension
        x = torch.cat((cls_tokens, x), dim=-2)  # (batch_dim, 1+seq_len, D)
        if x_mask is not None:
            cls_mask = torch.ones(
                (batch_dim, 1), dtype=torch.bool, device=x_mask.device
            )  # 1=keep; 0=mask
            x_mask_w_cls = torch.cat(
                (cls_mask, x_mask), dim=-1
            )  # (batch_dim, 1+seq_len)
            B, N = x_mask_w_cls.shape
            # torch expects (N, N) or (B*num_heads, N, N) mask (sadness ahead); torch masks
            x_mask_w_cls = (
                x_mask_w_cls.reshape(B, 1, 1, N)
                .expand(-1, self.self_attn.num_heads, N, -1)
                .reshape(B * self.self_attn.num_heads, N, N)
            )
            assert (
                x_mask_w_cls.dtype == x_mask_w_cls.bool().dtype
            ), "x_mask_w_cls.dtype != bool"
            x_mask_w_cls = ~x_mask_w_cls  # invert mask (1=mask)
        else:
            x_mask_w_cls = None

        # add positional embedding
        if self.add_pos_emb:
            seq_len = x.shape[
                1
            ]  # (don't even think about moving it before the CLS token concatenation)
            assert (
                seq_len <= self.pos_max_len
            ), f"Seq len ({seq_len}) > pos_max_len ({self.pos_max_len})"
            x = x + self.pos_emb[:, :seq_len, :]
            x = self.pos_drop(x)

        # apply encoder layer (calls nn.TransformerEncoderLayer.forward);
        x = super().forward(src=x, src_mask=x_mask_w_cls)  # (batch_dim, 1+seq_len, D)

        # CLS token is expected to hold spatial information for each frame
        x = x[:, 0, :]  # (batch_dim, D)

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token", "pos_emb"}


class SpatialTransformerEncoderLayer(BaseEncoderLayer):
    """Aggregates spatial dimensions by applying attention individually to each frame."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask=None,
        is_causal=False,
    ) -> torch.Tensor:
        """x is of shape (B*S, D, t, h, w) where S is the number of segments.
        if specified x_mask (B*S, t, h, w), 0=masked, 1=kept
        Returns a tensor of shape (B*S, t, D) pooling spatial information for each frame.
        """
        BS, D, t, h, w = x.shape

        # time as a batch dimension and flatten spatial dimensions as sequence
        x = einops.rearrange(x, "BS D t h w -> (BS t) (h w) D")
        # similar to mask
        if x_mask is not None:
            x_mask = einops.rearrange(x_mask, "BS t h w -> (BS t) (h w)")

        # apply encoder layer (BaseEncoderLayer.forward) - it will add CLS token and output its representation
        x = super().forward(x=x, x_mask=x_mask)  # (B*S*t, D)

        # reshape back to (B*S, t, D)
        x = einops.rearrange(x, "(BS t) D -> BS t D", BS=BS, t=t)

        # (B*S, t, D)
        return x


class AveragePooling(nn.Module):
    def __init__(
        self,
        avg_pattern: str,
        reduce_fn: str = "mean",
        then_permute_pattern: Optional[str] = None,
    ) -> None:
        """patterns are e.g. "bs t d -> bs d" """
        super().__init__()
        # TODO: need to register them as buffers (but fails because these are strings)
        self.reduce_fn = reduce_fn
        self.avg_pattern = avg_pattern
        self.then_permute_pattern = then_permute_pattern

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = einops.reduce(x, self.avg_pattern, self.reduce_fn)
        if self.then_permute_pattern is not None:
            x = einops.rearrange(x, self.then_permute_pattern)
        return x
