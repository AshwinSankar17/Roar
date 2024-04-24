from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from lightning_utilities.core.imports import RequirementCache

from roar.collections.tts.modules.submodules import (
    ConditionalLayerNorm,
)
from roar.collections.nlp.parts.submodules.positional_encodings import (
    apply_rotary_emb_func,
)
from roar.collections.tts.parts.utils.bert_padding import (
    pad_input,
    unpad_input_only,
)

HAVE_FLASH = RequirementCache("flash-attn>=2.0.0.post1")

RoPECache = Tuple[torch.Tensor, torch.Tensor]


# Multi Head Self Attention, may use flash attention or memory efficient attention
class MultiHeadAttn(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt=0.1,
        pre_lnorm=False,
        condition_types=[],
        **kwargs,
    ):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = 1 / (d_head**0.5)
        self.pre_lnorm = pre_lnorm

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head)
        self.drop = nn.Dropout(dropout)
        self.dropatt = dropatt
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = ConditionalLayerNorm(
            d_model, condition_dim=d_model, condition_types=condition_types
        )

    def forward(self, inp, attn_mask=None, conditioning=None):
        return self._forward(inp, attn_mask, conditioning)

    def _forward(self, inp, attn_mask=None, conditioning=None):
        residual = inp
        if self.pre_lnorm:
            # layer normalization
            inp = self.layer_norm(inp, conditioning)

        n_head, d_head = self.n_head, self.d_head

        head_q, head_k, head_v = torch.chunk(self.qkv_net(inp), 3, dim=2)

        head_q = head_q.view(inp.size(0), inp.size(1), n_head, d_head)
        head_k = head_k.view(inp.size(0), inp.size(1), n_head, d_head)
        head_v = head_v.view(inp.size(0), inp.size(1), n_head, d_head)

        q = head_q.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)
        k = head_k.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)
        v = head_v.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)

        if attn_mask is not None:  # prepare attention mask
            attn_mask = attn_mask.unsqueeze(1).to(q.dtype)  # [B, 1, T]
            attn_mask = attn_mask.repeat(n_head, attn_mask.size(2), 1)

        attn_vec = F.scaled_dot_product_attention(q, k, v, attn_mask, self.dropatt)

        attn_vec = attn_vec.view(n_head, inp.size(0), inp.size(1), d_head)
        attn_vec = (
            attn_vec.permute(1, 2, 0, 3)
            .contiguous()
            .view(inp.size(0), inp.size(1), n_head * d_head)
        )

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = residual + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(residual + attn_out, conditioning)

        return output


class FlashSelfAttention(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        n_query_groups: Optional[int] = None,
    ):
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.n_query_groups = n_query_groups

        shape = (self.n_head + self.n_query_groups * 2) * self.d_head
        self.qkv_net = nn.Linear(self.d_model, shape)

    def forward(
        self,
        inp: torch.Tensor,
        rope: RoPECache,
        cu_seqlens: torch.Tensor,
        max_seq_length: int,
        subset_idx: Optional[torch.Tensor],
        indices: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        conditioning: Optional[torch.Tensor] = None,
    ):
        B, T, C = inp.size()

        qkv = self.qkv_net(inp)
        qkv = pad_input(qkv, indices, cu_seqlens.shape[0] - 1, max_seq_length)

        q_per_kv = self.n_head // self.n_query_groups
        # total_qkv = q_per_kv + 2

        qkv = rearrange(
            qkv, "b t (h s d) -> b t h s d", h=self.n_query_groups, d=self.d_head
        )
        # qkv = qkv.view(
        #     B, T, self.config.n_query_groups, total_qkv, self.d_head
        # )

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        q = rearrange(q, "b t h 1 d -> b t h d")
        k = rearrange(k, "b t h 1 d -> b t h d")
        v = rearrange(v, "b t h 1 d -> b t h d")

        cos, sin = rope

        q = apply_rotary_emb_func(q, cos, sin, False, False)
        k = apply_rotary_emb_func(k, cos, sin, False, False)

        attn_vec = self.scaled_dot_product_attention(q, k, v, mask=attn_mask)

        attn_vec = unpad_input_only(attn_vec, torch.squeeze(attn_mask) == 1)

        attn_vec = rearrange(attn_vec, "... h d -> ... (h d)")

        return attn_vec

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.d_head)

        if (
            HAVE_FLASH
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            from flash_attn import flash_attn_func

            return flash_attn_func(
                q, k, v, dropout_p=0.0, softmax_scale=scale, causal=False
            )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)


class FlashCrossAttention(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        n_query_groups: Optional[int] = None,
    ):
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.n_query_groups = n_query_groups

        # shape = (self.n_head + self.n_query_groups * 2) * self.d_head
        self.q_net = nn.Linear(self.d_model, self.n_head * self.d_head)
        self.kv_net = nn.Linear(self.d_model, self.n_query_groups * self.d_head * 2)

    def forward(
        self,
        inp_q: torch.Tensor,
        inp_kv: torch.Tensor,
        rope: RoPECache,
        attn_mask: Optional[torch.Tensor] = None,
        conditioning: Optional[torch.Tensor] = None,
    ):
        B, T, C = inp_q.size()

        q = self.q_net(inp_q)
        # q = pad_input(q, indices, cu_seqlens.shape[0] - 1, max_seq_length)
        q = rearrange(q, "b t (h d) -> b t h d", h=self.n_query_groups, d=self.d_head)
        kv = self.kv_net(inp_kv)

        kv = rearrange(
            kv, "b t (h s d) -> b t h s d", h=self.n_query_groups, d=self.d_head
        )
        # qkv = qkv.view(
        #     B, T, self.config.n_query_groups, total_qkv, self.d_head
        # )

        # split batched computation into three
        k, v = kv.split(1, dim=-2)

        # q = rearrange(q, "b t h 1 d -> b t h d")
        k = rearrange(k, "b t h 1 d -> b t h d")
        v = rearrange(v, "b t h 1 d -> b t h d")

        cos, sin = rope

        q = apply_rotary_emb_func(q, cos, sin, False, False)
        k = apply_rotary_emb_func(k, cos, sin, False, False)

        attn_vec = self.scaled_dot_product_attention(q, k, v, mask=attn_mask)

        attn_vec = rearrange(attn_vec, "... h d -> ... (h d)")

        return attn_vec

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.d_head)

        if (
            HAVE_FLASH
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            from flash_attn import flash_attn_func

            return flash_attn_func(
                q, k, v, dropout_p=0.0, softmax_scale=scale, causal=False
            )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)
