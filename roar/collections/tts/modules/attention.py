from typing import Optional, TypedDict
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from roar.collections.tts.modules.submodules import (
    ConditionalInput,
    ConditionalLayerNorm,
    LinearNorm,
)

from roar.collections.tts.parts.utils.helpers import get_mask_from_lengths
from roar.core.classes import NeuralModule, adapter_mixins, typecheck
from roar.core.neural_types.elements import (
    EncodedRepresentation,
    LengthsType,
    MaskType,
    TokenIndex,
)
from roar.core.neural_types.neural_type import NeuralType
from roar.utils import logging

HAVE_FLASH = True
try:
    from flash_attn import flash_attn_qkvpacked_func
except ImportError:
    HAVE_FLASH = False


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


# Multi Head Cross Attention, may use flash attention or memory efficient attention
class MultiHeadCrossAttn(nn.Module):
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
        super(MultiHeadCrossAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = 1 / (d_head**0.5)
        self.pre_lnorm = pre_lnorm

        self.q_net = nn.Linear(d_model, n_head * d_head)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head)
        self.drop = nn.Dropout(dropout)
        self.dropatt = dropatt
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = ConditionalLayerNorm(
            d_model, condition_dim=d_model, condition_types=condition_types
        )

    def forward(self, inp, key_value, attn_mask=None, conditioning=None):
        return self._forward(inp, key_value, attn_mask, conditioning)

    def _forward(self, query, key_value, attn_mask=None, conditioning=None):
        residual = query
        if self.pre_lnorm:
            # layer normalization
            query = self.layer_norm(query, conditioning)
            key_value = self.layer_norm(key_value, conditioning)

        n_head, d_head = self.n_head, self.d_head

        head_q = self.q_net(query)
        head_k, head_v = torch.chunk(self.kv_net(key_value), 2, dim=2)

        head_q = head_q.view(query.size(0), query.size(1), n_head, d_head)
        head_k = head_k.view(key_value.size(0), key_value.size(1), n_head, d_head)
        head_v = head_v.view(key_value.size(0), key_value.size(1), n_head, d_head)

        q = head_q.permute(2, 0, 1, 3).reshape(-1, query.size(1), d_head)
        k = head_k.permute(2, 0, 1, 3).reshape(-1, key_value.size(1), d_head)
        v = head_v.permute(2, 0, 1, 3).reshape(-1, key_value.size(1), d_head)

        if attn_mask is not None:  # prepare attention mask
            attn_mask = attn_mask.unsqueeze(1).to(q.dtype)  # [B, 1, T]
            attn_mask = attn_mask.repeat(n_head, attn_mask.size(2), 1)

        attn_vec = F.scaled_dot_product_attention(q, k, v, attn_mask, self.dropatt)

        attn_vec = attn_vec.view(n_head, query.size(0), query.size(1), d_head)
        attn_vec = (
            attn_vec.permute(1, 2, 0, 3)
            .contiguous()
            .view(query.size(0), query.size(1), n_head * d_head)
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


# Multi Head Cross Attention, may use flash attention or memory efficient attention
class MultiHeadCrossAttn(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt=0.1,
        pre_lnorm=False,
        condition_types=[],
        **kwargs
    ):
        super(MultiHeadCrossAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = 1 / (d_head**0.5)
        self.pre_lnorm = pre_lnorm

        self.q_net = nn.Linear(d_model, n_head * d_head)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head)
        self.drop = nn.Dropout(dropout)
        self.dropatt = dropatt
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = ConditionalLayerNorm(
            d_model, condition_dim=d_model, condition_types=condition_types
        )

    def forward(self, inp, attn_mask=None, conditioning=None):
        return self._forward(inp, attn_mask, conditioning)

    def _forward(self, query, attn_mask=None, conditioning=None):
        key_value = conditioning if conditioning is not None else query
        residual = query
        if self.pre_lnorm:
            # layer normalization
            query = self.layer_norm(query, conditioning)
            key_value = self.layer_norm(key_value, conditioning)

        n_head, d_head = self.n_head, self.d_head

        head_q = self.q_net(query)
        head_k, head_v = torch.chunk(self.kv_net(key_value), 3, dim=2)

        head_q = head_q.view(query.size(0), query.size(1), n_head, d_head)
        head_k = head_k.view(key_value.size(0), key_value.size(1), n_head, d_head)
        head_v = head_v.view(key_value.size(0), key_value.size(1), n_head, d_head)

        q = head_q.permute(2, 0, 1, 3).reshape(-1, query.size(1), d_head)
        k = head_k.permute(2, 0, 1, 3).reshape(-1, key_value.size(1), d_head)
        v = head_v.permute(2, 0, 1, 3).reshape(-1, key_value.size(1), d_head)

        if attn_mask is not None:  # prepare attention mask
            attn_mask = attn_mask.unsqueeze(1).to(q.dtype)  # [B, 1, T]
            attn_mask = attn_mask.repeat(n_head, attn_mask.size(2), 1)

        attn_vec = F.scaled_dot_product_attention(q, k, v, attn_mask, self.dropatt)

        attn_vec = attn_vec.view(n_head, query.size(0), query.size(1), d_head)
        attn_vec = (
            attn_vec.permute(1, 2, 0, 3)
            .contiguous()
            .view(query.size(0), query.size(1), n_head * d_head)
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
