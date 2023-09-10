from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from roar.collections.tts.modules.submodules import (
    ConditionalInput,
    ConditionalLayerNorm,
    LinearNorm,
)
from roar.collections.tts.modules.attention import MultiHeadAttn
from roar.collections.tts.parts.utils.helpers import get_mask_from_lengths
from roar.core.classes import NeuralModule, adapter_mixins, typecheck
from roar.core.neural_types.elements import (
    EncodedRepresentation,
    LengthsType,
    MaskType,
    TokenIndex,
)
from roar.core.neural_types.neural_type import NeuralType


def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


# Implements sinusoidal positional encoding
class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        #        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        sinusoid_inp = torch.matmul(
            torch.unsqueeze(pos_seq, -1), torch.unsqueeze(self.inv_freq, 0)
        )

        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        if bsz is not None:
            return pos_emb[None, :, :].repeat(bsz, 1, 1)
        else:
            return pos_emb[None, :, :]


class PositionwiseConvFF(nn.Module):
    def __init__(
        self,
        d_model,
        d_inner,
        kernel_size,
        dropout,
        pre_lnorm=False,
        condition_types=[],
    ):
        super(PositionwiseConvFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)

        self.CoreNet = nn.Sequential(
            nn.Conv1d(d_model, d_inner, kernel_size[0], 1, (kernel_size[0] // 2)),
            nn.ReLU(),
            # nn.Dropout(dropout),  # worse convergence
            nn.Conv1d(d_inner, d_model, kernel_size[1], 1, (kernel_size[1] // 2)),
            nn.Dropout(dropout),
        )
        self.layer_norm = ConditionalLayerNorm(
            d_model, condition_dim=d_model, condition_types=condition_types
        )
        self.pre_lnorm = pre_lnorm

    def forward(self, inp, conditioning=None):
        return self._forward(inp, conditioning)

    def _forward(self, inp, conditioning=None):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(
                self.layer_norm(core_out, conditioning).to(inp.dtype)
            )
            core_out = core_out.transpose(1, 2)

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(core_out)
            core_out = core_out.transpose(1, 2)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out, conditioning).to(inp.dtype)

        return output


class TransformerLayer(nn.Module, adapter_mixins.AdapterModuleMixin):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size,
        dropout,
        condition_types=[],
        **kwargs
    ):
        super(TransformerLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(
            n_head, d_model, d_head, dropout, condition_types=condition_types, **kwargs
        )
        self.pos_ff = PositionwiseConvFF(
            d_model,
            d_inner,
            kernel_size,
            dropout,
            pre_lnorm=kwargs.get("pre_lnorm"),
            condition_types=condition_types,
        )

    def forward(self, dec_inp, mask=None, conditioning=None):
        output = self.dec_attn(
            dec_inp, attn_mask=~mask.squeeze(2), conditioning=conditioning
        )
        output *= mask
        output = self.pos_ff(output, conditioning)
        output *= mask

        if self.is_adapter_available():
            output = self.forward_enabled_adapters(output)
            output *= mask

        return output
