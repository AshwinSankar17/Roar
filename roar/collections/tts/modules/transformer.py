from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from roar.collections.tts.modules.submodules import (
    ConditionalInput,
    ConditionalLayerNorm,
    LinearNorm,
)
from roar.collections.tts.modules.attention import MultiHeadAttn, MultiHeadAttnFlash
from roar.collections.tts.modules.postional_embedding import PositionalEmbedding
from roar.collections.tts.parts.utils.helpers import get_mask_from_lengths
from roar.core.classes import NeuralModule, adapter_mixins, typecheck
from roar.core.neural_types.elements import (
    EncodedRepresentation,
    LengthsType,
    MaskType,
    TokenIndex,
)
from roar.core.neural_types.neural_type import NeuralType
from roar.utils.gpu_utils import is_gpu_ampere_or_newer
from roar.utils import logging

HAVE_FLASH = True
try:
    from flash_attn import flash_attn_qkvpacked_func
except ImportError:
    HAVE_FLASH = False


# TODO: move mask_from_lens to roar.collections.tts.parts.utils.helpers
def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


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
    ):  # TODO: add flash attention support for transformer
        super(TransformerLayer, self).__init__()
        AttentionBlock = MultiHeadAttn
        if kwargs.get("use_flash", False):
            if not HAVE_FLASH:
                logging.warning(
                    "Flash attention is not available. Falling back to regular attn."
                )
            elif not is_gpu_ampere_or_newer():
                logging.warning(
                    "Flash attention is only available on Ampere or newer GPUs. Falling back to regular attn."
                )
            else:
                AttentionBlock = MultiHeadAttnFlash

        self.dec_attn = AttentionBlock(
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


class FFTransformerDecoder(NeuralModule):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size,
        dropout,
        dropatt,
        dropemb=0.0,
        pre_lnorm=False,
        condition_types=[],
        use_flash=False,
    ):
        super(FFTransformerDecoder, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.pos_emb = PositionalEmbedding(self.d_model)
        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()
        self.cond_input = ConditionalInput(d_model, d_model, condition_types)

        for _ in range(n_layer):
            self.layers.append(
                TransformerLayer(
                    n_head,
                    d_model,
                    d_head,
                    d_inner,
                    kernel_size,
                    dropout,
                    dropatt=dropatt,
                    pre_lnorm=pre_lnorm,
                    condition_types=condition_types,
                )
            )

    @property
    def input_types(self):
        return {
            "input": NeuralType(("B", "T", "D"), EncodedRepresentation()),
            "seq_lens": NeuralType(("B"), LengthsType()),
            "conditioning": NeuralType(
                ("B", "T", "D"), EncodedRepresentation(), optional=True
            ),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(("B", "T", "D"), EncodedRepresentation()),
            "mask": NeuralType(("B", "T", "D"), MaskType()),
        }

    @typecheck()
    def forward(self, input, seq_lens, conditioning=None):
        return self._forward(input, mask_from_lens(seq_lens).unsqueeze(2), conditioning)

    def _forward(self, inp, mask, conditioning):
        pos_seq = torch.arange(inp.size(1), device=inp.device).to(inp.dtype)
        pos_emb = self.pos_emb(pos_seq) * mask
        inp += pos_emb
        inp = self.cond_input(inp, conditioning)
        out = self.drop(inp)

        for layer in self.layers:
            out = layer(out, mask=mask, conditioning=conditioning)

        # out = self.drop(out)
        return out, mask


class FFTransformerEncoder(FFTransformerDecoder):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size,
        dropout,
        dropatt,
        dropemb=0.0,
        pre_lnorm=False,
        n_embed=None,
        d_embed=None,
        padding_idx=0,
        condition_types=[],
        use_flash=False,
    ):
        super(FFTransformerEncoder, self).__init__(
            n_layer,
            n_head,
            d_model,
            d_head,
            d_inner,
            kernel_size,
            dropout,
            dropatt,
            dropemb,
            pre_lnorm,
            condition_types,
            use_flash,
        )

        self.padding_idx = padding_idx
        self.word_emb = nn.Embedding(
            n_embed, d_embed or d_model, padding_idx=self.padding_idx
        )

    @property
    def input_types(self):
        return {
            "input": NeuralType(("B", "T"), TokenIndex()),
            "conditioning": NeuralType(
                ("B", "T", "D"), EncodedRepresentation(), optional=True
            ),
        }

    def forward(self, input, conditioning=0):
        return self._forward(
            self.word_emb(input), (input != self.padding_idx).unsqueeze(2), conditioning
        )  # (B, L, 1)


class FFTransformer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=1,
        n_layers=6,
        n_head=1,
        d_head=64,
        d_inner=1024,
        kernel_size=3,
        dropout=0.1,
        dropatt=0.1,
        dropemb=0.0,
        use_flash=False,
    ):
        super(FFTransformer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_head = n_head
        self.d_head = d_head

        self.pos_emb = PositionalEmbedding(self.in_dim)
        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(
                TransformerLayer(
                    n_head,
                    in_dim,
                    d_head,
                    d_inner,
                    kernel_size,
                    dropout,
                    dropatt=dropatt,
                    use_flash=use_flash,
                )
            )

        self.dense = LinearNorm(in_dim, out_dim)

    def forward(self, dec_inp, in_lens):
        # B, C, T --> B, T, C
        inp = dec_inp.transpose(1, 2)
        mask = get_mask_from_lengths(in_lens)[..., None]

        pos_seq = torch.arange(inp.size(1), device=inp.device).to(inp.dtype)
        pos_emb = self.pos_emb(pos_seq) * mask

        out = self.drop(inp + pos_emb)

        for layer in self.layers:
            out = layer(out, mask=mask)

        out = self.dense(out).transpose(1, 2)
        return out
