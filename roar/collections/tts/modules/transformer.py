import warnings
from typing import Optional, Tuple, Iterable

import torch
import torch.nn as nn

# from xformers.ops import SwiGLU
from einops import rearrange
from lightning_utilities.core.imports import RequirementCache

from roar.collections.tts.modules.submodules import (
    ConditionalInput,
    ConditionalLayerNorm,
    ConditionalRMSNorm,
    LinearNorm,
)
from roar.collections.tts.modules.attention import MultiHeadAttn, FlashSelfAttention
from roar.collections.tts.modules.postional_embedding import PositionalEmbedding
from roar.collections.tts.parts.utils.helpers import (
    get_mask_from_lengths,
    build_rope_cache,
)
from roar.collections.tts.parts.utils.bert_padding import (
    unpad_input,
    pad_input,
    index_first_axis,
)
from roar.collections.tts.parts.submodules import SwiGLU

from roar.core.classes import NeuralModule, adapter_mixins, typecheck
from roar.core.neural_types.elements import (
    EncodedRepresentation,
    LengthsType,
    MaskType,
    TokenIndex,
)
from roar.core.neural_types.neural_type import NeuralType

HAVE_FLASH = RequirementCache("flash-attn>=2.0.0.post1")
RoPECache = Tuple[torch.Tensor, torch.Tensor]


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
            #TODO: FIXME
            core_out = self.layer_norm(inp, conditioning).to(inp.dtype)
            core_out = core_out.transpose(1, 2)
            core_out = self.CoreNet(core_out)
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
        **kwargs,
    ):  # TODO: add flash attention support for transformer
        super(TransformerLayer, self).__init__()
        AttentionBlock = MultiHeadAttn

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
        dtype=None,
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
                    use_flash=use_flash,
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


# class StyleAdaptiveTransformerDecoder(NeuralModule):
#     def __init__(
#         self,
#         n_layer,
#         n_head,
#         d_model,
#         d_head,
#         d_inner,
#         kernel_size,
#         dropout,
#         dropatt,
#         dropemb=0.0,
#         pre_lnorm=False,
#         condition_types=[],
#         use_flash=False,
#     ):
#         super(FFTransformerDecoder, self).__init__()
#         self.d_model = d_model
#         self.n_head = n_head
#         self.d_head = d_head

#         self.pos_emb = PositionalEmbedding(self.d_model)
#         self.drop = nn.Dropout(dropemb)
#         self.layers = nn.ModuleList()
#         self.style_adapters = nn.ModuleList()
#         self.cond_input = ConditionalInput(d_model, d_model, condition_types)
#         for _ in range(n_layer):
#             self.layers.append(
#                 TransformerLayer(
#                     n_head,
#                     d_model,
#                     d_head,
#                     d_inner,
#                     kernel_size,
#                     dropout,
#                     dropatt=dropatt,
#                     pre_lnorm=pre_lnorm,
#                     condition_types=condition_types,
#                     use_flash=use_flash,
#                 )
#             )
#             self.style_adapters.append(
#                 MultiHeadCrossAttn(n_head, d_model, d_head, dropout, pre_lnorm),
#             )

#     @property
#     def input_types(self):
#         return {
#             "input": NeuralType(("B", "T", "D"), EncodedRepresentation()),
#             "seq_lens": NeuralType(("B"), LengthsType()),
#             "conditioning": NeuralType(
#                 ("B", "T", "D"), EncodedRepresentation(), optional=True
#             ),
#         }

#     @property
#     def output_types(self):
#         return {
#             "out": NeuralType(("B", "T", "D"), EncodedRepresentation()),
#             "mask": NeuralType(("B", "T", "D"), MaskType()),
#         }

#     @typecheck()
#     def forward(self, input, seq_lens, conditioning=None):
#         return self._forward(input, mask_from_lens(seq_lens).unsqueeze(2), conditioning)

#     def _forward(self, inp, mask, conditioning):
#         pos_seq = torch.arange(inp.size(1), device=inp.device).to(inp.dtype)
#         pos_emb = self.pos_emb(pos_seq) * mask
#         inp += pos_emb
#         inp = self.cond_input(inp, conditioning)
#         out = self.drop(inp)

#         for layer, style_adapter in zip(self.layers, self.style_adapters):
#             conditioning = style_adapter(out, conditioning, mask=~mask.squeeze(2))
#             out = layer(out, mask=mask, conditioning=conditioning)

#         # out = self.drop(out)
#         return out, mask


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
        dtype=None,
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


class BiLLaMaLayer(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        n_query_groups: Optional[int] = None,
        pre_lnorm: bool = True,
        condition_types: Iterable[str] = [],
    ):
        super(BiLLaMaLayer, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.pre_lnorm = pre_lnorm
        
        self.attention = FlashSelfAttention(
            n_head, d_model, d_head, n_query_groups
        )

        self.dropout = nn.Dropout(dropout)
        # self.o_net = nn.Linear(self.n_head * self.d_head, self.d_model)
        self.o_net = SwiGLU(self.d_model, self.d_model)
        self.norm_1 = ConditionalRMSNorm(
            self.d_model, self.d_model, condition_types=condition_types
        )
        self.norm_2 = ConditionalRMSNorm(
            self.d_model, self.d_model, condition_types=condition_types
        )
        if not HAVE_FLASH:
            warnings.warn(
                "Unable to import flash attention. Defaulting to pytorch implementation. This will reduce the throughput of the model"
            )

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
        if subset_idx is not None:
            residual = index_first_axis(inp, subset_idx)
        else:
            residual = pad_input(inp, indices, attn_mask.shape[0], max_seq_length)
        if self.pre_lnorm:
            # layer normalization
            inp = pad_input(inp, indices, attn_mask.shape[0], max_seq_length)
            # print("CU SEQLENS ", cu_seqlens.shape)
            # print("INP SHAPE ", inp.shape)
            # print("conditioning SHAPE ", conditioning.shape)
            inp = self.norm_1(inp, conditioning)
            inp, _, _, _ = unpad_input(inp, attn_mask)
            
        attn_vec = self.attention(
            inp,
            rope,
            cu_seqlens,
            max_seq_length,
            subset_idx,
            indices,
            attn_mask=attn_mask,
            conditioning=conditioning,
        )
        if self.pre_lnorm:
            # residual connection
            attn_vec = pad_input(attn_vec, indices, attn_mask.shape[0], max_seq_length)
            # print("ATTN VEC SHAPE ", attn_vec.shape)
            # print("conditioning SHAPE ", conditioning.shape)
            intermidiate = self.norm_2(residual + attn_vec, conditioning)
            # intermidiate, _, _, _ = unpad_input(intermidiate, attn_mask)
        else:
            # residual connection + layer normalization
            attn_vec = pad_input(attn_vec, indices, attn_mask.shape[0], max_seq_length)
            intermidiate = self.norm_1(residual + attn_vec, conditioning)
            # intermidiate, _, _, _ = unpad_input(intermidiate, attn_mask)

        if subset_idx is not None:
            attn_out = self.o_net(index_first_axis(intermidiate, subset_idx))
        else:
            attn_out = self.o_net(intermidiate)

        attn_out = self.dropout(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = intermidiate + attn_out
        else:
            # residual connection + layer normalization
            output = self.norm_2(intermidiate + attn_out, conditioning)
        output, *_ = unpad_input(output, attn_mask)
        return output


class FlashTransformerLayer(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size,
        dropout,
        dropatt=0.0,
        n_query_groups=None,
        condition_types=[],
        pre_lnorm=True,
    ):
        super(FlashTransformerLayer, self).__init__()

        self.dec_attn = BiLLaMaLayer(
            n_head,
            d_model,
            d_head,
            dropout,
            n_query_groups=n_query_groups,
            pre_lnorm=pre_lnorm,
            condition_types=condition_types,
        )
        self.pos_ff = PositionwiseConvFF(
            d_model,
            d_inner,
            kernel_size,
            dropout,
            pre_lnorm=pre_lnorm,
            condition_types=condition_types,
        )

    def forward(
        self,
        dec_inp: torch.Tensor,
        rope: RoPECache,
        cu_seqlens: torch.Tensor,
        batch: int,
        max_seq_length: int,
        subset_idx: Optional[torch.Tensor],
        indices: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        conditioning: Optional[torch.Tensor] = None,
    ):
        output = self.dec_attn(
            dec_inp,
            rope,
            cu_seqlens,
            max_seq_length,
            subset_idx,
            indices,
            attn_mask=attn_mask,
            conditioning=conditioning,
        )
        output = pad_input(output, indices, batch, max_seq_length)
        output = self.pos_ff(output, conditioning)
        output, _, _, _ = unpad_input(output, attn_mask)
        return output


class FlashTransformerDecoder(NeuralModule):
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
        rope_base=10_000,
        rope_condense_ratio=1.0,
        rotary_percentage=0.25,
        n_query_groups=None,
        condition_types=[],
        pre_lnorm=True,
        dtype=None,
    ):
        super(FlashTransformerDecoder, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.dtype = dtype

        if n_query_groups is None:
            n_query_groups = n_head

        self.rope_cache = None
        self.rope_base = rope_base
        self.rope_condense_ratio = rope_condense_ratio
        self.rotary_percentage = rotary_percentage

        self.layers = nn.ModuleList()
        self.cond_input = ConditionalInput(d_model, d_model, condition_types)

        for _ in range(n_layer):
            self.layers.append(
                FlashTransformerLayer(
                    n_head,
                    d_model,
                    d_head,
                    d_inner,
                    kernel_size,
                    dropout,
                    dropatt=dropatt,
                    pre_lnorm=pre_lnorm,
                    n_query_groups=n_query_groups,
                    condition_types=condition_types,
                )
            )

    def build_rope_cache(self, inp: torch.Tensor):
        return build_rope_cache(
            seq_len=inp.size(1),
            n_elem=int(self.rotary_percentage * self.d_head),
            dtype=self.dtype, #TODO: figure out a neat way to do this
            device=inp.device,
            base=self.rope_base,
            condense_ratio=self.rope_condense_ratio,
        )

    @property
    def input_types(self):
        return {
            "input": NeuralType(("B", "T", "D"), EncodedRepresentation()),
            "seq_lens": NeuralType(("B"), LengthsType()),
            "conditioning": NeuralType(
                ("B", "T", "D"), EncodedRepresentation(), optional=True
            ),
            "subset_mask": NeuralType(
                ("B", "T"), EncodedRepresentation(), optional=True
            ),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(("B", "T", "D"), EncodedRepresentation()),
            "mask": NeuralType(("B", "T"), MaskType()),
        }

    @typecheck()
    def forward(self, input, seq_lens, conditioning=None, subset_mask=None):
        return self._forward(input, mask_from_lens(seq_lens), conditioning, subset_mask)

    def _forward(self, inp, mask, conditioning, subset_mask):
        B, T, _ = inp.size()
        if self.rope_cache is None or self.rope_cache[0].size(0) < T:
            self.rope_cache = self.build_rope_cache(inp)

        inp = self.cond_input(inp, conditioning)
        # print("PRE UNPAD SHAPE ", inp.shape)
        out, indices, cu_seqlens, _ = unpad_input(inp, mask)

        cos, sin = self.rope_cache
        cos, sin = cos[:T], sin[:T]

        if subset_mask is None:
            for layer in self.layers:
                out = layer(
                    out,
                    (cos, sin),
                    cu_seqlens,
                    B,
                    T,
                    None,
                    indices,
                    attn_mask=mask,
                    conditioning=conditioning,
                )
            # print("PRE PAD", out.shape)
            out = pad_input(out, indices, B, T)
            # print("POST PAD", out.shape)
        else:
            for i, layer in enumerate(self.layers):
                out = layer(
                    out,
                    (cos, sin),
                    cu_seqlens,
                    B,
                    T,
                    None,
                    indices,
                    attn_mask=mask,
                    conditioning=conditioning,
                )
            subset_idx = torch.nonzero(subset_mask[mask], as_tuple=False).flatten()

            out = self.layers[-1](
                out,
                (cos, sin),
                cu_seqlens,
                T,
                subset_idx=subset_idx,
                indices=indices,
                attn_mask=mask,
            )

        return out, mask


class FlashTransformerEncoder(FlashTransformerDecoder):
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
        n_embed=None,
        d_embed=None,
        padding_idx=0,
        rope_base=10_000,
        rope_condense_ratio=1.0,
        rotary_percentage=0.25,
        n_query_groups=None,
        condition_types=[],
        pre_lnorm=True,
        dtype=None,
    ):
        super(FlashTransformerEncoder, self).__init__(
            n_layer,
            n_head,
            d_model,
            d_head,
            d_inner,
            kernel_size,
            dropout,
            dropatt,
            rope_base=rope_base,
            rope_condense_ratio=rope_condense_ratio,
            rotary_percentage=rotary_percentage,
            n_query_groups=n_query_groups,
            condition_types=condition_types,
            pre_lnorm=pre_lnorm,
            dtype=dtype,
        )

        self.padding_idx = padding_idx
        self.word_emb = nn.Embedding(
            n_embed, d_embed or d_model, padding_idx=padding_idx
        )

    @property
    def input_types(self):
        return {
            "input": NeuralType(("B", "T"), TokenIndex()),
            "conditioning": NeuralType(
                ("B", "T", "D"), EncodedRepresentation(), optional=True
            ),
        }

    def forward(self, input, conditioning=0, subset_mask=None):
        return self._forward(
            self.word_emb(input), (input != self.padding_idx).unsqueeze(2), conditioning, subset_mask
        )  # (B, L, 1)
