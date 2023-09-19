from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from roar.collections.tts.modules.submodules import (
    ConditionalInput,
    ConditionalLayerNorm,
    LinearNorm,
    Swish,
)
import roar.collections.tts.modules.submodules as submodules
from roar.collections.tts.modules.postional_embedding import PositionalEmbedding
from roar.collections.tts.modules.attention import MultiHeadAttn
from roar.collections.tts.parts.utils.helpers import get_mask_from_lengths
from roar.core.classes import NeuralModule, adapter_mixins, typecheck, access_mixins
from roar.core.neural_types import NeuralType
from roar.core.neural_types.elements import (
    EncodedRepresentation,
    LengthsType,
    MaskType,
    TokenIndex,
)
from roar.core.neural_types.neural_type import NeuralType


# TODO: move mask_from_lens to roar.collections.tts.parts.utils.helpers
def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


class ConvolutionalModule(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_channels,
        kernel_size,  # depthwise_kernel_size
        dropout,
        pre_lnorm=True,  # https://arxiv.org/pdf/2005.08100.pdf figure 2
        condition_types=[],
        activation_fn="GLU",
        bias=True,
    ) -> None:
        assert kernel_size - 1 % 2 == 1, "kernel size must be odd for 'SAME' padding"
        assert hasattr(nn, activation_fn) or hasattr(
            submodules, activation_fn
        ), f"Activation function {activation_fn} not found in torch.nn"
        super(ConvolutionalModule, self).__init__()
        self.embed_dim = embed_dim
        self.n_channels = n_channels
        self.dropout = dropout

        if hasattr(nn, activation_fn):
            self.depthwise_activation = getattr(nn, activation_fn)()
        else:  # alias for SiLU in Swish
            self.depthwise_activation = getattr(submodules, activation_fn)()

        self.CoreNet = nn.Sequential(
            nn.Conv1d(
                embed_dim, 2 * n_channels, kernel_size=1, stride=1, padding=0, bias=bias
            ),  # pointwise_conv_in B x 2C x T done for glu which uses one half of the input as a gating mechanism for the other half.
            nn.GLU(dim=1),  # B x C x T
            nn.Conv1d(
                n_channels,
                n_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=n_channels,
                bias=bias,
            ),  # depth-wise conv B x C x T
            nn.BatchNorm1d(n_channels),
            self.depthwise_activation(),
            nn.Conv1d(
                n_channels, embed_dim, kernel_size=1, stride=1, padding=0, bias=bias
            ),  # pointwise_conv_out B x C x T
            nn.Dropout(dropout),
        )
        self.layer_norm = ConditionalLayerNorm(
            embed_dim, condition_dim=embed_dim, condition_types=condition_types
        )
        self.pre_lnorm = pre_lnorm

    def forward(self, x, conditioning=None):
        self._forward(x, conditioning)

    def _forward(self, x, conditioning=None):
        """
        Args:
            x: Input of shape B x T x C
        Returns:
            Tensor of shape B x T x C
        """
        x = x.transpose(1, 2)  # B x C x T

        if self.pre_lnorm:
            x = self.layer_norm(x, conditioning).to(x.dtype)
            x = self.CoreNet(x)
            x = x.transpose(1, 2)  # B x T x C

        else:
            x = self.CoreNet(x)
            x = x.transpose(1, 2)  # B x T x C
            x = self.layer_norm(x, conditioning).to(x.dtype)

        return x


class PositionwiseFF(nn.Module):
    """
    Position-wise Feedforward layer from Conformer paper.
    """

    def __init__(
        self,
        embed_dim,
        hidden_size,
        dropout_l1=0.0,
        dropout_l2=0.0,
        activation_fn="Swish",
        pre_lnorm=False,
        bias=True,
    ) -> None:
        assert hasattr(nn, activation_fn) or hasattr(
            submodules, activation_fn
        ), f"Activation function {activation_fn} not found in torch.nn"
        super(PositionwiseFF, self).__init__()
        if hasattr(nn, activation_fn):
            self.activation = getattr(nn, activation_fn)()
        else:  # alias for SiLU in Swish
            self.activation = getattr(submodules, activation_fn)()
        self.CoreNet = nn.Sequential(
            nn.Linear(embed_dim, hidden_size, bias=bias),
            self.activation(),
            nn.Dropout(dropout_l1),
            nn.Linear(hidden_size, embed_dim, bias=bias),
            nn.Dropout(dropout_l2),
        )
        self.layer_norm = ConditionalLayerNorm(embed_dim)
        self.pre_lnorm = pre_lnorm

    def forward(self, x, conditioning=None):
        self._forward(x, conditioning)

    def _forward(self, x, conditioning=None):
        """
        Args:
            x: Input of shape B x T x C
        Returns:
            Tensor of shape B x T x C
        """
        if self.pre_lnorm:
            x = self.layer_norm(x, conditioning).to(x.dtype)
            x = self.CoreNet(x)
        else:
            x = self.CoreNet(x)
            x = self.layer_norm(x, conditioning).to(x.dtype)

        return x


class ConformerLayer(
    nn.Module, adapter_mixins.AdapterModuleMixin, access_mixins.AccessMixin
):  # TODO: add flash attention support for conformer and finish the docstrings
    """A single block of the Conformer encoder.

    Args:
        n_heads (int): number of heads for multi-head attention
        conv_kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
    """

    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        conv_kernel_size=31,
        dropout=0.1,
        dropout_att=0.1,
        condition_types=[],
        **kwargs,
    ) -> None:
        super(ConformerLayer, self).__init__()

        self.ff1 = PositionwiseFF(
            d_model,
            d_inner,
            dropout_l1=dropout,
            activation_fn=kwargs.get("ff_activation", "Swish"),
            pre_lnorm=True,
        )

        self.convolutional_block = ConvolutionalModule(
            d_model,
            d_model,
            kernel_size=conv_kernel_size,
            dropout=dropout,
            pre_lnorm=True,
            condition_types=condition_types,
            activation_fn=kwargs.get("conv_activation", "Swish"),
        )

        self.self_attn = MultiHeadAttn(
            n_head,
            d_model,
            d_head,
            dropout_att,
            condition_types=condition_types,
            **kwargs,
        )

        self.ff2 = PositionwiseFF(
            d_model,
            d_inner,
            dropout_l1=dropout,
            dropout_l2=dropout,
            activation_fn=kwargs.get("ff_activation", "Swish"),
            pre_lnorm=False,
        )

    def forward(self, x, mask=None, conditioning=None):
        """
        Args:
            x: Input of shape B x T x C #TODO: recheck shapes
            mask: Input mask of shape B x T x T
            conditioning: Input conditioning of shape B x C
        Returns:
            Tensor of shape B x T x C
        """
        residual = x
        output = self.ff1(x, conditioning)
        residual = output * 0.5 + residual
        output = residual

        output = self.self_attn(
            output, attn_mask=~mask.squeeze(2), conditioning=conditioning
        )

        residual = output + residual
        output = residual

        if self.is_adapter_available():
            o_x = {
                "inp": output,
                "loc": "mha",
                "attn_mask": mask,
                "conditioning": conditioning,
            }
            o_y = self.forward_enabled_adapters(o_x)
            residual = o_y["output"]
            output = residual

        output = self.convolutional_block(output, conditioning)

        residual = output + residual
        output = residual

        output = self.ff2(output, conditioning)
        residual = output * 0.5 + residual
        output = residual

        if self.is_adapter_available():
            o_x = {
                "inp": output,
                "loc": "post_ff",
            }
            o_y = self.forward_enabled_adapters(output)
            output = o_y["output"]

        if self.is_access_enabled() and self.access_cfg.get(
            "save_encoded_tensors", False
        ):
            self.register_accessible_tensor(name="encoder", tensor=output)

        return output

    def forward_single_enabled_adapter_(
        self,
        input: Dict,
        adapter_module: nn.Module,
        *,
        adapter_name: str,
        adapter_strategy: "roar.core.classes.mixins.adapter_mixin_strategies.AbstractAdapterStrategy",
    ):
        """
        Forward step of a single adapter module.
        Args:
            input: Dictionary containing the input tensors
                inp: Input to the adapter module (output of mha or ff2).
                loc: context location of the adapter module call.
                attn_mask: Optional attention mask.
                conditioning: Optional conditioning tensor.
            adapter_module: The adapter module to be used.
        """
        assert all([hasattr(input, k) for k in ["inp", "loc"]]), (
            "Input must be a dictionary with keys: "
            "'inp': input to the adapter module, "
            "'loc': location of the adapter module call."
            "'attn_mask': Optional attention mask."
        )
        inp = input["inp"]
        loc = input["loc"]
        attn_mask = input.get("attn_mask", None)
        conditioning = input.get("conditioning", None)

        if isinstance(adapter_module, adapter_module.LinearAdapter) and loc == ["post"]:
            output = adapter_strategy(inp, adapter_module, module=self)
        elif isinstance(adapter_module, MultiHeadAttn) and loc == ["mha"]:
            mha_inp = {
                "inp": inp,
                "attn_mask": attn_mask,
                "conditioning": conditioning,
            }
            output = adapter_strategy(mha_inp, adapter_module, module=self)
        else:
            output = inp

        input["output"] = output
        return input


class FFConformerDecoder(NeuralModule):
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
        **kwargs,
    ):
        super(FFConformerDecoder, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.pos_emb = PositionalEmbedding(self.d_model)
        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()
        self.cond_input = ConditionalInput(d_model, d_model, condition_types)

        for _ in range(n_layer):
            self.layers.append(
                ConformerLayer(
                    n_head,
                    d_model,
                    d_head,
                    d_inner,
                    conv_kernel_size=kernel_size,
                    dropout=dropout,
                    dropout_att=dropatt,
                    condition_types=condition_types,
                    pre_lnorm=pre_lnorm,
                    ff_activation=kwargs.get("ff_activation", "Swish"),
                    conv_activation=kwargs.get("conv_activation", "Swish"),
                )
            )

    @property
    def input_types(self) -> Dict[str, NeuralType] | None:
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

    def _forward(self, inp, mask, conditioning=None):
        pos_seq = torch.arange(inp.size(1), device=inp.device).to(inp.dtype)
        pos_emb = self.pos_emb(pos_seq) * mask
        inp += pos_emb
        inp = self.cond_input(inp, conditioning)
        out = self.drop(inp)

        for layer in self.layers:
            out = layer(out, mask=mask, conditioning=conditioning)

        # out = self.drop(out)
        return out, mask


class FFConformerEncoder(FFConformerDecoder):
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
    ):
        super(FFConformerEncoder, self).__init__(
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


class FFConformer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=1,
        n_layers=6,
        n_head=1,
        d_head=64,
        d_inner=1024,
        kernel_size=31,
        dropout=0.1,
        dropatt=0.1,
        dropemb=0.0,
        **kwargs,
    ):
        # TODO: Expand kwargs and set deterministic number of parameters
        super(FFConformer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_head = n_head
        self.d_head = d_head

        self.pos_emb = PositionalEmbedding(self.in_dim)
        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(
                ConformerLayer(
                    n_head,
                    in_dim,
                    d_head,
                    d_inner,
                    conv_kernel_size=kernel_size,
                    dropout=dropout,
                    dropout_att=dropatt,
                    **kwargs,
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
