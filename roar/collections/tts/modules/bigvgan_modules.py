from typing import Tuple, Optional, Iterable

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d
from torch.nn.utils import spectral_norm, weight_norm

from roar.core.classes.common import typecheck
from roar.core.classes.module import NeuralModule
from roar.core.neural_types.elements import (
    AudioSignal,
    VoidType,
    LengthsType,
    EncodedRepresentation,
)
from roar.core.neural_types.neural_type import NeuralType

from roar.collections.tts.modules.alias_free_torch import Activation1d
from roar.collections.tts.parts.utils.activations import Snake, SnakeBeta
from roar.collections.tts.parts.utils.helpers import mask_sequence_tensor

LRELU_SLOPE = 0.1
_ACTIVATIONS = {
    "snake": Snake,
    "snakebeta": SnakeBeta,
}


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def get_padding_2d(
    kernel_size: Tuple[int, int], dilation: Tuple[int, int]
) -> Tuple[int, int]:
    paddings = (
        get_padding(kernel_size[0], dilation[0]),
        get_padding(kernel_size[1], dilation[1]),
    )
    return paddings


def get_down_sample_padding(kernel_size: int, stride: int) -> int:
    return (kernel_size - stride + 1) // 2


def get_up_sample_padding(kernel_size: int, stride: int) -> Tuple[int, int]:
    output_padding = (kernel_size - stride) % 2
    padding = (kernel_size - stride + 1) // 2
    return padding, output_padding


class Conv1dNorm(NeuralModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: Optional[int] = None,
    ):
        super().__init__()
        if not padding:
            padding = get_padding(kernel_size=kernel_size, dilation=dilation)
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode="reflect",
        )
        self.conv = nn.utils.weight_norm(conv)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(("B", "C", "T"), VoidType()),
            "input_len": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(("B", "C", "T"), VoidType()),
        }

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    @typecheck()
    def forward(self, inputs, input_len):
        out = self.conv(inputs)
        out = mask_sequence_tensor(out, input_len)
        return out


class ConvTranspose1dNorm(NeuralModule):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
    ):
        super().__init__()
        padding, output_padding = get_up_sample_padding(kernel_size, stride)
        conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            padding_mode="zeros",
        )
        self.conv = nn.utils.weight_norm(conv)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(("B", "C", "T"), VoidType()),
            "input_len": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(("B", "C", "T"), VoidType()),
        }

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    @typecheck()
    def forward(self, inputs, input_len):
        out = self.conv(inputs)
        out = mask_sequence_tensor(out, input_len)
        return out


class Conv2dNorm(NeuralModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
    ):
        super().__init__()
        assert len(kernel_size) == len(dilation)
        padding = get_padding_2d(kernel_size, dilation)
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode="reflect",
        )
        self.conv = nn.utils.weight_norm(conv)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(("B", "C", "H", "T"), VoidType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(("B", "C", "H", "T"), VoidType()),
        }

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    @typecheck()
    def forward(self, inputs):
        return self.conv(inputs)


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        filters: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        activation: str = "snakebeta",
        snake_logscale: bool = True,
        drop: float = 0.0,
    ):
        self.in_activation = Activation1d(
            activation=_ACTIVATIONS.get(activation, SnakeBeta)(
                channels, alpha_logscale=snake_logscale
            )
        )
        self.skip_activation = Activation1d(
            activation=_ACTIVATIONS.get(activation, SnakeBeta)(
                filters, alpha_logscale=snake_logscale
            )
        )
        self.input_conv = Conv1dNorm(
            in_channels=channels,
            out_channels=filters,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
        )
        self.skip_conv = Conv1dNorm(
            in_channels=filters,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.dropout = nn.Dropout(drop)

    def remove_weight_norm(self):
        self.input_conv.remove_weight_norm()
        self.skip_conv.remove_weight_norm()

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(("B", "C", "T"), VoidType()),
            "input_len": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        return {"out": NeuralType(("B", "C", "T"), EncodedRepresentation())}

    @typecheck()
    def forward(self, inputs, input_len):
        conv_input = self.input_activation(inputs)
        skip_input = self.input_conv(inputs=conv_input, input_len=input_len)
        skip_input = self.skip_activation(skip_input)
        res = self.skip_conv(inputs=skip_input, input_len=input_len)
        res = self.dropout(res)
        out = inputs + res
        return out


class AMPBlock1(NeuralModule):
    """
    Residual block wrapper for BigV-GAN which creates a block for multiple dilations.

    Args:
        channels: Input dimension.
        kernel_size: Kernel size of the residual blocks.
        dilations: List of dilations. One residual block will be created for each dilation in the list.
        activation: Activation for the residual blocks.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: Iterable[int],
        activation: str = "snakebeta",
        alpha_logscale: bool = True,
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    channels=channels,
                    filters=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation=activation,
                    snake_logscale=alpha_logscale,
                )
                for dilation in dilations
            ]
        )

    def remove_weight_norm(self):
        for res_block in self.res_blocks:
            res_block.remove_weight_norm()

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(("B", "C", "T"), VoidType()),
            "input_len": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        return {"out": NeuralType(("B", "C", "T"), VoidType())}

    @typecheck()
    def forward(self, inputs, input_len):
        out = inputs
        for res_block in self.res_blocks:
            out = res_block(inputs=out, input_len=input_len)
        return out


class AMPBlock2(NeuralModule):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: Iterable[int],
        activation: str = "snakebeta",
        alpha_logscale: bool = True,
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                Conv1dNorm(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation,
                )
                for dilation in dilations
            ]
        )
        self.activations = nn.ModuleList()
        for _ in dilations:
            self.activations.append(
                Activation1d(
                    activation=_ACTIVATIONS.get(activation, SnakeBeta)(
                        channels, alpha_logscale=alpha_logscale
                    )
                )
            )

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(("B", "C", "T"), VoidType()),
            "input_len": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        return {"out": NeuralType(("B", "C", "T"), VoidType())}

    def forward(self, inputs, input_lengths):
        for c, a in zip(self.convs, self.activations):
            xt = a(inputs)
            xt = c(xt, input_lengths)
            x = xt + inputs
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            l.remove_weight_norm()


class Generator(NeuralModule):
    __constants__ = ["lrelu_slope", "num_kernels", "num_upsamples"]

    def __init__(
        self,
        resblock,
        upsample_rates,
        upsample_kernel_sizes,
        upsample_initial_channel,
        in_kernel_size: int = 7,
        out_kernel_size: int = 3,
        resblock_kernel_sizes: Iterable[int] = (3, 7, 11),
        resblock_dilation_sizes: Iterable[int] = (1, 3, 5),
        initial_input_size: int = 80,
        activation: str = "snakebeta",
        alpha_logscale: bool = True,
    ):
        assert in_kernel_size > 0
        assert out_kernel_size > 0
        super().__init__()
        self.upsample_rates = upsample_rates
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upsample_factor = np.prod(list(upsample_rates))
        self.conv_pre = Conv1dNorm(
            initial_input_size, upsample_initial_channel, in_kernel_size, 1
        )
        self.lrelu_slope = LRELU_SLOPE
        resblock = AMPBlock1 if resblock == 1 else AMPBlock2

        # in_channels = upsample_initial_channel
        self.activations = nn.ModuleList([])
        self.up_sample_conv_layers = nn.ModuleList([])
        self.res_layers = nn.ModuleList([])
        for i, (up_sample_rate, upsample_kernel_size) in enumerate(
            zip(upsample_rates, upsample_kernel_sizes)
        ):
            in_channels = upsample_initial_channel // (2**i)
            out_channels = in_channels // 2

            act = Activation1d(
                activation=_ACTIVATIONS.get(activation, SnakeBeta)(
                    in_channels, alpha_logscale=alpha_logscale
                )
            )
            self.activations.append(act)

            up_sample_conv = ConvTranspose1dNorm(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=upsample_kernel_size,
                stride=up_sample_rate,
            )

            self.up_sample_conv_layers.append(up_sample_conv)

            res_layer = resblock(
                channels=in_channels,
                kernel_sizes=resblock_kernel_sizes,
                dilations=resblock_dilation_sizes,
                activation=activation,
                alpha_logscale=alpha_logscale,
            )
            self.res_layers.append(res_layer)

        self.post_activation = Activation1d(
            activation=_ACTIVATIONS.get(activation, SnakeBeta)(
                in_channels, alpha_logscale=alpha_logscale
            )
        )
        self.post_conv = Conv1dNorm(
            in_channels=in_channels, out_channels=1, kernel_size=out_kernel_size
        )
        self.out_activation = nn.Tanh()

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(("B", "D", "T_encoded"), VoidType()),
            "input_len": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "audio": NeuralType(("B", "T_audio"), AudioSignal()),
            "audio_len": NeuralType(tuple("B"), LengthsType()),
        }

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        for up_sample_conv in self.up_sample_conv_layers:
            up_sample_conv.remove_weight_norm()
        for res_layer in self.res_layers:
            res_layer.remove_weight_norm()

    @typecheck()
    def forward(self, inputs, input_len):
        audio_len = input_len
        # [B, C, T_encoded]
        out = self.pre_conv(inputs=inputs, input_len=audio_len)
        for act, res_layer, up_sample_conv, up_sample_rate in zip(
            self.activations,
            self.res_layers,
            self.up_sample_conv_layers,
            self.up_sample_rates,
        ):
            audio_len = audio_len * up_sample_rate
            out = act(out)
            # [B, C / 2, T * up_sample_rate]
            out = up_sample_conv(inputs=out, input_len=audio_len)
            out = res_layer(inputs=out, input_len=audio_len)

        out = self.post_activation(out)
        # [B, 1, T_audio]
        out = self.post_conv(inputs=out, input_len=audio_len)
        audio = self.out_activation(out)
        # audio = rearrange(audio, "B 1 T -> B T")
        return audio, audio_len


class DiscriminatorP(NeuralModule):
    def __init__(
        self, period, kernel_size=5, stride=3, use_spectral_norm=False, debug=False
    ):
        super().__init__()
        self.period = period
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        conv_ch = [32, 128, 512, 1024] if not debug else [8, 12, 32, 64]
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        conv_ch[0],
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        conv_ch[0],
                        conv_ch[1],
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        conv_ch[1],
                        conv_ch[2],
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        conv_ch[2],
                        conv_ch[3],
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(conv_ch[3], conv_ch[3], (kernel_size, 1), 1, padding=(2, 0))
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(conv_ch[3], 1, (3, 1), 1, padding=(1, 0)))

    @property
    def input_types(self):
        return {
            "x": NeuralType(("B", "S", "T"), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "decision": NeuralType(("B", "T"), VoidType()),
            "feature_maps": [NeuralType(("B", "C", "H", "W"), VoidType())],
        }

    @typecheck()
    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(NeuralModule):
    def __init__(self, debug=False):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(2, debug=debug),
                DiscriminatorP(3, debug=debug),
                DiscriminatorP(5, debug=debug),
                DiscriminatorP(7, debug=debug),
                DiscriminatorP(11, debug=debug),
            ]
        )

    @property
    def input_types(self):
        return {
            "y": NeuralType(("B", "S", "T"), AudioSignal()),
            "y_hat": NeuralType(("B", "S", "T"), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "real_scores": [NeuralType(("B", "T"), VoidType())],
            "fake_scores": [NeuralType(("B", "T"), VoidType())],
            "real_feature_maps": [[NeuralType(("B", "C", "H", "W"), VoidType())]],
            "fake_feature_maps": [[NeuralType(("B", "C", "H", "W"), VoidType())]],
        }

    @typecheck()
    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(NeuralModule):
    def __init__(self, use_spectral_norm=False, debug=False):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        conv_ch = [128, 256, 512, 1024] if not debug else [16, 32, 32, 64]
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, conv_ch[0], 15, 1, padding=7)),
                norm_f(Conv1d(conv_ch[0], conv_ch[0], 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(conv_ch[0], conv_ch[1], 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(conv_ch[1], conv_ch[2], 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(conv_ch[2], conv_ch[3], 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(conv_ch[3], conv_ch[3], 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(conv_ch[3], conv_ch[3], 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(conv_ch[3], 1, 3, 1, padding=1))

    @property
    def input_types(self):
        return {
            "x": NeuralType(("B", "S", "T"), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "decision": NeuralType(("B", "T"), VoidType()),
            "feature_maps": [NeuralType(("B", "C", "T"), VoidType())],
        }

    @typecheck()
    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(NeuralModule):
    def __init__(self, debug=False):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True, debug=debug),
                DiscriminatorS(debug=debug),
                DiscriminatorS(debug=debug),
            ]
        )
        self.meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)]
        )

    @property
    def input_types(self):
        return {
            "y": NeuralType(("B", "S", "T"), AudioSignal()),
            "y_hat": NeuralType(("B", "S", "T"), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "real_scores": [NeuralType(("B", "T"), VoidType())],
            "fake_scores": [NeuralType(("B", "T"), VoidType())],
            "real_feature_maps": [[NeuralType(("B", "C", "T"), VoidType())]],
            "fake_feature_maps": [[NeuralType(("B", "C", "T"), VoidType())]],
        }

    @typecheck()
    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(NeuralModule):
    def __init__(self, use_spectral_norm, resolution):
        super().__init__()

        self.resolution = resolution
        assert (
            len(self.resolution) == 3
        ), "MRD layer requires list with len=3, got {}".format(self.resolution)
        self.lrelu_slope = LRELU_SLOPE

        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
            ]
        )
        self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    @property
    def input_types(self):
        return {
            "x": NeuralType(("B", "S", "T"), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "decision": NeuralType(("B", "T"), VoidType()),
            "feature_maps": [NeuralType(("B", "C", "T"), VoidType())],
        }

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(
            x,
            (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
            mode="reflect",
        )
        x = x.squeeze(1)
        x = torch.view_as_real(
            torch.stft(
                x,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                center=False,
                return_complex=True,
            )
        )  # [B, F, TT, 2] (Note: torch.stft() returns complex tensor [B, F, TT]; converted via view_as_real)
        mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

        return mag


class MultiResolutionDiscriminator(NeuralModule):
    def __init__(
        self,
        resolutions: Iterable[Tuple(int, int, int)],
        use_spectral_norm: bool = False,
        debug=False,
    ):
        super().__init__()
        self.resolutions = resolutions
        assert (
            len(self.resolutions) == 3
        ), "MRD requires list of list with len=3, each element having a list with len=3. got {}".format(
            self.resolutions
        )
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorR(use_spectral_norm, resolution)
                for resolution in self.resolutions
            ]
        )

    @property
    def input_types(self):
        return {
            "y": NeuralType(("B", "S", "T"), AudioSignal()),
            "y_hat": NeuralType(("B", "S", "T"), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "real_scores": [NeuralType(("B", "T"), VoidType())],
            "fake_scores": [NeuralType(("B", "T"), VoidType())],
            "real_feature_maps": [[NeuralType(("B", "C", "T"), VoidType())]],
            "fake_feature_maps": [[NeuralType(("B", "C", "T"), VoidType())]],
        }

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
