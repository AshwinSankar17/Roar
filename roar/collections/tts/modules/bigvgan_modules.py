from typing import Tuple, Optional, Iterable

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import spectral_norm, weight_norm, remove_weight_norm

from roar.core.classes.common import typecheck
from roar.core.classes.module import NeuralModule
from roar.core.neural_types.elements import (
    AudioSignal,
    VoidType,
    LengthsType,
    EncodedRepresentation,
    MelSpectrogramType
)
from roar.core.neural_types.neural_type import NeuralType

from roar.collections.tts.modules.alias_free_torch import Activation1d
from roar.collections.tts.parts.submodules.activations import Snake, SnakeBeta
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


class AMPBlock1(torch.nn.Module):

    def __init__(self, channels, kernel_size, dilation, activation="snakebeta", alpha_logscale=True):
        super().__init__()
        # self.lrelu_slope = LRELU_SLOPE
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2)

        if activation == 'snake': # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=Snake(channels, alpha_logscale=alpha_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == 'snakebeta': # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=SnakeBeta(channels, alpha_logscale=alpha_logscale))
                 for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")


    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):

    def __init__(self, channels, kernel_size, dilation, activation="snakbeta", alpha_logscale=True):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)

        if activation == 'snake': # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=Snake(channels, alpha_logscale=alpha_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == 'snakebeta': # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=SnakeBeta(channels, alpha_logscale=alpha_logscale))
                 for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x):
        for c, a in zip (self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


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
        apply_weight_init_conv_pre=False,
    ):
        assert in_kernel_size > 0
        assert out_kernel_size > 0
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        # self.upsample_rates = upsample_rates
        self.upsample_factor = np.prod(list(upsample_rates))
        self.conv_pre = weight_norm(
            Conv1d(
                initial_input_size, upsample_initial_channel, in_kernel_size, 1, padding=get_padding(in_kernel_size, 1)
            )
        )
        # self.lrelu_slope = LRELU_SLOPE
        resblock = AMPBlock1 if resblock == 1 else AMPBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            resblock_list = nn.ModuleList()
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                resblock_list.append(resblock(ch, k, d, activation=activation, alpha_logscale=alpha_logscale))
            self.resblocks.append(resblock_list)
        
        if activation == "snake": # periodic nonlinearity with snake function and anti-aliasing
            activation_post = Snake(ch, alpha_logscale=alpha_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        elif activation == "snakebeta": # periodic nonlinearity with snakebeta function and anti-aliasing
            activation_post = SnakeBeta(ch, alpha_logscale=alpha_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        if apply_weight_init_conv_pre:
            self.conv_pre.apply(init_weights)

    @property
    def input_types(self):
        return {
            "x": NeuralType(("B", "D", "T"), MelSpectrogramType()),
        }

    @property
    def output_types(self):
        return {
            "audio": NeuralType(("B", "S", "T"), AudioSignal()),
        }

    @typecheck()
    def forward(self, x):
        x = self.conv_pre(x)
        for upsample_layer, resblock_group in zip(self.ups, self.resblocks):
            # x = F.leaky_relu(x, self.lrelu_slope)
            x = upsample_layer(x)
            xs = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
            for resblock in resblock_group:
                xs += resblock(x)
            x = xs / self.num_kernels
        # x = F.leaky_relu(x)
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for group in self.resblocks:
            for block in group:
                block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


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
                window=torch.hann_window(win_length).to(dtype=torch.float, device=x.device),
                center=False,
                return_complex=True,
            )
        )  # [B, F, TT, 2] (Note: torch.stft() returns complex tensor [B, F, TT]; converted via view_as_real)
        mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

        return mag


class MultiResolutionDiscriminator(NeuralModule):
    def __init__(
        self,
        resolutions: Iterable[Tuple[int, int, int]],
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
    
class Discriminator(NeuralModule):
    """
    Wrapper class which takes a list of discriminators and aggregates the results across them.
    """

    def __init__(self, discriminators: Iterable[NeuralModule]):
        super().__init__()
        self.discriminators = nn.ModuleList(discriminators)

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'S', 'T_audio'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'S', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "scores_real": [NeuralType(('B', 'T_out'), VoidType())],
            "scores_gen": [NeuralType(('B', 'T_out'), VoidType())],
            "fmaps_real": [[NeuralType(('B', 'T_layer', 'C'), VoidType())]],
            "fmaps_gen": [[NeuralType(('B', 'T_layer', 'C'), VoidType())]],
        }

    @typecheck()
    def forward(self, audio_real, audio_gen):
        scores_real = []
        scores_gen = []
        fmaps_real = []
        fmaps_gen = []
        for discriminator in self.discriminators:
            score_real, score_gen, fmap_real, fmap_gen = discriminator(y=audio_real, y_hat=audio_gen)
            scores_real += score_real
            fmaps_real += fmap_real
            scores_gen += score_gen
            fmaps_gen += fmap_gen

        return scores_real, scores_gen, fmaps_real, fmaps_gen