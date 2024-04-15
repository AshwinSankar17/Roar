from typing import List

import torch
import torch.nn.functional as F
from einops import rearrange

from roar.collections.asr.parts.preprocessing.features import FilterbankFeatures

from roar.core.classes import Loss, typecheck
from roar.core.neural_types import (
    AudioSignal,
    LengthsType,
    LossType,
    NeuralType,
    PredictionsType,
    RegressionValuesType,
)


class MaskedLoss(Loss):
    def __init__(self, loss_fn, loss_scale: float = 1.0):
        super(MaskedLoss, self).__init__()
        self.loss_scale = loss_scale
        self.loss_fn = loss_fn

    @property
    def input_types(self):
        return {
            "predicted": NeuralType(("B", "D", "T"), PredictionsType()),
            "target": NeuralType(("B", "D", "T"), RegressionValuesType()),
            "target_len": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, predicted, target, target_len):
        assert target.shape[2] == predicted.shape[2]

        # [B, D, T]
        loss = self.loss_fn(input=predicted, target=target)
        # [B, T]
        loss = torch.mean(loss, dim=1)
        # [B]
        loss = torch.sum(loss, dim=1) / torch.clamp(target_len, min=1.0)

        # [1]
        loss = torch.mean(loss)
        loss = self.loss_scale * loss

        return loss


class MaskedMAELoss(MaskedLoss):
    def __init__(self, loss_scale: float = 1.0):
        loss_fn = torch.nn.L1Loss(reduction="none")
        super(MaskedMAELoss, self).__init__(loss_fn=loss_fn, loss_scale=loss_scale)


class MaskedMSELoss(MaskedLoss):
    def __init__(self, loss_scale: float = 1.0):
        loss_fn = torch.nn.MSELoss(reduction="none")
        super(MaskedMSELoss, self).__init__(loss_fn=loss_fn, loss_scale=loss_scale)


class TimeDomainLoss(Loss):
    def __init__(self):
        super(TimeDomainLoss, self).__init__()
        self.loss_fn = MaskedMAELoss()

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(("B", "T"), AudioSignal()),
            "audio_gen": NeuralType(("B", "T"), AudioSignal()),
            "audio_len": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, audio_real, audio_gen, audio_len):
        audio_real = rearrange(audio_real, "B T -> B 1 T")
        audio_gen = rearrange(audio_gen, "B T -> B 1 T")
        loss = self.loss_fn(
            target=audio_real, predicted=audio_gen, target_len=audio_len
        )
        return loss


class MultiResolutionMelLoss(Loss):
    """
    Multi-resolution log mel spectrogram loss.

    Args:
        sample_rate: Sample rate of audio.
        resolutions: List of resolutions, each being 3 integers ordered [num_fft, hop_length, window_length]
        mel_dims: Dimension of mel spectrogram to compute for each resolution. Should be same length as 'resolutions'.
        log_guard: Value to add to mel spectrogram to avoid taking log of 0.
    """

    def __init__(
        self,
        sample_rate: int,
        resolutions: List[List],
        mel_dims: List[int],
        scale_factor: float = 1.0,
        log_guard: float = 1.0,
    ):
        super(MultiResolutionMelLoss, self).__init__()
        assert len(resolutions) == len(mel_dims)

        self.l1_loss_fn = MaskedMAELoss(loss_scale=scale_factor)
        self.l2_loss_fn = MaskedMSELoss(loss_scale=scale_factor)

        self.mel_features = torch.nn.ModuleList()
        for mel_dim, (n_fft, hop_len, win_len) in zip(mel_dims, resolutions):
            mel_feature = FilterbankFeatures(
                sample_rate=sample_rate,
                nfilt=mel_dim,
                n_window_size=win_len,
                n_window_stride=hop_len,
                n_fft=n_fft,
                pad_to=1,
                mag_power=1.0,
                log_zero_guard_type="add",
                log_zero_guard_value=log_guard,
                mel_norm=None,
                normalize=None,
                preemph=None,
                dither=0.0,
                use_grads=True,
            )
            self.mel_features.append(mel_feature)

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(("B", "T"), AudioSignal()),
            "audio_gen": NeuralType(("B", "T"), AudioSignal()),
            "audio_len": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "l1_loss": NeuralType(elements_type=LossType()),
            "l2_loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, audio_real, audio_gen, audio_len):
        l1_loss = 0.0
        l2_loss = 0.0
        for mel_feature in self.mel_features:
            mel_real, mel_real_len = mel_feature(x=audio_real, seq_len=audio_len)
            mel_real.requires_grad = False
            mel_gen, _ = mel_feature(x=audio_gen, seq_len=audio_len)
            l1_loss += self.l1_loss_fn(
                predicted=mel_gen, target=mel_real, target_len=mel_real_len
            )
            l2_loss += self.l2_loss_fn(
                predicted=mel_gen, target=mel_real, target_len=mel_real_len
            )

        l1_loss /= len(self.mel_features)
        l2_loss /= len(self.mel_features)

        return l1_loss, l2_loss
