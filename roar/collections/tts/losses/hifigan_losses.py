import torch
import torch.nn.functional as F

from roar.core.classes import Loss, typecheck
from roar.core.neural_types.elements import LossType, VoidType, MelSpectrogramType
from roar.core.neural_types.neural_type import NeuralType


class FeatureMatchingLoss(Loss):
    """Feature Matching Loss module"""

    @property
    def input_types(self):
        return {
            "fmap_r": [[NeuralType(elements_type=VoidType())]],
            "fmap_g": [[NeuralType(elements_type=VoidType())]],
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss * 2


class DiscriminatorLoss(Loss):
    """Discriminator Loss module"""

    @property
    def input_types(self):
        return {
            "disc_real_outputs": [NeuralType(("B", "T"), VoidType())],
            "disc_generated_outputs": [NeuralType(("B", "T"), VoidType())],
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
            "real_losses": [NeuralType(elements_type=LossType())],
            "fake_losses": [NeuralType(elements_type=LossType())],
        }

    @typecheck()
    def forward(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg**2)
            loss += r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses


class GeneratorLoss(Loss):
    """Generator Loss module"""

    @property
    def input_types(self):
        return {
            "disc_outputs": [NeuralType(("B", "T"), VoidType())],
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
            "fake_losses": [NeuralType(elements_type=LossType())],
        }

    @typecheck()
    def forward(self, disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses


class MelLoss(Loss):
    @property
    def input_types(self):
        return {
            "spect_predicted": NeuralType(("B", "D", "T"), MelSpectrogramType()),
            "spect_tgt": NeuralType(("B", "D", "T"), MelSpectrogramType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, spect_predicted, spect_tgt):
        spect_tgt.requires_grad = False
        spect_tgt = spect_tgt.transpose(1, 2)  # (B, T, H)
        spect_predicted = spect_predicted.transpose(1, 2)  # (B, T, H)

        ldiff = spect_tgt.size(1) - spect_predicted.size(1)
        spect_predicted = F.pad(spect_predicted, (0, 0, 0, ldiff, 0, 0), value=0.0)
        mel_mask = spect_tgt.ne(0).float()
        loss_fn = F.l1_loss
        mel_loss = loss_fn(spect_predicted, spect_tgt, reduction="none")
        mel_loss = (mel_loss * mel_mask).sum() / mel_mask.sum()

        return mel_loss
