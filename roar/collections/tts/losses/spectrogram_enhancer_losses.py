import torch
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import grad as torch_grad

from roar.collections.tts.parts.utils.helpers import mask_sequence_tensor


class GradientPenaltyLoss(torch.nn.Module):
    """
    R1 loss from [1], used following [2]
    [1] Mescheder et. al. - Which Training Methods for GANs do actually Converge? 2018, https://arxiv.org/abs/1801.04406
    [2] Karras et. al. - A Style-Based Generator Architecture for Generative Adversarial Networks, 2018 (https://arxiv.org/abs/1812.04948)
    """

    def __init__(self, weight: float = 10.0):
        super().__init__()
        self.weight = weight

    def __call__(self, images, output):
        batch_size, *_ = images.shape
        gradients = torch_grad(
            outputs=output,
            inputs=images,
            grad_outputs=torch.ones(output.size(), device=images.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.reshape(batch_size, -1)
        return self.weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


class GeneratorLoss(torch.nn.Module):
    def __call__(self, fake_logits):
        return fake_logits.mean()


class HingeLoss(torch.nn.Module):
    def __call__(self, real_logits, fake_logits):
        return (F.relu(1 + real_logits) + F.relu(1 - fake_logits)).mean()


class ConsistencyLoss(torch.nn.Module):
    """
    Loss to keep SpectrogramEnhancer from generating extra sounds.
    L1 distance on x0.25 Mel scale (20 bins for typical 80-bin scale)
    """

    def __init__(self, weight: float = 10):
        super().__init__()
        self.weight = weight

    def __call__(self, condition, output, lengths):
        *_, w, h = condition.shape
        w, h = w // 4, h

        condition = F.interpolate(
            condition, size=(w, h), mode="bilinear", antialias=True
        )
        output = F.interpolate(output, size=(w, h), mode="bilinear", antialias=True)

        dist = (condition - output).abs()
        dist = mask_sequence_tensor(dist, lengths)
        return (dist / rearrange(lengths, "b -> b 1 1 1")).sum(
            dim=-1
        ).mean() * self.weight
