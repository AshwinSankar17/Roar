import torch
import torch.nn as nn


class Swish(torch.nn.SiLU):
    """
    Swish activation function introduced in 'https://arxiv.org/abs/1710.05941'
    Mathematically identical to SiLU. See note in nn.SiLU for references.
    """


@torch.jit.script
def snake(
    x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, eps: float = 1e-9
) -> torch.Tensor:
    """
    equation for snake activation function: x + (alpha + eps)^-1 * sin(alpha * x)^2
    """
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (beta + eps).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake(nn.Module):
    """
    Snake activation function introduced in 'https://arxiv.org/abs/2006.08195'
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha, self.alpha)


class SnakeBeta(nn.Module):
    """
    Adapted from Amphion--FACodec
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(1, 256)
        >>> x = a1(x)
    """

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(1, in_features, 1) * alpha)
            self.beta = nn.Parameter(torch.zeros(1, in_features, 1) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(1, in_features, 1) * alpha)
            self.beta = nn.Parameter(torch.ones(1, in_features, 1) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta := x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha
        beta = self.beta
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = snake(x, alpha=self.alpha, beta=self.beta)

        return x
