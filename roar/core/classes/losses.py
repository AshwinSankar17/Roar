import torch

from roar.core.classes.common import Serialization, Typing

__all__ = ['Loss']


class Loss(torch.nn.modules.loss._Loss, Typing, Serialization):
    """Inherit this class to implement custom loss."""

    def __init__(self, **kwargs):
        super(Loss, self).__init__(**kwargs)
