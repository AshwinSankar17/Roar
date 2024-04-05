import torch
import torch.nn as nn

from roar.core.classes import NeuralModule, adapter_mixins, typecheck
from roar.core.neural_types.neural_type import NeuralType
from einops import rearrange
