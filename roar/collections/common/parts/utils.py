import math
import os
from typing import Iterable, List

import torch.nn as nn

__all__ = ["if_exist", "_compute_softmax", "flatten"]

activation_registry = {
    "identity": nn.Identity,
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


def if_exist(outfold: str, files: List[str]):
    """
    Returns true if all given files exist in the given folder
    Args:
        outfold: folder path
        files: list of file names relative to outfold
    """
    if not os.path.exists(outfold):
        return False
    for file in files:
        if not os.path.exists(f"{outfold}/{file}"):
            return False
    return True


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def flatten_iterable(iter: Iterable) -> Iterable:
    """Flatten an iterable which contains values or
    iterables with values.

    Args:
        iter: iterable containing values at the deepest level.

    Returns:
        A flat iterable containing values.
    """
    for it in iter:
        if isinstance(it, str) or not isinstance(it, Iterable):
            yield it
        else:
            yield from flatten_iterable(it)


def flatten(list_in: List) -> List:
    """Flatten a list of (nested lists of) values into a flat list.

    Args:
        list_in: list of values, possibly nested

    Returns:
        A flat list of values.
    """
    return list(flatten_iterable(list_in))
