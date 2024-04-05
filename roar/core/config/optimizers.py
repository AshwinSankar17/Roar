from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple

from omegaconf import MISSING, OmegaConf

__all__ = [
    'OptimizerParams',
    'SGDParams',
    'AdamParams',
    'AdamWParams',
    'AdadeltaParams',
    'AdamaxParams',
    'AdagradParams',
    'RpropParams',
    'RMSpropParams',
]


@dataclass
class OptimizerParams:
    """
    Base Optimizer params with no values. User can chose it to explicitly override via
    command line arguments
    """

    lr: Optional[float] = MISSING


@dataclass
class SGDParams(OptimizerParams):
    """
    Default configuration for Adam optimizer.
    """

    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False


@dataclass
class AdamParams(OptimizerParams):
    """
    Default configuration for Adam optimizer.
    """

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0
    amsgrad: bool = False


@dataclass
class AdamWParams(OptimizerParams):
    """
    Default configuration for AdamW optimizer.
    """

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0
    amsgrad: bool = False


@dataclass
class AdadeltaParams(OptimizerParams):
    """
    Default configuration for Adadelta optimizer.
    """

    rho: float = 0.9
    eps: float = 1e-6
    weight_decay: float = 0


@dataclass
class AdamaxParams(OptimizerParams):
    """
    Default configuration for Adamax optimizer.
    """

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0


@dataclass
class AdagradParams(OptimizerParams):
    """
    Default configuration for Adagrad optimizer.
    """

    lr_decay: float = 0
    weight_decay: float = 0
    initial_accumulator_value: float = 0
    eps: float = 1e-10


@dataclass
class RMSpropParams(OptimizerParams):
    """
    Default configuration for RMSprop optimizer.
    """

    alpha: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0
    momentum: float = 0
    centered: bool = False


@dataclass
class RpropParams(OptimizerParams):
    """
    Default configuration for RpropParams optimizer.
    """

    etas: Tuple[float, float] = (0.5, 1.2)
    step_sizes: Tuple[float, float] = (1e-6, 50)

def register_optimizer_params(name: str, optimizer_params: OptimizerParams):
    """
    Checks if the optimizer param name exists in the registry, and if it doesnt, adds it.

    This allows custom optimizer params to be added and called by name during instantiation.

    Args:
        name: Name of the optimizer. Will be used as key to retrieve the optimizer.
        optimizer_params: Optimizer class
    """
    if name in AVAILABLE_OPTIMIZER_PARAMS:
        raise ValueError(f"Cannot override pre-existing optimizers. Conflicting optimizer name = {name}")

    AVAILABLE_OPTIMIZER_PARAMS[name] = optimizer_params


def get_optimizer_config(name: str, **kwargs: Optional[Dict[str, Any]]) -> OptimizerParams:
    """
    Convenience method to obtain a OptimizerParams class and partially instantiate it with optimizer kwargs.

    Args:
        name: Name of the OptimizerParams in the registry.
        kwargs: Optional kwargs of the optimizer used during instantiation.

    Returns:
        a partially instantiated OptimizerParams
    """
    if name is None:
        return kwargs

    if name not in AVAILABLE_OPTIMIZER_PARAMS:
        raise ValueError(
            f"Cannot resolve optimizer parameters '{name}'. Available optimizer parameters are : "
            f"{AVAILABLE_OPTIMIZER_PARAMS.keys()}"
        )

    scheduler_params = AVAILABLE_OPTIMIZER_PARAMS[name]

    if kwargs is not None and len(kwargs) != 0:
        kwargs = OmegaConf.create(kwargs)
        OmegaConf.merge(scheduler_params(), kwargs)

    scheduler_params = partial(scheduler_params, **kwargs)
    return scheduler_params


AVAILABLE_OPTIMIZER_PARAMS = {
    'optim_params': OptimizerParams,
    'adam_params': AdamParams,
    'sgd_params': SGDParams,
    'adadelta_params': AdadeltaParams,
    'adamax_params': AdamaxParams,
    'adagrad_params': AdagradParams,
    'adamw_params': AdamWParams,
    'rmsprop_params': RMSpropParams,
    'rprop_params': RpropParams,
}
