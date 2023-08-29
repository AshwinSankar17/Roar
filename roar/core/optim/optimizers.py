import copy
from functools import partial
from typing import Dict, Optional, Union, Any

import hydra
import torch
from torch import optim
from omegaconf import DictConfig, OmegaConf
from torch.optim import adadelta, adagrad, adamax, rmsprop, rprop
from torch.optim.optimizer import Optimizer

from roar.core.config import OptimizerParams, get_optimizer_config, register_optimizer_params
from roar.utils import logging
from roar.utils.model_utils import maybe_update_config_version

AVAILABLE_OPTIMIZERS = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "adadelata": adadelta.Adadelta,
    'adadelta': adadelta.Adadelta,
    'adamax': adamax.Adamax,
    'adagrad': adagrad.Adagrad,
    'rmsprop': rmsprop.RMSprop,
    'rprop': rprop.Rprop,
}

__all__ = ["get_optimizer", "register_optimizer", "parse_optimizer_args"]

def parse_optimizer_args(
    optimizer_name: str, optimizer_kwargs: Union[DictConfig, Dict[str, Any]]
) -> Union[Dict[str, Any], DictConfig]:
    
    kwargs = {}
    
    if not optimizer_kwargs:
        return kwargs

    optimizer_kwargs = copy.deepcopy(optimizer_kwargs)
    optimizer_kwargs = maybe_update_config_version(optimizer_kwargs)
    
    if isinstance(optimizer_kwargs, DictConfig):
        optimizer_kwargs = OmegaConf.to_container(optimizer_kwargs, resolve=True)
    
    # If it is a dictionary, perform stepwise resolution
    if hasattr(optimizer_kwargs, "keys"):
        # Attempt class path resolution
        if "_target_" in optimizer_kwargs: # captures (target, _target_) pair
            optimizer_kwargs_config = OmegaConf.create(optimizer_kwargs)
            optimizer_instance = hydra.utils.instantiate(optimizer_kwargs_config)
            optimizer_instance = vars(optimizer_instance)
            return optimizer_instance
    
        # If class path was not provided, perhaps `name` was provided
        if "name" in optimizer_kwargs:
            # If auto is passed as the name for resolution then lookup optimizer name and resolve its parameter config
            if optimizer_kwargs["name"] == "auto":
                optimizer_params_name = f"{optimizer_name}_params"
                optimizer_kwargs.pop("name")
            else:
                optimizer_params_name = optimizer_kwargs.pop("name")
                
            # Override arguments with those provided in the config yml file
            optimizer_kwargs = optimizer_kwargs.get("params", optimizer_kwargs) # If optimizer kwarg overrides are wrapped in yaml params
            
            if isinstance(optimizer_params_override, DictConfig):
                optimizer_params_override = OmegaConf.to_container(optimizer_params_override, resolve=True)
            
            optimizer_params_cls = get_optimizer_config(optimizer_params_name, **optimizer_params_override)

            # If we are provided just a Config object, simply return the dictionary of that object
            if optimizer_params_name is None:
                optimizer_params = vars(optimizer_params_cls)
                return optimizer_params

            else:
                # If we are provided a partial class instantiation of a Config,
                # Instantiate it and retrieve its vars as a dictionary
                optimizer_params = optimizer_params_cls()  # instantiate the parameters object
                optimizer_params = vars(optimizer_params)
                return optimizer_params

        # simply return the dictionary that was provided
        return optimizer_kwargs

    return kwargs


def register_optimizer(name: str, optimizer: Optimizer, optimizer_params: OptimizerParams):
    """
    Checks if the optimizer name exists in the registry, and if it doesnt, adds it.

    This allows custom optimizers to be added and called by name during instantiation.

    Args:
        name: Name of the optimizer. Will be used as key to retrieve the optimizer.
        optimizer: Optimizer class
        optimizer_params: The parameters as a dataclass of the optimizer
    """
    if name in AVAILABLE_OPTIMIZERS:
        raise ValueError(f"Cannot override pre-existing optimizers. Conflicting optimizer name = {name}")

    AVAILABLE_OPTIMIZERS[name] = optimizer

    optim_name = "{}_params".format(optimizer.__name__)
    register_optimizer_params(name=optim_name, optimizer_params=optimizer_params)


def get_optimizer(name: str, **kwargs: Optional[Dict[str, Any]]) -> Optimizer:
    """
    Convenience method to obtain an Optimizer class and partially instantiate it with optimizer kwargs.

    Args:
        name: Name of the Optimizer in the registry.
        kwargs: Optional kwargs of the optimizer used during instantiation.

    Returns:
        a partially instantiated Optimizer
    """
    if name not in AVAILABLE_OPTIMIZERS:
        raise ValueError(
            f"Cannot resolve optimizer '{name}'. Available optimizers are : " f"{AVAILABLE_OPTIMIZERS.keys()}"
        )
    if name == 'fused_adam':
        if not torch.cuda.is_available():
            raise ValueError(f'CUDA must be available to use fused_adam.')

    optimizer = AVAILABLE_OPTIMIZERS[name]
    optimizer = partial(optimizer, **kwargs)
    return optimizer
