from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import MISSING

__all__ = ['DataLoaderConfig']


@dataclass
class DataLoaderConfig:
    """
    Configuration of PyTorch DataLoader.
    """

    batch_size: int = MISSING
    shuffle: bool = False
    sampler: Optional[Any] = None
    batch_sampler: Optional[Any] = None
    num_workers: int = 0
    collate_fn: Optional[Any] = None
    pin_memory: bool = False
    drop_last: bool = False
    timeout: int = 0
    worker_init_fn: Optional[Any] = None
    multiprocessing_context: Optional[Any] = None
