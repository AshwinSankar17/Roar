from dataclasses import dataclass
from typing import Optional

__all__ = ['Config']


@dataclass
class Config:
    """
    Abstract Configuration class.

    Args:
        name: name of the module/dataset/loss/model object (used in serialization, DEFAULT: None)
    """

    name: Optional[str] = None
