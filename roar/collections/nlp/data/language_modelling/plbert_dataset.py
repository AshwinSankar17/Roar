import array
import os
import pickle
import random
from typing import Dict, List, Optional

import h5py
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from tqdm import tqdm

from roar.core.classes import Dataset


__all__ = ["PLBertPretrainingDataset", "PLBertPretrainingPreprocessedDataloader"]


def load_h5(input_file: str):
    return h5py.File(input_file, "r")


class PLBertPretrainingDataset(Dataset):
    """
    Dataset for bert pretraining when using data preprocessing including tokenization
    """

    def __init__(
        self,
        tokenizer: object,
        data_file: str,
        max_seq_length: Optional[int] = 128,
        mask_prob: Optional[float] = 0.15,
        short_seq_prob: Optional[float] = 0.1,
        seq_a_ratio: Optional[float] = 0.6,
        sentence_idx_file: Optional[str] = None,
    ):
        """
        Args:
            tokenizer: tokenizer
            data_file: path to data
            max_seq_length: maximum sequence length of input tensors
            mask_probability: proability to mask token
            short_seq_prob: probability to create a sequence shorter than max_seq_length
            seq_a_ratio: ratio between lengths of first and second sequence
            sentence_idx_file: sentence indices file for caching
        """
