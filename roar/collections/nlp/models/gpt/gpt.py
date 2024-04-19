from typing import Optional, Dict, List

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import warnings
from lightning_utilities.core.imports import RequirementCache

from roar.collections.nlp.modules.common.text_generation import (
    TextGeneration,
    LengthParam,
    OutputType,
    SamplingParam,
)
from roar.collections.nlp.modules.gpt import GPT
from roar.collections.nlp.losses import FusedCrossEntropyLoss
from roar.collections.nlp.parts.utils import utils_funcs
from roar.collections.nlp.parts.utils.helpers import (
    find_multiple,
    chunked_cross_entropy,
)

from roar.core.neural_types import ChannelType, NeuralType, Index
from roar.core.classes.common import PretrainedModelInfo
from roar.core.classes import Exportable


from roar.utils import logging, model_utils


_APEX_AVAILABLE = RequirementCache("apex")


class GPTExportableModel(torch.nn.Module, Exportable):
    """
    GPT Wrapper for ONNX export
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.dtype = utils_funcs.torch_dtype_from_precision(model._cfg.precision)

    def forward(self, tokens, max_seq_length, input_pos):
        with torch.no_grad(), torch.inference_mode(), torch.autocast(
            "cuda", dtype=self.dtype
        ), warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", category=torch.jit.TracerWarning, module=r".*"
            )
            # assert tokens.shape == position_ids.shape
            # assert (
            #     attention_mask.shape[2]
            #     == attention_mask.shape[3]
            #     == tokens.shape[1]
            #     == position_ids.shape[1]
            # )
            output_tensor = self.model.forward(
                idx=tokens.cuda(),
                max_seq_length=max_seq_length,
                input_pos=input_pos.cuda() if input_pos else None,
            )
        return output_tensor

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    # def input_example(self, max_batch=1, max_dim=768, seq_len=6):
    #     ids = [
    #         self.model.tokenizer.text_to_ids(text)
    #         for text in ["how is the weather on           Sunday"]
    #     ]
    #     id_tensors = [
    #         torch.unsqueeze(torch.LongTensor(id_list), dim=0) for id_list in ids
    #     ]
    #     masks_and_position_ids = [
    #         get_ltor_masks_and_position_ids(
    #             id_tensor, self.model.tokenizer.eos_id, False, False, False
    #         )
    #         for id_tensor in id_tensors
    #     ]
    #     for tokens, attn_mask_and_pos_ids in zip(id_tensors, masks_and_position_ids):
    #         attn_mask, _, pos_ids = attn_mask_and_pos_ids
    #         return tokens, pos_ids, attn_mask

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(("B", "T"), ChannelType()),
            "max_seq_length": Index(),
            "input_pos": NeuralType(("B", "T"), ChannelType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"logits": NeuralType(("B", "T", "D"), ChannelType())}

    @property
    def input_names(self) -> List[str]:
        return ["input_ids", "position_ids", "attention_mask"]

    @property
    def output_names(self) -> List[str]:
        return ["logits"]


class GPTModel(TextGeneration):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        super().__init__(cfg=cfg, trainer=trainer)
        self.ds_class = cfg.train_ds.dataset._target_
        self.ds_class_name = self.ds_class.split(".")[-1]
        assert self.ds_class in [
            "roar.collections.nlp.data.dataset.TextGenerationDataset",
            "lightning.data.CombinedStreamingDataset",
        ], f"Unknown dataset class: {self.ds_class}"

        self._setup_tokenizer(cfg)
        assert self.text_tokenizer is not None
        if self._cfg.padded_vocab_size is None:
            self._cfg.padded_vocab_size = find_multiple(
                self.text_tokenizer.vocab_size, self.padding_multiple
            )
        assert (
            self._cfg.n_embd % self._cfg.n_head == 0
        ), "Make sure that your embedding size is divisible by number of attention heads"
        if self._cfg.n_query_groups is not None:
            assert self._cfg.n_head % self._cfg.n_query_groups == 0
        else:
            self._cfg.n_query_groups = self._cfg.n_head
        # gpt_module_kwargs = {
        #     "n_embed": (self.text_tokenizer.vocab_size),
        #     "padding_idx": self.text_tokenizer.pad,
        # }

        self._parser = None
        self.loss_scale = cfg.get("loss_scale", 1.0)

        self.ce_loss_fn = FusedCrossEntropyLoss()
        self.gpt = GPT(self._cfg)

    def _setup_tokenizer(self, cfg: DictConfig):
        self.text_tokenizer = instantiate(cfg.text_tokenizer)

    @property
    def parser(self):
        if self._parser:
            return self._parser
        else:
            self._parser = self.text_tokenizer.encode
            return self._parser
