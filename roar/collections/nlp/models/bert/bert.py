import os
from typing import Optional, Dict

import torch
import torch.nn as nn
from einops import rearrange

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer


from roar.core.classes import NeuralModule, typecheck
from roar.core.neural_types.elements import (
    EncodedRepresentation,
    LengthsType,
    MaskType,
    TokenIndex,
)
from roar.core.neural_types.neural_type import NeuralType

from roar.collections.nlp.losses import FusedCrossEntropyLoss
from roar.collections.tts.parts.utils.helpers import (
    get_mask_from_lengths,
    build_rope_cache,
)
from roar.collections.tts.parts.utils.utils_funcs import find_multiple
from roar.collections.tts.parts.utils.bert_padding import (
    pad_input,
    unpad_input,
    index_put_first_axis,
)

from roar.collections.common.metrics import Perplexity
from roar.collections.nlp.data.language_modeling.lm_bert_dataset import (
    BertPretrainingDataset,
    BertPretrainingPreprocessedDataloader,
)
from roar.collections.nlp.modules.bert import BertModel, BertPretrainingTokenClassifier
from roar.collections.nlp.modules.common.tokenizer_utils import get_tokenizer

from roar.core.classes.common import PretrainedModelInfo
from roar.core.classes.modelPT import ModelPT
from roar.utils import logging

__all__ = ["BERTLMModel"]


class BERTLMModel(ModelPT):
    """
    BERT language model pretraining.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        output_types_dict = {
            "mlm_logits": self.mlm_classifier.output_types["log_probs"]
        }
        if not self.only_mlm_loss:
            output_types_dict["nsp_logits"] = self.nsp_classifier.output_types["logits"]
        return output_types_dict

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # vocab_file = None
        config_dict = None
        # config_file = None

        if cfg.tokenizer is not None:
            if cfg.tokenizer.get("tokenizer_name") and cfg.tokenizer.get(
                "tokenizer_model"
            ):
                self._setup_tokenizer(cfg.tokenizer)
            if cfg.get("tokenizer.vocab_file"):
                self.register_artifact("tokenizer.vocab_file", cfg.tokenizer.vocab_file)
                # vocab_file = self.register_artifact(
                #     "tokenizer.vocab_file", cfg.tokenizer.vocab_file
                # )
        else:
            self.tokenizer = None

        super().__init__(cfg=cfg, trainer=trainer)

        if cfg.get("language_model.config"):
            config_dict = OmegaConf.to_container(
                cfg.language_model.config, resolve=True
            )
        if cfg.get("language_model.config_file"):
            self.register_artifact(
                "language_model.config_file", cfg.language_model.config_file
            )
            # config_file = self.register_artifact(
            #     "language_model.config_file", cfg.language_model.config_file
            # )

        self.padding_multiple = cfg.get("padding_multiple", 64)
        self.vocab_size = find_multiple(
            self.tokenizer.vocab_size, self.padding_multiple
        )

        self.word_vocab_size = find_multiple(cfg.word_vocab_size, self.padding_multiple)

        self.bert_model = BertModel(**config_dict, vocab_size=self.vocab_size)

        self.hidden_size = self.bert_model.config.hidden_size

        self.mlm_classifier = BertPretrainingTokenClassifier(
            hidden_size=self.hidden_size,
            n_vocab=self.vocab_size,
        )

        self.grapheme_classifier = BertPretrainingTokenClassifier(
            hidden_size=self.hidden_size,
            n_vocab=self.target_vocab_size,
        )

        self.mlm_loss = FusedCrossEntropyLoss()

        # # tie weights of MLM softmax layer and embedding layer of the encoder
        if (
            self.mlm_classifier.decoder.w3.weight
            != self.bert_model.embeddings.word_embeddings.weight.shape
        ):
            raise ValueError(
                "Final classification layer does not match embedding layer."
            )
        self.mlm_classifier.decoder.w3.weight = (
            self.bert_model.embeddings.word_embeddings.weight
        )
        # create extra bias

        # setup to track metrics
        self.validation_perplexity = Perplexity()

        self.setup_optimization(cfg.optim)

    @typecheck()
    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        hidden_states = self.bert_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        mlm_logits = self.mlm_classifier(hidden_states=hidden_states)
        if self.only_mlm_loss:
            return (mlm_logits,)
        nsp_logits = self.nsp_classifier(hidden_states=hidden_states)
        return mlm_logits, nsp_logits

    def _compute_losses(self, mlm_logits, nsp_logits, output_ids, output_mask, labels):
        mlm_loss = self.mlm_loss(
            log_probs=mlm_logits, labels=output_ids, output_mask=output_mask
        )
        if self.only_mlm_loss:
            loss, nsp_loss = mlm_loss, None
        else:
            nsp_loss = self.nsp_loss(logits=nsp_logits, labels=labels)
            loss = self.agg_loss(loss_1=mlm_loss, loss_2=nsp_loss)
        return mlm_loss, nsp_loss, loss

    def _parse_forward_outputs(self, forward_outputs):
        if self.only_mlm_loss:
            return forward_outputs[0], None
        else:
            return forward_outputs

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, output_ids, output_mask, labels = batch
        forward_outputs = self.forward(
            input_ids=input_ids,
            token_type_ids=input_type_ids,
            attention_mask=input_mask,
        )
        mlm_logits, nsp_logits = self._parse_forward_outputs(forward_outputs)
        _, _, loss = self._compute_losses(
            mlm_logits, nsp_logits, output_ids, output_mask, labels
        )
        lr = self._optimizer.param_groups[0]["lr"]
        self.log("train_loss", loss)
        self.log("lr", lr, prog_bar=True)
        return {"loss": loss, "lr": lr}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, output_ids, output_mask, labels = batch
        forward_outputs = self.forward(
            input_ids=input_ids,
            token_type_ids=input_type_ids,
            attention_mask=input_mask,
        )
        mlm_logits, nsp_logits = self._parse_forward_outputs(forward_outputs)
        _, _, loss = self._compute_losses(
            mlm_logits, nsp_logits, output_ids, output_mask, labels
        )
        self.validation_perplexity(logits=mlm_logits)
        loss = {"val_loss": loss}
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        """Called at the end of validation to aggregate outputs.

        Args:
            outputs (list): The individual outputs of each validation step.

        Returns:
            dict: Validation loss and tensorboard logs.
        """
        if self.validation_step_outputs:
            avg_loss = torch.stack(
                [x["val_loss"] for x in self.validation_step_outputs]
            ).mean()
            perplexity = self.validation_perplexity.compute()
            logging.info(f"evaluation perplexity {perplexity.cpu().item()}")
            self.log("val_loss", avg_loss)
            self.validation_step_outputs.clear()  # free memory

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = (
            self._setup_preprocessed_dataloader(train_data_config)
            if self.tokenizer is None
            else self._setup_dataloader(train_data_config)
        )

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = (
            self._setup_preprocessed_dataloader(val_data_config)
            if self.tokenizer is None
            else self._setup_dataloader(val_data_config)
        )

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        pass

    def _setup_preprocessed_dataloader(self, cfg: Optional[DictConfig]):
        dataset = cfg.data_file
        max_predictions_per_seq = cfg.max_predictions_per_seq
        batch_size = cfg.batch_size

        if os.path.isdir(dataset):
            files = [
                os.path.join(dataset, f)
                for f in os.listdir(dataset)
                if os.path.isfile(os.path.join(dataset, f))
            ]
        else:
            files = [dataset]
        files.sort()
        dl = BertPretrainingPreprocessedDataloader(
            data_files=files,
            max_predictions_per_seq=max_predictions_per_seq,
            batch_size=batch_size,
        )
        return dl

    def _setup_tokenizer(self, cfg: DictConfig):
        tokenizer = get_tokenizer(
            tokenizer_name=cfg.tokenizer_name,
            tokenizer_model=cfg.tokenizer_model,
            special_tokens=OmegaConf.to_container(cfg.special_tokens)
            if cfg.special_tokens
            else None,
            vocab_file=cfg.vocab_file,
        )
        self.tokenizer = tokenizer

    def _setup_dataloader(self, cfg: DictConfig):
        dataset = BertPretrainingDataset(
            tokenizer=self.tokenizer,
            data_file=cfg.data_file,
            max_seq_length=cfg.max_seq_length,
            mask_prob=cfg.mask_prob,
            short_seq_prob=cfg.short_seq_prob,
        )
        dl = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            collate_fn=dataset.collate_fn,
            drop_last=cfg.get("drop_last", False),
            shuffle=cfg.shuffle,
            num_workers=cfg.get("num_workers", 0),
        )
        return dl

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []

        result.append(
            PretrainedModelInfo(
                pretrained_model_name="bertbaseuncased",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/roar/bertbaseuncased/versions/1.0.0rc1/files/bertbaseuncased.roar",
                description="The model was trained EN Wikipedia and BookCorpus on a sequence length of 512.",
            )
        )

        result.append(
            PretrainedModelInfo(
                pretrained_model_name="bertlargeuncased",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/roar/bertlargeuncased/versions/1.0.0rc1/files/bertlargeuncased.roar",
                description="The model was trained EN Wikipedia and BookCorpus on a sequence length of 512.",
            )
        )
        return result
