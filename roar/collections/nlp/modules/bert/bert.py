from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from roar.collections.tts.modules.submodules import (
    ConditionalInput,
    ConditionalLayerNorm,
    LinearNorm,
)
from roar.collections.tts.modules.attention import (
    MultiHeadAttn,
    MultiHeadAttnFlash,
    BiDirectionalLLaMaSelfAttention,
)
from roar.collections.tts.modules.postional_embedding import PositionalEmbedding
from roar.collections.tts.parts.utils.helpers import (
    get_mask_from_lengths,
    build_rope_cache,
)
from roar.collections.tts.parts.utils.bert_padding import unpad_input, pad_input
from roar.core.neural_types.elements import (
    EncodedRepresentation,
    LengthsType,
    MaskType,
    TokenIndex,
)
from roar.core.neural_types.neural_type import NeuralType


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, padding_idx, dropout_prob=0.0):
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=padding_idx
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, tokens):
        out = self.word_embeddings(tokens)
        out = self.layer_norm(out)
        out = self.dropout(out)
        return out


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(
        self, hidden_states: torch.Tensor, pool: Optional[bool] = True
    ) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] if pool else hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertEncoder(nn.Encoder):
    def __init__(
        self,
        vocab_size,
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size,
        dropout,
        dropatt,
        n_embed=None,
        d_embed=None,
        padding_idx=0,
        rope_base=10_000,
        rope_condense_ratio=1.0,
        rotary_percentage=0.25,
        n_query_groups=None,
        condition_types=[],
        pre_lnorm=True,
        add_pooling_layer=True,
    ):
        super(BertEncoder, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.padding_idx = padding_idx
        self.word_emb = BertEmbeddings(
            vocab_size,
            d_model or d_embed,
            padding_idx=self.padding_idx,
            dropout_prob=dropout,
        )
        self.pooler = BertPooler(d_model) if add_pooling_layer else None

        if n_query_groups is None:
            n_query_groups = n_head

        self.rope_cache = None
        self.rope_base = rope_base
        self.rope_condense_ratio = rope_condense_ratio
        self.rotary_percentage = rotary_percentage

        self.layers = nn.ModuleList()
        self.cond_input = ConditionalInput(d_model, d_model, condition_types)

        for _ in range(n_layer):
            self.layers.append(
                BiDirectionalLLaMaSelfAttention(
                    n_head,
                    d_model,
                    d_head,
                    d_inner,
                    kernel_size,
                    dropout,
                    dropatt=dropatt,
                    pre_lnorm=pre_lnorm,
                    n_query_groups=n_query_groups,
                    condition_types=condition_types,
                )
            )

    @property
    def input_types(self):
        return {
            "input": NeuralType(("B", "T"), TokenIndex()),
            "conditioning": NeuralType(
                ("B", "T", "D"), EncodedRepresentation(), optional=True
            ),
        }

    def forward(self, input, masked_tokens_mask=None, conditioning=0):
        word_embeddings = self.word_emb(input)
        attention_mask = (input != self.padding_idx).unsqueeze(2)
        subset_mask = []
        first_col_mask = []

        if masked_tokens_mask is None:
            subset_mask = None
        else:
            first_col_mask = torch.zeros_like(masked_tokens_mask)
            first_col_mask[:, 0] = True
            subset_mask = masked_tokens_mask | first_col_mask

        encoder_outputs = self.encoder(
            word_embeddings,
            attention_mask,
            subset_mask=subset_mask,
        )

        if masked_tokens_mask is None:
            sequence_output = encoder_outputs
            pooled_output = (
                self.pooler(sequence_output) if self.pooler is not None else None
            )
        else:
            # TD [2022-03-01]: the indexing here is very tricky.
            subset_idx = subset_mask[attention_mask]  # type: ignore
            sequence_output = encoder_outputs[
                masked_tokens_mask[attention_mask][subset_idx]
            ]
            if self.pooler is not None:
                pool_input = encoder_outputs[first_col_mask[attention_mask][subset_idx]]
                pooled_output = self.pooler(pool_input, pool=False)
            else:
                pooled_output = None

        encoder_outputs = sequence_output

        if self.pooler is not None:
            return encoder_outputs, pooled_output

        return encoder_outputs, None


class BertPredictionHeadTransform(nn.Module):
    def __init__(
        self,
        hidden_size,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size, bert_model_embedding_weights):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1), bert_model_embedding_weights.size(0)
        )
        self.decoder.weight = bert_model_embedding_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, hidden_size, bert_model_embedding_weights):
        super().__init__()
        self.predictions = BertLMPredictionHead(
            hidden_size, bert_model_embedding_weights
        )

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
