# Adapted from https://huggingface.co/mosaicml/mosaic-bert-base/blob/main/bert_layers.py

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from xformers.ops import SwiGLU

from roar.collections.tts.modules.submodules import (
    ConditionalInput,
)

from roar.core.classes import NeuralModule, typecheck
from roar.core.neural_types.elements import (
    EncodedRepresentation,
    LengthsType,
    MaskType,
    TokenIndex,
)
from roar.core.neural_types.neural_type import NeuralType

from roar.collections.tts.modules.transformer import BiLLaMaLayer
from roar.collections.nlp.parts.submodules.normalization import FusedRMSNorm
from roar.collections.nlp.losses import FusedCrossEntropyLoss
from roar.collections.tts.parts.utils.helpers import (
    get_mask_from_lengths,
    build_rope_cache,
)
from roar.collections.tts.parts.utils.bert_padding import (
    pad_input,
    unpad_input,
    index_put_first_axis,
)


class BertEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        type_vocab_size,
        max_position_embeddings,
        padding_idx,
        dropout_prob=0.0,
    ):
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=padding_idx
        )
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = FusedRMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.register_buffer(
            "token_type_ids",
            torch.zeros(max_position_embeddings, dtype=torch.long),
            persistent=False,
        )

    def forward(self, input_ids, token_type_ids):
        B, T = input_ids.size()

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                assert isinstance(self.token_type_ids, torch.LongTensor)
                buffered_token_type_ids = self.token_type_ids[:, :T]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(B, T)
                token_type_ids = buffered_token_type_ids_expanded  # type: ignore
            else:
                token_type_ids = torch.zeros(
                    (B, T),  # type: ignore
                    dtype=torch.long,
                    device=self.word_embeddings.device,
                )
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        input_embeddings = self.word_embeddings(input_ids)
        embeddings = token_type_embeddings + input_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


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
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size,
        dropout,
        dropatt,
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
                BiLLaMaLayer(
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
            "hidden_states": NeuralType(("B", "T", "D"), TokenIndex()),
            "conditioning": NeuralType(
                ("B", "T", "D"), EncodedRepresentation(), optional=True
            ),
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        subset_mask: Optional[torch.Tensor] = None,
        output_all_encoded_layers: Optional[bool] = False,
        conditioning: Optional[torch.Tensor] = 0,
    ):
        B, T, C = hidden_states.size()

        if self.rope_cache is None or self.rope_cache[0].size(1) < T:
            self.rope_cache = self.build_rope_cache(hidden_states)

        attention_mask_bool = attention_mask.bool()
        hidden_states, indices, cu_seqlens, _ = unpad_input(
            hidden_states, attention_mask_bool
        )

        cos, sin = self.rope_cache
        cos, sin = cos[:T], sin[:T]

        all_encoder_layers = []
        if subset_mask is None:
            for layer_module in self.layers:
                hidden_states = layer_module(
                    hidden_states,
                    (cos, sin),
                    cu_seqlens,
                    T,
                    None,
                    indices,
                    attn_mask=attention_mask,
                    conditioning=conditioning,
                )
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            # Pad inputs and mask. It will insert back zero-padded tokens.
            # Assume ntokens is total number of tokens (padded and non-padded)
            # and ntokens_unpad is total number of non-padded tokens.
            # Then padding performs the following de-compression:
            #     hidden_states[ntokens_unpad,hidden] -> hidden_states[ntokens,hidden]
            hidden_states = pad_input(hidden_states, indices, B, T)
        else:
            for i in range(len(self.layers) - 1):
                layer_module = self.layers[i]
                hidden_states = layer_module(
                    hidden_states,
                    (cos, sin),
                    cu_seqlens,
                    T,
                    None,
                    indices,
                    attn_mask=attention_mask,
                    conditioning=conditioning,
                )
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            subset_idx = torch.nonzero(
                subset_mask[attention_mask_bool], as_tuple=False
            ).flatten()
            hidden_states = self.layers[-1](
                hidden_states,
                (cos, sin),
                cu_seqlens,
                T,
                subset_idx=subset_idx,
                indices=indices,
                attn_mask=attention_mask,
                conditioning=conditioning,
            )

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

    def build_rope_cache(self, inp: torch.Tensor):
        return build_rope_cache(
            seq_len=inp.size(1),
            n_elem=int(self.rotary_percentage * self.d_head),
            dtype=inp.dtype,
            device=inp.device,
            base=self.rope_base,
            condense_ratio=self.condense_ratio,
        )


class BertPredictionHeadTransform(nn.Module):
    def __init__(
        self,
        hidden_size,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = FusedRMSNorm(hidden_size, eps=1e-12)

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


class BertPretrainingTokenClassifier(nn.Module):
    def __init__(self, hidden_size, n_vocab):
        self.transform = BertPredictionHeadTransform(hidden_size)
        self.decoder = SwiGLU(hidden_size, n_vocab)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, hidden_size, bert_model_embedding_weights):
        super().__init__()
        self.predictions = BertPretrainingTokenClassifier(
            hidden_size, bert_model_embedding_weights
        )

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertModel(NeuralModule):
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
        d_embed=None,
        padding_idx=0,
        rope_base=10_000,
        rope_condense_ratio=1.0,
        rotary_percentage=0.25,
        n_query_groups=None,
        condition_types=[],
        pre_lnorm=True,
        add_pooling_layer=False,
    ):
        self.embeddings = BertEmbeddings(
            vocab_size,
            d_model or d_embed,
            padding_idx=padding_idx,
            dropout_prob=dropout,
        )
        self.pooler = BertPooler(d_model) if add_pooling_layer else None

        self.encoder = BertEncoder(
            n_layer,
            n_head,
            d_model,
            d_head,
            d_inner,
            kernel_size,
            dropout,
            dropatt,
            rope_base=rope_base,
            rope_condense_ratio=rope_condense_ratio,
            rotary_percentage=rotary_percentage,
            n_query_groups=n_query_groups,
            condition_types=condition_types,
            pre_lnorm=pre_lnorm,
            add_pooling_layer=add_pooling_layer,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_all_encoded_layers: bool = True,
        masked_tokens_mask: Optional[torch.Tensor] = None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embedding_output = self.embeddings(input_ids, token_type_ids)

        subset_mask = []
        first_col_mask = []

        if masked_tokens_mask is None:
            subset_mask = None
        else:
            first_col_mask = torch.zeros_like(masked_tokens_mask)
            first_col_mask[:, 0] = True
            subset_mask = masked_tokens_mask | first_col_mask

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            subset_mask=subset_mask,
        )

        if masked_tokens_mask is None:
            sequence_output = encoder_outputs[-1]
            pooled_output = (
                self.pooler(sequence_output) if self.pooler is not None else None
            )
        else:
            # TD [2022-03-01]: the indexing here is very tricky.
            attention_mask_bool = attention_mask.bool()
            subset_idx = subset_mask[attention_mask_bool]  # type: ignore
            sequence_output = encoder_outputs[-1][
                masked_tokens_mask[attention_mask_bool][subset_idx]
            ]
            if self.pooler is not None:
                pool_input = encoder_outputs[-1][
                    first_col_mask[attention_mask_bool][subset_idx]
                ]
                pooled_output = self.pooler(pool_input, pool=False)
            else:
                pooled_output = None

        if not output_all_encoded_layers:
            encoder_outputs = sequence_output

        if self.pooler is not None:
            return encoder_outputs, pooled_output

        return encoder_outputs, None


class BertForMaskedLM(NeuralModule):
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
        self.bert = BertModel(
            vocab_size,
            n_layer,
            n_head,
            d_model,
            d_head,
            d_inner,
            kernel_size,
            dropout,
            dropatt,
            d_embed=d_embed,
            padding_idx=padding_idx,
            rope_base=rope_base,
            rope_condense_ratio=rope_condense_ratio,
            rotary_percentage=rotary_percentage,
            n_query_groups=n_query_groups,
            condition_types=condition_types,
            pre_lnorm=pre_lnorm,
            add_pooling_layer=add_pooling_layer,
        )
        self.cls = BertOnlyMLMHead(d_model, self.bert.embeddings.word_embeddings.weight)

    def forward(
        self,
        inp,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = False,
        conditioning=0,
    ):
        if labels is None:
            masked_tokens_mask = None
        else:
            masked_tokens_mask = labels > 0

        outputs = self.bert(
            inp,
            attention_mask=attention_mask,
            masked_tokens_mask=masked_tokens_mask,
            conditioning=conditioning,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        loss = None
        if labels is not None:
            # Compute loss
            loss_fct = FusedCrossEntropyLoss()
            masked_token_idx = torch.nonzero(
                labels.flatten() > 0, as_tuple=False
            ).flatten()

            loss = loss_fct(prediction_scores, labels.flatten()[masked_token_idx])

            assert inp is not None, "Coding error; please open an issue"
            batch, seqlen = inp.shape[:2]
            prediction_scores = rearrange(
                index_put_first_axis(
                    prediction_scores, masked_token_idx, batch * seqlen
                ),
                "(b s) d -> b s d",
                b=batch,
            )
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "prediction_scores": prediction_scores,
            "hidden_states": None,
            "attentions": None,
        }
