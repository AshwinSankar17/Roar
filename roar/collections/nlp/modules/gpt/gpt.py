import math
from typing import Optional, Any, Tuple
from typing_extensions import Self

import torch
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from xformers.ops import SwiGLU

from roar.collections.nlp.modules.gpt.config import Config
from roar.core.classes import NeuralModule
from roar.collections.nlp.parts.submodules.llm import KVCache
from roar.collections.nlp.parts.submodules.normalization import RMSNorm, FusedRMSNorm
from roar.collections.nlp.parts.submodules.positional_encodings import (
    apply_rotary_emb_func,
)

RoPECache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")
APEX_AVAILABLE = RequirementCache("apex")


def build_mask_cache(max_seq_length: int, device: torch.device) -> torch.Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), dtype=torch.bool, device=device)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
    condense_ratio: int = 1,
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # added by peiyuan to ensure same data type with q, k, to use fused rotary embedding
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    # ptl takes care of type casting
    # this is to mimic the behaviour of complex32, else we will get different results
    if cos.dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


class GPT(NeuralModule):
    def __init__(
        self,
        padded_vocab_size: int,
        n_embd: int,
        n_layers: int,
        block_size: int = 4096,
        rope_base: int = 10000,
        n_head: int = 8,
        rope_condense_ratio: float = 1.0,
        rotary_percentage: int = 0.25,
        norm_class: nn.Module = FusedRMSNorm,
        norm_eps: float = 1e-5,
        lm_head_bias: bool = False,
        scale_embeddings: bool = False,
        config: Config = None,
    ) -> None:
        super().__init__()
        assert padded_vocab_size is not None

        self.padded_vocab_size = padded_vocab_size
        self.n_embd = n_embd
        self.n_layers = n_layers
        self.block_size = block_size
        self.norm_eps = norm_eps
        self.scale_embeddings = scale_embeddings
        self.rope_base = rope_base
        self.n_head = n_head
        self.rope_condense_ratio = rope_condense_ratio
        self.head_size = self.n_embd // self.n_head
        self.rotary_percentage = rotary_percentage
        norm_class = self._maybe_bump_to_fused_norm(norm_class)
        # self.lm_head_bias = lm_head_bias

        self.config = config

        self.lm_head = nn.Linear(self.n_embd, padded_vocab_size, bias=lm_head_bias)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(padded_vocab_size, self.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(self.n_layers)),
                ln_f=norm_class(self.n_embd, eps=self.norm_eps),
            )
        )

        # self.max_seq_length = self.config.block_size
        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None

    def _maybe_bump_to_fused_norm(self, norm_class: nn.Module) -> nn.Module:
        if APEX_AVAILABLE:
            from apex.normalization import FusedLayerNorm

            return {"LayerNorm": FusedLayerNorm}.get(norm_class.__name__, norm_class)
        return norm_class

    def _init_weights(self, module: nn.Module, n_layer: Optional[int] = None) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.n_embd)
            )
            # RWKV: set it to 1e-4
            # torch.nn.init.uniform_(module.weight,  -1e-4, 1e-4)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.n_embd)
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # GPT-NeoX
        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, LLaMAMLP)) or (
                name == "w3.weight"
                and isinstance(module, SwiGLU)
                or (name == "proj.weight" and isinstance(module, CausalSelfAttention))
            ):  # if use xformer swiglu, fc2 layer will be renamed to w3
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.n_embd) / n_layer)

    def forward(
        self,
        idx: torch.Tensor,
        max_seq_length: Optional[int],
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        T = idx.size(1)
        use_kv_cache = input_pos is not None
        block_size = self.block_size

        if max_seq_length is None:
            max_seq_length = block_size

        if use_kv_cache:
            assert (
                max_seq_length >= T
            ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}."

        assert (
            max_seq_length <= block_size
        ), f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert (
            block_size >= T
        ), f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)

        cos, sin = self.rope_cache
        if use_kv_cache:  # use the kv cache
            cos = cos.index_select(0, input_pos)
            sin = sin.index_select(0, input_pos)
            if self.mask_cache is None:
                self.mask_cache = self.build_mask_cache(idx)
                # raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
            self.set_kv_cache(idx, max_seq_length, cos.size(-1) * 2)
        else:
            cos = cos[:T]
            sin = sin[:T]
            mask = None

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if self.scale_embeddings:
            x = x * (self.n_embd**0.5)

        for block in self.transformer.h:
            x = block(x, (cos, sin), mask, input_pos)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)  # (b, t, vocab_size)

    def build_rope_cache(self, idx: "torch.tensor") -> RoPECache:
        return build_rope_cache(
            seq_len=self.block_size,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=idx.dtype,
            condense_ratio=self.config.rope_condense_ratio,
            device=idx.device,
            base=self.config.rope_base,
        )

    def build_mask_cache(self, idx: "torch.tensor") -> "torch.tensor":
        return build_mask_cache(self.block_size, device=idx.device)

    def set_kv_cache(
        self,
        idx: "torch.tensor",
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        batch_size = idx.size(0)
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)
        # max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size,
                max_seq_length,
                rope_cache_length,
                idx.device,
                dtype,
            )

        # if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
        #     # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
        #     # for the kv-cache support (only during inference), we only create it in that situation
        #     # self.mask_cache = build_mask_cache(max_seq_length, device)
        #     self.mask_cache = build_mask_cache(max_seq_length)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
        self.proj = nn.Linear(
            config.head_size * config.n_head, config.n_embd, bias=config.bias
        )
        # disabled by default
        self.kv_cache: Optional[KVCache] = None

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(
            B, T, self.config.n_query_groups, total_qkv, self.config.head_size
        )
        # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        # if self.config.n_query_groups != self.config.n_head and (
        #     input_pos is None or self.config.n_query_groups != 1
        # ):
        #     k = k.expand(
        #         B, self.config.n_query_groups, q_per_kv, T, self.config.head_size
        #     )
        #     v = v.expand(
        #         B, self.config.n_query_groups, q_per_kv, T, self.config.head_size
        #     )

        q = q.reshape(B, T, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B, T, -1, self.config.head_size)  # (B, T, nh_k, hs)
        v = v.reshape(B, T, -1, self.config.head_size)  # (B, T, nh_v, hs)

        cos, sin = rope

        q = apply_rotary_emb_func(q, cos, sin, False, False)
        k = apply_rotary_emb_func(k, cos, sin, False, False)
        # q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        # k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        y = self.scaled_dot_product_attention(q, k, v, mask)

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.head_size)

        if (
            FlashAttention2Available
            and mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            from flash_attn import flash_attn_func

            return flash_attn_func(
                q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True
            )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
        v_shape = (batch_size, max_seq_length, heads, self.config.head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError(
                    "Please pass the `rope_cache_length=gpt.cos.size(-1)` value"
                )
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                max_seq_length,
                heads,
                rope_cache_length
                + self.config.head_size
                - int(self.config.rotary_percentage * self.config.head_size),
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.swiglu = SwiGLU(
            config.n_embd, config.intermediate_size, bias=False, _pack_weights=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.swiglu(x)


# class LLaMAMLP(nn.Module):
#     def __init__(self, config: Config) -> None:
#         super().__init__()
#         self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
#         self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
#         self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

#         self.config = config

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x_fc_1 = self.fc_1(x)
#         x_fc_2 = self.fc_2(x)
#         x = torch.nn.functional.silu(x_fc_1) * x_fc_2
#         return self.proj(x)


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = (
            None
            if config.shared_attention_norm
            else config.norm_class(config.n_embd, eps=config.norm_eps)
        )
        self.mlp = config.mlp_class(config)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Non-parallel residual       Parallel residual
            ┌─ x                     ┌─ x ────────────┐             Note: if `shared_attention_norm` is True,
            │  ↓                     │  ↓             ↓                   the output from `norm_1` is reused
            │  norm_1                │  norm_1  ───►  norm_2
            │  ↓                     │  ↓             ↓
            │  attn                  │  attn          mlp
            │  ↓                     │  ↓             │
        ┌───└► +                     └► + ◄───────────┘
        │     norm_2
        │     ↓
        │     mlp
        │     ↓
        └───► +
        """

        x_normed = self.norm_1(x)
        attention_output = self.attn(x_normed, rope, mask, input_pos)

        if self.config.parallel_residual:
            x_normed = x_normed if self.config.shared_attention_norm else self.norm_2(x)
            x = self.mlp(x_normed) + attention_output + x
        else:
            x = attention_output + x
            x = self.mlp(self.norm_2(x)) + x
        return x
