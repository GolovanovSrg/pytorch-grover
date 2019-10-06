import copy
import json
import math
import six

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from typing import Any, Dict, List, Optional, Tuple, Union


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    from torch.nn import LayerNorm


class GroverConfig:
    """Configuration for `GroverModel`"""

    def __init__(self,
                 vocab_size: int,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 hidden_act: str = 'gelu',
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 max_position_embeddings: int = 512,
                 initializer_range: float = 0.02) -> None:
        """Constructs NewsConfig.

        Args:
          vocab_size: Vocabulary size of  context_ids` in `GroverModel`.
          hidden_size: Size of the layers
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.pad_token_id = 0

    @classmethod
    def from_dict(cls, json_object: Dict[str, Any]) -> 'GroverConfig':
        """Constructs a `NewsConfig` from a Python dictionary of parameters."""
        config = GroverConfig(vocab_size=-1)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file: str) -> 'GroverConfig':
        """Constructs a `NewsConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class CombinedEmbedding(nn.Module):
    """
    Combined embedding layer, include token embeddings and positional embeddings.
    """

    def __init__(self, n_embeddings: int, n_pos_embeddings: int, embedding_dim: int,
                 padding_idx: Optional[int] = None, adapter_mode: bool = False) -> None:
        """
        :param n_embeddings: size of dictionary of token embeddings (include padding token)
        :param n_pos_embeddings: size of dictionary of positional embeddings
        :param embedding_dim: size of each embedding vector
        :param padding_idx: if given, pads output with embedding vector at
                            padding_idx (initialized to zeros) whenever it encounters index
        :param adapter_mode: if True disable gradients
        """

        super().__init__()

        self.tok_padding_idx = padding_idx
        self.pos_padding_idx = 0
        self.n_embeddings = n_embeddings
        self.adapter_mode = adapter_mode

        self.tok_embedding = nn.Embedding(n_embeddings, embedding_dim, padding_idx=self.tok_padding_idx)
        self.pos_embedding = nn.Embedding(n_pos_embeddings + 1, embedding_dim, padding_idx=self.pos_padding_idx)
        self.emb_norm = LayerNorm(embedding_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        # TODO: tf.truncated_normal_initializer
        nn.init.normal_(self.tok_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def _embedding(self, x: torch.Tensor, add_length: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padding_mask = x.eq(self.tok_padding_idx)

        positions = torch.cumsum(~padding_mask, dim=-1, dtype=torch.long)
        if add_length is not None:
            positions = positions + add_length
        positions.masked_fill_(padding_mask, self.pos_padding_idx)
        lens = positions[:, -1].unsqueeze(-1)

        x = self.tok_embedding(x) + self.pos_embedding(positions)
        x = self.emb_norm(x)

        return x, padding_mask, lens

    def forward(self, x: torch.Tensor, add_length: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input tensor with indexes to extract, shape=[batch_size, sequence_length]
        :param add_length: lengths for positional embedding to add for current part of sequences, shape=[batch_size]
        :return: output tensor with normalized embeddings with shape=[batch_size, sequence_length, embedding_dim],
                 padding mask with shape=[batch_size, sequence_length] and sequences lengths with shape=[batch_size, 1]
        """

        assert x.dim() == 2

        if self.adapter_mode:
            with torch.no_grad():
                x, padding_mask, lens = self._embedding(x, add_length)
        else:
            x, padding_mask, lens = self._embedding(x, add_length)

        return x, padding_mask, lens


LayerStateType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class Adapter(nn.Module):
    """
    https://arxiv.org/pdf/1902.00751.pdf
    """

    def __init__(self, in_features: int, middle_features: int) -> None:
        super().__init__()

        self.layer_1 = nn.Linear(in_features, middle_features)
        self.layer_2 = nn.Linear(middle_features, in_features)

        self._init_weights()

    def _init_weights(self) -> None:
        # TODO: tf.truncated_normal_initializer
        nn.init.normal_(self.layer_1.weight, std=0.02)
        nn.init.normal_(self.layer_2.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_1(x)
        x = gelu(x)
        x = self.layer_2(x)
        x = residual + x

        return x


class MultiheadAttention(nn.Module):
    """
    Multihead self-attention layer
    """

    @classmethod
    def _get_future_mask(cls, size: Tuple[int, int], device: torch.device) -> torch.Tensor:
        nd, ns = size
        max_size = max(nd, ns)
        if not hasattr(cls, '_future_mask') or \
                cls._future_mask.device != device or \
                any(s < max_size for s in cls._future_mask.shape):
            cls._future_mask = torch.triu(torch.ones(max_size, max_size, dtype=torch.uint8, device=device,
                                                     requires_grad=False), 1).bool()

        # future mask when we already may have past pre-computed values: take a slice at the end of the mask
        mask = cls._future_mask[ns-nd:ns, :ns]

        return mask

    def __init__(self, n_features: int, n_heads: int, dropout: float, adapter_mode: bool = False) -> None:
        """
        :param n_features: size of each input and output sample
        :param n_heads: number of heads
        :param dropout: dropout probability
        :param adapter_mode: if True use adapter (for fine-tuning)
        """

        super().__init__()

        assert n_features % n_heads == 0

        self.n_features = n_features
        self.n_heads = n_heads
        self.qkv_proj = nn.Linear(n_features, 3 * n_features)
        self.out_proj = nn.Linear(n_features, n_features)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.adapter_mode = adapter_mode
        if adapter_mode:
            self.adapter = Adapter(n_features, n_features // 16)

        self._init_weights()

    def _init_weights(self) -> None:
        # TODO: tf.truncated_normal_initializer
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def _split_heads(self, x: torch.Tensor, is_key: bool = False) -> torch.Tensor:
        x = x.view(x.shape[0], x.shape[1], self.n_heads, self.n_features // self.n_heads)
        x = x.permute(0, 2, 3, 1) if is_key else x.permute(0, 2, 1, 3)

        return x

    def _calc_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   padding_mask: torch.Tensor) -> torch.Tensor:
        w = torch.matmul(q, k) / math.sqrt(self.n_features // self.n_heads)

        future_mask = MultiheadAttention._get_future_mask(w.shape[-2:], w.device).unsqueeze(0).unsqueeze(0)
        w.masked_fill_(future_mask, float('-inf'))

        if padding_mask is not None:
            w.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        mask = (w == float('-inf')).all(dim=-1)
        w = F.softmax(w, dim=-1)
        if mask.any().item():
            w = w.masked_fill(mask.unsqueeze(-1), 0)
        out = torch.matmul(w, v)

        return out

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[0], x.shape[1], self.n_features)

        return x

    def _attn(self, x: torch.Tensor, padding_mask: torch.Tensor, past: Optional[LayerStateType] = None,
              return_past: bool = True) -> Tuple[torch.Tensor, LayerStateType]:
        query, key, value = self.qkv_proj(x).split(self.n_features, dim=-1)

        if past is not None:
            past_key, past_value, past_padding_mask = past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
            padding_mask = torch.cat((past_padding_mask, padding_mask), dim=-1)

        # we can reuse: key/value/padding_mask for next forward steps, query for next attention ops
        saved = (key, value, padding_mask) if return_past else None

        query = self._split_heads(query)
        key = self._split_heads(key, is_key=True)
        value = self._split_heads(value)

        if self.training and not self.adapter_mode:
            x = checkpoint(self._calc_attn, query, key, value, padding_mask)
        else:
            x = self._calc_attn(query, key, value, padding_mask)

        x = self._merge_heads(x)
        x = self.out_proj(x)
        x = self.dropout(x)

        return x, saved

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, past: Optional[LayerStateType] = None,
                return_past: bool = True) -> Tuple[torch.Tensor, LayerStateType]:
        """
        :param x: input tensor, shape=[batch_size, sequence_length, embedding_dim]
        :param padding_mask: padding mask for input (1 is padded position), shape=[batch_size, sequence_length]
        :param past: cache for layer (speed up decoding), tuple with cached keys, values and padding mask
        :param return_past: if True return cache for layer
        :return: output tensor with shape=[batch_size, sequence_length, embedding_dim] and cache for layer
        """

        residual = x

        if self.adapter_mode:
            with torch.no_grad():
                x, saved = self._attn(x, padding_mask, past, return_past)
            x = self.adapter(x)
        else:
            x, saved = self._attn(x, padding_mask, past, return_past)

        x = residual + x

        return x, saved


def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation: https://arxiv.org/abs/1606.08415
    """
    sqrt_two = 1.4142135623730951
    cdf = (x / sqrt_two).erf_().add_(1.0).mul_(0.5)
    return x * cdf


class FeedForward(nn.Module):
    """
    Feed forward layer with residual connection
    """

    def __init__(self, in_features: int, middle_features: int, dropout: float, adapter_mode: bool = False) -> None:
        """
        :param in_features: size of each input and output sample
        :param middle_features: middle size of each sample
        :param dropout: dropout probability
        :param adapter_mode: if True use adapter (for fine-tuning)
        """

        super().__init__()

        self.layer_1 = nn.Linear(in_features, middle_features)
        self.layer_2 = nn.Linear(middle_features, in_features)
        self.ff_norm_1 = LayerNorm(in_features)
        self.ff_norm_2 = LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.adapter_mode = adapter_mode
        if adapter_mode:
            self.adapter = Adapter(in_features, in_features // 16)

        self._init_weights()

    def _init_weights(self) -> None:
        # TODO: tf.truncated_normal_initializer
        nn.init.normal_(self.layer_1.weight, std=0.02)
        nn.init.normal_(self.layer_2.weight, std=0.02)

    def _ff(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_1(x)
        if self.training and not self.adapter_mode:
            x = checkpoint(gelu, x)
        else:
            x = gelu(x)
        x = self.layer_2(x)
        x = self.dropout(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor, shape=[batch_size, sequence_length, embedding_dim]
        :return: output tensor, shape=[batch_size, sequence_length, embedding_dim]
        """

        residual = x
        x = self.ff_norm_1(x)

        if self.adapter_mode:
            with torch.no_grad():
                x = self._ff(x)
            x = self.adapter(x)
        else:
            x = self._ff(x)

        x = residual + x
        x = self.ff_norm_2(x)

        return x


class TransformerBlock(nn.Module):
    """
    Transformer layer (attention layer and feed forward layer)
    """

    def __init__(self, n_attn_features: int, n_heads: int, n_ff_features: int,
                 attn_dropout: float = 0, ff_dropout: float = 0, adapter_mode: bool = False) -> None:
        """

        :param n_attn_features: size of each sample in attention layer
        :param n_heads: number of heads in attention layer
        :param n_ff_features: middle size of each sample in feed forward layer
        :param attn_dropout: dropout probability in attention layer
        :param ff_dropout: dropout probability in feed forward layer
        :param adapter_mode: if True use adapters (for fine-tuning)
        """

        super().__init__()

        self.attn = MultiheadAttention(n_attn_features, n_heads, attn_dropout, adapter_mode)
        self.ff = FeedForward(n_attn_features, n_ff_features, ff_dropout, adapter_mode)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, layer_past: Optional[LayerStateType] = None,
                return_layer_past: bool = True) -> Tuple[torch.Tensor, LayerStateType]:
        """
        :param x: input tensor, shape=[batch_size, sequence_length, embedding_dim]
        :param padding_mask: padding mask for input (1 is padded position), shape=[batch_size, sequence_length]
        :param layer_past: cache for layer (speed up decoding), tuple with cached keys, values and padding mask
        :param return_layer_past: if True return cache for layer
        :return: output tensor, shape=[batch_size, sequence_length, embedding_dim]
        """

        x, saved_kv = self.attn(x, padding_mask, layer_past, return_layer_past)
        x = self.ff(x)

        return x, saved_kv


ModelStateType = Tuple[torch.Tensor, List[LayerStateType]]


class GroverModel(nn.Module):
    """
    Grover model (decoder only transformer)
    """

    def __init__(self, n_layers: int, n_embeddings: int, n_pos_embeddings: int, embedding_dim: int, n_heads: int,
                 n_ff_features: int, padding_idx: int, attn_dropout: float = 0, ff_dropout: float = 0,
                 adapter_mode: bool = False) -> None:
        """
        :param n_layers: number of layers in model
        :param n_embeddings: size of dictionary of token embeddings (include padding token)
        :param n_pos_embeddings: size of dictionary of positional embeddings
        :param embedding_dim: size of each embedding vector
        :param n_heads: number of heads in attention layers
        :param n_ff_features: middle size of each sample in feed forward layers
        :param padding_idx: if given, pads output with embedding vector at
                            padding_idx (initialized to zeros) whenever it encounters index
        :param attn_dropout: dropout probability in attention layers
        :param ff_dropout: dropout probability in feed forward layers
        :param adapter_mode: if True use adapters (for fine-tuning)
        """

        super().__init__()

        self.adapter_mode = adapter_mode

        self.embedding = CombinedEmbedding(n_embeddings=n_embeddings,
                                           n_pos_embeddings=n_pos_embeddings,
                                           embedding_dim=embedding_dim,
                                           padding_idx=padding_idx)

        base_block = TransformerBlock(n_attn_features=embedding_dim,
                                      n_heads=n_heads,
                                      n_ff_features=n_ff_features,
                                      attn_dropout=attn_dropout,
                                      ff_dropout=ff_dropout,
                                      adapter_mode=adapter_mode)
        self.layers = nn.ModuleList([copy.deepcopy(base_block) for _ in range(n_layers)])

    def adapter_named_parameters(self):
        assert self.adapter_mode, 'Adapters are not enabled in the model'

        params = []
        for layer in self.layers:
            params.extend(layer.attn.adapter.named_parameters())
            params.extend(layer.ff.adapter.named_parameters())
            params.extend(layer.ff.ff_norm_1.named_parameters())
            params.extend(layer.ff.ff_norm_2.named_parameters())

        return params

    def forward(self, x: torch.Tensor, past: Optional[ModelStateType] = None, return_past: bool = True) -> \
            Tuple[torch.Tensor, torch.Tensor, ModelStateType]:
        """
        :param x: input tensor with indexes (context for each generation, maybe padded),
                  shape=[batch_size, sequence_length]
        :param past: cache for model (speed up decoding), tuple with past lengths of sequences
                     and list of caches for each layer
        :param return_past: if True return cache for model
        :return: tuple of output tensor with logits for each position in sequences,
                 padding mask and cache for next decoding step
        """

        past_length: Optional[torch.Tensor]
        past_layers: Union[List[None], List[LayerStateType]]

        if past is None:
            past_length, past_layers = None, [None] * len(self.layers)
        else:
            past_length, past_layers = past

        x, padding_mask, lens = self.embedding(x, past_length)

        saved_layers = []
        for layer, layer_past in zip(self.layers, past_layers):
            x, saved = layer(x, padding_mask, layer_past=layer_past, return_layer_past=return_past)
            saved_layers.append(saved)

        x = F.linear(x, self.embedding.tok_embedding.weight)
        state = (lens, saved_layers) if return_past else None

        return x, padding_mask, state


def sample_sequences(model: GroverModel, contexts: torch.Tensor, bos_id: int, eos_id: int, max_seq_len: int,
                     ignore_ids: Optional[torch.Tensor] = None, temperature: float = 1, top_p: float = 1) -> \
                         List[torch.Tensor]:
    """
    Nucleus sampling for Grover model
    :param model: Grover model
    :param contexts: context (indexes) for each generation, maybe padded
    :param bos_id: begin of generation index
    :param eos_id: end of generation index
    :param max_seq_len: max number of tokens in sequence (include context and generation)
    :param ignore_ids: indexes which will not sampled
    :param temperature: temperature parameter for output logits
    :param top_p: top probability parameter for nucleus sampling
    :return: list of generations
    """

    assert 0 < top_p <= 1
    model.eval()

    with torch.no_grad():
        sequences = [torch.full((contexts.shape[0], 1), fill_value=bos_id, dtype=torch.long, device=contexts.device)]
        lens = torch.ones(contexts.shape[0], dtype=torch.long, device=contexts.device)
        is_end = torch.zeros(contexts.shape[0], dtype=torch.uint8, device=contexts.device)
        _, _, state = model(contexts)

        while contexts.shape[1] + len(sequences) <= max_seq_len:
            logits, _, state = model(sequences[-1], past=state)
            logits = logits.squeeze(1) / temperature

            if ignore_ids is not None:
                d = torch.zeros_like(logits)
                d.index_fill_(1, ignore_ids, -1e10)
                logits.add_(d)

            sorted_logits, sorted_indices = torch.sort(logits, dim=1, descending=True)
            cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=1), dim=1)
            mask_to_remove = cumulative_probabilities > top_p
            mask_to_remove[:, 1:] = mask_to_remove[:, :-1].clone()
            mask_to_remove[:, 0] = 0
            sorted_logits.add_(mask_to_remove.float() * -1e10)
            logits.scatter_(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=1)
            next_ids = torch.multinomial(probs, num_samples=1)
            sequences.append(next_ids)

            lens[~is_end] += 1
            is_end[next_ids.view(-1) == eos_id] = 1

            if all(is_end):
                break

        sequences = torch.cat(sequences, dim=1)
        sequences = [s[:l] for s, l in zip(sequences, lens)]

        return sequences


def _length_penalty(sequence_lengths: torch.Tensor, length_penalty_coef: float) -> torch.Tensor:
    """
    https://arxiv.org/abs/1609.08144
    """
    return (5 + sequence_lengths) ** length_penalty_coef / (5 + 1) ** length_penalty_coef


def _fix_state(state: ModelStateType, beam_idxs: torch.Tensor, beam_size: int) -> ModelStateType:
    def fix_tensor(t: torch.Tensor) -> torch.Tensor:
        n_dims = t.dim()
        if n_dims == 2:
            t = t.unsqueeze(-1)

        t_size = t.shape
        tile_size = t_size[-2] * t_size[-1]
        new_t = t.contiguous().view(-1, beam_size, tile_size)
        new_t = new_t.gather(1, beam_idxs.unsqueeze(-1).repeat([1, 1, tile_size]))
        new_t = new_t.view(*t_size)

        if n_dims == 2:
            new_t = new_t.squeeze(-1)

        return new_t

    past_length, past_layers = state

    past_length[...] = fix_tensor(past_length)
    for past_layer in past_layers:
        for t in past_layer:
            t[...] = fix_tensor(t)

    return past_length, past_layers


def beamsearch_sequences(model: GroverModel, contexts: torch.Tensor, bos_id: int, eos_id: int, beam_size: int,
                         vocab_size: int, length_penalty_coef: float = 0.8, diversity_coef: float = 0,
                         diversity_groups: int = 1, return_beams: bool = False, max_seq_len: int = 1024,
                         ignore_ids: Optional[torch.Tensor] = None) -> \
                            Union[List[torch.Tensor], List[List[torch.Tensor]]]:
    """
    Diverse Beam Search: https://arxiv.org/abs/1610.02424.
    Implementation of the Hamming Diversity penalty, which performed best in the original paper.
    It is classical beam-search if diversity_groups = 1.

    :param model: Grover model
    :param contexts: context (indexes) for each generation, maybe padded
    :param bos_id: begin of generation index
    :param eos_id: end of generation index
    :param beam_size: number of generation in beam
    :param vocab_size: size of vocabulary
    :param length_penalty_coef: length penalty coefficient
    :param diversity_coef: diversity penalty coefficient
    :param diversity_groups: number of diversity groups
    :param return_beams: return all generations in beam or not
    :param max_seq_len: max number of tokens in sequence (include context and generation)
    :param ignore_ids: indexes which will not sampled
    :return: list of generations or list of beams if return_beams = True
    """

    assert beam_size % diversity_groups == 0
    model.eval()

    with torch.no_grad():
        batch_size = contexts.shape[0]
        contexts_len = contexts.shape[1]
        group_size = beam_size // diversity_groups
        device = contexts.device

        sequences = torch.full((batch_size, beam_size, 1), fill_value=bos_id, dtype=torch.long, device=device)
        beam_scores = torch.zeros(batch_size, beam_size, device=device)
        lens = torch.ones(batch_size, beam_size, dtype=torch.long, device=device)
        is_end = torch.zeros(batch_size, beam_size, dtype=torch.uint8, device=device)
        diversity_penalty = torch.zeros((batch_size, vocab_size), device=device)

        beam_contexts = contexts.unsqueeze(1).repeat((1, beam_size, 1)).view(batch_size * beam_size, -1)
        _, _, state = model(beam_contexts)
        while contexts_len + sequences.shape[-1] <= max_seq_len:
            logits, _, state = model(sequences[..., -1].view(-1, 1), past=state)
            logits = logits.squeeze(1)

            if ignore_ids is not None:
                logits.index_fill_(1, ignore_ids, -1e10)  # -1e10 instead of -inf for prevent Nan's

            log_probs = F.log_softmax(logits, dim=-1).view(batch_size, beam_size, -1)
            end_marker = 1 - is_end.float()
            beam_scores = beam_scores.unsqueeze(-1) + log_probs * end_marker.unsqueeze(-1)
            penalty = _length_penalty(lens.float() + end_marker, length_penalty_coef).unsqueeze(-1)
            beam_scores = beam_scores / penalty

            if sequences.shape[-1] == 1:  # first beams
                penalty, beam_scores = penalty[:, 0, :], beam_scores[:, 0, :]
                beam_scores, idxs = beam_scores.topk(beam_size, dim=-1)
                beam_idxs = torch.zeros((batch_size, beam_size), dtype=torch.long, device=device)
            else:
                penalty = penalty.view(batch_size, diversity_groups, group_size, -1)
                beam_scores = beam_scores.view(batch_size, diversity_groups, group_size, -1)

                all_scores, all_idxs = [], []
                for g in range(diversity_groups):
                    g_beam_scores, g_penalty = beam_scores[:, g, :, :], penalty[:, g, :, :]
                    g_beam_scores -= diversity_coef * diversity_penalty.unsqueeze(1) / g_penalty
                    g_beam_scores = g_beam_scores.view(batch_size, -1)

                    g_scores, g_idxs = g_beam_scores.topk(group_size, dim=-1)
                    g_idxs += g * group_size * vocab_size

                    all_scores.append(g_scores)
                    all_idxs.append(g_idxs)

                    g_next_ids = torch.fmod(g_idxs, vocab_size)
                    diversity_penalty.scatter_add_(1, g_next_ids, torch.ones((batch_size, group_size), device=device))

                diversity_penalty.fill_(0)
                penalty = penalty.view(batch_size, -1)
                beam_scores = torch.cat(all_scores, dim=-1)
                idxs = torch.cat(all_idxs, dim=-1)

                beam_idxs = (idxs.float() / vocab_size).long()

            next_ids = torch.fmod(idxs, vocab_size)
            is_end = torch.gather(is_end, 1, beam_idxs)
            lens = torch.gather(lens, 1, beam_idxs)
            lens[~is_end] += 1
            is_end[next_ids == eos_id] = 1

            sequences = torch.gather(sequences, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, sequences.shape[-1]))
            sequences = torch.cat([sequences, next_ids.view(batch_size, beam_size, 1)], dim=-1)

            state = _fix_state(state, beam_idxs, beam_size)

            if all(is_end.view(-1)):
                break

            beam_scores *= penalty

        sequences = [[sequences[batch_i, beam_i, :lens[batch_i, beam_i]] for beam_i in range(beam_size)]
                     for batch_i in range(batch_size)]

        if return_beams:
            return sequences

        bests = beam_scores.argmax(dim=-1)
        return [sequences[batch_i][bests[batch_i]] for batch_i in range(batch_size)]


def get_grover_model(config: GroverConfig, adapter_mode: bool = False) -> GroverModel:
    """
    Create Grover model by config
    :param adapter_mode: if True use adapters (for fine-tuning)
    :param config: config for model
    :return: Grover model
    """

    model = GroverModel(n_layers=config.num_hidden_layers,
                        n_embeddings=config.vocab_size,
                        n_pos_embeddings=config.max_position_embeddings,
                        embedding_dim=config.hidden_size,
                        n_heads=config.num_attention_heads,
                        n_ff_features=config.intermediate_size,
                        padding_idx=config.pad_token_id,
                        attn_dropout=config.hidden_dropout_prob,
                        ff_dropout=config.hidden_dropout_prob,
                        adapter_mode=adapter_mode)

    return model
