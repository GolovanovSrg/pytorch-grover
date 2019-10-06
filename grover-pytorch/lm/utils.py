import tensorflow as tf
import torch

from typing import Any, Dict, Collection, List


def load_tf_checkpoint(checkpoint_path: str, n_layers: int) -> Dict[str, torch.Tensor]:
    """
    Loading of tf checkpoint of Grover model as pytorch module state dict
    :param checkpoint_path: path of tf checkpoint
    :param n_layers: number of layers in checkpoint
    :return: pytorch module state dict
    """

    def get_weights(name: str, transpose: bool = False) -> torch.Tensor:
        weights = tf.train.load_variable(checkpoint_path, name)
        if transpose:
            weights = weights.T
        return torch.from_numpy(weights)

    def get_pos_embedding(name: str) -> torch.Tensor:
        weights = get_weights(name)
        weights_with_padding = torch.cat([torch.zeros(1, weights.shape[-1]), weights], dim=0)
        return weights_with_padding

    def get_qkv_proj(layer_id: int, weight_name: str) -> torch.Tensor:
        assert weight_name in ['kernel', 'bias']
        transpose = True if weight_name == 'kernel' else False
        layers = []
        for name in ['query', 'key', 'value']:
            layer = get_weights(f'newslm/layer{layer_id:02d}/{name}_layer/{weight_name}', transpose=transpose)
            layers.append(layer)
        return torch.cat(layers, dim=0)

    model_state = {'embedding.tok_embedding.weight': get_weights('newslm/embeddings/word_embed'),
                   'embedding.pos_embedding.weight': get_pos_embedding('newslm/embeddings/pos_embed'),
                   'embedding.emb_norm.weight': get_weights('newslm/embeddings/LayerNorm_embed_norm/gamma'),
                   'embedding.emb_norm.bias': get_weights('newslm/embeddings/LayerNorm_embed_norm/beta')}

    for layer_id in range(n_layers):
        layer_state = {f'layers.{layer_id}.attn.qkv_proj.weight': get_qkv_proj(layer_id, 'kernel'),
                       f'layers.{layer_id}.attn.qkv_proj.bias': get_qkv_proj(layer_id, 'bias'),
                       f'layers.{layer_id}.attn.out_proj.weight': get_weights(f'newslm/layer{layer_id:02d}/context_projection_layer/kernel', transpose=True),  # noqa
                       f'layers.{layer_id}.attn.out_proj.bias': get_weights(f'newslm/layer{layer_id:02d}/context_projection_layer/bias'),  # noqa
                       f'layers.{layer_id}.ff.layer_1.weight': get_weights(f'newslm/layer{layer_id:02d}/intermediate/kernel', transpose=True),  # noqa
                       f'layers.{layer_id}.ff.layer_1.bias': get_weights(f'newslm/layer{layer_id:02d}/intermediate/bias'),  # noqa
                       f'layers.{layer_id}.ff.layer_2.weight': get_weights(f'newslm/layer{layer_id:02d}/output/kernel', transpose=True),  # noqa
                       f'layers.{layer_id}.ff.layer_2.bias': get_weights(f'newslm/layer{layer_id:02d}/output/bias'),
                       f'layers.{layer_id}.ff.ff_norm_1.weight': get_weights(f'newslm/layer{layer_id:02d}/LayerNorm_mlp_ln0/gamma'),  # noqa
                       f'layers.{layer_id}.ff.ff_norm_1.bias': get_weights(f'newslm/layer{layer_id:02d}/LayerNorm_mlp_ln0/beta'),  # noqa
                       f'layers.{layer_id}.ff.ff_norm_2.weight': get_weights(f'newslm/layer{layer_id:02d}/LayerNorm_mlp_ln1/gamma'),  # noqa
                       f'layers.{layer_id}.ff.ff_norm_2.bias': get_weights(f'newslm/layer{layer_id:02d}/LayerNorm_mlp_ln1/beta')}  # noqa

        model_state.update(layer_state)

    return model_state


def pad_sequence(sequences: Collection[torch.Tensor], batch_first: bool = False, padding_value: int = 0,
                 left: bool = False) -> torch.Tensor:
    if not len(sequences):
        return torch.empty(0)

    trailing_dims = sequences[0].shape[1:]
    max_len = max([s.shape[0] for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        s_slice = slice(-length, None) if left else slice(None, length)
        b_slice = (i, s_slice) if batch_first else (s_slice, i)
        out_tensor[b_slice] = tensor

    return out_tensor


def chunks(sequences: Collection[Any], chunk_size: int) -> List[Collection[Any]]:
    return [sequences[i:i + chunk_size] for i in range(0, len(sequences), chunk_size)]


def set_seed(seed: int = 0) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
