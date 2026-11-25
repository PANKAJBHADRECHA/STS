"""Load packed tensors without quantizing random dense model parameters."""
from pathlib import Path


def packed_layer(layer, weights, prefix, group_size=32):
    import mlx.core as mx
    import mlx.nn as nn
    if isinstance(layer, nn.Linear):
        cls = nn.QuantizedLinear
    elif isinstance(layer, nn.Embedding):
        cls = nn.QuantizedEmbedding
        if hasattr(layer, 'zero_idx'):
            if getattr(layer, 'low_rank', None) is not None or getattr(layer, 'demux_second_stream', False):
                raise ValueError('Low-rank/demultiplexed embeddings require a dedicated loader')

            class ZeroEmbedding(nn.QuantizedEmbedding):
                def __call__(self, inputs):
                    values = super().__call__(mx.maximum(inputs, 0))
                    return mx.where((inputs == -1)[..., None], mx.zeros_like(values), values)

            cls = ZeroEmbedding
    else:
        raise ValueError(f'Unsupported packed module: {prefix}')
    width = layer.weight.shape[-1]
    packed_width = weights[prefix + '.weight'].shape[-1] * 32
    if packed_width % width:
        raise ValueError(f'Invalid packed width for {prefix}')
    bits = packed_width // width
    if bits not in (2, 3, 4, 6, 8):
        raise ValueError(f'Unsupported bit width for {prefix}')
    packed = cls.__new__(cls)
    nn.Module.__init__(packed)
    packed.group_size, packed.bits, packed.mode = group_size, bits, 'affine'
    for suffix in ('weight', 'scales', 'biases', 'bias'):
        key = prefix + '.' + suffix
        if key in weights:
            setattr(packed, suffix, weights[key])
    packed.freeze()
    return packed


def load_packed_model(path):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from moshi_mlx import models
    weights = mx.load(str(Path(path)))
    config = models.config_v0_1()
    model = models.Lm(config)
    model.set_dtype(mx.bfloat16)
    names = dict(model.named_modules())
    replacements = []
    for key in weights:
        if key.endswith('.scales'):
            prefix = key.removesuffix('.scales')
            replacements.append((prefix, packed_layer(names[prefix], weights, prefix)))
    model.update_modules(tree_unflatten(replacements))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model, config
