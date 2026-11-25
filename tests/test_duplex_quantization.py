"""Verify checkpoint integrity and numeric behavior, without giant model loads."""
import importlib.util
import contextlib
import io
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('mlx'), 'Apple MLX environment required')
class PackedCheckpointTests(unittest.TestCase):
    def test_requantization_has_no_gaps_or_overlapping_tensors(self):
        import mlx.core as mx
        from safetensors import safe_open
        from scripts.requantize_moshi import convert
        for factor in (False, True):
            with self.subTest(factor=factor), tempfile.TemporaryDirectory() as tmp:
                source, target = Path(tmp)/'source.safetensors', Path(tmp)/'result.safetensors'
                values = mx.arange(32*64).reshape(32,64).astype(mx.bfloat16) / 1000
                prefix = 'transformer.layers.0.linear'
                if factor:
                    weights={prefix+'.weight':values}
                else:
                    q,s,b=mx.quantize(values, group_size=32, bits=4)
                    weights={prefix+'.biases':b,prefix+'.scales':s,prefix+'.weight':q}
                mx.save_safetensors(str(source), weights)
                with contextlib.redirect_stdout(io.StringIO()):
                    convert(source,target,4 if factor else 2,factor)
                with safe_open(target,framework='numpy') as checkpoint:
                    self.assertEqual(len(checkpoint.keys()),3)
                loaded=mx.load(str(target))
                dense=mx.dequantize(loaded[prefix+'.weight'],loaded[prefix+'.scales'],loaded[prefix+'.biases'],group_size=32,bits=4 if factor else 2)
                self.assertEqual(dense.shape, values.shape)
                self.assertLess(float(mx.max(mx.abs(dense-values)).item()),.04)

    def test_packed_linear_and_embedding_match_quantized_values(self):
        import mlx.core as mx
        import mlx.nn as nn
        from sts_lab.packed_moshi import packed_layer
        mx.random.seed(7)
        for layer in (nn.Linear(64,32),nn.Embedding(32,64)):
            for bits in (2,3,4):
                with self.subTest(layer=type(layer).__name__,bits=bits):
                    q,s,b=mx.quantize(layer.weight,group_size=32,bits=bits)
                    weights={'test.weight':q,'test.scales':s,'test.biases':b}
                    if 'bias' in layer:weights['test.bias']=layer.bias
                    packed=packed_layer(layer,weights,'test')
                    dense=mx.dequantize(q,s,b,group_size=32,bits=bits)
                    if isinstance(layer,nn.Linear):
                        inputs=mx.ones((1,64));expected=inputs@dense.T+layer.bias
                    else:
                        inputs=mx.array([0,7,31]);expected=dense[inputs]
                    self.assertTrue(bool(mx.allclose(packed(inputs),expected,atol=.005)))

    def test_unsupported_module_rejected(self):
        import mlx.nn as nn
        from sts_lab.packed_moshi import packed_layer
        with self.assertRaises(ValueError):packed_layer(nn.ReLU(),{},'test')

    def test_zero_embedding_preserves_negative_sentinel(self):
        import mlx.core as mx
        import mlx.nn as nn
        from sts_lab.packed_moshi import packed_layer
        class SentinelEmbedding(nn.Embedding):
            zero_idx=-1
            low_rank=None
            demux_second_stream=False
        layer=SentinelEmbedding(32,64)
        q,s,b=mx.quantize(layer.weight,group_size=32,bits=4)
        packed=packed_layer(layer,{'test.weight':q,'test.scales':s,'test.biases':b},'test')
        actual=packed(mx.array([-1,0,5]))
        self.assertEqual(float(mx.max(mx.abs(actual[0])).item()),0)
        self.assertTrue(bool(mx.allclose(actual[1:],packed(mx.array([0,5])))))

    def test_invalid_packed_width_rejected(self):
        import mlx.core as mx
        import mlx.nn as nn
        from sts_lab.packed_moshi import packed_layer
        with self.assertRaises(ValueError):
            packed_layer(nn.Linear(64,32),{'test.weight':mx.zeros((32,1),dtype=mx.uint32)},'test')
