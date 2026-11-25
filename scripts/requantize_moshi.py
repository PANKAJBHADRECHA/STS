"""Experimental, bounded-memory requantization of a local Moshi q4 checkpoint.

Only the temporal transformer is changed. Codec and depth decoder stay intact.
Writes directly to safetensors with one tensor group resident at a time.
"""
import argparse
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def convert(source, target, bits, factor=False):
    import mlx.core as mx
    import numpy as np
    if source.resolve() == target.resolve():
        raise ValueError('Source and destination must differ')
    if target.exists():
        raise FileExistsError(target)
    with source.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(size))
    source_start = 8 + size
    dtypes = {'BF16': np.uint16, 'F16': np.float16, 'F32': np.float32,
              'U32': np.uint32, 'I32': np.int32, 'I64': np.int64}
    reserve = 1024 * 1024
    temporary = target.with_suffix('.partial')
    result = {}
    offset = 0
    done = set()
    mx.set_cache_limit(32 * 1024**2)

    def read(name):
        item = header[name]
        raw = np.memmap(source, dtype=dtypes[item['dtype']], mode='r',
                        offset=source_start + item['data_offsets'][0], shape=tuple(item['shape']))
        value = mx.array(raw)
        return value.view(mx.bfloat16) if item['dtype'] == 'BF16' else value

    with temporary.open('wb') as output:
        output.write(struct.pack('<Q', reserve))
        output.write(b' ' * reserve)

        def write(name, value):
            nonlocal offset
            mx.eval(value)
            dtype = {mx.bfloat16: 'BF16', mx.float16: 'F16', mx.float32: 'F32',
                     mx.uint32: 'U32', mx.int32: 'I32', mx.int64: 'I64'}[value.dtype]
            raw = np.array(value.view(mx.uint16) if dtype == 'BF16' else value).tobytes()
            output.write(raw)
            result[name] = {'dtype': dtype, 'shape': list(value.shape), 'data_offsets': [offset, offset + len(raw)]}
            offset += len(raw)
            done.add(name)

        for name in header:
            if name == '__metadata__' or name in done:
                continue
            prefix = name.rsplit('.', 1)[0]
            if not factor and name.startswith('transformer.') and prefix + '.scales' in header:
                # Process companions together even when biases precede weight.
                name = prefix + '.weight'
            if (factor and name.endswith('.weight') and len(header[name]['shape']) == 2
                    and header[name]['shape'][-1] % 32 == 0):
                packed, scales, biases = mx.quantize(read(name), group_size=32, bits=bits)
                write(name, packed)
                write(prefix + '.scales', scales)
                write(prefix + '.biases', biases)
                del packed, scales, biases
                mx.clear_cache()
            elif (name.endswith('.weight') and name.startswith('transformer.')
                    and prefix + '.scales' in header):
                dense = mx.dequantize(read(name), read(prefix + '.scales'),
                                      read(prefix + '.biases'), group_size=32, bits=4)
                packed, scales, biases = mx.quantize(dense, group_size=32, bits=bits)
                write(name, packed)
                write(prefix + '.scales', scales)
                write(prefix + '.biases', biases)
                del dense, packed, scales, biases
                mx.clear_cache()
            else:
                write(name, read(name))
        encoded = json.dumps(result, separators=(',', ':')).encode()
        if len(encoded) > reserve:
            raise ValueError('Safetensors header reserve exceeded')
        output.seek(8)
        output.write(encoded.ljust(reserve, b' '))
    temporary.rename(target)
    target.with_suffix('.json').write_text(json.dumps({'source': source.name, 'bits': bits, 'mode': 'factor_all_linear' if factor else 'moshi_temporal',
        'group_size': 32, 'status': 'experimental_not_quality_qualified'}, indent=2))
    print(json.dumps({'path': str(target), 'bytes': target.stat().st_size, 'bits': bits}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bits', type=int, choices=(2, 3, 4), required=True)
    parser.add_argument('--source', type=Path, default=ROOT / '.runtime/native/models/model.q4.safetensors')
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--factor', action='store_true')
    args = parser.parse_args()
    convert(args.source, args.target, args.bits, args.factor)
