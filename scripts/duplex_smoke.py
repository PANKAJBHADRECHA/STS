"""Actual simultaneous audio-in/audio-out Moshi inference, with resource limits.

Offline recorded speech is fed on every model frame, including while it speaks.
This tests native duplex inference, not microphone hardware or human quality.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sts_lab.evaluation import atomic_json


def main():
    import mlx.core as mx
    import numpy as np
    import rustymimi
    import sentencepiece
    import sphn
    from moshi_mlx import models, utils
    from sts_lab.packed_moshi import load_packed_model
    from scripts.native_probe import fixtures
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--model', type=Path, default=ROOT / '.runtime/native/models/model.q4.safetensors')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/duplex-packed-smoke.json')
    parser.add_argument('--case', choices=['all', 'greeting', 'correction', 'interruption', 'noise', 'silence'], default='all')
    args = parser.parse_args()
    if not 4 <= args.frames <= 750:
        parser.error('--frames must be between 4 and 750')
    directory = ROOT / '.runtime/native/models'
    result = {'mode': 'actual_duplex_model_frame_probe', 'status': 'starting', 'cases': [],
              'checkpoint': 'kyutai/moshiko-mlx-q4', 'revision': '18e4df760a34d5977a34517d7d1580e07acbb2f1',
              'local_variant': args.model.name, 'live_qualified': False, 'physical_microphone_tested': False,
              'quality_qualified': False, 'input_during_output': True}
    atomic_json(args.report, result)
    mx.set_cache_limit(32 * 1024**2)
    mx.set_memory_limit(5 * 1024**3)
    mx.random.seed(42)
    started = time.perf_counter()
    model, config = load_packed_model(args.model)
    result.update(status='loaded', load_seconds=time.perf_counter()-started, load_peak_mlx_bytes=mx.get_peak_memory())
    atomic_json(args.report, result)
    print('Packed model loaded', result['load_seconds'], flush=True)
    tokenizer = sentencepiece.SentencePieceProcessor(str(directory/'tokenizer_spm_32k_3.model'))
    fixture_dir = ROOT / '.runtime/duplex-fixtures'
    fixture_dir.mkdir(exist_ok=True)
    sources = fixtures(fixture_dir)
    for source in sources:
        name = Path(source).stem
        if args.case != 'all' and name != args.case:
            continue
        codec = rustymimi.Tokenizer(str(directory/'tokenizer-e351c8d8-checkpoint125.safetensors'), num_codebooks=8)
        for cache in model.transformer_cache + model.depformer_cache:
            cache.reset()
        gen = models.LmGen(model=model, max_steps=args.frames+5,
                          text_sampler=utils.Sampler(top_k=25, temp=.8),
                          audio_sampler=utils.Sampler(top_k=250, temp=.8), check=False)
        pcm, _ = sphn.read(source, sample_rate=24000)
        pcm = np.pad(pcm, ((0,0),(0,max(0,args.frames*1920-pcm.shape[-1]))))
        times, output, pieces = [], [], []
        case = {'name': name, 'status': 'running', 'frames': 0}
        result['cases'].append(case)
        start = time.perf_counter()
        for i in range(args.frames):
            before = time.perf_counter()
            encoded = codec.encode_step(pcm[None,0:1,i*1920:(i+1)*1920])
            tokens = mx.array(encoded).transpose(0,2,1)[:,:,:8]
            text = gen.step(tokens[0])[0].item()
            if text not in (0,1,2,3):
                pieces.append(tokenizer.id_to_piece(text).replace('▁',' '))
            codes = gen.last_audio_tokens()
            if codes is not None:
                output.append(codec.decode_step(np.array(codes[:,:,None]).astype(np.uint32)))
            elapsed = time.perf_counter()-before
            times.append(elapsed)
            case.update(frames=i+1, last_frame_ms=elapsed*1000)
            atomic_json(args.report, result)
            if i % 10 == 0:
                print(name, 'frame', i, round(elapsed*1000), flush=True)
            if elapsed > 30 or time.perf_counter()-start > 180:
                case.update(status='resource_stopped', reason='Frame exceeded 30 seconds or case exceeded 180 seconds')
                break
        else:
            case['status'] = 'completed'
        wav = fixture_dir / (name+'-reply.wav')
        if output:
            audio = np.concatenate(output, axis=-1)[0,0]
            rustymimi.write_wav(str(wav), audio, sample_rate=24000)
            case.update(output_seconds=len(audio)/24000, output_rms=float(np.sqrt(np.mean(audio**2))))
        case.update(audio_seconds=len(times)*.08, wall_seconds=time.perf_counter()-start,
                    frame_p50_ms=float(np.median(times)*1000), frame_p95_ms=float(np.percentile(times,95)*1000),
                    real_time_factor=sum(times)/(len(times)*.08), model_text=''.join(pieces), peak_mlx_bytes=mx.get_peak_memory())
        atomic_json(args.report, result)
        del gen, codec, output
        gc.collect()
        mx.clear_cache()
        if case['status'] == 'resource_stopped':
            break
    result['status'] = 'completed' if all(c['status']=='completed' for c in result['cases']) else 'resource_stopped'
    # Timing is necessary, never sufficient, for voice quality qualification.
    result['timing_qualified'] = bool(result['cases']) and all(c['status']=='completed' and c['real_time_factor']<=1 for c in result['cases'])
    atomic_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
