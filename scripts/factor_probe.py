"""Run five native duplex audio cases on a fixed input/output timeline."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import re
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from sts_lab.evaluation import atomic_json


def main():
    import mlx.core as mx
    import numpy as np
    import rustymimi
    import sphn
    from tokenizers import Tokenizer
    from sts_lab.factor_mlx import load_factor
    from scripts.native_probe import fixtures
    parser=argparse.ArgumentParser()
    parser.add_argument('--frames',type=int,default=160)
    parser.add_argument('--case', default='all')
    parser.add_argument('--pipeline', action='store_true')
    parser.add_argument('--fused-depth', action='store_true')
    parser.add_argument('--casual', action='store_true')
    parser.add_argument('--starts-first', action='store_true')
    parser.add_argument('--report', type=Path, default=ROOT/'reports/factor-duplex.json')
    args=parser.parse_args()
    directory=ROOT/'.runtime/factor-source'
    report=args.report
    result={'status':'starting','mode':'actual_native_duplex_inference','checkpoint':'JSALT2026-Conv-AI-Simulator/f-actor-mimi-behavior-sd',
            'revision':'7e45bf192ab1c7a3db5e7a26715119f5b6f4f1f1','quantization_bits':4,'cases':[],
            'physical_microphone_tested':False,'inference_mocked':False,'live_qualified':False,'quality_qualified':False,'codec_pipeline':args.pipeline,'starts_first':args.starts_first,'casual_narrative':args.casual,'depth_evaluation':'single_sync' if args.fused_depth else 'per_codebook_sync'}
    atomic_json(report,result)
    mx.set_cache_limit(32*1024**2);mx.random.seed(42)
    start=time.perf_counter();model=load_factor(directory);model.fast_depth=args.fused_depth
    result.update(load_seconds=time.perf_counter()-start,load_peak_mlx_bytes=mx.get_peak_memory())
    print('Loaded',result['load_seconds'],flush=True)
    tokenizer=Tokenizer.from_file(str(directory/'tokenizer.json'))
    speaker=json.loads((ROOT/'.runtime/factor-speaker_embeddings.json').read_text())['tom']
    fixture_dir=ROOT/('.runtime/factor-casual-fixtures' if args.casual else '.runtime/factor-fixtures');fixture_dir.mkdir(exist_ok=True)
    paths=fixtures(fixture_dir)
    pool=ThreadPoolExecutor(max_workers=1)
    for source in paths:
        name=Path(source).stem
        if args.case!='all' and name!=args.case:continue
        prompt=('Generate a dialogue between you (Tom) and another speaker (User) based on the given narrative. Follow the specific behavior instructions for you.\n\n'
                'Narrative:\n- You are a friendly assistant. Listen and answer the user briefly and accurately.\n\n'
                'Your behaviors:\n- backchannels: 2\n- interruptions: 0\n- starts the dialogue: False\n\n'
                'Ensure that the dialogue reflects the behaviours of you.\n<|SOS|>')
        if args.casual:prompt=prompt.replace('You are a friendly assistant. Listen and answer the user briefly and accurately.', 'You and User have a friendly conversation about your day. Greet User and listen to what they say.')
        if args.starts_first:prompt=prompt.replace('starts the dialogue: False','starts the dialogue: True')
        model.reset(tokenizer.encode(prompt).ids,speaker)
        codec=rustymimi.Tokenizer(str(ROOT/'.runtime/native/models/tokenizer-e351c8d8-checkpoint125.safetensors'),num_codebooks=8)
        pcm,_=sphn.read(source,sample_rate=24000)
        pcm=np.pad(pcm,((0,0),(0,max(0,args.frames*1920-pcm.shape[-1]))))
        if name=='interruption':
            greeting,_=sphn.read(str(fixture_dir/'greeting.wav'),sample_rate=24000)
            mixed=np.zeros_like(pcm)
            mixed[:,:min(greeting.shape[-1],mixed.shape[-1])]=greeting[:,:mixed.shape[-1]]
            length=min(pcm.shape[-1],mixed.shape[-1]-4*24000)
            mixed[:,4*24000:4*24000+length]+=pcm[:,:length]
            pcm=mixed
        output=[];texts=[];times=[];overlap=0
        case={'name':name,'status':'running','frames':0,'input_during_output':True};result['cases'].append(case);atomic_json(report,result)
        started=time.perf_counter()
        for i in range(args.frames):
            before=time.perf_counter()
            incoming=pcm[None,0:1,i*1920:(i+1)*1920]
            if args.pipeline:future=pool.submit(codec.encode_step,incoming)
            else:encoded=codec.encode_step(incoming)
            own,text=model.predict()
            if args.pipeline:encoded=future.result()
            codes=np.array(own,dtype=np.uint32)
            if np.any(codes>=2048):
                case.update(status='reserved_audio_token',reserved_tokens=codes.tolist());break
            if args.pipeline:future=pool.submit(codec.decode_step,codes[:,:,None])
            else:audio=codec.decode_step(codes[:,:,None])
            texts.append(int(text.item()))
            model.advance(own,mx.array(encoded[:,:,0]),text)
            if args.pipeline:audio=future.result()
            output.append(audio)
            if np.sqrt(np.mean(incoming**2))>.005 and np.sqrt(np.mean(audio**2))>.005:overlap+=1
            elapsed=time.perf_counter()-before;times.append(elapsed)
            case.update(frames=i+1,last_frame_ms=elapsed*1000)
            if i%20==0:
                print(name,i,round(elapsed*1000),tokenizer.decode(texts),flush=True);atomic_json(report,result)
            if elapsed>30 or time.perf_counter()-started>240:
                case.update(status='resource_stopped');break
        else:case['status']='completed'
        if output:
            audio=np.concatenate(output,axis=-1)[0,0]
            rustymimi.write_wav(str(fixture_dir/(name+'-reply.wav')),audio,sample_rate=24000)
            case.update(output_seconds=len(audio)/24000,output_rms=float(np.sqrt(np.mean(audio**2))))
        raw_text=tokenizer.decode(texts,skip_special_tokens=False)
        case.update(model_text=re.sub(r'\[[A-Z_]+\]',' ',raw_text).strip(),
                    control_tokens=dict(Counter(re.findall(r'\[[A-Z_]+\]',raw_text))),overlapping_speech_frames=overlap,
                    frames=len(times),wall_seconds=time.perf_counter()-started,
                    real_time_factor=sum(times)/(len(times)*.08) if times else None,
                    frame_p50_ms=float(np.median(times)*1000) if times else None,
                    frame_p95_ms=float(np.percentile(times,95)*1000) if times else None,
                    peak_mlx_bytes=mx.get_peak_memory())
        atomic_json(report,result);del codec
    pool.shutdown()
    result['status']='completed'
    result['timing_qualified']=all(c['status']=='completed' and c['real_time_factor']<=1 for c in result['cases'])
    atomic_json(report,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
