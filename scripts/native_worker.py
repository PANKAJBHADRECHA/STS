"""Moshi/Mimi direct audio tokens; adapted from Kyutai's MIT run_inference path.

See https://github.com/kyutai-labs/moshi. Only local model files are accepted.
"""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from sts_lab.audio import inspect_wav
from sts_lab.evaluation import atomic_json


def main():
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    import rustymimi
    import sentencepiece
    import sphn
    from moshi_mlx import models, utils

    job=json.loads(Path(sys.argv[1]).read_text())
    weights,codec,text_tokenizer=job["assets"]
    mx.random.seed(299792458)
    mx.set_memory_limit(6*1024**3)
    mx.set_cache_limit(128*1024**2)
    start=time.perf_counter()
    config=models.config_v0_1()
    model=models.Lm(config)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model,bits=4,group_size=32)
    model.load_weights(weights,strict=True)
    mx.eval(model.parameters())
    text=sentencepiece.SentencePieceProcessor(text_tokenizer)
    report={"load_seconds":time.perf_counter()-start,"cases":[]}
    atomic_json(Path(job["output"]),report)
    for source in job["inputs"]:
        tokenizer=rustymimi.Tokenizer(codec,num_codebooks=8)
        pcm,_=sphn.read(source,sample_rate=24000)
        steps=pcm.shape[-1]//1920
        gen=models.LmGen(model=model,max_steps=steps+5,text_sampler=utils.Sampler(top_k=25,temp=.8),
                         audio_sampler=utils.Sampler(top_k=250,temp=.8),check=False)
        generated=[];pieces=[];latencies=[]
        case_start=time.perf_counter()
        for index in range(steps):
            frame_start=time.perf_counter()
            encoded=tokenizer.encode_step(pcm[None,0:1,index*1920:(index+1)*1920])
            tokens=mx.array(encoded).transpose(0,2,1)[:,:,:8]
            token=gen.step(tokens[0])[0].item()
            if token not in (0,1,2,3):pieces.append(text.id_to_piece(token).replace("▁"," "))
            audio=gen.last_audio_tokens()
            if audio is not None:
                generated.append(tokenizer.decode_step(np.array(audio[:,:,None]).astype(np.uint32)))
            latencies.append((time.perf_counter()-frame_start)*1000)
        destination=Path(source).with_name(Path(source).stem+"-output.wav")
        if generated:
            output=np.concatenate(generated,axis=-1)
            rustymimi.write_wav(str(destination),output[0,0],sample_rate=24000)
        elapsed=time.perf_counter()-case_start
        report["cases"].append({"case":Path(source).stem,"input_seconds":steps*.08,"wall_seconds":elapsed,
                                "real_time_factor":elapsed/(steps*.08),"frames":steps,
                                "frame_p50_ms":float(np.median(latencies)),"frame_p95_ms":float(np.percentile(latencies,95)),
                                "generated_model_text":"".join(pieces),"output":inspect_wav(destination) if generated else None,
                                "speech_quality_evaluated":False})
        atomic_json(Path(job["output"]),report)
        print(Path(source).stem,elapsed,flush=True)
        del tokenizer,gen,generated
        mx.clear_cache()


if __name__=="__main__":main()
