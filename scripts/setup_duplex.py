"""Install pinned, public F-Actor/Mimi assets and quantize locally for MLX.

Weights and reference speaker vectors remain ignored local artifacts.
"""
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
REPO='JSALT2026-Conv-AI-Simulator/f-actor-mimi-behavior-sd'
REVISION='7e45bf192ab1c7a3db5e7a26715119f5b6f4f1f1'
CODE_REVISION='9f93fd079fb26efb7a0d8d400b5ba51d60a8c5e1'


def main():
    from huggingface_hub import snapshot_download,hf_hub_download
    from scripts.requantize_moshi import convert
    destination=ROOT/'.runtime/factor-source'
    if shutil.disk_usage(ROOT).free<8*1024**3:
        raise SystemExit('At least 8 GiB free disk space is required for setup')
    print('Downloading F-Actor (Llama 3.2 model license applies).',flush=True)
    snapshot_download(REPO,revision=REVISION,local_dir=destination,
                      allow_patterns=['model.safetensors','*.json','README.md'])
    target=destination/'model.q4.safetensors'
    if not target.exists():convert(destination/'model.safetensors',target,4,factor=True)
    models=ROOT/'.runtime/native/models'
    models.mkdir(parents=True,exist_ok=True)
    hf_hub_download('kyutai/moshiko-mlx-q4','tokenizer-e351c8d8-checkpoint125.safetensors',
                    revision='18e4df760a34d5977a34517d7d1580e07acbb2f1',local_dir=models)
    voice=ROOT/'.runtime/factor-speaker_embeddings.json'
    if not voice.exists():
        url=f'https://raw.githubusercontent.com/MaikeZuefle/f-actor/{CODE_REVISION}/training/example_speakers/speaker_embeddings.json'
        with urllib.request.urlopen(url,timeout=30) as response:data=response.read(100000)
        voice.write_bytes(data)
    print('Local assets prepared. Run scripts/factor_probe.py before qualification.',flush=True)


if __name__=='__main__':main()
