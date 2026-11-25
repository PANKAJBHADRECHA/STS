"""Fetch only the pinned native voice assets; never fall back to a larger model."""
import argparse
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent.parent
MODELS={
    '4bit':('mlx-community/LFM2.5-Audio-1.5B-4bit','55d4de65fa10fa935561aaaa995271d8b8ff2ee0'),
    '6bit':('mlx-community/LFM2.5-Audio-1.5B-6bit','c875bcf10d0b18bd811ab7755f20d3b356a61890'),
}


def main():
    from huggingface_hub import snapshot_download
    parser=argparse.ArgumentParser();parser.add_argument('--precision',choices=MODELS,default='4bit');args=parser.parse_args()
    destination=ROOT/'.runtime'/f'lfm-{args.precision}'
    if shutil.disk_usage(ROOT).free<17*1024**3:
        raise SystemExit('Keep at least 17 GiB free before downloading native voice assets')
    repo,revision=MODELS[args.precision]
    snapshot_download(repo,revision=revision,local_dir=destination,allow_patterns=['*.json','*.safetensors'],max_workers=2)
    print(f'Native speech assets installed: {destination}')


if __name__=='__main__':main()
