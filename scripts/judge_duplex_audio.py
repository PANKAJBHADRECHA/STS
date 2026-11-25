"""Independent ASR checks of generated fixture replies; never used for serving."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent


def main():
    import sphn
    from faster_whisper import WhisperModel
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=ROOT/'.runtime/factor-fixtures')
    parser.add_argument('--report',type=Path,default=ROOT/'reports/factor-audio-transcripts.json')
    args=parser.parse_args()
    model=WhisperModel('tiny.en',device='cpu',compute_type='int8',cpu_threads=2)
    rows=[]
    for path in sorted(args.directory.glob('*-reply.wav')):
        audio,_=sphn.read(str(path),sample_rate=16000)
        segments,_=model.transcribe(audio[0],beam_size=3,vad_filter=True)
        text=' '.join(s.text for s in segments).strip()
        print(path.name,text,flush=True)
        rows.append({'file':path.name,'transcript':text})
    args.report.write_text(json.dumps({'evaluation_only':'Whisper tiny.en, not part of the STS serving path.',
        'limitations':'ASR can make mistakes. No human listening or quality rating is claimed.',
        'outputs':rows},indent=2)+'\n')


if __name__=='__main__':main()
