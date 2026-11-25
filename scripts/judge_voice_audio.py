"""Independent ASR of generated speech, used only for evaluation, never serving."""
import array
import json
from pathlib import Path
import sys
import wave

ROOT=Path(__file__).resolve().parent.parent


def main():
    import numpy as np
    from scipy.signal import resample_poly
    from faster_whisper import WhisperModel
    model=WhisperModel('tiny.en',device='cpu',compute_type='int8',cpu_threads=2)
    rows=[]
    for path in sorted((ROOT/'.runtime/conversation-fixtures').glob('*-reply.wav')):
        with wave.open(str(path),'rb') as stream:
            samples=np.asarray(array.array('h',stream.readframes(stream.getnframes())),dtype=np.float32)/32768
            rate=stream.getframerate()
        # Passing samples avoids PyAV's changing file-open API entirely.
        samples=resample_poly(samples,2,3).astype(np.float32) if rate==24000 else samples
        segments,_=model.transcribe(samples,beam_size=3,vad_filter=False)
        text=' '.join(s.text for s in segments).strip();print(path.name,text,flush=True)
        rows.append({'file':path.name,'transcript':text})
    (ROOT/'reports/lfm-audio-transcripts.json').write_text(json.dumps({
        'evaluation_only':'Whisper tiny.en independently transcribes generated WAVs; not part of the STS inference path.',
        'limitations':'ASR errors remain possible; no human listening or acoustic-quality benchmark is claimed.',
        'outputs':rows},indent=2)+'\n')


if __name__=='__main__':main()
