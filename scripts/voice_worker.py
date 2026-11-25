"""One main-thread MLX worker, controlled by the loopback server via JSON lines."""
import contextlib
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))


def reply(value):
    print(json.dumps(value),flush=True)


def main():
    from sts_lab.voice_engine import VoiceEngine, BRIEF_PROMPT
    with contextlib.redirect_stdout(sys.stderr):
        engine=VoiceEngine(ROOT/'.runtime/lfm-4bit',system_prompt=BRIEF_PROMPT)
    reply({'ready':True,'load_seconds':engine.load_seconds,'model':'LFM2.5-Audio-1.5B / MLX 4-bit'})
    current_session=None
    for line in sys.stdin:
        try:
            job=json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if job['session'] != current_session:
                    engine.reset();current_session=job['session']
                if job['command']=='reset':
                    engine.reset();result={'reset':True}
                else:
                    result=engine.respond(job['input'],job['output'],reset=job.get('reset',False))
            reply({'ok':True,'result':result})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            engine.reset()
            reply({'ok':False,'error':str(error),'type':type(error).__name__})


if __name__=='__main__':main()
