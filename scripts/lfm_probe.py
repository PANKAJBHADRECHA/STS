"""Real audio-input conversations with LFM2.5, independent of the simulator."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from sts_lab.evaluation import atomic_json
from sts_lab.voice_engine import VoiceEngine,BRIEF_PROMPT


def spoken(directory, name, text):
    path=directory/f'{name}.wav'
    if not path.exists():
        aiff=directory/f'{name}.aiff'
        subprocess.run(['say','-v','Samantha','-r','170','-o',str(aiff),text],check=True)
        subprocess.run(['afconvert','-f','WAVE','-d','LEI16@24000',str(aiff),str(path)],check=True)
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--sampling',choices=['upstream','greedy'],default='upstream');p.add_argument('--model',default='.runtime/lfm-4bit');p.add_argument('--max-tokens',type=int,default=256);p.add_argument('--smoke',action='store_true');p.add_argument('--brief',action='store_true');args=p.parse_args()
    directory=ROOT/'.runtime/conversation-fixtures';directory.mkdir(exist_ok=True)
    sources=[
        ('greeting',spoken(directory,'greeting','Hello. How are you today?'),True),
        ('arithmetic',spoken(directory,'arithmetic','What is two plus three? Give a short answer.'),True),
        ('memory_start',spoken(directory,'memory-start','My name is Maya and my favorite drink is coffee. Please remember that.'),True),
        ('memory_followup',spoken(directory,'memory-followup','What is my name and my favorite drink?'),False),
        ('correction',spoken(directory,'correction','I want to meet on Tuesday. Sorry, I mean Friday. Which day did I choose?'),True),
        ('upstream_recording',directory/'upstream-question.wav',True),
    ]
    if args.smoke:sources=sources[:1]
    report={'mode':'actual_native_audio_conversations','checkpoint':str(args.model),'sampling':args.sampling,'cases':[],
            'limitations':['User fixture speech is synthesized except the upstream recorded question.','No live human microphone session or full-duplex barge-in is claimed.']}
    tag=('6bit-' if '6bit' in args.model else '')+('brief-' if args.brief else '')
    path=ROOT/f'reports/lfm-{tag}{args.sampling}.json'
    atomic_json(path,report)
    system_prompt=None
    if args.brief:
        system_prompt=BRIEF_PROMPT
    report['system_prompt']=system_prompt or 'Respond with interleaved text and audio.'
    engine=VoiceEngine(ROOT/args.model,max_tokens=args.max_tokens,sampling=args.sampling,system_prompt=system_prompt)
    report['load_seconds']=engine.load_seconds
    print('Model loaded',engine.load_seconds,flush=True)
    for name,source,reset in sources:
        try:
            result=engine.respond(source,directory/f'{tag}{args.sampling}-{name}-reply.wav',reset=reset)
            report['cases'].append({'case':name,**result})
            print(name,result['text'],round(result['wall_seconds'],2),flush=True)
        except Exception as error:
            report['cases'].append({'case':name,'error':str(error),'type':type(error).__name__})
            print(name,type(error).__name__,str(error),flush=True)
        atomic_json(path,report)


if __name__=='__main__':main()
