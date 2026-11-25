"""Select the best tested real-audio recipe; five disclosed semantic spot checks."""
import json
from pathlib import Path
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from sts_lab.evaluation import atomic_json


def main():
    candidates=[]
    for filename in ['lfm-upstream.json','lfm-greedy.json','lfm-6bit-upstream.json','lfm-brief-upstream.json']:
        report=json.loads((ROOT/'reports'/filename).read_text())
        cases={x['case']:x for x in report['cases']}
        text=lambda case:cases.get(case,{}).get('text','').lower()
        checks={
            'greeting':bool(re.search(r'well|good|hello|hi\b',text('greeting'))),
            'arithmetic':bool(re.search(r'five|\b5\b',text('arithmetic'))),
            'two_turn_facts':all(w in text('memory_followup') for w in ['maya','coffee']) and not any(w in text('memory_followup') for w in ['latte','vanilla','cinnamon']),
            'latest_correction':'friday' in text('correction'),
            'recorded_question':any(w in text('upstream_recording') for w in ['wood','craft','furniture']),
        }
        audio=sum('output' in c and c['output']['rms']>.001 and c['output']['clipped_fraction']<.01 and not c['truncated'] for c in cases.values())
        times=[c['wall_seconds'] for c in cases.values() if 'wall_seconds' in c]
        candidates.append({'recipe':filename,'checks':checks,'spot_checks_passed':sum(checks.values()),
                           'audio_turns_passed':audio,'audio_turns_total':len(cases),'median_wall_seconds':statistics.median(times),
                           'peak_mlx_memory_bytes':max(c.get('model_peak_memory_bytes',0) for c in cases.values())})
    eligible=[c for c in candidates if c['audio_turns_total']==6 and c['audio_turns_passed']==6]
    winner=sorted(eligible,key=lambda c:(-c['spot_checks_passed'],c['median_wall_seconds'],c['peak_mlx_memory_bytes']))[0]
    result={'winner':winner['recipe'],'model':'mlx-community/LFM2.5-Audio-1.5B-4bit',
            'revision':'55d4de65fa10fa935561aaaa995271d8b8ff2ee0',
            'mode':'native_sts_turn_based','full_duplex':False,'candidates':candidates,
            'selection_rule':'All six turns must generate valid non-silent untruncated audio; then five semantic spot checks, median wall time, memory.',
            'limitations':['Tiny known test set; prompt was tuned after observing failures, not a blind benchmark.',
                           'Keyword checks do not certify overall reasoning, speaker-role correctness or voice quality.',
                           'Tests used synthesized input speech plus an upstream recorded question, not a live human caller.',
                           'Host load was not isolated; timing comparisons are descriptive.',
                           'The model can still confuse user/assistant pronouns; factual spot checks do not remove that limitation.']}
    if winner['recipe']!='lfm-brief-upstream.json':
        raise SystemExit('Measured winner changed; update voice_worker.py before publishing')
    atomic_json(ROOT/'reports/voice-selection.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
