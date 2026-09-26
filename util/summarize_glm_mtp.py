"""Preserve absolute MTP rates, telemetry and acceptance/rollback distributions."""
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from util.summarize_glm_decode import main as summarize_telemetry


def main():
    with redirect_stdout(io.StringIO()): summarize_telemetry()
    directory=Path(sys.argv[1])
    result=json.loads((directory/'summary.json').read_text())
    modes={}
    for native in (False,True):
        runs=[r for r in result['runs'] if r['native']==native]
        if not runs: continue
        rates=[r['effective_output_tokens_per_second'] for r in runs]
        modes[str(native)]=dict(rates=rates,mean=statistics.mean(rates),median=statistics.median(rates))
    result['effective_modes']=modes
    if len(modes)==2:
        result['effective_improvement_percent']=100*(modes['True']['mean']/modes['False']['mean']-1)
    for run in result['runs']:
        stats=run.get('draft_stats')
        if stats:
            run['accepted_per_window']=dict(Counter(s[2] for s in stats))
            run['rejected_per_window']=dict(Counter(s[1]-s[2] for s in stats))
            assert sum(s[1] for s in stats)==run['proposed']
            assert sum(s[2] for s in stats)==run['accepted']
    (directory/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','modes')},indent=2))


if __name__=='__main__': main()
