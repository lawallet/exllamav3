"""Require completed staged correctness/performance gates before packaging."""
from collections import Counter
import json
from pathlib import Path
import re
import statistics

BASE=Path(__file__).resolve().parents[1]/'native-results/deepseek-v41/omp-policy'


def read(path):return json.loads((BASE/path).read_text(encoding='utf-8-sig'))


def main():
    tests=read('native-regressions.json')
    assert tests['success'] and tests['tests']==234 and not tests['skipped']
    cpu=read('candidate-cpu/complete.json')
    assert cpu['exact'] and cpu['torch_threads']==56
    for setting in ('unset','0','1','50'):
        result=read(f'runtime-integration/{setting}.json')
        expected=200 if setting=='unset' else int(setting)
        assert result['threads']==56
        assert all(r['status']['actual_ms']==expected for r in result['outcomes'])
    cases=('qwen','glm','nemotron','deepseek4','gemma','devstral','glm-tp')
    for case in cases:
        assert read(f'candidate-{case}/complete.json')['exact']
        assert read(f'candidate-{case}/metadata.json')['implemented']
        for path in (BASE/f'candidate-{case}').glob('*block0.json'):
            result=json.loads(path.read_text())
            assert result['exact'] and all(r['milliseconds']==0 and r['implementation']['successful']
                for r in result['runtime']),path
    tp=read('candidate-glm-tp/tp-native-evidence.json')
    assert len(tp)==2 and all(sum(b['graph_replays'] for b in r['blocks'])>0 for r in tp)
    ds=read('candidate-deepseek/final-status.json')
    placement=Counter(m['device'] for m in ds['status']['placement'] if re.fullmatch(r'layers\.\d+',m['key']))
    assert placement=={'cuda:0':26,'cuda:1':14},placement
    assert ds['status']['generator']==dict(batch=1,drafts=3,dynamic=False,cache_tokens=65536)
    assert ds['status']['ngrams'] and all(n['mode']=='fp16_disk' for n in ds['status']['ngrams'])
    command=read('candidate-deepseek/server.json')['command'];index=command.index('--gpu-split')
    assert list(map(float,command[index+1:index+4]))==[76,46,0]
    assert ds['status']['graph_replays']>0 and ds['extra']['torch_threads']==56
    assert ds['extra']['implemented_policy']['actual_ms']==0
    assert read('parity-staged-final.json')['exact']
    assert read('old-release-final-control-run.json')['code']==0
    tp_checks=read('tp-process-rechecks.json')
    assert len(tp_checks)==3 and all(r['code']==0 for r in tp_checks)
    rates={}
    for label in ('candidate-deepseek','old-release-final-control'):
        values=[]
        for path in (BASE/label).glob('steady-r*-ir1.json'):
            job=json.loads(path.read_text())['result']['state']['jobs'][0]
            values.append(job['new_tokens']/job['time_generate'])
        assert len(values)>2
        rates[label]=statistics.median(values)
    assert rates['candidate-deepseek']>rates['old-release-final-control'],rates
    result=dict(passed=True,tests=tests['tests'],models=list(cases),placement=dict(placement),
                native_medians=rates,incremental_percent=100*(rates['candidate-deepseek']/rates['old-release-final-control']-1))
    (BASE/'release-gates.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
