"""Check installed Tabby startup policy without toggling validation controls."""
import json
from pathlib import Path
import time
import httpx
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'native-results/deepseek-v41/installed'
key=yaml.safe_load(Path('D:/tabbyAPI/api_tokens.yml').read_text())['admin_key']
with httpx.Client(base_url='http://127.0.0.1:5008',headers={'Authorization':'Bearer '+key},timeout=7200,trust_env=False) as c:
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        try:
            r=c.get('/__validation/status',timeout=5)
            if r.status_code==200 and r.json()['model_loaded']:break
        except httpx.HTTPError:pass
        time.sleep(3)
    else:raise RuntimeError('Installed server did not load')
    before=r.json()
    assert before['ir'] and before['generator']['drafts']==3
    assert before['version']=='1.6.6+nativeplan.v2.tp.moe1.ds41.1'
    assert all(Path(before[k]).is_relative_to(Path('D:/tabbyAPI/venv')) for k in ('package','extension'))
    prompt=json.loads((ROOT/'native-results/deepseek-v41/graph-http/timing/prompt.json').read_text())['text']
    r=c.post('/v1/completions',json=dict(model='DeepSeek-V4.1-Flash-1017-Heretic-EXL3-1.68bpw',prompt=prompt,
        max_tokens=128,min_tokens=128,temperature=0,top_k=1,top_p=1,min_p=0,repetition_penalty=1,
        frequency_penalty=0,presence_penalty=0,add_bos_token=False,token_healing=False,stream=False))
    r.raise_for_status()
    after=c.get('/__validation/results').json();s=after['status']
    assert s['graph_captures']>0 and s['graph_replays']>s['graph_captures']
    assert s['vision'] and len(s['ngrams'])==2 and all(n['mode']=='fp16_disk' for n in s['ngrams'])
    reference=json.loads((ROOT/'native-results/deepseek-v41/splits/budget76/audit/c178-r0-mtp1-ir0-audit.json').read_text())['result']['state']['jobs'][0]
    job=after['state']['jobs'][0]
    assert len(after['state']['jobs'])==1 and job['new_tokens']==128
    for k in ('ids','accepted_draft_tokens','rejected_draft_tokens'):assert job[k]==reference[k],k
    (OUT/'startup-proof.json').write_text(json.dumps(dict(before=before,result=after,response=r.json()),indent=2))
    print(json.dumps(dict(version=s['version'],package=s['package'],extension=s['extension'],
        captures=s['graph_captures'],replays=s['graph_replays'],engram='fp16_disk',tokens_exact=True),indent=2))
