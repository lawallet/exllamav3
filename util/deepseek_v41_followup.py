"""Fresh-process follow-up. Uses the installed wheel; no in-request telemetry by default."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import httpx
import yaml

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'native-results/deepseek-v41'
TABBY=Path('D:/tabbyAPI')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=BASE/'followup/fresh')
    p.add_argument('--processes',type=int,default=3)
    p.add_argument('--pairs',type=int,default=5)
    p.add_argument('--warmup-pairs',type=int,default=2)
    p.add_argument('--existing',action='store_true')
    p.add_argument('--client-torch',action='store_true',help='Controlled reproduction of the original client imports')
    p.add_argument('--split',type=float,nargs=3,default=[76,46,0])
    p.add_argument('--context',choices=['short','8k','65k'],default='short')
    p.add_argument('--tokens',type=int,default=512)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    proof=json.loads((BASE/'package/installed-proof.json').read_text())
    binary=TABBY/'venv/Lib/site-packages/exllamav3_ext.cp313-win_amd64.pyd'
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==proof['extension_sha256']
    assert importlib.metadata.version('exllamav3')==proof['version']
    prompt_path=BASE/('graph-http/timing/prompt.json' if a.context=='short' else f'graph-http/context{a.context}/prompt.json')
    prompt=json.loads(prompt_path.read_text())['text']
    if a.client_torch:
        import torch
        from exllamav3 import Config,Tokenizer
        tokenizer=Tokenizer.from_config(Config.from_directory(str(TABBY/'models/DeepSeek-V4.1-Flash-1017-Heretic-EXL3-1.68bpw')))
        assert tokenizer.encode(prompt,encode_special_tokens=True).shape[-1]==177
    (out/'protocol.json').write_text(json.dumps(dict(arguments=vars(a)|{'output':str(out)},
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),extension_sha256=proof['extension_sha256'],
        version=proof['version'],client_imports_torch=a.client_torch,telemetry=False,
        environment={k:v for k,v in os.environ.items() if k.startswith(('CUDA','EXL','OMP','MKL','OPENBLAS'))}),indent=2))
    key=yaml.safe_load((TABBY/'api_tokens.yml').read_text())['admin_key']
    c=httpx.Client(base_url='http://127.0.0.1:5008',headers={'Authorization':'Bearer '+key},timeout=7200,trust_env=False)
    def get(route):
        r=c.get(route);r.raise_for_status();return r.json()
    def post(route,body):
        r=c.post(route,json=body);r.raise_for_status();return r.json()
    def save(path,data):path.write_text(json.dumps(data,indent=2))
    reference=json.loads((BASE/'splits/budget76/timing/c178-r0-mtp1-ir0.json').read_text())['result']['state']['jobs'][0]['ids'] if a.context=='short' and a.tokens==512 else None
    for process in range(a.processes):
        folder=out/f'process{process}';folder.mkdir()
        started=time.time_ns()
        if not a.existing:
            subprocess.run([sys.executable,'-I',str(ROOT/'util/deepseek_v41_campaign.py'),'start',
                '--output',str(folder),'--split',*map(str,a.split)],check=True,cwd=ROOT)
        deadline=time.monotonic()+300
        while time.monotonic()<deadline:
            try:
                ready=get('/__validation/status')
                if ready['model_loaded']:break
            except httpx.HTTPError:pass
            time.sleep(2)
        else:raise RuntimeError('Server startup timeout')
        blocks=[m for m in ready['placement'] if m['key'].startswith('layers.') and m['key'][7:].isdigit()]
        if a.split==[76,46,0]:assert [sum(m['device']==f'cuda:{i}' for m in blocks) for i in (0,1)]==[26,14]
        assert all(n['mode']=='fp16_disk' for n in ready['ngrams']) and ready['generator']['drafts']==3
        save(folder/'startup.json',dict(start_ns=started,ready_ns=time.time_ns(),status=ready))
        rows=[]
        for round_ in range(a.warmup_pairs+a.pairs):
            phase='warmup' if round_<a.warmup_pairs else 'steady'
            order=(False,True) if (round_+process)%2==0 else (True,False)
            for ir in order:
                name=f'{phase}-r{round_}-ir{int(ir)}'
                body=dict(name=name,mtp=True,ir=ir,graphs=True,profile=False,trace=False,audit=False)
                control_start=time.time_ns();before=post('/__validation/control',body)
                start_ns=time.time_ns();start=time.perf_counter()
                response=post('/v1/completions',dict(model='DeepSeek-V4.1-Flash-1017-Heretic-EXL3-1.68bpw',prompt=prompt,
                    max_tokens=a.tokens,min_tokens=a.tokens,temperature=0,top_k=1,top_p=1,min_p=0,repetition_penalty=1,
                    frequency_penalty=0,presence_penalty=0,add_bos_token=False,token_healing=False,stream=False))
                end_ns=time.time_ns();http_wall=time.perf_counter()-start
                result=get('/__validation/results');job=result['state']['jobs'][0]
                save(folder/(name+'.json'),dict(process=process,phase=phase,round=round_,ir=ir,control_start_ns=control_start,
                    request_start_ns=start_ns,response_end_ns=end_ns,http_wall=http_wall,before=before,result=result,response=response))
                assert job['new_tokens']==a.tokens and job['cached_tokens']==0
                row=dict(name=name,phase=phase,ir=ir,tps=a.tokens/job['time_generate'],prefill=job['time_prefill'],
                    prompt_tokens=job['prompt_tokens'],accepted=job['accepted_draft_tokens'],rejected=job['rejected_draft_tokens'],
                    token_sha256=hashlib.sha256(json.dumps(job['ids']).encode()).hexdigest(),
                    original_tokens_exact=(job['ids']==reference if reference is not None else None),http_wall=http_wall)
                rows.append(row);save(folder/'rows.json',rows)
                print(json.dumps(dict(process=process,**row)),flush=True)
        summary={}
        for ir in (False,True):
            sample=[r['tps'] for r in rows if r['phase']=='steady' and r['ir']==ir]
            summary[str(ir)]=dict(n=len(sample),median=statistics.median(sample),minimum=min(sample),maximum=max(sample),
                stdev=statistics.stdev(sample) if len(sample)>1 else 0,variance=statistics.variance(sample) if len(sample)>1 else 0)
        save(folder/'summary.json',summary);print('SUMMARY',process,json.dumps(summary),flush=True)
        if not a.existing:
            post('/__validation/shutdown',{})
            deadline=time.monotonic()+90
            while time.monotonic()<deadline:
                with socket.socket() as probe:
                    try:
                        probe.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1);probe.bind(('127.0.0.1',5008));break
                    except OSError:time.sleep(1)
            else:raise RuntimeError('Own server did not shut down')

if __name__=='__main__':main()
