"""Focused generation-pipeline investigation against an isolated Tabby process.

No runtime source patches: baseline uses only existing diagnostic switches.
"""
import argparse, hashlib, json, os, socket, subprocess, sys, time
from pathlib import Path
import httpx, yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'native-results/deepseek-v41'
TABBY = Path('D:/tabbyAPI')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=['baseline','measure','candidate','candidate-profile','candidate-upload','environment'], default='baseline')
    p.add_argument('--package-dir', type=Path)
    p.add_argument('--long', action='store_true')
    p.add_argument('--combined', action='store_true',help='Pair both implemented policies against both disabled')
    p.add_argument('--environment-policy',choices=['unset','mask','seed','both'],default='both')
    p.add_argument('--pairs', type=int, default=5)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    def save(name, value): (out / name).write_text(json.dumps(value, indent=2))
    for name in ('deepseek_v41_pipeline.py', 'serve_deepseek_v41_followup.py', 'serve_deepseek_v41_ir.py', 'serve_deepseek_v41_pipeline.py'):
        (out / name).write_bytes((ROOT / 'util' / name).read_bytes())
    save('source.json', dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)))
    old = json.loads((BASE / 'followup/fresh/process0/server.json').read_text())
    configs = {n: hashlib.sha256((TABBY/n).read_bytes()).hexdigest() for n in old['config_hashes']}
    key = yaml.safe_load((TABBY/'api_tokens.yml').read_text())['admin_key']
    c = httpx.Client(base_url='http://127.0.0.1:5008', headers={'Authorization':'Bearer '+key}, timeout=7200, trust_env=False)
    def post(path, body):
        r=c.post(path,json=body);r.raise_for_status();return r.json()
    def get(path):
        r=c.get(path);r.raise_for_status();return r.json()
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1);probe.bind(('127.0.0.1',5008))
    cmd=old['command'];cmd[2]=str(ROOT/'util/serve_deepseek_v41_followup.py')
    candidate_controls=a.phase.startswith('candidate')
    if candidate_controls or a.phase=='environment':cmd[2]=str(ROOT/'util/serve_deepseek_v41_pipeline.py')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,0,1',EXL3_NGRAM_STREAM='1',EXL3_NATIVE_PLAN='1',
             EXL3_OMP_BLOCKTIME_MS='0',EXL3_NATIVE_MOE_ASSIGN='0',PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',DS41_RESULTS=str(out))
    for k in ('DS41_PACKAGE_ROOT','DS41_EXTENSION_DIR'):env.pop(k,None)
    env.pop('EXL3_STOP_MASK_CACHE',None)
    env.pop('EXL3_ASYNC_DRAFT_SEED',None)
    if a.phase=='environment':
        if a.environment_policy in ('mask','both'):env['EXL3_STOP_MASK_CACHE']='1'
        if a.environment_policy in ('seed','both'):env['EXL3_ASYNC_DRAFT_SEED']='1'
    if a.package_dir:
        env['DS41_PACKAGE_ROOT']=str(a.package_dir.resolve())
        env['DS41_EXTENSION_DIR']=str(a.package_dir.resolve())
    log=(out/'server.log').open('w',encoding='utf-8')
    server=subprocess.Popen(cmd,cwd=TABBY,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    save('server.json',dict(pid=server.pid,command=cmd,start_ns=time.time_ns(),config_hashes=configs))
    monitor=None
    signatures={}
    try:
        deadline=time.monotonic()+360
        while time.monotonic()<deadline:
            if server.poll() is not None:raise RuntimeError('Server exited; inspect server.log')
            try:
                status=get('/__validation/status')
                if status['model_loaded']:break
            except httpx.HTTPError:pass
            time.sleep(2)
        else:raise RuntimeError('startup timeout')
        save('startup.json',dict(time_ns=time.time_ns(),status=status,extra=get('/__followup/status')))
        def run(name, minimum=512, serial=False, mtp=True, tokens=512, mode='off', profile=False, trace=False, context='short', candidate=False, ir=True, async_seed=None):
            if async_seed is None:async_seed=bool(candidate and a.combined)
            before=post('/__validation/control',dict(name=name,mtp=mtp,ir=ir,graphs=True,audit=False,profile=profile,trace=trace))
            policy=None
            if a.phase=='environment':
                candidate=a.environment_policy in ('mask','both')
                async_seed=a.environment_policy in ('seed','both')
                policy=get('/__pipeline/status')
                assert policy['enabled']==candidate and policy['async_seed']==async_seed
                assert policy['seed_implementation']!='diagnostic prototype'
            if candidate_controls:
                policy=post('/__pipeline/control',dict(enabled=candidate,async_seed=async_seed))
                assert policy['enabled']==candidate
                assert policy['async_seed']==async_seed
            extra=post('/__followup/control',dict(mode=mode,serial_readback=serial))
            assert extra['blocktime']['milliseconds']==0
            path=BASE/('graph-http/timing/prompt.json' if context=='short' else f'graph-http/context{context}/prompt.json')
            prompt=json.loads(path.read_text())['text']
            start_ns=time.time_ns();start=time.perf_counter()
            response=post('/v1/completions',dict(model='DeepSeek-V4.1-Flash-1017-Heretic-EXL3-1.68bpw',prompt=prompt,
                max_tokens=tokens,min_tokens=min(minimum,tokens),temperature=0,top_k=1,top_p=1,min_p=0,repetition_penalty=1,
                frequency_penalty=0,presence_penalty=0,add_bos_token=False,token_healing=False,stream=False))
            end_ns=time.time_ns();wall=time.perf_counter()-start;result=get('/__validation/results')
            job=result['state']['jobs'][0]
            save(name+'.json',dict(name=name,context=context,minimum=minimum,serial_readback=serial,mtp=mtp,mode=mode,
                tokens=tokens,candidate=candidate,async_seed=async_seed,ir=ir,request_start_ns=start_ns,response_end_ns=end_ns,http_wall=wall,before=before,
                extra_before=extra,pipeline_policy=policy,
                result=result,response=response,extra_after=get('/__followup/status')))
            key=(context,tokens,mtp,min(minimum,tokens),ir)
            signature=(job['ids'],job['accepted_draft_tokens'],job['rejected_draft_tokens'])
            assert key not in signatures or signatures[key]==signature,(name,'token/MTP mismatch')
            signatures.setdefault(key,signature)
            assert job['cached_tokens']==0 and job['new_tokens']==tokens,(name,'workload mismatch')
            assert len(job['ids'][0])==job['prompt_tokens']+tokens
            print(json.dumps(dict(name=name,tps=tokens/job['time_generate'],prefill=job['time_prefill'],
                  accepted=job.get('accepted_draft_tokens'),rejected=job.get('rejected_draft_tokens'))),flush=True)
        # Specialization/warmup is distinct from profiling and primary throughput.
        for i in range(2):
            run(f'warm-masked-{i}');run(f'warm-other-{i}',minimum=512 if candidate_controls else 0,candidate=candidate_controls)
        if a.phase=='baseline':
            run('profile-masked',mode='host')
            run('profile-free',minimum=0,mode='host')
            run('profile-free-serial',minimum=0,serial=True,mode='host')
            run('profile-ordinary',mtp=False,mode='host',tokens=256)
            run('profile-cpython',profile=True)
            run('profile-cuda',trace=True,tokens=256)
        if a.phase=='environment':
            for i in range(a.pairs):run(f'steady-env-r{i}')
            if a.long:
                for context in ('8k','65k'):
                    run(f'{context}-env-warm',context=context)
                    run(f'steady-{context}-env',context=context)
        elif a.phase=='candidate-upload':
            cases=[(False,False),(True,False),(False,True),(True,True)]
            for cache,upload in cases:run(f'upload-warm-cache{int(cache)}-async{int(upload)}',candidate=cache,async_seed=upload)
            for i in range(a.pairs):
                order=cases[i%len(cases):]+cases[:i%len(cases)]
                if i%2:order=list(reversed(order))
                for cache,upload in order:
                    run(f'steady-upload-r{i}-cache{int(cache)}-async{int(upload)}',candidate=cache,async_seed=upload)
        elif a.phase=='candidate-profile':
            run('upload-warm',async_seed=True)
            run('combined-warm',candidate=True,async_seed=True)
            monitor=subprocess.Popen([sys.executable,'-I',str(ROOT/'util/deepseek_v41_telemetry.py'),
                '--pid',str(server.pid),'--output',str(out/'telemetry')],creationflags=subprocess.CREATE_NO_WINDOW)
            time.sleep(3)
            for i in range(2):
                cases=[(False,False),(True,False),(False,True),(True,True)]
                for cache,upload in (cases if i==0 else list(reversed(cases))):
                    run(f'monitored-r{i}-cache{int(cache)}-async{int(upload)}',candidate=cache,async_seed=upload)
            (out/'telemetry/stop').touch();monitor.wait(timeout=20);monitor=None
            for candidate in (False,True):
                run(f'profile-mask-cache{int(candidate)}',candidate=candidate,mode='host')
            run('profile-upload-only',async_seed=True,mode='host')
            run('profile-combined',candidate=True,async_seed=True,mode='host')
            run('profile-optimized-cpython',candidate=True,async_seed=True,profile=True)
            run('profile-optimized-cuda',candidate=True,async_seed=True,trace=True,tokens=256)
        elif candidate_controls:
            for i in range(a.pairs):
                for candidate in ([False,True] if i%2==0 else [True,False]):
                    run(f'steady-masked-r{i}-cache{int(candidate)}',candidate=candidate)
            for context in (('8k','65k') if a.long else ()):
                for candidate in (False,True):run(f'{context}-warm-cache{int(candidate)}',context=context,candidate=candidate)
                for i in range(2):
                    for candidate in ([False,True] if i%2==0 else [True,False]):
                        run(f'steady-{context}-r{i}-cache{int(candidate)}',context=context,candidate=candidate)
            for i in range(2):
                for candidate in ([False,True] if i%2==0 else [True,False]):
                    run(f'steady-ordinary-r{i}-cache{int(candidate)}',candidate=candidate,mtp=False,tokens=256)
                    run(f'steady-free-r{i}-cache{int(candidate)}',candidate=candidate,minimum=0)
                    run(f'steady-stock-r{i}-cache{int(candidate)}',candidate=candidate,ir=False,tokens=256)
        else:
            # Uninstrumented runs follow, alternating execution order.
            for i in range(a.pairs):
                for serial in ([True,False] if i%2==0 else [False,True]):
                    run(f'steady-free-r{i}-serial{int(serial)}',minimum=0,serial=serial)
                run(f'steady-masked-r{i}')
        save('final-status.json',dict(status=get('/__validation/status'),extra=get('/__followup/status')))
    finally:
        if monitor:
            (out/'telemetry/stop').touch()
            try:monitor.wait(timeout=20)
            except subprocess.TimeoutExpired:monitor.terminate();monitor.wait(timeout=10)
        if server.poll() is None:
            try:post('/__validation/shutdown',{})
            except httpx.HTTPError:pass
            try:server.wait(timeout=90)
            except subprocess.TimeoutExpired:server.terminate();server.wait(timeout=15)
        log.close()
        assert configs=={n:hashlib.sha256((TABBY/n).read_bytes()).hexdigest() for n in configs}
if __name__=='__main__':main()
