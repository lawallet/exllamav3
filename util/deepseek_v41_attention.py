"""Fresh attention diagnostics; all existing release policies remain enabled."""
import argparse,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
import httpx,yaml
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'native-results/deepseek-v41';TABBY=Path('D:/tabbyAPI')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--phase',choices=['profile','pairs','modes','audit','lifecycle','prefix','telemetry','ablation'],default='profile')
    ap.add_argument('--pairs',type=int,default=5);ap.add_argument('--long',action='store_true')
    ap.add_argument('--package-dir',type=Path);ap.add_argument('--implemented',action='store_true')
    ap.add_argument('--modes-tail',action='store_true',help='Skip the already-passed greedy MTP cases; repeat seeded controls')
    ap.add_argument('--seed',type=int,help='Verified backend seed for an entire paired diagnostic cohort')
    ap.add_argument('--minimum',type=int,default=512)
    ap.add_argument('--profile-candidate',action='store_true');ap.add_argument('--telemetry',action='store_true');a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    def save(name,data):(out/name).write_text(json.dumps(data,indent=2,default=str))
    for name in ('deepseek_v41_attention.py','serve_deepseek_v41_attention.py','deepseek_v41_projection_probe.py','serve_deepseek_v41_pipeline.py','serve_deepseek_v41_followup.py','serve_deepseek_v41_ir.py'):
        (out/name).write_bytes((ROOT/'util'/name).read_bytes())
    save('source.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)))
    old=json.loads((BASE/'generation-pipeline/installed-enabled/server.json').read_text())
    configs={n:hashlib.sha256((TABBY/n).read_bytes()).hexdigest() for n in old['config_hashes']}
    key=yaml.safe_load((TABBY/'api_tokens.yml').read_text())['admin_key']
    client=httpx.Client(base_url='http://127.0.0.1:5008',headers={'Authorization':'Bearer '+key},timeout=7200,trust_env=False)
    def get(path):r=client.get(path);r.raise_for_status();return r.json()
    def post(path,body):r=client.post(path,json=body);r.raise_for_status();return r.json()
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1);probe.bind(('127.0.0.1',5008))
    command=old['command'];command[2]=str(ROOT/'util/serve_deepseek_v41_attention.py')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,0,1',EXL3_NATIVE_PLAN='1',EXL3_OMP_BLOCKTIME_MS='0',
        EXL3_STOP_MASK_CACHE='1',EXL3_ASYNC_DRAFT_SEED='1',EXL3_NGRAM_STREAM='1',
        PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',DS41_RESULTS=str(out))
    for k in ('DS41_PACKAGE_ROOT','DS41_EXTENSION_DIR'):env.pop(k,None)
    if a.package_dir:
        env['DS41_PACKAGE_ROOT']=env['DS41_EXTENSION_DIR']=str(a.package_dir.resolve())
    env['DS41_ATTENTION_IMPLEMENTED']='1' if a.implemented else '0'
    env['EXL3_NATIVE_DSV41_PROJECTIONS']='1' if a.implemented else '0'
    log=(out/'server.log').open('w',encoding='utf-8')
    server=subprocess.Popen(command,cwd=TABBY,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    save('server.json',dict(pid=server.pid,command=command,config_hashes=configs,start_ns=time.time_ns(),
        environment={k:v for k,v in env.items() if k.startswith(('EXL3_','CUDA_VISIBLE','DS41_'))}))
    refs={};monitor=None
    try:
        deadline=time.monotonic()+420
        while time.monotonic()<deadline:
            if server.poll() is not None:raise RuntimeError('server exited')
            try:
                if get('/__validation/status')['model_loaded']:break
            except httpx.HTTPError:pass
            time.sleep(2)
        else:raise TimeoutError('model loading')
        startup=dict(status=get('/__validation/status'),extra=get('/__followup/status'),
                     attention=get('/__attention/results'))
        save('startup.json',startup)
        if a.implemented:
            assert startup['attention']['candidate']['enabled'] and startup['attention']['candidate']['available']
        from collections import Counter
        assert Counter(p['device'] for p in startup['status']['placement'] if p['attention'])=={'cuda:0':26,'cuda:1':14}
        assert startup['status']['generator']==dict(batch=1,drafts=3,dynamic=False,cache_tokens=65536)
        assert startup['status']['ngrams'] and all(n['mode']=='fp16_disk' for n in startup['status']['ngrams'])
        if a.telemetry:
            monitor=subprocess.Popen([sys.executable,'-I',str(ROOT/'util/deepseek_v41_telemetry.py'),
                '--pid',str(server.pid),'--output',str(out/'telemetry')],cwd=ROOT,env=env,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        def run(name,context='short',profile=False,candidate=False,minimum=None,mtp=True,seed=None,tokens=512,audit=False,native_graphs=True,trace=False):
            if seed is None:seed=a.seed
            if minimum is None:minimum=a.minimum
            before=post('/__validation/control',dict(name=name,mtp=mtp,ir=True,graphs=True,audit=audit,profile=False,trace=trace))
            post('/__followup/control',dict(mode='off',serial_readback=False))
            policy=post('/__attention/control',dict(profile=profile,candidate=candidate,native_graphs=native_graphs,seed=seed));extra=get('/__followup/status')
            assert policy['pipeline']['enabled'] and policy['pipeline']['async_seed']
            assert extra['blocktime']['milliseconds']==0 and extra['torch_threads']==56
            path=BASE/('graph-http/timing/prompt.json' if context=='short' else f'graph-http/context{context}/prompt.json')
            prompt=json.loads(path.read_text())['text'];start_ns=time.time_ns();start=time.perf_counter()
            response=post('/v1/completions',dict(model='DeepSeek-V4.1-Flash-1017-Heretic-EXL3-1.68bpw',prompt=prompt,
                max_tokens=tokens,min_tokens=minimum,temperature=0 if seed is None else 0.8,
                top_k=1 if seed is None else 40,top_p=1,min_p=0,repetition_penalty=1,
                frequency_penalty=0,presence_penalty=0,add_bos_token=False,token_healing=False,stream=False,
                **({} if seed is None else {'seed':seed})))
            wall=time.perf_counter()-start;end_ns=time.time_ns();result=get('/__validation/results')
            j,=result['state']['jobs'];signature=(j['ids'],j['accepted_draft_tokens'],j['rejected_draft_tokens'])
            if audit:signature=(*signature,result['state']['logits'])
            refkey=(context,minimum,mtp,seed,tokens,audit)
            after=get('/__attention/results')
            save(name+'.json',dict(name=name,context=context,tokens=j['new_tokens'],minimum=minimum,mtp=mtp,ir=True,
                request_start_ns=start_ns,response_end_ns=end_ns,http_wall=wall,before=before,extra_before=extra,
                result=result,response=response,extra_after=get('/__followup/status'),attention=policy,candidate=candidate,
                attention_after=after,native_graphs=native_graphs,seed=seed))
            assert refkey not in refs or signature==refs[refkey],(name,'token/MTP/forward mismatch')
            refs.setdefault(refkey,signature)
            assert (minimum!=tokens or j['new_tokens']==tokens) and j['cached_tokens']==0
            if seed is not None:assert after['observed_seeds']==[seed],after['observed_seeds']
            if a.implemented and candidate:
                assert after['candidate']['available'] and after['candidate']['replays']>0
                if not native_graphs:
                    assert after['candidate']['replays']==policy['candidate']['replays'],'unwarmed ablation shape'
            if profile:save(name+'-attention.json',get('/__attention/results'))
            print(json.dumps(dict(name=name,tps=j['new_tokens']/j['time_generate'],prefill=j['time_prefill'],accepted=j['accepted_draft_tokens'],rejected=j['rejected_draft_tokens'])),flush=True)
        if a.phase in ('lifecycle','prefix'):
            reference=None
            for candidate in (False,True):
                post('/__validation/control',dict(name='lifecycle',mtp=True,ir=True,graphs=True))
                post('/__attention/control',dict(candidate=candidate))
                prompt_path='graph-http/context8k/prompt.json' if a.phase=='prefix' else 'graph-http/timing/prompt.json'
                prompt=json.loads((BASE/prompt_path).read_text())['text']
                result=post('/__attention/lifecycle',dict(prompt=prompt,prefix_only=a.phase=='prefix'))
                save(f'lifecycle-candidate{int(candidate)}.json',result)
                assert reference is None or result['records']==reference
                reference=result['records']
                print(json.dumps(dict(candidate=candidate,lifecycle_exact=True)),flush=True)
        elif a.phase=='profile':
            for i in range(2):run(f'warm-short-{i}',candidate=a.profile_candidate)
            run('profile-short',profile=True,candidate=a.profile_candidate)
            run('warm-65k',context='65k',candidate=a.profile_candidate)
            run('profile-65k',context='65k',profile=True,candidate=a.profile_candidate)
            run('trace-short',tokens=128,minimum=128,trace=True,candidate=a.profile_candidate)
        elif a.phase=='ablation':
            assert a.implemented
            for i in range(2):run(f'warm-ablation-{i}',candidate=True)
            settings=[('baseline',False,True),('native-eager',True,False),('native-graph',True,True)]
            for i in range(3):
                order=settings[i:]+settings[:i]
                for label,candidate,graphs in order:
                    run(f'ablation-r{i}-{label}',candidate=candidate,native_graphs=graphs)
        elif a.phase=='telemetry':
            assert a.telemetry
            for candidate in (False,True):run(f'warm-monitored-candidate{int(candidate)}',candidate=candidate)
            for context in ('short','65k'):
                for i in range(2):
                    for candidate in ((False,True) if i%2==0 else (True,False)):
                        run(f'monitored-{context}-r{i}-candidate{int(candidate)}',context=context,candidate=candidate)
        elif a.phase=='audit':
            for context in ('short','65k'):
                for candidate in (False,True):
                    run(f'audit-{context}-candidate{int(candidate)}',context=context,candidate=candidate,
                        audit=True,tokens=64,minimum=64)
        elif a.phase=='modes':
            for mtp in (True,False):
                for minimum,seed in ((512,None),(0,None),(0,841)):
                    if a.modes_tail and mtp and seed is None:continue
                    settings=(False,False,True,True) if seed is not None else (False,True)
                    for repeat,candidate in enumerate(settings):
                        run(f'mode-mtp{int(mtp)}-min{minimum}-seed{seed}-candidate{int(candidate)}-repeat{repeat}',
                            candidate=candidate,minimum=minimum,mtp=mtp,seed=seed)
        else:
            for i in range(2):
                for candidate in (False,True):run(f'warm-r{i}-candidate{int(candidate)}',candidate=candidate)
            for i in range(a.pairs):
                for candidate in ((False,True) if i%2==0 else (True,False)):
                    run(f'steady-r{i}-candidate{int(candidate)}',candidate=candidate)
            for context in (('8k','65k') if a.long else ()):
                for candidate in (False,True):run(f'{context}-warm-candidate{int(candidate)}',context=context,candidate=candidate)
                for i in range(2):
                    for candidate in ((False,True) if i%2==0 else (True,False)):
                        run(f'steady-{context}-r{i}-candidate{int(candidate)}',context=context,candidate=candidate)
        save('final-status.json',dict(status=get('/__validation/status'),extra=get('/__followup/status')))
    finally:
        if monitor is not None:
            (out/'telemetry'/'stop').touch()
            try:monitor.wait(timeout=15)
            except subprocess.TimeoutExpired:monitor.terminate();monitor.wait(timeout=10)
        if server.poll() is None:
            try:post('/__validation/shutdown',{})
            except httpx.HTTPError:pass
            try:server.wait(timeout=60)
            except subprocess.TimeoutExpired:server.terminate();server.wait(timeout=15)
        log.close()
        assert configs=={n:hashlib.sha256((TABBY/n).read_bytes()).hexdigest() for n in configs}
if __name__=='__main__':main()
