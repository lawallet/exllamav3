"""Follow-up diagnostic cohorts; unchanged wheel, temporary server and monitors."""
import argparse, hashlib, json, os, socket, subprocess, sys, time
from pathlib import Path
import httpx,yaml
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'native-results/deepseek-v41';TABBY=Path('D:/tabbyAPI')
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['host','cpu','blocktime','omp-sweep','omp-repeat','omp-implemented','prefill76','prefill94','prefill94-small','prefill76-return'],required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--package-dir',type=Path)
    p.add_argument('--runtime-policy',type=int)
    p.add_argument('--pairs',type=int,default=5,help='Measured pairs for implemented-policy validation')
    p.add_argument('--skip-long',action='store_true',help='Short-only installed/unset control')
    p.add_argument('--monitor-implemented',action='store_true',help='Separate telemetry pair after uninstrumented implemented-policy timing')
    a=p.parse_args()
    if a.phase=='prefill76-return' and not (BASE/'followup/prefill94-small/final-status.json').exists():
        subprocess.run([sys.executable,'-I',str(Path(__file__).resolve()),'--phase','prefill94-small'],check=True,cwd=ROOT)
    out=(a.output or BASE/'followup'/a.phase).resolve();out.mkdir(parents=True,exist_ok=False)
    for script in ('deepseek_v41_experiments.py','serve_deepseek_v41_followup.py','deepseek_v41_telemetry.py','deepseek_v41_dxgi.py'):
        (out/script).write_bytes((ROOT/'util'/script).read_bytes())
    key=yaml.safe_load((TABBY/'api_tokens.yml').read_text())['admin_key']
    c=httpx.Client(base_url='http://127.0.0.1:5008',headers={'Authorization':'Bearer '+key},timeout=7200,trust_env=False)
    def post(path,body):r=c.post(path,json=body);r.raise_for_status();return r.json()
    def get(path):r=c.get(path);r.raise_for_status();return r.json()
    def save(path,obj):path.write_text(json.dumps(obj,indent=2))
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1);probe.bind(('127.0.0.1',5008))
    old=json.loads((BASE/'followup/fresh/process0/server.json').read_text())
    cmd=old['command'];cmd[2]=str(ROOT/'util/serve_deepseek_v41_followup.py')
    cmd[cmd.index('--gpu-split')+1]='94' if a.phase.startswith('prefill94') else '76'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,0,1',EXL3_NGRAM_STREAM='1',EXL3_NATIVE_PLAN='0',EXL3_NATIVE_MOE_ASSIGN='0',PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',DS41_RESULTS=str(out))
    for k in ('DS41_PACKAGE_ROOT','DS41_EXTENSION_DIR'):env.pop(k,None)
    env.pop('EXL3_OMP_BLOCKTIME_MS',None)
    if a.runtime_policy is not None:env['EXL3_OMP_BLOCKTIME_MS']=str(a.runtime_policy)
    if a.package_dir:
        env['DS41_PACKAGE_ROOT']=str(a.package_dir.resolve())
        env['DS41_EXTENSION_DIR']=str(a.package_dir.resolve())
    log=(out/'server.log').open('w',encoding='utf-8')
    server=subprocess.Popen(cmd,cwd=TABBY,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    save(out/'server.json',dict(pid=server.pid,command=cmd,start_ns=time.time_ns(),config_hashes=old['config_hashes']))
    monitor=None
    try:
        deadline=time.monotonic()+300
        while time.monotonic()<deadline:
            if server.poll() is not None:raise RuntimeError('Server exited; inspect server.log')
            try:
                status=get('/__validation/status')
                if status['model_loaded']:break
            except httpx.HTTPError:pass
            time.sleep(2)
        else:raise RuntimeError('startup timeout')
        save(out/'startup.json',dict(time_ns=time.time_ns(),status=status,extra=get('/__followup/status')))
        def run(name,context='short',tokens=512,ir=True,mode='off',profile=False,trace=False,empty=False,minimum=None,serial=False,threads=None,main_group=None,chunk=None,blocktime=None):
            before=post('/__validation/control',dict(name=name,mtp=True,ir=ir,graphs=True,audit=False,profile=profile,trace=trace))
            options=dict(mode=mode,empty_cache=empty,serial_readback=serial)
            if threads is not None:options['threads']=threads
            if main_group is not None:options['main_group']=main_group
            if chunk is not None:options['chunk_size']=chunk
            if blocktime is not None:options['blocktime']=blocktime
            extra=post('/__followup/control',options)
            if isinstance(blocktime,int):assert extra['blocktime']['milliseconds']==blocktime,extra['blocktime']
            if a.phase=='omp-implemented':
                assert blocktime is None,'implemented policy must not use experimental setter'
                expected=a.runtime_policy if a.runtime_policy is not None else 200
                assert extra['blocktime']['milliseconds']==expected,extra['blocktime']
            path=BASE/('graph-http/timing/prompt.json' if context=='short' else f'graph-http/context{context}/prompt.json')
            prompt=json.loads(path.read_text())['text']
            start_ns=time.time_ns();start=time.perf_counter()
            response=post('/v1/completions',dict(model='DeepSeek-V4.1-Flash-1017-Heretic-EXL3-1.68bpw',prompt=prompt,
                max_tokens=tokens,min_tokens=tokens if minimum is None else minimum,temperature=0,top_k=1,top_p=1,min_p=0,repetition_penalty=1,
                frequency_penalty=0,presence_penalty=0,add_bos_token=False,token_healing=False,stream=False))
            end_ns=time.time_ns();wall=time.perf_counter()-start;result=get('/__validation/results');job=result['state']['jobs'][0]
            save(out/(name+'.json'),dict(name=name,context=context,ir=ir,mode=mode,minimum=minimum,serial_readback=serial,empty_cache=empty,request_start_ns=start_ns,
                response_end_ns=end_ns,http_wall=wall,before=before,extra_before=extra,result=result,response=response,extra_after=get('/__followup/status')))
            print(json.dumps(dict(phase=a.phase,name=name,tps=tokens/job['time_generate'],prefill=job['time_prefill'],accepted=job['accepted_draft_tokens'],rejected=job['rejected_draft_tokens'])),flush=True)
        def pairs(prefix,n):
            for i in range(n):
                for ir in ([False,True] if i%2==0 else [True,False]):run(f'{prefix}-r{i}-ir{int(ir)}',ir=ir)
        def start_monitor():
            nonlocal monitor
            monitor=subprocess.Popen([sys.executable,'-I',str(ROOT/'util/deepseek_v41_telemetry.py'),'--pid',str(server.pid),'--output',str(out/'telemetry')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
            time.sleep(3)
            if monitor.poll() is not None:raise RuntimeError(monitor.stdout.read())
        if a.phase=='omp-implemented':
            pairs('warmup',2);pairs('steady',a.pairs)
            for context in (() if a.skip_long else ('8k','65k')):
                run(f'{context}-warmup',context=context,tokens=512)
                run(f'{context}-steady',context=context,tokens=512)
            if a.monitor_implemented:
                start_monitor();pairs('monitored',1)
        elif a.phase in ('omp-sweep','omp-repeat'):
            settings=[0,1,50] if a.phase=='omp-sweep' else [0,1]
            run('baseline-warmup',blocktime='restore')
            run('baseline-control',blocktime='restore')
            for bt in settings:run(f'warmup-block{bt}',blocktime=bt)
            for i in range(5 if a.phase=='omp-sweep' else 3):
                if len(settings)==2:
                    order=settings if i%2==0 else list(reversed(settings))
                else:
                    order=settings[i%len(settings):]+settings[:i%len(settings)]
                    if i%2:order=list(reversed(order))
                for bt in order:run(f'steady-r{i}-block{bt}',blocktime=bt)
            if a.phase=='omp-sweep':
                start_monitor()
                for i in range(2):
                    for bt in (settings if i%2==0 else list(reversed(settings))):run(f'monitored-r{i}-block{bt}',blocktime=bt)
                (out/'telemetry/stop').touch();monitor.wait(timeout=15);save(out/'monitor-exit.json',dict(code=monitor.returncode,output=monitor.stdout.read()));monitor=None
                for context in ('8k','65k'):
                    run(f'{context}-warmup',context=context,tokens=512,blocktime=0)
                    for i in range(2):
                        for bt in (settings if i%2==0 else list(reversed(settings))):run(f'{context}-r{i}-block{bt}',context=context,tokens=512,blocktime=bt)
            run('baseline-restored',blocktime='restore')
        elif a.phase=='host':
            pairs('warmup',2);pairs('unmonitored',2)
            start_monitor();pairs('monitored',3)
            (out/'telemetry/stop').touch();monitor.wait(timeout=15);save(out/'monitor-exit.json',dict(code=monitor.returncode,output=monitor.stdout.read()));monitor=None
            subprocess.run([sys.executable,'-I',str(ROOT/'util/deepseek_v41_followup.py'),'--existing','--processes','1','--pairs','3','--warmup-pairs','0','--client-torch','--output',str(out/'torch-client')],check=True,cwd=ROOT)
            pairs('post-client',2)
            run('native-host',mode='host');post('/__followup/control',dict(guard_microbench=True))
            run('stock-host',mode='host',ir=False)
            run('native-cprofile',profile=True)
            run('native-trace',trace=True,tokens=256)
            pairs('post-profile',1)
            for i in range(3):
                for minimum in ([512,0] if i%2==0 else [0,512]):run(f'readback-r{i}-min{minimum}',minimum=minimum)
            run('min0-host',minimum=0,mode='host')
            for i in range(3):
                for serial in ([True,False] if i%2==0 else [False,True]):run(f'min0-r{i}-serial{int(serial)}',minimum=0,serial=serial)
        elif a.phase=='cpu':
            run('warmup56');run('warmup1',threads=1)
            for i in range(3):
                for n in ([56,1] if i%2==0 else [1,56]):run(f'unmonitored-r{i}-threads{n}',threads=n)
            start_monitor()
            for i in range(3):
                cases=[(56,0),(56,1),(1,1),(1,0)]
                for n,g in (cases if i%2==0 else list(reversed(cases))):run(f'monitored-r{i}-threads{n}-group{g}',threads=n,main_group=g)
            (out/'telemetry/stop').touch();monitor.wait(timeout=15);save(out/'monitor-exit.json',dict(code=monitor.returncode,output=monitor.stdout.read()));monitor=None
            for i in range(2):
                for n in ([56,1] if i%2==0 else [1,56]):run(f'min0-r{i}-threads{n}',threads=n,minimum=0,main_group='restore')
            run('threads1-host',threads=1,mode='host',main_group='restore')
            run('threads1-cprofile',threads=1,profile=True)
            run('restore56',threads=56,main_group='restore')
            wpr_status=subprocess.run(['wpr','-status'],capture_output=True,text=True)
            save(out/'wpr-status.json',dict(code=wpr_status.returncode,stdout=wpr_status.stdout,stderr=wpr_status.stderr))
            if 'not recording' in (wpr_status.stdout+wpr_status.stderr).lower():
                start=subprocess.run(['wpr','-start',str(ROOT/'util/deepseek_v41_cpu.wprp')+'!DS41CPU','-filemode'],capture_output=True,text=True)
                save(out/'wpr-start.json',dict(code=start.returncode,stdout=start.stdout,stderr=start.stderr))
                if start.returncode==0:
                    try:run('native-etw',threads=56,tokens=128)
                    finally:
                        stop=subprocess.run(['wpr','-stop',str(out/'native-cpu.etl')],capture_output=True,text=True)
                        save(out/'wpr-stop.json',dict(code=stop.returncode,stdout=stop.stdout,stderr=stop.stderr))
                        if stop.returncode:
                            cancel=subprocess.run(['wpr','-cancel'],capture_output=True,text=True)
                            save(out/'wpr-cancel.json',dict(code=cancel.returncode,stdout=cancel.stdout,stderr=cancel.stderr))
            for context in ('8k','65k'):
                for n in (1,56):run(f'{context}-threads{n}',context=context,tokens=64,threads=n,mode='prefill')
        elif a.phase=='blocktime':
            run('warmup-default',blocktime='restore');run('warmup-zero',blocktime=0)
            for i in range(3):
                for bt in (['restore',0] if i%2==0 else [0,'restore']):run(f'unmonitored-r{i}-block{bt}',blocktime=bt)
            start_monitor()
            for i in range(3):
                for bt in (['restore',0] if i%2==0 else [0,'restore']):run(f'monitored-r{i}-block{bt}',blocktime=bt)
            (out/'telemetry/stop').touch();monitor.wait(timeout=15);save(out/'monitor-exit.json',dict(code=monitor.returncode,output=monitor.stdout.read()));monitor=None
            run('zero-host',blocktime=0,mode='host')
            for context in ('8k','65k'):
                run(f'{context}-warmup',context=context,tokens=64,blocktime='restore')
                for bt in ('restore',0):run(f'{context}-block{bt}',context=context,tokens=64,blocktime=bt,mode='prefill')
            run('restored-default',blocktime='restore')
        else:
            start_monitor()
            run('short-warmup',tokens=128)
            for context in ('8k','65k'):
                count=2 if a.phase=='prefill76-return' else 3 if a.phase=='prefill94-small' else 4
                for i in range(count):run(f'{context}-r{i}',context=context,tokens=64,mode='prefill',
                    chunk=2048 if a.phase=='prefill94-small' else None,empty=a.phase=='prefill94-small')
                if a.phase=='prefill94':
                    for i in range(3):run(f'{context}-empty-r{i}',context=context,tokens=64,mode='prefill',empty=True)
        save(out/'final-status.json',dict(status=get('/__validation/status'),extra=get('/__followup/status')))
    finally:
        if monitor:
            (out/'telemetry/stop').touch()
            try:monitor.wait(timeout=15)
            except subprocess.TimeoutExpired:monitor.terminate();monitor.wait(timeout=10)
            save(out/'monitor-exit.json',dict(code=monitor.returncode,output=monitor.stdout.read()))
        if server.poll() is None:
            try:post('/__validation/shutdown',{})
            except httpx.HTTPError:pass
            try:server.wait(timeout=90)
            except subprocess.TimeoutExpired:server.terminate();server.wait(timeout=15)
        log.close()
        current={f:hashlib.sha256((TABBY/f).read_bytes()).hexdigest() for f in old['config_hashes']}
        assert current==old['config_hashes']
if __name__=='__main__':main()
