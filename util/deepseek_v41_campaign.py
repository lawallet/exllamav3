"""Start/benchmark/stop the isolated, authenticated Tabby validation server."""
import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import subprocess
import socket
import sys
import time
import httpx
import yaml

ROOT=Path(__file__).resolve().parents[1]
TABBY=Path('D:/tabbyAPI')
MODEL='DeepSeek-V4.1-Flash-1017-Heretic-EXL3-1.68bpw'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['start','status','run','stop'])
    p.add_argument('--output',type=Path,default=ROOT/'native-results/deepseek-v41/baseline')
    p.add_argument('--split',type=float,nargs='+',default=[90,46,0])
    p.add_argument('--tokens',type=int,default=512)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--context',type=int,default=178)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--trace',action='store_true')
    p.add_argument('--audit',action='store_true')
    p.add_argument('--graphs',action='store_true')
    p.add_argument('--telemetry',action='store_true')
    p.add_argument('--package-root',type=Path)
    p.add_argument('--extension-dir',type=Path)
    p.add_argument('--ir',action='store_true')
    p.add_argument('--compare',action='store_true',help='Alternate stock/native with each MTP setting')
    p.add_argument('--compare-eager',action='store_true',help='Alternate eager and graph-enabled native execution')
    p.add_argument('--mtp',choices=['both','on','off'],default='both')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    key=yaml.safe_load((TABBY/'api_tokens.yml').read_text())['admin_key']
    with httpx.Client(base_url='http://127.0.0.1:5008',headers={'Authorization':'Bearer '+key},timeout=7200,trust_env=False) as c:
        def get(path):
            r=c.get(path);r.raise_for_status();return r.json()
        def post(path,body):
            r=c.post(path,json=body);r.raise_for_status();return r.json()
        def save(name,x): (out/name).write_text(json.dumps(x,indent=2),encoding='utf-8')
        if a.action=='start':
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                probe.bind(('127.0.0.1',5008))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,0,1',EXL3_NGRAM_STREAM='1',
                EXL3_NATIVE_PLAN='0',EXL3_NATIVE_MOE_ASSIGN='0',PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',DS41_RESULTS=str(out))
            if a.package_root: env['DS41_PACKAGE_ROOT']=str(a.package_root.resolve())
            if a.extension_dir: env['DS41_EXTENSION_DIR']=str(a.extension_dir.resolve())
            cmd=[sys.executable,'-I',str(ROOT/'util/serve_deepseek_v41_ir.py'),
                '--host','127.0.0.1','--port','5008','--model-name',MODEL,
                '--max-seq-len','65536','--cache-size','65536','--max-batch-size','1',
                '--gpu-split',*[str(v) for v in a.split], '--draft-gpu-split','90','0','0',
                '--draft-mode','mtp','--draft-num-tokens','3','--dynamic-draft','false',
                '--ngram-ram','false','--vision','true','--tensor-parallel','false',
                '--log-live-status','false','--log-generation-params','false',
                '--use-as-default','draft_mode','draft_num_tokens','dynamic_draft','max_batch_size','chunk_size']
            before={f:hashlib.sha256((TABBY/f).read_bytes()).hexdigest() for f in ('config.yml','api_tokens.yml')}
            with (out/'server.log').open('w',encoding='utf-8') as log:
                proc=subprocess.Popen(cmd,cwd=TABBY,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            save('server.json',dict(pid=proc.pid,command=cmd,config_hashes=before));print('STARTED',proc.pid,flush=True)
            return
        if a.action=='status':
            s=get('/__validation/status');save('status.json',s);print(json.dumps(s,indent=2));return
        if a.action=='stop': print(post('/__validation/shutdown',{}));return
        import torch
        from exllamav3 import Config,Tokenizer
        tok=Tokenizer.from_config(Config.from_directory(str(TABBY/'models'/MODEL)))
        if a.telemetry:
            log=(out/f'telemetry-c{a.context}.csv').open('w',encoding='utf-8')
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate',
                '--format=csv','-lms','500'],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            def close_monitor():
                if monitor.poll() is None:monitor.terminate();monitor.wait(timeout=10)
                log.close()
            atexit.register(close_monitor)
        head=tok.encode('<｜begin▁of▁sentence｜><｜User｜>Explain how computers execute programs. Use the following notes as background.\n',encode_special_tokens=True)
        filler=tok.encode('A program uses instructions, memory, data structures, functions, and operating system services.\n')
        if a.context>1024:
            filler=tok.encode('\n'.join(
                f'def transform_{i}(items):\n    """Stable unique values for task {i}."""\n    return list(dict.fromkeys(items))\n'
                for i in range(4000)))
        tail=tok.encode('\nWrite a detailed tutorial with Python examples, tracing execution, loops, memory, and common errors.<｜Assistant｜></think>',encode_special_tokens=True)
        n=a.context-head.shape[-1]-tail.shape[-1]
        assert n>=0
        ids=torch.cat((head,filler.repeat(1,(n+filler.shape[-1]-1)//filler.shape[-1])[:,:n],tail),dim=1)
        text=tok.decode(ids,decode_special_tokens=True)[0]
        save('prompt.json',dict(text=text,ids=ids.tolist(),roundtrip=tok.encode(text,encode_special_tokens=True).shape[-1]))
        modes=[False,True] if a.mtp=='both' else [a.mtp=='on']
        pairs=([(mtp,True,graphs) for mtp in modes for graphs in (False,True)] if a.compare_eager else
               [(mtp,ir,a.graphs) for mtp in modes for ir in ([False,True] if a.compare else [a.ir])])
        for repeat in range(a.repeats):
            for mtp,ir,graphs in (pairs if repeat%2==0 else list(reversed(pairs))):
                name=f'c{a.context}-r{repeat}-mtp{int(mtp)}-ir{int(ir)}'+('-profile' if a.profile else '')+('-trace' if a.trace else '')+('-audit' if a.audit else '')
                if a.compare_eager:name+=f'-g{int(graphs)}'
                assert not (out/(name+'.json')).exists(),name
                before=post('/__validation/control',dict(name=name,mtp=mtp,ir=ir,profile=a.profile,trace=a.trace,audit=a.audit,graphs=graphs))
                print('RUN',name,flush=True);start=time.perf_counter()
                body=dict(model=MODEL,prompt=text,max_tokens=a.tokens,min_tokens=a.tokens,
                    temperature=0,top_k=1,top_p=1,min_p=0,repetition_penalty=1,
                    frequency_penalty=0,presence_penalty=0,add_bos_token=False,token_healing=False,stream=False)
                response=post('/v1/completions',body)
                result=get('/__validation/results')
                assert len(result['state']['jobs'])==1 and result['state']['jobs'][0]['new_tokens']==a.tokens,result['state']['jobs']
                save(name+'.json',dict(before=before,result=result,response=response,wall=time.perf_counter()-start))
                print('DONE',name,[{k:v for k,v in j.items() if k not in ('ids','memory')} for j in result['state']['jobs']],flush=True)

if __name__=='__main__': main()
