"""Fresh-process real OpenMP API and cross-thread initialization checks (CPU only)."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]


def child(package, expected):
    sys.path.insert(0,str(package))
    import torch
    import exllamav3
    from exllamav3 import Model,Generator
    from exllamav3.util.openmp import openmp_status
    runtime=ctypes.CDLL('libiomp5md.dll')
    runtime.kmp_set_blocktime.argtypes=[ctypes.c_int]
    runtime.kmp_get_blocktime.restype=ctypes.c_int
    threads=torch.get_num_threads()
    before=openmp_status()
    assert before['actual_ms']==200 and not before['applied_on_thread'],before
    model=Model.__new__(Model)
    model.loaded_tp=False;model._get_prefetch_layers=[];model.modules=[];model.last_kv_module_idx=-1
    model.prepare_inputs=lambda x,p:x
    model.forward_ls=lambda x,p:x
    model.prefill_ls=lambda *args:args[0]
    outcomes=[];errors=[]
    def worker(entry):
        try:
            runtime.kmp_set_blocktime(200)
            if entry=='generator':
                from types import SimpleNamespace
                gen=Generator.__new__(Generator);gen.cache=SimpleNamespace(initialized=False)
                try:gen.iterate()
                except AssertionError as error:assert 'Cache tensors' in str(error)
            else:
                x=torch.ones(1,1,dtype=torch.long)
                assert torch.equal(getattr(model,entry)(x),x)
            status=openmp_status()
            assert status['actual_ms']==expected,status
            assert torch.get_num_threads()==threads
            outcomes.append(dict(entry=entry,status=status))
            if entry=='forward':
                runtime.kmp_set_blocktime(50)
                model.forward(torch.ones(1,1,dtype=torch.long))
                assert runtime.kmp_get_blocktime()==50,'setter repeated on hot path'
        except BaseException as error:errors.append(repr(error))
    for entry in ('forward','prefill','generator'):
        thread=threading.Thread(target=worker,args=(entry,));thread.start();thread.join(timeout=20)
        assert not thread.is_alive(),'initialization deadlock'
    assert not errors,errors
    assert runtime.kmp_get_blocktime()==200,'worker changed caller policy'
    return dict(package=exllamav3.__file__,expected=expected,threads=threads,before=before,outcomes=outcomes)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--package-dir',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--child',type=int)
    a=p.parse_args()
    if a.child is not None:
        result=child(a.package_dir.resolve(),a.child)
        a.output.write_text(json.dumps(result,indent=2));return
    a.output.mkdir(parents=True,exist_ok=False)
    for value in (None,0,1,50):
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='')
        env.pop('EXL3_OMP_BLOCKTIME_MS',None)
        if value is not None:env['EXL3_OMP_BLOCKTIME_MS']=str(value)
        name='unset' if value is None else str(value)
        command=[sys.executable,'-I',str(Path(__file__).resolve()),'--package-dir',str(a.package_dir.resolve()),
                 '--output',str(a.output/(name+'.json')),'--child',str(200 if value is None else value)]
        r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=90)
        (a.output/(name+'.log')).write_text(r.stdout+r.stderr)
        assert r.returncode==0,(name,r.stderr)
        print('PASS',name,flush=True)


if __name__=='__main__':main()
