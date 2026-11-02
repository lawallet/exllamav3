"""Bounded CPU-only parallel computation / two-submitter policy stress cases."""
import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--package-dir',type=Path)
    p.add_argument('--implemented',type=int)
    a=p.parse_args()
    if a.package_dir:sys.path.insert(0,str(a.package_dir.resolve()))
    import torch
    if a.implemented is not None:
        from exllamav3.util.openmp import configure_openmp_for_current_thread,openmp_status
        assert os.environ['EXL3_OMP_BLOCKTIME_MS']==str(a.implemented)
    a.output.mkdir(parents=True,exist_ok=False)
    runtime=ctypes.CDLL('libiomp5md.dll')
    runtime.kmp_set_blocktime.argtypes=[ctypes.c_int]
    runtime.kmp_get_blocktime.restype=ctypes.c_int
    original=runtime.kmp_get_blocktime();threads=torch.get_num_threads()
    torch.manual_seed(1234)
    x=torch.randn(2048,2048);y=torch.randn(2048,2048)
    settings=[a.implemented] if a.implemented is not None else [200,0,1,50]
    results=[];reference=None
    def compute(value,kind):
        if a.implemented is not None:
            configure_openmp_for_current_thread()
        else:runtime.kmp_set_blocktime(value)
        assert runtime.kmp_get_blocktime()==value
        if kind=='matmul':
            z=x@y
            for _ in range(12):torch.mm(x,y,out=z)
        else:
            z=torch.empty_like(x)
            for _ in range(200):
                torch.add(x,y,out=z)
        return z
    for kind in ('matmul','elementwise'):
        for count in (1,2):
            reference=None
            for value in settings:
                if count==1:compute(value,kind)
                else:
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        list(pool.map(lambda _:compute(value,kind),range(2)))
            for repeat in range(3):
                for value in (settings if repeat%2==0 else list(reversed(settings))):
                    start=time.perf_counter();cpu=time.process_time()
                    if count==1:outputs=[compute(value,kind)]
                    else:
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            outputs=list(pool.map(lambda _:compute(value,kind),range(2)))
                    wall=time.perf_counter()-start;cpu=time.process_time()-cpu
                    if reference is None:reference=outputs[0].clone()
                    assert all(torch.equal(reference,z) for z in outputs),(kind,count,value)
                    assert torch.get_num_threads()==threads
                    row=dict(kind=kind,submitters=count,repeat=repeat,blocktime=value,wall_s=wall,cpu_s=cpu,
                             core_equivalents=cpu/wall,torch_threads=threads,exact=True)
                    results.append(row)
                    (a.output/'results.json').write_text(json.dumps(results,indent=2))
                    print(json.dumps(row),flush=True)
    if a.implemented is None:runtime.kmp_set_blocktime(original)
    (a.output/'complete.json').write_text(json.dumps(dict(exact=True,torch_threads=threads,original_blocktime=original)))


if __name__=='__main__':main()
